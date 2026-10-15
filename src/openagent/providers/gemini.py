"""Google Gemini REST API adapter.

Speaks the native Generative Language API (v1beta) using Server-Sent Events (SSE).
Handles message conversion to contents/parts, function declarations, and streaming
candidate parts back to normalized StreamEvents.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from typing import Any
from urllib.parse import urlencode

import httpx

from ..core.events import (
    DoneEvent,
    ErrorEvent,
    StartEvent,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageEvent,
)
from ..core.provider import (
    ChatProvider,
    ProviderError,
    coerce_arguments,
    render_system_prompt,
)
from ..core.types import (
    ChatRequest,
    FinishReason,
    ImagePart,
    Message,
    ModelInfo,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolProtocol,
    Usage,
)
from ..utils.http import HttpTransport
from ..utils.sse import iter_sse


class GeminiProvider(ChatProvider):
    """Native adapter for Google's Gemini Generative Language REST API."""

    name = "gemini"
    tool_protocol = ToolProtocol.NATIVE
    supports_vision = True
    supports_reasoning = True
    supports_native_tool_results = True
    supports_system_role = False
    default_context_window = 1_000_000

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        api_key: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        extra_headers: Mapping[str, str] | None = None,
        context_window: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
        client: Any = None,
    ) -> None:
        self.extra_body = dict(extra_body or {})
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.context_window = context_window or 1_000_000

        headers = dict(extra_headers or {})
        if api_key:
            headers["x-goog-api-key"] = api_key

        self._transport = HttpTransport(
            self.base_url,
            provider=self.name,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

    @property
    def transport(self) -> HttpTransport:
        return self._transport

    # -- request rendering -------------------------------------------------- #

    def build_payload(self, request: ChatRequest) -> dict[str, Any]:
        system_text = render_system_prompt(request)

        contents: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "system":
                continue

            if msg.role == "tool":
                part: dict[str, Any] = {
                    "functionResponse": {
                        "name": msg.name or msg.tool_call_id or "function",
                        "response": {"output": msg.tool_text()},
                    }
                }
                if contents and contents[-1]["role"] == "user":
                    contents[-1]["parts"].append(part)
                else:
                    contents.append({"role": "user", "parts": [part]})
                continue

            role = "model" if msg.role == "assistant" else "user"
            if role == "model" and msg.metadata.get("gemini_parts"):
                contents.append({"role": role, "parts": deepcopy(msg.metadata["gemini_parts"])})
                continue
            parts: list[dict[str, Any]] = []

            if msg.reasoning:
                parts.append({"text": msg.reasoning, "thought": True})

            for p in msg.content:
                match p:
                    case TextPart(text=t) if t:
                        parts.append({"text": t})
                    case ThinkingPart(text=t) if t:
                        parts.append({"text": t, "thought": True})
                    case ImagePart(data=d, mime_type=m):
                        if d and m:
                            parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": m,
                                        "data": d,
                                    }
                                }
                            )

            for call in msg.tool_calls:
                parts.append(
                    {
                        "functionCall": {
                            "name": call.name,
                            "args": call.arguments or {},
                        }
                    }
                )

            if parts:
                if contents and contents[-1]["role"] == role:
                    contents[-1]["parts"].extend(parts)
                else:
                    contents.append({"role": role, "parts": parts})

        payload: dict[str, Any] = {
            "contents": contents,
        }

        if system_text:
            payload["system_instruction"] = {"parts": [{"text": system_text}]}

        generation_config: dict[str, Any] = {}
        if request.response_format == "json_object":
            generation_config["responseMimeType"] = "application/json"
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.top_p is not None:
            generation_config["topP"] = request.top_p
        if request.max_tokens is not None or self.default_max_tokens is not None:
            generation_config["maxOutputTokens"] = (
                request.max_tokens if request.max_tokens is not None else self.default_max_tokens
            )
        if request.stop:
            generation_config["stopSequences"] = list(request.stop)
        if generation_config:
            payload["generationConfig"] = generation_config

        if request.tools:
            payload["tools"] = [
                {"function_declarations": [spec.to_gemini_schema() for spec in request.tools]}
            ]
            if request.tool_choice == "required":
                payload["tool_config"] = {"function_calling_config": {"mode": "ANY"}}
            elif request.tool_choice == "none":
                payload.pop("tools", None)

        payload.update(self.extra_body)
        return payload

    # -- streaming ---------------------------------------------------------- #

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        payload = self.build_payload(request)
        model_name = self.model
        model_path = model_name if model_name.startswith("models/") else f"models/{model_name}"

        url = f"/{model_path}:streamGenerateContent?alt=sse"

        try:
            response = await self._transport.stream_post(url, payload)
        except ProviderError as exc:
            yield ErrorEvent(error=exc)
            return

        try:
            yield StartEvent(model=request.model, provider=self.name)

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            calls: list[ToolCall] = []
            response_parts: list[dict[str, Any]] = []
            usage = Usage()
            finish_reason = FinishReason.STOP
            terminal_received = False

            async for event in iter_sse(response):
                if event.data == "[DONE]":
                    break
                if not event.data.strip():
                    continue

                try:
                    chunk = json.loads(event.data)
                except json.JSONDecodeError:
                    continue

                if "error" in chunk:
                    err = chunk["error"]
                    msg = err.get("message", "Unknown error") if isinstance(err, dict) else str(err)
                    yield ErrorEvent(error=ProviderError(msg, provider=self.name))
                    return

                if "usageMetadata" in chunk:
                    meta = chunk["usageMetadata"]
                    candidate_tokens = int(
                        meta.get(
                            "candidatesTokenCount", usage.completion_tokens - usage.reasoning_tokens
                        )
                    )
                    reasoning_tokens = int(meta.get("thoughtsTokenCount", usage.reasoning_tokens))
                    usage = Usage(
                        prompt_tokens=int(meta.get("promptTokenCount", usage.prompt_tokens)),
                        completion_tokens=candidate_tokens + reasoning_tokens,
                        reasoning_tokens=reasoning_tokens,
                        cached_tokens=int(meta.get("cachedContentTokenCount", usage.cached_tokens)),
                    )
                    yield UsageEvent(usage=usage)

                candidates = chunk.get("candidates") or []
                for candidate in candidates:
                    cand_finish = candidate.get("finishReason")
                    if cand_finish:
                        terminal_received = True
                        match cand_finish:
                            case "STOP":
                                finish_reason = FinishReason.STOP
                            case "MAX_TOKENS":
                                finish_reason = FinishReason.LENGTH
                            case (
                                "SAFETY"
                                | "RECITATION"
                                | "BLOCKLIST"
                                | "PROHIBITED_CONTENT"
                                | "SPII"
                            ):
                                finish_reason = FinishReason.CONTENT_FILTER
                            case _:
                                finish_reason = FinishReason.STOP

                    content = candidate.get("content") or {}
                    parts = content.get("parts") or []
                    for part in parts:
                        if set(part) == {"thoughtSignature"} and response_parts:
                            response_parts[-1]["thoughtSignature"] = part["thoughtSignature"]
                        else:
                            response_parts.append(deepcopy(part))
                        if "text" in part:
                            txt = part["text"]
                            if part.get("thought"):
                                reasoning_parts.append(txt)
                                yield ThinkingDelta(text=txt, index=len(reasoning_parts) - 1)
                            else:
                                text_parts.append(txt)
                                yield TextDelta(text=txt, index=len(text_parts) - 1)

                        if "functionCall" in part:
                            if request.tool_choice == "none":
                                raise ProviderError(
                                    "provider returned forbidden tool calls", provider=self.name
                                )
                            fc = part["functionCall"]
                            call_name = fc.get("name", "")
                            args = fc.get("args") or {}
                            call_id = f"call_{len(calls)}"
                            raw_args = json.dumps(args) if isinstance(args, dict) else str(args)
                            parsed_args = coerce_arguments(args)

                            yield ToolCallStart(index=len(calls), id=call_id, name=call_name)
                            yield ToolCallDelta(index=len(calls), arguments_delta=raw_args)
                            call = ToolCall(
                                id=call_id,
                                name=call_name,
                                arguments=parsed_args,
                                raw_arguments=raw_args,
                            )
                            yield ToolCallEnd(index=len(calls), call=call)
                            calls.append(call)

            if not terminal_received:
                raise ProviderError("stream ended before finishReason", provider=self.name)
            if calls:
                finish_reason = FinishReason.TOOL_CALLS

            assistant_message = Message.assistant(
                "".join(text_parts),
                tool_calls=calls,
                reasoning="".join(reasoning_parts) or None,
            )
            assistant_message.metadata["gemini_parts"] = response_parts
            yield DoneEvent(
                finish_reason=finish_reason,
                message=assistant_message,
                usage=usage,
            )
        except httpx.HTTPError as exc:
            error = ProviderError(f"network error: {exc}", retryable=True, provider=self.name)
            yield ErrorEvent(error=error, retryable=True)
        except ProviderError as exc:
            yield ErrorEvent(error=exc, retryable=exc.retryable)
        finally:
            await response.aclose()

    async def list_models(self) -> list[ModelInfo]:
        items: list[Any] = []
        cursor = None
        seen: set[str] = set()
        while True:
            path = "/models" + ("?" + urlencode({"pageToken": cursor}) if cursor else "")
            data = await self._transport.get_json(path)
            items.extend(data.get("models", []))
            cursor = data.get("nextPageToken")
            if not cursor:
                break
            if cursor in seen:
                raise ProviderError("repeated model discovery cursor", provider=self.name)
            seen.add(cursor)
        return [
            ModelInfo(
                id=m.get("name", "").removeprefix("models/"),
                provider=self.name,
                context_window=m.get("inputTokenLimit"),
                max_output=m.get("outputTokenLimit"),
            )
            for m in items
            if isinstance(m, dict)
        ]

    async def close(self) -> None:
        await self._transport.aclose()
