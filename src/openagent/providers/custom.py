"""Custom JSONPath HTTP provider adapter.

Extracts text deltas, tool calls, and token usage from arbitrary HTTP JSON / SSE / NDJSON
APIs using configurable jsonpath-ng expressions.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
import jsonpath_ng

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
)
from ..core.types import (
    ChatRequest,
    FinishReason,
    Message,
    ModelInfo,
    ToolCall,
    ToolProtocol,
    Usage,
)
from ..utils.http import HttpTransport
from ..utils.sse import iter_sse


class CustomJsonPathProvider(ChatProvider):
    """Adapter for arbitrary JSON APIs using jsonpath-ng expressions."""

    name = "custom"
    tool_protocol = ToolProtocol.JSON_SCHEMA
    supports_vision = False
    supports_reasoning = True
    supports_native_tool_results = True
    supports_system_role = True
    default_context_window = 128_000

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        text_path: str | None = "choices[0].delta.content",
        thinking_path: str | None = None,
        tool_calls_path: str | None = None,
        usage_prompt_path: str | None = None,
        usage_completion_path: str | None = None,
        chat_endpoint: str = "/chat/completions",
        api_key: str | None = None,
        auth_header: str = "authorization",
        auth_prefix: str = "Bearer ",
        headers: Mapping[str, str] | None = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        context_window: int | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.chat_endpoint = chat_endpoint
        self.context_window = context_window or 128_000

        self.text_path = text_path
        self.thinking_path = thinking_path
        self.tool_calls_path = tool_calls_path
        self.usage_prompt_path = usage_prompt_path
        self.usage_completion_path = usage_completion_path

        self._text_expr = jsonpath_ng.parse(text_path) if text_path else None
        self._thinking_expr = jsonpath_ng.parse(thinking_path) if thinking_path else None
        self._tool_calls_expr = jsonpath_ng.parse(tool_calls_path) if tool_calls_path else None
        self._usage_prompt_expr = (
            jsonpath_ng.parse(usage_prompt_path) if usage_prompt_path else None
        )
        self._usage_completion_expr = (
            jsonpath_ng.parse(usage_completion_path) if usage_completion_path else None
        )

        request_headers = dict(headers or {})
        if api_key and auth_header:
            header_name = auth_header.lower()
            if header_name == "authorization":
                request_headers["Authorization"] = f"{auth_prefix}{api_key}"
            else:
                request_headers[header_name] = api_key

        self._transport = HttpTransport(
            self.base_url,
            provider=self.name,
            headers=request_headers,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

    @property
    def transport(self) -> HttpTransport:
        return self._transport

    # -- chunk parsing ------------------------------------------------------ #

    def parse_chunk(
        self, chunk: dict[str, Any], partials: dict[int, dict[str, str]] | None = None
    ) -> list[StreamEvent]:
        """Tolerantly extract events from a parsed JSON chunk."""
        events: list[StreamEvent] = []

        # 1. Text deltas
        if self._text_expr:
            try:
                matches = self._text_expr.find(chunk)
                for m in matches:
                    if m.value is not None:
                        val = str(m.value)
                        if val:
                            events.append(TextDelta(text=val))
            except Exception:
                pass

        # 2. Tool calls
        if self._thinking_expr:
            for match in self._thinking_expr.find(chunk):
                if match.value is not None and str(match.value):
                    events.append(ThinkingDelta(text=str(match.value)))

        if self._tool_calls_expr:
            try:
                matches = self._tool_calls_expr.find(chunk)
                call_number = 0
                for match in matches:
                    values = match.value if isinstance(match.value, list) else [match.value]
                    for c in values:
                        if not isinstance(c, dict):
                            continue
                        fn = c.get("function") or c
                        index = int(c.get("index", call_number))
                        call_number += 1
                        name = fn.get("name", "")
                        args = fn.get("arguments", fn.get("args", ""))
                        raw = (
                            json.dumps(args) if isinstance(args, (dict, list)) else str(args or "")
                        )
                        call_id = c.get("id") or f"call_{index}"
                        if partials is not None:
                            slot = partials.setdefault(index, {"id": "", "name": "", "args": ""})
                            if name and not slot["name"]:
                                events.append(ToolCallStart(index=index, id=call_id, name=name))
                            slot["name"] += name
                            if c.get("id"):
                                slot["id"] = call_id
                            slot["args"] += raw
                            if raw:
                                events.append(ToolCallDelta(index=index, arguments_delta=raw))
                        else:
                            events.append(ToolCallStart(index=index, id=call_id, name=name))
                            events.append(ToolCallDelta(index=index, arguments_delta=raw))
                            call = ToolCall(
                                name=name,
                                arguments=coerce_arguments(raw),
                                id=call_id,
                                raw_arguments=raw,
                            )
                            events.append(ToolCallEnd(index=index, call=call))
            except (ProviderError, ValueError, TypeError) as exc:
                events.append(
                    ErrorEvent(
                        error=ProviderError(f"could not parse tool call: {exc}", provider=self.name)
                    )
                )

        # 3. Usage
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        if self._usage_prompt_expr:
            try:
                matches = self._usage_prompt_expr.find(chunk)
                if matches and matches[0].value is not None:
                    prompt_tokens = int(matches[0].value)
            except Exception:
                pass

        if self._usage_completion_expr:
            try:
                matches = self._usage_completion_expr.find(chunk)
                if matches and matches[0].value is not None:
                    completion_tokens = int(matches[0].value)
            except Exception:
                pass

        if prompt_tokens is not None or completion_tokens is not None:
            events.append(
                UsageEvent(
                    usage=Usage(
                        prompt_tokens=prompt_tokens or 0,
                        completion_tokens=completion_tokens or 0,
                    )
                )
            )

        return events

    # -- streaming ---------------------------------------------------------- #

    @staticmethod
    def _render_message(message: Message) -> dict[str, Any]:
        out: dict[str, Any] = {"role": message.role, "content": message.text}
        if message.role == "assistant" and message.tool_calls:
            out["content"] = message.text or None
            out["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": c.raw_arguments or json.dumps(c.arguments),
                    },
                }
                for c in message.tool_calls
            ]
        if message.role == "tool":
            out["tool_call_id"] = message.tool_call_id or ""
        return out

    def build_payload(self, request: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [self._render_message(m) for m in request.messages],
            "stream": True,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = [t.to_openai_schema() for t in request.tools]
        return payload

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        payload = self.build_payload(request)
        try:
            response = await self._transport.stream_post(self.chat_endpoint, payload)
        except ProviderError as exc:
            yield ErrorEvent(error=exc)
            return

        try:
            yield StartEvent(model=request.model, provider=self.name)

            text_parts: list[str] = []
            thinking_parts: list[str] = []
            calls: list[ToolCall] = []
            partials: dict[int, dict[str, str]] = {}
            usage = Usage()

            content_type = response.headers.get("content-type", "").lower()
            is_sse = "text/event-stream" in content_type

            if is_sse:
                async for event in iter_sse(response):
                    if event.data == "[DONE]":
                        break
                    if not event.data.strip():
                        continue
                    try:
                        chunk = json.loads(event.data)
                    except json.JSONDecodeError:
                        continue
                    for ev in self.parse_chunk(chunk, partials):
                        if isinstance(ev, ErrorEvent):
                            yield ev
                            return
                        match ev:
                            case ThinkingDelta(text=t):
                                thinking_parts.append(t)
                                yield ev
                            case TextDelta(text=t):
                                text_parts.append(t)
                                yield ev
                            case ToolCallEnd(call=c) if c is not None:
                                calls.append(c)
                                yield ev
                            case UsageEvent(usage=u):
                                usage = usage + u
                                yield ev
                            case _:
                                yield ev
            else:
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                        if line == "[DONE]":
                            break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for ev in self.parse_chunk(chunk, partials):
                        if isinstance(ev, ErrorEvent):
                            yield ev
                            return
                        match ev:
                            case ThinkingDelta(text=t):
                                thinking_parts.append(t)
                                yield ev
                            case TextDelta(text=t):
                                text_parts.append(t)
                                yield ev
                            case ToolCallEnd(call=c) if c is not None:
                                calls.append(c)
                                yield ev
                            case UsageEvent(usage=u):
                                usage = usage + u
                                yield ev
                            case _:
                                yield ev

            for index, slot in sorted(partials.items()):
                call = ToolCall(
                    slot["name"],
                    coerce_arguments(slot["args"]),
                    id=slot["id"] or f"call_{index}",
                    raw_arguments=slot["args"],
                )
                calls.append(call)
                yield ToolCallEnd(index=index, call=call)
            finish = FinishReason.TOOL_CALLS if calls else FinishReason.STOP
            yield DoneEvent(
                finish_reason=finish,
                message=Message.assistant(
                    "".join(text_parts),
                    tool_calls=calls,
                    reasoning="".join(thinking_parts) or None,
                ),
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
        return []

    async def close(self) -> None:
        await self._transport.aclose()
