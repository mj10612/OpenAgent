"""Anthropic Messages API adapter.

Differs from the OpenAI shape in several ways that matter:
- system is a top-level parameter, not a message
- tool results are ``user`` messages containing ``tool_result`` blocks
- assistant turns carry thinking blocks that must be echoed back verbatim
- content is a typed block array rather than a string
- the API version header is mandatory

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

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
from ..core.provider import ChatProvider, ProviderError, coerce_arguments, content_to_plain
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

API_VERSION = "2023-06-01"


class AnthropicProvider(ChatProvider):
    """Native adapter for Anthropic's Messages API."""

    name = "anthropic"
    tool_protocol = ToolProtocol.NATIVE
    supports_vision = True
    supports_reasoning = True
    supports_native_tool_results = True
    default_context_window = 200_000

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.anthropic.com",
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
        self.context_window = context_window or 200_000

        headers = {
            "anthropic-version": API_VERSION,
            **dict(extra_headers or {}),
        }
        if api_key:
            headers["x-api-key"] = api_key

        self._transport = HttpTransport(
            self.base_url,
            provider=self.name,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

    # -- request rendering -------------------------------------------------- #

    def _render_content(self, msg: Message) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []

        # Thinking blocks must precede text and be echoed unmodified.
        # Legacy unsigned reasoning cannot be replayed as a native thinking block.

        for part in msg.content:
            match part:
                case ThinkingPart(text=t, signature=s) if s:
                    block: dict[str, Any] = {"type": "thinking", "thinking": t}
                    if s:
                        block["signature"] = s
                    blocks.append(block)
                case ImagePart(data=d, mime_type=m, url=u):
                    if d:
                        blocks.append(
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": m, "data": d},
                            }
                        )
                    elif u:
                        blocks.append({"type": "image", "source": {"type": "url", "url": u}})
                case TextPart(text=t) if t:
                    blocks.append({"type": "text", "text": t})

        return blocks

    def _render_tool_results(self, msgs: Sequence[Message]) -> list[dict[str, Any]]:
        """Group consecutive tool results into one ``user`` turn.

        The API requires every tool_result for a parallel batch to appear in a
        single user message, otherwise the request is rejected.
        """
        blocks: list[dict[str, Any]] = []
        for msg in msgs:
            if msg.role != "tool":
                continue
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.tool_text(),
                    **({"is_error": True} if msg.is_error else {}),
                }
            )
        return blocks

    def build_payload(self, request: ChatRequest) -> dict[str, Any]:
        system = "\n\n".join(m.text for m in request.messages if m.role == "system" and m.text)

        wire: list[dict[str, Any]] = []
        index = 0
        msgs = [m for m in request.messages if m.role != "system"]
        while index < len(msgs):
            msg = msgs[index]

            if msg.role == "tool":
                batch: list[dict[str, Any]] = []
                while index < len(msgs) and msgs[index].role == "tool":
                    batch.extend(self._render_tool_results([msgs[index]]))
                    index += 1
                if batch:
                    wire.append({"role": "user", "content": batch})
                continue

            role = "assistant" if msg.role == "assistant" else "user"
            if role == "assistant" and msg.metadata.get("anthropic_content"):
                wire.append({"role": role, "content": msg.metadata["anthropic_content"]})
                index += 1
                continue
            content = self._render_content(msg)
            for call in msg.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments or {},
                    }
                )
            if content:
                wire.append({"role": role, "content": content})
            index += 1

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": wire,
            "max_tokens": request.max_tokens
            if request.max_tokens is not None
            else (self.default_max_tokens or 8192),
        }
        if system:
            payload["system"] = system
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop

        if request.tools:
            payload["tools"] = [spec.to_anthropic_schema() for spec in request.tools]
            if request.tool_choice == "required":
                payload["tool_choice"] = {"type": "any"}
            elif request.tool_choice == "none":
                payload.pop("tools")

        if request.thinking.enabled:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": request.thinking.budget_tokens or 4096,
            }
            # Extended thinking requires the default temperature.
            payload.pop("temperature", None)
            payload.pop("top_p", None)

        payload.update(self.extra_body)
        return payload

    # -- streaming ---------------------------------------------------------- #

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        payload = self.build_payload(request)
        payload["stream"] = True

        try:
            response = await self._transport.stream_post("/v1/messages", payload)
        except ProviderError as exc:
            yield ErrorEvent(error=exc)
            return

        try:
            yield StartEvent(model=request.model, provider=self.name)

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            calls: list[ToolCall] = []
            # Tool blocks stream in three phases: start (id/name), delta (args),
            # stop. These tables live in this scope on purpose — module-level state
            # would be shared across concurrent sessions.
            partial_args: dict[int, str] = {}
            pending_names: dict[int, str] = {}
            pending_ids: dict[int, str] = {}
            content_blocks: dict[int, TextPart | ThinkingPart] = {}
            native_blocks: dict[int, dict[str, Any]] = {}
            stop_reason = "end_turn"
            usage = Usage()

            try:
                async for event in iter_sse(response):
                    if event.data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(event.data)
                    except json.JSONDecodeError:
                        continue

                    kind = chunk.get("type")
                    data = chunk.get("data") if isinstance(chunk.get("data"), dict) else chunk

                    match kind:
                        case "message_start":
                            msg_usage = (data.get("message") or {}).get("usage") or {}
                            usage = Usage(
                                prompt_tokens=int(msg_usage.get("input_tokens", 0)),
                                completion_tokens=int(msg_usage.get("output_tokens", 0)),
                                cached_tokens=int(msg_usage.get("cache_read_input_tokens", 0)),
                            )

                        case "content_block_start":
                            block = data.get("content_block") or {}
                            block_index = data.get("index", 0)
                            native_blocks[block_index] = dict(block)
                            if block.get("type") == "thinking":
                                content_blocks[block_index] = ThinkingPart(
                                    block.get("thinking", ""), block.get("signature")
                                )
                            elif block.get("type") == "text":
                                content_blocks[block_index] = TextPart(block.get("text", ""))
                            if block.get("type") == "tool_use":
                                block_index = data.get("index", len(partial_args))
                                partial_args[block_index] = ""
                                pending_names[block_index] = block.get("name", "")
                                pending_ids[block_index] = (
                                    block.get("id", "") or f"call_{block_index}"
                                )
                                yield ToolCallStart(
                                    index=block_index,
                                    id=pending_ids[block_index],
                                    name=pending_names[block_index],
                                )

                        case "content_block_delta":
                            delta = data.get("delta") or {}
                            block_index = data.get("index", 0)
                            match delta.get("type"):
                                case "text_delta":
                                    chunk_text = delta.get("text", "")
                                    text_parts.append(chunk_text)
                                    block = native_blocks.setdefault(
                                        block_index, {"type": "text", "text": ""}
                                    )
                                    block["text"] = block.get("text", "") + chunk_text
                                    part = content_blocks.setdefault(block_index, TextPart(""))
                                    part.text += chunk_text
                                    yield TextDelta(text=chunk_text)
                                case "thinking_delta":
                                    chunk_text = delta.get("thinking", "")
                                    reasoning_parts.append(chunk_text)
                                    block = native_blocks.setdefault(
                                        block_index, {"type": "thinking", "thinking": ""}
                                    )
                                    block["thinking"] = block.get("thinking", "") + chunk_text
                                    part = content_blocks.setdefault(block_index, ThinkingPart(""))
                                    part.text += chunk_text
                                    yield ThinkingDelta(text=chunk_text)
                                case "input_json_delta":
                                    json_delta = delta.get("partial_json", "")
                                    partial_args[block_index] = (
                                        partial_args.get(block_index, "") + json_delta
                                    )
                                    yield ToolCallDelta(
                                        index=block_index, arguments_delta=json_delta
                                    )
                                case "signature_delta":
                                    part = content_blocks.setdefault(block_index, ThinkingPart(""))
                                    if isinstance(part, ThinkingPart):
                                        part.signature = (part.signature or "") + delta.get(
                                            "signature", ""
                                        )
                                        native_blocks[block_index]["signature"] = part.signature

                        case "content_block_stop":
                            block_index = data.get("index", 0)
                            if block_index in partial_args:
                                call = _finalise_call(
                                    partial_args.pop(block_index),
                                    block_index,
                                    pending_names.pop(block_index, ""),
                                    pending_ids.pop(block_index, f"call_{block_index}"),
                                )
                                calls.append(call)
                                native_blocks[block_index]["input"] = call.arguments
                                yield ToolCallEnd(index=block_index, call=call)

                        case "message_delta":
                            delta = data.get("delta") or {}
                            if stop := delta.get("stop_reason"):
                                stop_reason = stop
                            if delta_usage := data.get("usage"):
                                usage = Usage(
                                    prompt_tokens=usage.prompt_tokens,
                                    completion_tokens=int(
                                        delta_usage.get("output_tokens", usage.completion_tokens)
                                    ),
                                    cached_tokens=usage.cached_tokens,
                                )

                        case "message_stop":
                            pass

                        case "error":
                            err = data.get("error") or {}
                            yield ErrorEvent(
                                error=ProviderError(err.get("message", "stream error"))
                            )
                            return

                        case _:
                            continue

            except ProviderError as exc:
                yield ErrorEvent(error=exc, retryable=exc.retryable)
                return
            except Exception as exc:
                yield ErrorEvent(error=exc)
                return

            # A tool block may be open at the moment the stream ends.
            for block_index, raw in sorted(partial_args.items()):
                call = _finalise_call(
                    raw,
                    block_index,
                    pending_names.get(block_index, ""),
                    pending_ids.get(block_index, f"call_{block_index}"),
                )
                calls.append(call)
                native_blocks[block_index]["input"] = call.arguments
                yield ToolCallEnd(index=block_index, call=call)
            partial_args.clear()

            yield UsageEvent(usage=usage)
            yield DoneEvent(
                finish_reason=_map_stop(stop_reason, bool(calls)),
                usage=usage,
                message=Message(
                    role="assistant",
                    content=[part for _, part in sorted(content_blocks.items())],
                    tool_calls=calls,
                    reasoning="".join(reasoning_parts) or None,
                    metadata={
                        "anthropic_content": [block for _, block in sorted(native_blocks.items())]
                    },
                ),
            )
        finally:
            await response.aclose()

    async def list_models(self) -> list[ModelInfo]:
        body = await self._transport.get_json("/v1/models")
        return [
            ModelInfo(
                id=entry.get("id", ""),
                provider=self.name,
                context_window=200_000,
                max_output=8192,
                supports_vision=True,
                supports_tools=True,
            )
            for entry in body.get("data") or ()
            if entry.get("id")
        ]

    async def close(self) -> None:
        await self._transport.aclose()


def _finalise_call(raw: str, index: int, name: str, call_id: str) -> ToolCall:
    """Turn accumulated ``input_json_delta`` fragments into a ToolCall.

    Invalid JSON raises :class:`ToolCallParseError` so the loop can report a
    precise error instead of silently executing nothing. The ``index`` is used
    only to build a fallback id when the stream omitted one.
    """
    return ToolCall(
        name=name,
        arguments=coerce_arguments(raw),
        id=call_id or f"call_{index}",
        raw_arguments=raw or "{}",
    )


def _map_stop(stop_reason: str, has_tool_calls: bool) -> FinishReason:
    match stop_reason:
        case "tool_use":
            return FinishReason.TOOL_CALLS
        case "max_tokens":
            return FinishReason.LENGTH
        case "end_turn" | "stop_sequence":
            return FinishReason.TOOL_CALLS if has_tool_calls else FinishReason.STOP
        case "refusal":
            return FinishReason.CONTENT_FILTER
        case _:
            return FinishReason.UNKNOWN


def blocks_to_text(blocks: Sequence[Mapping[str, Any]]) -> str:
    """Flatten Anthropic content blocks to plain text (used by tests/tooling)."""
    return "\n".join(
        b.get("text", "") for b in blocks if isinstance(b, Mapping) and b.get("type") == "text"
    )


def _unused(*_args: object) -> None:
    """Placeholder to keep linters from flagging helper imports."""
    _ = content_to_plain
