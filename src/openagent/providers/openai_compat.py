"""OpenAI-compatible provider adapter.

This single adapter covers OpenAI itself plus the large population of services
that reimplemented its wire format (DeepSeek, Groq, Together, Fireworks,
OpenRouter, Cerebras, vLLM, Ollama, LM Studio, llama.cpp, NVIDIA NIM, …). That
is why OpenAgent can claim near-universal model support without a maintenance
burden proportional to the number of vendors.

Divergences handled here rather than by bloating the request:
- ``/v1`` suffix present or absent on ``base_url``
- reasoning models that reject ``temperature``/``top_p``
- ``max_tokens`` vs ``max_completion_tokens``
- tool calls streamed as index-tagged argument fragments

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
import re
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
)
from ..core.provider import (
    ChatProvider,
    ProviderError,
    coerce_arguments,
    content_to_plain,
    render_system_prompt,
)
from ..core.types import (
    ChatRequest,
    ImagePart,
    Message,
    ModelInfo,
    TextPart,
    ToolCall,
    ToolProtocol,
    ToolSpec,
    Usage,
)
from ..utils.http import HttpTransport
from ..utils.sse import SSEvent
from .text_protocol import TextToolCodec

#: Substrings identifying reasoning models that reject sampling parameters.
REASONING_HINTS = (
    "o1",
    "o3",
    "o4",
    "gpt-5",
    "gpt5",
    "deepseek-reasoner",
    "reasoner",
    "thinking",
    "qwq",
    "r1",
)

#: Known context windows for popular ids, used to pre-size the token budget.
CONTEXT_HINTS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o3": 200_000,
    "claude-3": 200_000,
    "claude-sonnet": 200_000,
    "claude-opus": 200_000,
    "llama-3": 128_000,
    "llama3": 128_000,
    "llama-4": 1_000_000,
    "qwen": 32_768,
    "qwen2.5": 32_768,
    "qwen3": 131_072,
    "deepseek": 64_000,
    "mistral": 32_768,
    "mixtral": 32_768,
    "command-r": 128_000,
    "gemini": 1_000_000,
    "glm": 128_000,
    "kimi": 200_000,
    "grok": 131_072,
}


def guess_context_window(model: str, default: int = 128_000) -> int:
    """Best-effort context window for an arbitrary model id.

    Deliberately substring-based: new models appear faster than any registry can
    track, so a conservative guess plus a config override beats a stale table.
    """
    name = model.lower()
    for hint, window in CONTEXT_HINTS.items():
        if hint in name:
            return window
    return default


def is_reasoning_model(model: str) -> bool:
    name = model.lower()
    return any(hint in name for hint in REASONING_HINTS)


class OpenAICompatProvider(ChatProvider):
    """Adapter for the OpenAI Chat Completions wire format."""

    tool_protocol = ToolProtocol.JSON_SCHEMA
    supports_streaming = True
    supports_vision = True
    supports_native_tool_results = True
    supports_system_role = True

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        provider_name: str = "openai-compat",
        headers: Mapping[str, str] | None = None,
        auth_header: str = "authorization",
        auth_prefix: str = "Bearer ",
        timeout: float = 600.0,
        max_retries: int = 3,
        extra_body: Mapping[str, Any] | None = None,
        context_window: int | None = None,
        client: Any = None,
    ) -> None:
        self.name = provider_name
        self.model = model
        self.base_url = self._normalise_base_url(base_url)
        self.context_window = context_window or guess_context_window(model)
        self.extra_body = dict(extra_body or {})

        request_headers = dict(headers or {})
        if api_key:
            header = auth_header.lower()
            if header == "authorization":
                request_headers["Authorization"] = f"{auth_prefix}{api_key}"
            else:
                request_headers[header] = api_key

        self._codec = TextToolCodec()
        self._transport = HttpTransport(
            self.base_url,
            provider=self.name,
            headers=request_headers,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

    @staticmethod
    def _normalise_base_url(base_url: str) -> str:
        """Accept bare hosts, ``/v1`` endpoints and full chat URLs alike."""
        url = base_url.rstrip("/")
        if url.endswith("/chat/completions"):
            url = url[: -len("/chat/completions")]
        if not re.search(r"/v\d+[a-z]*(?:/|$)", url):
            # Most OpenAI-compatible servers expose their API under /v1.
            url = f"{url}/v1"
        return url

    @property
    def transport(self) -> HttpTransport:
        return self._transport

    # -- request rendering -------------------------------------------------- #

    def _render_message(self, msg: Message) -> dict[str, Any]:
        """Convert a canonical message to the OpenAI wire shape."""
        match msg.role:
            case "assistant":
                out: dict[str, Any] = {"role": "assistant"}
                text = msg.text
                # Some providers reject a null/empty content alongside tool_calls.
                out["content"] = text if text else None
                if msg.tool_calls:
                    out["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": c.raw_arguments or "{}"},
                        }
                        for c in msg.tool_calls
                    ]
                if msg.reasoning:
                    out["reasoning_content"] = msg.reasoning
                return out

            case "tool":
                return {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id or "",
                    "content": msg.tool_text() or "(no output)",
                }

            case "system":
                return {"role": "system", "content": msg.text}

            case _:
                return self._render_user(msg)

    def _render_user(self, msg: Message) -> dict[str, Any]:
        images = [p for p in msg.content if isinstance(p, ImagePart)]
        text = content_to_plain(msg.content)
        if not images:
            return {"role": "user", "content": text}

        parts: list[dict[str, Any]] = []
        for part in msg.content:
            match part:
                case ImagePart(data=d, mime_type=m, url=u):
                    url = u or f"data:{m};base64,{d}"
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                case TextPart(text=t):
                    parts.append({"type": "text", "text": t})
                case _:
                    parts.append({"type": "text", "text": content_to_plain([part])})
        return {"role": "user", "content": parts}

    def build_payload(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        """Assemble the JSON body for a completion request."""
        use_text_protocol = bool(request.tools) and self.tool_protocol == ToolProtocol.TEXT

        if use_text_protocol:
            system = self._codec.render(request)
            messages = [{"role": "system", "content": system}]
            messages += [self._render_message(m) for m in request.messages if m.role != "system"]
        else:
            system = render_system_prompt(request)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages += [self._render_message(m) for m in request.messages if m.role != "system"]

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}

        # Reasoning models reject sampling parameters outright.
        reasoning = is_reasoning_model(request.model)
        if request.temperature is not None and not reasoning:
            payload["temperature"] = request.temperature
        if request.top_p is not None and not reasoning:
            payload["top_p"] = request.top_p
        if request.seed is not None and not reasoning:
            payload["seed"] = request.seed

        max_tokens = (
            request.max_tokens if request.max_tokens is not None else self.default_max_tokens
        )
        if max_tokens is not None:
            if reasoning:
                payload["max_completion_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens

        if request.stop:
            payload["stop"] = request.stop

        if request.tools and not use_text_protocol:
            payload["tools"] = [spec.to_openai_schema() for spec in request.tools]
            payload["tool_choice"] = request.tool_choice

        if request.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}

        if request.thinking.enabled:
            payload["reasoning_effort"] = request.thinking.effort or "medium"

        payload.update(self.extra_body)
        return payload

    # -- streaming ---------------------------------------------------------- #

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        payload = self.build_payload(request, stream=True)
        try:
            response = await self._transport.stream_post("/chat/completions", payload)
        except ProviderError as exc:
            yield ErrorEvent(error=exc)
            return

        try:
            use_text_protocol = bool(request.tools) and self.tool_protocol == ToolProtocol.TEXT

            yield StartEvent(model=request.model, provider=self.name)

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            usage = Usage()
            finish_reason = "stop"
            text_protocol_calls: list[ToolCall] = []

            # Partial tool calls keyed by streaming index.
            partials: dict[int, dict[str, str]] = {}
            codec_stream = self._codec.parse_stream() if use_text_protocol else None

            try:
                async for event in _iter_openai_sse(response):
                    if event.event not in ("message", ""):
                        continue
                    if event.data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(event.data)
                    except json.JSONDecodeError:
                        continue

                    if usage_payload := chunk.get("usage"):
                        usage = _parse_usage(usage_payload)

                    for choice in chunk.get("choices") or ():
                        if reason := choice.get("finish_reason"):
                            finish_reason = reason
                        delta = choice.get("delta") or choice.get("message") or {}

                        if reasoning_text := delta.get("reasoning_content") or delta.get(
                            "reasoning"
                        ):
                            reasoning_parts.append(reasoning_text)
                            yield ThinkingDelta(text=reasoning_text)

                        if content := delta.get("content"):
                            if isinstance(content, list):
                                # Some gateways return content parts even when streaming.
                                content = "".join(
                                    p.get("text", "") for p in content if isinstance(p, dict)
                                )
                            if content:
                                if codec_stream is not None:
                                    codec_stream.feed(content)
                                    continue
                                text_parts.append(content)
                                yield TextDelta(text=content)

                        for raw_call in delta.get("tool_calls") or ():
                            index = int(raw_call.get("index", len(partials)))
                            slot = partials.setdefault(index, {"id": "", "name": "", "args": ""})

                            if fn := raw_call.get("function"):
                                if fn_name := fn.get("name"):
                                    if not slot["name"]:
                                        yield ToolCallStart(
                                            index=index, name=fn_name, id=slot["id"]
                                        )
                                    slot["name"] += fn_name
                                if args_delta := fn.get("arguments"):
                                    slot["args"] += args_delta
                                    yield ToolCallDelta(index=index, arguments_delta=args_delta)
                            if call_id := raw_call.get("id"):
                                slot["id"] = call_id

                if codec_stream is not None:
                    text_protocol_calls, remaining_text = codec_stream.finish()
                    if remaining_text:
                        text_parts.append(remaining_text)
                        yield TextDelta(text=remaining_text)

                final_tool_calls: list[ToolCall] = []
                for index, slot in sorted(partials.items()):
                    try:
                        arguments = coerce_arguments(slot["args"])
                    except ProviderError as exc:
                        yield ErrorEvent(error=exc)
                        return
                    call = ToolCall(
                        name=slot["name"],
                        arguments=arguments,
                        id=slot["id"] or f"call_{index}",
                        raw_arguments=slot["args"] or "{}",
                    )
                    final_tool_calls.append(call)
                    yield ToolCallEnd(index=index, call=call)

                for call in text_protocol_calls:
                    yield ToolCallEnd(index=len(final_tool_calls), call=call)
                    final_tool_calls.append(call)

                if not final_tool_calls and text_parts:
                    yield TextDelta(text="")

            except ProviderError as exc:
                yield ErrorEvent(error=exc, retryable=exc.retryable)
                return
            except Exception as exc:
                yield ErrorEvent(error=exc)
                return

            from ..core.types import Message

            yield DoneEvent(
                finish_reason=_map_finish(finish_reason),
                usage=usage,
                message=Message.assistant(
                    "".join(text_parts),
                    tool_calls=final_tool_calls,
                    reasoning="".join(reasoning_parts) or None,
                ),
            )
        finally:
            await response.aclose()

    # -- discovery ---------------------------------------------------------- #

    async def list_models(self) -> list[ModelInfo]:
        body = await self._transport.get_json("/models")
        raw_models = body.get("data") or body.get("models") or []
        out: list[ModelInfo] = []
        for entry in raw_models:
            if isinstance(entry, str):
                out.append(ModelInfo(id=entry, provider=self.name))
                continue
            model_id = entry.get("id") or entry.get("name")
            if not model_id:
                continue
            out.append(
                ModelInfo(
                    id=model_id,
                    provider=self.name,
                    context_window=entry.get("context_length")
                    or entry.get("context_window")
                    or guess_context_window(model_id),
                    max_output=entry.get("max_output_tokens"),
                    supports_tools=entry.get("tool_calling"),
                    supports_vision=entry.get("vision"),
                )
            )
        return out

    async def close(self) -> None:
        await self._transport.aclose()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def _iter_openai_sse(response: Any) -> AsyncIterator[SSEvent]:
    """Yield decoded SSE events, tolerating servers that ignore SSE framing.

    A few OpenAI-compatible servers answer with a plain JSON array or one JSON
    object per line instead of ``data:`` lines; both are handled so a
    non-conforming gateway still works.
    """
    from ..utils.sse import SSEDecoder, iter_sse

    headers = getattr(response, "headers", {})
    content_type = ""
    if isinstance(headers, Mapping):
        content_type = headers.get("content-type", "")

    if "text/event-stream" in content_type:
        async for event in iter_sse(response):
            yield event
        return

    decoder = SSEDecoder()
    saw_sse = False
    buffer: list[str] = []

    async for line in response.aiter_lines():
        if not saw_sse:
            if line.startswith(("data:", "event:", ":")):
                saw_sse = True
            elif line.strip() != "":
                buffer.append(line)
                continue

        if saw_sse:
            if buffer:
                for prev in buffer:
                    evt = decoder.decode_line(prev)
                    if evt is not None and evt.data.strip():
                        yield evt
                buffer.clear()
            evt = decoder.decode_line(line)
            if evt is not None and evt.data.strip():
                yield evt

    if saw_sse:
        trailing = decoder.flush()
        if trailing is not None and trailing.data.strip():
            yield trailing
        return

    # Fallback: try to interpret the whole payload as one JSON document, then
    # as newline-delimited JSON.
    blob = "\n".join(buffer).strip()
    if not blob:
        return
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        chunks = parsed if isinstance(parsed, list) else [parsed]
        for chunk in chunks:
            yield SSEvent(data=json.dumps(chunk))
        return

    for line in buffer:
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        yield SSEvent(data=line)


def _parse_usage(raw: Mapping[str, Any]) -> Usage:
    details = raw.get("completion_tokens_details") or raw.get("output_tokens_details") or {}
    prompt_details = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
    return Usage(
        prompt_tokens=int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0),
        completion_tokens=int(raw.get("completion_tokens") or raw.get("output_tokens") or 0),
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
        cached_tokens=int(prompt_details.get("cached_tokens") or 0),
    )


def _map_finish(reason: str) -> Any:
    from ..core.types import FinishReason

    match reason:
        case "tool_calls" | "function_call":
            return FinishReason.TOOL_CALLS
        case "length" | "max_tokens":
            return FinishReason.LENGTH
        case "content_filter":
            return FinishReason.CONTENT_FILTER
        case "stop" | "end_turn" | "stop_sequence" | "eos":
            return FinishReason.STOP
        case _:
            return FinishReason.UNKNOWN


def tool_call_from_dict(raw: Mapping[str, Any]) -> ToolCall:
    """Parse an OpenAI-shaped tool call. Exposed for adapters and tests."""
    fn = raw.get("function") or {}
    return ToolCall(
        name=str(fn.get("name", "")),
        arguments=coerce_arguments(fn.get("arguments")),
        id=str(raw.get("id") or ""),
        raw_arguments=str(fn.get("arguments") or "{}"),
    )


def specs_to_openai(specs: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [spec.to_openai_schema() for spec in specs]
