"""Provider abstraction and the registry that instantiates them.

The contract is intentionally tiny: implement :meth:`ChatProvider.stream` and,
optionally, :meth:`ChatProvider.list_models`. Everything else — message
rendering, argument coercion, error mapping — lives in the shared HTTP layer.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import abc
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .events import StreamEvent
from .types import (
    ChatRequest,
    ChatResponse,
    ImagePart,
    Message,
    ModelInfo,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolProtocol,
    Usage,
)


class ProviderError(Exception):
    """Base class for provider failures.

    ``retryable`` drives the agent loop's backoff policy; ``status`` carries the
    HTTP code when one exists so callers can distinguish auth problems from
    transient ones without string matching.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        provider: str = "",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.retryable = retryable
        self.provider = provider

    def __str__(self) -> str:
        prefix = f"[{self.provider}] " if self.provider else ""
        code = f" (HTTP {self.status})" if self.status else ""
        return f"{prefix}{self.message}{code}"


class AuthenticationError(ProviderError):
    """Missing or rejected credentials."""


class RateLimitError(ProviderError):
    """HTTP 429 or an equivalent provider-side throttle."""

    def __init__(self, message: str, *, retry_after: float | None = None, **kw: Any) -> None:
        super().__init__(message, status=kw.pop("status", 429), retryable=True, **kw)
        self.retry_after = retry_after


class ContextWindowError(ProviderError):
    """The request exceeded the model's context window."""


class ModelNotFoundError(ProviderError):
    """The requested model id is unknown to the provider."""


class ToolCallParseError(ProviderError):
    """The model produced tool calls we could not decode."""


class ChatProvider(abc.ABC):
    """A source of chat completions.

    Implementations must be safe to instantiate without network access and must
    not perform I/O in ``__init__``; this lets the CLI validate a whole
    configuration offline before the first request.
    """

    #: Stable identifier used in config and logs.
    name: str = "base"
    #: How tool calls are exchanged with this provider.
    tool_protocol: ToolProtocol = ToolProtocol.NONE
    #: Whether :meth:`stream` yields incremental events.
    supports_streaming: bool = True
    #: Whether the provider can accept image parts.
    supports_vision: bool = False
    #: Whether the provider exposes reasoning text.
    supports_reasoning: bool = False
    #: Whether the provider can consume ``tool`` role messages natively.
    supports_native_tool_results: bool = False
    #: Whether system messages are a separate parameter or part of ``messages``.
    supports_system_role: bool = True
    #: Default context window, used for budgeting when unknown.
    default_context_window: int = 128_000
    default_max_tokens: int | None = None

    # -- required ---------------------------------------------------------- #

    @abc.abstractmethod
    def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Yield normalised events for ``request``.

        Implementations must emit exactly one terminal event (``DoneEvent`` or
        ``ErrorEvent``).
        """
        raise NotImplementedError

    # -- optional ---------------------------------------------------------- #

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Non-streaming convenience wrapper around :meth:`stream`.

        Used by background machinery (compaction, sub-agents, title
        generation) that does not need incremental output.
        """
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: list[ToolCall] = []
        usage = Usage()
        message: Message | None = None
        finish = "stop"

        async for event in self.stream(request):
            from .events import (
                DoneEvent,
                ErrorEvent,
                TextDelta,
                ThinkingDelta,
                ToolCallEnd,
                UsageEvent,
            )

            match event:
                case TextDelta(text=t):
                    text_parts.append(t)
                case ThinkingDelta(text=t):
                    reasoning_parts.append(t)
                case ToolCallEnd(call=c) if c is not None:
                    calls.append(c)
                case UsageEvent(usage=u):
                    usage = usage + u
                case DoneEvent() as done:
                    finish = done.finish_reason.value
                    # Terminal usage is authoritative; streamed usage is a fallback.
                    if any(
                        (
                            done.usage.total_tokens,
                            done.usage.cached_tokens,
                            done.usage.reasoning_tokens,
                        )
                    ):
                        usage = done.usage
                    message = done.message
                case ErrorEvent(error=e):
                    raise e if isinstance(e, BaseException) else ProviderError(str(e))

        if message is None:
            message = Message.assistant(
                "".join(text_parts),
                tool_calls=calls,
                reasoning="".join(reasoning_parts) or None,
            )
        return ChatResponse(message=message, usage=usage, finish_reason=finish, model=request.model)

    async def list_models(self) -> list[ModelInfo]:
        """Discover available models. Adapters that cannot introspect return ``[]``."""
        return []

    async def close(self) -> None:
        """Release transport resources."""
        return None

    async def __aenter__(self) -> ChatProvider:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


# --------------------------------------------------------------------------- #
# Shared rendering helpers
# --------------------------------------------------------------------------- #


def render_system_prompt(request: ChatRequest) -> str:
    """Join all system messages in a request."""
    return "\n\n".join(m.text for m in request.messages if m.role == "system" and m.text)


def coerce_arguments(raw: str | Mapping[str, Any] | None) -> dict[str, Any]:
    """Best-effort conversion of tool-call arguments into a mapping.

    Models occasionally emit a bare JSON scalar, or JSON wrapped in a markdown
    fence. Both are handled rather than raising, because refusing here would
    abort an otherwise recoverable turn.
    """
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    text = raw.strip()
    if not text:
        return {}

    # Strip markdown fences some models wrap JSON in.
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.lstrip("`")
        text = text.removeprefix("json").strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Recover single quotes / trailing commas, common with small models.
        candidate = text.replace("'", '"')
        candidate = candidate.replace(",}", "}").replace(",]", "]")
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ToolCallParseError(f"could not decode tool arguments: {exc}") from exc

    if isinstance(parsed, Mapping):
        return dict(parsed)
    return {"value": parsed}


def strip_empty(messages: Sequence[Message]) -> list[Message]:
    """Drop assistant messages that carry neither text nor tool calls.

    Providers reject these outright, and they carry no information anyway.
    """
    out: list[Message] = []
    for msg in messages:
        if msg.role == "assistant" and not msg.text and not msg.tool_calls:
            continue
        out.append(msg)
    return out


def content_to_plain(parts: Sequence[Any]) -> str:
    """Flatten content parts to a string, dropping images.

    Providers without vision get the text plus a placeholder so the model does
    not silently forget that an image was attached.
    """
    chunks: list[str] = []
    for part in parts:
        match part:
            case TextPart(text=t):
                chunks.append(t)
            case ThinkingPart(text=t):
                chunks.append(t)
            case ImagePart():
                chunks.append("[image omitted: model does not support vision]")
            case _:
                text = getattr(part, "content", None)
                if text:
                    chunks.append(str(text))
    return "\n".join(chunks)


@dataclass(slots=True)
class ProviderDescriptor:
    """Static metadata about a provider implementation."""

    kind: str
    module: str
    factory: str = "create"
    protocol: ToolProtocol = ToolProtocol.NONE
    streaming: bool = True
    vision: bool = False
    reasoning: bool = False
    doc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "module": self.module,
            "factory": self.factory,
            "protocol": self.protocol.value,
            "streaming": self.streaming,
            "vision": self.vision,
            "reasoning": self.reasoning,
            "doc": self.doc,
        }


@dataclass(slots=True)
class ProviderHealth:
    """Result of a lightweight connectivity probe."""

    provider: str
    ok: bool
    detail: str = ""
    models: list[ModelInfo] = field(default_factory=list)
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "ok": self.ok,
            "detail": self.detail,
            "latency_ms": self.latency_ms,
            "models": [m.id for m in self.models],
        }
