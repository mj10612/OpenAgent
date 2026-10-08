"""Provider-agnostic core data model.

Everything in this module is intentionally free of any provider wire format.
Adapters translate between these types and whatever their API expects, so the
agent loop, context manager and tools never need to know which vendor is
serving the request.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Self

Role = Literal["system", "user", "assistant", "tool"]
"""Conversation roles. Kept deliberately small and provider-independent."""


# --------------------------------------------------------------------------- #
# Content parts
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TextPart:
    """Plain text. The overwhelmingly common case."""

    text: str

    type: Literal["text"] = "text"


@dataclass(slots=True)
class ThinkingPart:
    """Reasoning / thinking content, when a model exposes it.

    Providers that hide chain-of-thought simply never emit this.
    """

    text: str
    signature: str | None = None

    type: Literal["thinking"] = "thinking"


@dataclass(slots=True)
class ImagePart:
    """An image, either inlined as base64 or referenced by URL."""

    data: str | None = None
    mime_type: str = "image/png"
    url: str | None = None
    detail: Literal["auto", "low", "high"] = "auto"

    type: Literal["image"] = "image"


@dataclass(slots=True)
class ToolResultPart:
    """The rendered output of a tool call, embedded in a message."""

    call_id: str
    name: str
    content: str
    is_error: bool = False

    type: Literal["tool_result"] = "tool_result"


ContentPart = TextPart | ThinkingPart | ImagePart | ToolResultPart


def text_of(parts: Iterable[ContentPart] | str | None) -> str:
    """Concatenate only the textual parts.

    Used by token estimation, logging and compaction, where images and tool
    payloads would only add noise.
    """
    if parts is None:
        return ""
    if isinstance(parts, str):
        return parts
    chunks: list[str] = []
    for p in parts:
        if isinstance(p, TextPart):
            chunks.append(p.text)
        elif isinstance(p, ToolResultPart):
            chunks.append(p.content)
    return "\n".join(chunks)


# --------------------------------------------------------------------------- #
# Tool calls
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ToolCall:
    """A model request to invoke a tool.

    ``arguments`` is the parsed mapping. ``raw_arguments`` keeps the exact
    string the model produced so that malformed JSON can be surfaced verbatim
    instead of being lost.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    raw_arguments: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": dict(self.arguments)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        args = data.get("arguments")
        if isinstance(args, str):
            raw = args
            try:
                parsed = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {"value": parsed}
        elif isinstance(args, Mapping):
            raw = json.dumps(dict(args))
            parsed = dict(args)
        else:
            raw, parsed = "", {}

        return cls(
            name=str(data.get("name", "")),
            arguments=parsed,
            id=str(data.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
            raw_arguments=raw,
        )


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Message:
    """One entry in the conversation.

    ``content`` is always a list, even for text-only models. This uniformity is
    what makes adding vision support later a non-breaking change.
    """

    role: Role
    content: list[ContentPart] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = None
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- constructors ------------------------------------------------------ #

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role="system", content=[TextPart(text=text)])

    @classmethod
    def user(cls, text: str, images: Sequence[ImagePart] | None = None) -> Message:
        parts: list[ContentPart] = []
        if images:
            parts.extend(images)
        parts.append(TextPart(text=text))
        return cls(role="user", content=parts)

    @classmethod
    def assistant(
        cls,
        text: str | None = None,
        *,
        tool_calls: Sequence[ToolCall] | None = None,
        reasoning: str | None = None,
    ) -> Message:
        parts: list[ContentPart] = []
        if text:
            parts.append(TextPart(text=text))
        return cls(
            role="assistant",
            content=parts,
            tool_calls=list(tool_calls or ()),
            reasoning=reasoning,
        )

    @classmethod
    def tool_result(
        cls,
        call_id: str,
        name: str,
        content: str,
        *,
        is_error: bool = False,
    ) -> Message:
        return cls(
            role="tool",
            content=[TextPart(content)],
            tool_call_id=call_id,
            name=name,
            metadata={"is_error": is_error},
        )

    # -- accessors --------------------------------------------------------- #

    @property
    def text(self) -> str:
        return text_of(self.content)

    @property
    def images(self) -> list[ImagePart]:
        return [p for p in self.content if isinstance(p, ImagePart)]

    @property
    def is_error(self) -> bool:
        return bool(self.metadata.get("is_error"))

    def tool_text(self) -> str:
        """Text of a tool-result message (stored as TextPart for provider ease)."""
        return text_of(self.content)

    # -- serialisation ----------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": self.role,
            "content": [part_to_dict(p) for p in self.content],
            "created_at": self.created_at,
        }
        if self.tool_calls:
            data["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        if self.reasoning:
            data["reasoning"] = self.reasoning
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Message:
        raw_content = data.get("content")
        parts: list[ContentPart] = []
        if isinstance(raw_content, str):
            parts = [TextPart(text=raw_content)]
        elif isinstance(raw_content, Sequence):
            for item in raw_content:
                if isinstance(item, Mapping):
                    parts.append(part_from_dict(item))

        role = data.get("role", "user")
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"

        return cls(
            role=role,  # type: ignore[arg-type]
            content=parts,
            tool_calls=[ToolCall.from_dict(c) for c in data.get("tool_calls") or ()],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            reasoning=data.get("reasoning"),
            created_at=float(data.get("created_at", time.time())),
            metadata=dict(data.get("metadata") or {}),
        )


def TextPartText(content: str) -> TextPart:
    """Factory kept separate so tool results read clearly at call sites."""
    return TextPart(text=content)


def part_to_dict(part: ContentPart) -> dict[str, Any]:
    match part:
        case TextPart(text=t):
            return {"type": "text", "text": t}
        case ThinkingPart(text=t, signature=s):
            return {"type": "thinking", "text": t, "signature": s}
        case ImagePart(data=d, mime_type=m, url=u, detail=det):
            return {"type": "image", "data": d, "mime_type": m, "url": u, "detail": det}
        case ToolResultPart(call_id=c, name=n, content=ct, is_error=e):
            return {
                "type": "tool_result",
                "call_id": c,
                "name": n,
                "content": ct,
                "is_error": e,
            }
    raise TypeError(f"unsupported content part: {type(part)!r}")


def part_from_dict(data: Mapping[str, Any]) -> ContentPart:
    kind = data.get("type", "text")
    if kind == "thinking":
        return ThinkingPart(text=str(data.get("text", "")), signature=data.get("signature"))
    if kind == "image":
        return ImagePart(
            data=data.get("data"),
            mime_type=str(data.get("mime_type", "image/png")),
            url=data.get("url"),
            detail=data.get("detail", "auto"),
        )
    if kind == "tool_result":
        return ToolResultPart(
            call_id=str(data.get("call_id", "")),
            name=str(data.get("name", "")),
            content=str(data.get("content", "")),
            is_error=bool(data.get("is_error")),
        )
    return TextPart(text=str(data.get("text", "")))


# --------------------------------------------------------------------------- #
# Tool specifications
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ToolParam:
    """One parameter of a tool, in JSON-Schema form."""

    name: str
    type: str
    description: str = ""
    required: bool = True
    enum: list[Any] | None = None
    default: Any = None
    items: dict[str, Any] | None = None

    def to_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum:
            schema["enum"] = list(self.enum)
        if self.items:
            schema["items"] = self.items
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass(slots=True)
class ToolSpec:
    """The provider-independent description of a tool.

    Providers render this into whichever schema their API understands. A text
    protocol provider renders it into prose; a JSON-schema provider renders it
    into ``parameters``.
    """

    name: str
    description: str
    params: list[ToolParam] = field(default_factory=list)
    danger: Literal["none", "write", "execute", "network"] = "none"
    source: Literal["builtin", "mcp"] = "builtin"
    input_schema: dict[str, Any] | None = None

    def schema(self) -> dict[str, Any]:
        """Return the complete input schema without losing nested constraints."""
        if self.input_schema is not None:
            return deepcopy(self.input_schema)
        return {
            "type": "object",
            "properties": {p.name: p.to_schema() for p in self.params},
            "required": [p.name for p in self.params if p.required],
        }

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema(),
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.schema(),
        }

    def to_gemini_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parametersJsonSchema"
            if self.input_schema is not None
            else "parameters": self.schema(),
        }

    def signature(self) -> str:
        """Compact signature used by the text tool protocol."""
        args = ", ".join(f"{p.name}{'' if p.required else '?'}: {p.type}" for p in self.params)
        return f"{self.name}({args})"


# --------------------------------------------------------------------------- #
# Requests / responses
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ModelInfo:
    """A model as advertised by a provider's discovery endpoint."""

    id: str
    provider: str
    context_window: int | None = None
    max_output: int | None = None
    supports_tools: bool | None = None
    supports_vision: bool | None = None
    supports_reasoning: bool | None = None
    aliases: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.id}"


@dataclass(slots=True)
class ThinkingBudget:
    """Extended-thinking configuration, expressed provider-independently."""

    enabled: bool = False
    budget_tokens: int | None = None
    effort: Literal["low", "medium", "high"] | None = None


@dataclass(slots=True)
class ChatRequest:
    """Everything needed to obtain a completion, in neutral terms."""

    messages: list[Message]
    model: str
    tools: list[ToolSpec] = field(default_factory=list)
    tool_choice: Literal["auto", "none", "required"] = "auto"
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    stream: bool = True
    seed: int | None = None
    thinking: ThinkingBudget = field(default_factory=ThinkingBudget)
    response_format: Literal["text", "json_object"] = "text"
    extra: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Usage:
    """Token accounting. All fields optional: many providers report partials."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass(slots=True)
class ChatResponse:
    """A completed (non-streamed) response."""

    message: Message
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    model: str = ""


# --------------------------------------------------------------------------- #
# Enums shared across layers
# --------------------------------------------------------------------------- #


class ToolProtocol(StrEnum):
    """How a provider conveys tool calls.

    The agent loop branches on this and nothing else, which is what makes
    adding a novel tool-call dialect a one-file change.
    """

    NATIVE = "native"
    """Provider API accepts a tool schema and returns structured tool calls."""

    JSON_SCHEMA = "json_schema"
    """OpenAI-shaped ``tools``/``tool_calls``; request/response fields remapped."""

    TEXT = "text"
    """No native support; the model is instructed to emit a fenced block."""

    NONE = "none"
    """No tool use at all."""


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"


class Permission(StrEnum):
    """Three-state authorisation for a tool invocation."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"

    @classmethod
    def parse(cls, value: str | None) -> Permission:
        match (value or "ask").strip().lower():
            case "allow" | "always" | "yes" | "y":
                return cls.ALLOW
            case "deny" | "no" | "n" | "never":
                return cls.DENY
            case _:
                return cls.ASK


def iter_text_chunks(parts: Iterable[ContentPart], *, chunk_size: int = 4000) -> Iterator[str]:
    """Split text into bounded chunks, never splitting a ``\\n\\n`` paragraph."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    buffer = ""
    for part in parts:
        if not isinstance(part, TextPart):
            continue
        buffer += part.text
        while len(buffer) >= chunk_size:
            split = buffer.rfind("\n\n", 0, chunk_size)
            cut = split + 2 if split > 0 else chunk_size
            yield buffer[:cut]
            buffer = buffer[cut:]
    if buffer:
        yield buffer
