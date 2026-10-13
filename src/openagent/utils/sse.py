"""Server-Sent Events parsing and tolerant JSON decoding.

Streaming is the part of the stack most likely to be fed malformed bytes, so
everything here is deliberately forgiving: a truncated line is emitted rather
than dropped, and unknown event types are ignored rather than fatal.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.provider import ProviderError


@dataclass(slots=True)
class SSEvent:
    """A single decoded Server-Sent Event."""

    event: str = "message"
    data: str = ""
    id: str | None = None
    retry: int | None = None


class SSEDecoder:
    """Stateful SSE line parser."""

    def __init__(self) -> None:
        self.event_name = "message"
        self.data_lines: list[str] = []
        self.event_id: str | None = None
        self.retry: int | None = None

    def decode_line(self, raw: str) -> SSEvent | None:
        line = raw.rstrip("\r\n")

        if not line:
            return self.flush()

        if line.startswith(":"):
            return None

        if ":" not in line:
            return None

        field_name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]

        match field_name:
            case "event":
                self.event_name = value
            case "data":
                self.data_lines.append(value)
            case "id":
                self.event_id = value
            case "retry":
                try:
                    self.retry = int(value)
                except ValueError:
                    self.retry = None
            case _:
                pass
        return None

    def flush(self) -> SSEvent | None:
        if not self.data_lines and self.event_name == "message" and self.event_id is None:
            return None
        evt = SSEvent(
            event=self.event_name,
            data="\n".join(self.data_lines),
            id=self.event_id,
            retry=self.retry,
        )
        self.event_name = "message"
        self.data_lines = []
        self.event_id = None
        self.retry = None
        return evt


def decode_sse_lines(lines: Iterator[str]) -> Iterator[SSEvent]:
    """Fold a stream of raw lines into events.

    Spec-driven: fields are ``field: value``, a blank line dispatches the
    buffer, and multiple ``data:`` lines are joined with newlines per the SSE
    specification.
    """
    decoder = SSEDecoder()
    for raw in lines:
        evt = decoder.decode_line(raw)
        if evt is not None and evt.data.strip():
            yield evt

    trailing = decoder.flush()
    if trailing is not None and trailing.data.strip():
        yield trailing


class LineResponse(Protocol):
    def aiter_lines(self) -> AsyncIterator[str]: ...


async def iter_sse(response: LineResponse) -> AsyncIterator[SSEvent]:
    """Decode an open streaming HTTP response into events.

    ``response.aiter_lines`` is used rather than ``iter_bytes`` so the framing
    is handled by the HTTP layer rather than by us.
    """
    decoder = SSEDecoder()
    async for raw in response.aiter_lines():
        evt = decoder.decode_line(raw)
        if evt is not None and evt.data.strip():
            yield evt

    trailing = decoder.flush()
    if trailing is not None and trailing.data.strip():
        yield trailing


def parse_json_payload(text: str, *, provider: str = "") -> Any:
    """Decode a JSON body, mapping failures to a provider error."""
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # Some gateways double-encode or wrap payloads in ```json fences.
        cleaned = text.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            snippet = text[:200]
            raise ProviderError(
                f"invalid JSON in response: {exc}; body starts with: {snippet!r}",
                provider=provider,
            ) from exc


@dataclass(slots=True)
class ErrorInfo:
    """Normalised error extracted from a provider response body."""

    message: str
    type: str = ""
    code: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_body(cls, body: Any, *, provider: str = "") -> ErrorInfo:
        """Dig a message out of the many error shapes in circulation.

        Recognises OpenAI (``error.message``), Anthropic, Gemini
        (``error.message``), Ollama (``error``), and a bare string body.
        """
        if isinstance(body, str):
            return cls(message=body or "unknown error")

        if not isinstance(body, dict):
            return cls(message=str(body))

        err = body.get("error", body.get("detail", body.get("message")))
        if isinstance(err, str):
            return cls(message=err, raw=body)
        if isinstance(err, dict):
            return cls(
                message=str(err.get("message") or err.get("detail") or err.get("status") or err),
                type=str(err.get("type", "")),
                code=str(err.get("code", "")),
                raw=body,
            )

        # Gemini: {"error": {"message": ...}} is covered above; Ollama: {"error": "..."}.
        return cls(message=json.dumps(body)[:500], raw=body)
