"""Text-based tool-calling protocol (tier T2).

Many capable models have no native function-calling API, or it is disabled by
the gateway. Rather than let them degrade to plain chat, we describe the tools
in the system prompt and parse a fenced block back out of the reply.

The block dialect is JSON-first because it is what small models handle most
reliably, with a simple line-oriented syntax as a fallback for models that
struggle to emit well-formed JSON:

    ```tool_code
    {"name": "read_file", "arguments": {"path": "README.md"}}
    ```

    <<<TOOL>>>
    read_file
    {"path": "README.md"}

Design notes
------------
- Multiple calls in one block are supported, so the model can parallelise.
- A trailing text segment after the block is preserved as the assistant message.
- Unknown tool names are *not* an error here; the agent loop decides, since it
  is the component that knows which tools exist.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from typing import Any

from ..core.provider import ToolCallParseError
from ..core.types import ChatRequest, ToolCall, ToolSpec
from ..utils.tool_metadata import render_tool_metadata

#: Fenced JSON block, the primary dialect.
FENCED_RE = re.compile(
    r"```(?:tool_code|tool_use|tool|json_tool)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

#: Line-oriented dialect for models that emit invalid JSON.
LINE_MARKER_RE = re.compile(r"<<<TOOL>>>\s*\n", re.IGNORECASE)

#: Reusable call id generator, sequential per process for readable transcripts.
_call_counter = 0


def _next_call_id() -> str:
    global _call_counter
    _call_counter += 1
    return f"txt_{_call_counter:04d}"


class TextToolCodec:
    """Renders tool specs for a text-only model and parses replies back."""

    block_open = "```tool_code"
    block_close = "```"
    line_marker = "<<<TOOL>>>"

    # -- prompt rendering --------------------------------------------------- #

    def render(self, request: ChatRequest) -> str:
        """Build the system prompt, appending a tool contract section."""
        base = "\n\n".join(m.text for m in request.messages if m.role == "system" and m.text)
        if not request.tools or request.tool_choice == "none":
            return base

        lines = [
            base.rstrip(),
            "",
            "# Tool use",
            "",
            "You can call the following tools. This interface is a text protocol:",
            "write your tool calls inside a fenced block, then wait for the result.",
            "",
            "Format:",
            "```tool_code",
            '{"name": "<tool>", "arguments": {<arguments>}}',
            "```",
            "",
            "Rules:",
            "- Emit the block only when you want a tool to run.",
            "- You may emit several calls in one block; they run in order.",
            "- After the block, write nothing. The next message is the tool output.",
            "- Do not invent tool names or arguments that are not listed.",
            "- Use exact values from the schema; quote every string.",
            "",
            "# Available tools",
            "",
        ]

        for spec in request.tools:
            lines.extend(self._render_spec(spec))

        lines.extend(
            [
                "",
                "If no tool is needed, reply normally without a tool block.",
            ]
        )
        return "\n".join(lines)

    def _render_spec(self, spec: ToolSpec) -> list[str]:
        return [render_tool_metadata(spec), ""]

    # -- response parsing --------------------------------------------------- #

    def parse_stream(self) -> _StreamingParser:
        """Incremental parser for use during a text stream."""
        return _StreamingParser(self)

    def parse(self, text: str) -> tuple[list[ToolCall], str]:
        """Split a reply into ``(tool_calls, remaining_text)``."""
        return _StreamingParser(self).consume(text).finish()


class _StreamingParser:
    """Accumulates streamed text, emitting calls as blocks complete.

    Needed because a tool block may straddle many SSE chunks; parsing the reply
    only at the end would delay feedback and make the TUI jitter.
    """

    def __init__(self, codec: TextToolCodec) -> None:
        self._codec = codec
        self._buffer = ""
        self._calls: list[ToolCall] = []
        self._emitted_upto = 0

    def consume(self, text: str) -> _StreamingParser:
        """Consume input text incrementally, returning self for chaining."""
        self.feed(text)
        return self

    def feed(self, text: str) -> list[ToolCall]:
        """Consume a chunk, returning any tool calls that completed."""
        self._buffer += text
        calls, text_out = self._consume_complete(self._buffer)
        self._buffer = text_out
        self._calls.extend(calls)
        return calls

    def finish(self) -> tuple[list[ToolCall], str]:
        """Flush trailing content and return (calls, remaining_text)."""
        if self._buffer.strip():
            calls, remaining = self._consume_complete(self._buffer, force=True)
            self._calls.extend(calls)
            self._buffer = remaining
        return self._calls, self._buffer

    @property
    def pending_text(self) -> str:
        return self._buffer

    # -- internals ---------------------------------------------------------- #

    def _consume_complete(self, text: str, *, force: bool = False) -> tuple[list[ToolCall], str]:
        """Extract every fully-formed block, returning the remainder."""
        calls: list[ToolCall] = []
        remainder = text

        while True:
            match = FENCED_RE.search(remainder)
            if match and (force or match.end() <= len(remainder)):
                body = match.group(1)
                calls.extend(self._parse_body(body))
                remainder = remainder[: match.start()] + remainder[match.end() :]
                continue

            marker = LINE_MARKER_RE.search(remainder)
            if marker:
                tail = remainder[marker.end() :]
                # The line form runs until a blank line or a terminator.
                stop = len(tail)
                for terminator in ("\n\n", self._codec.line_marker, self._codec.block_open):
                    idx = tail.find(terminator)
                    if idx != -1:
                        stop = min(stop, idx)
                body, rest = tail[:stop], tail[stop:]
                if force or stop < len(tail):
                    calls.extend(self._parse_body(body))
                    remainder = remainder[: marker.start()] + rest
                    continue
                break

            if force and re.search(r"```(?:tool_code|tool_use|tool|json_tool)\b", remainder, re.I):
                raise ToolCallParseError("unterminated tool block")
            break

        return calls, remainder

    def _parse_body(self, body: str) -> list[ToolCall]:
        calls: list[ToolCall] = []
        stripped = body.strip()
        if not stripped:
            return calls

        # JSON array of calls.
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                for entry in parsed:
                    call = _call_from_obj(entry)
                    if call:
                        calls.append(call)
                return calls

        # Single JSON object.
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                if isinstance(parsed.get("calls"), list):
                    for entry in parsed["calls"]:
                        call = _call_from_obj(entry)
                        if call:
                            calls.append(call)
                else:
                    call = _call_from_obj(parsed)
                    if call:
                        calls.append(call)
                return calls

            # Repair the most common malformations before giving up.
            try:
                repaired = json.loads(
                    stripped.replace("'", '"').replace(",}", "}").replace(",]", "]")
                )
            except json.JSONDecodeError:
                repaired = None
            if isinstance(repaired, dict):
                call = _call_from_obj(repaired)
                if call:
                    calls.append(call)
                return calls

        # Line form: first line is the name, remainder is the JSON arguments.
        lines = stripped.splitlines()
        if lines and lines[0].strip() and not lines[0].strip().startswith(("{", "[")):
            name = lines[0].strip().strip("`").strip()
            arg_text = "\n".join(lines[1:]).strip()
            arguments: dict[str, Any] = {}
            if arg_text:
                arguments = _strict_arguments(arg_text)
            calls.append(
                ToolCall(
                    name=name,
                    arguments=arguments,
                    id=_next_call_id(),
                    raw_arguments=arg_text or "{}",
                )
            )

        if stripped and not calls:
            raise ToolCallParseError("invalid tool block")
        return calls


def _strict_arguments(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolCallParseError(f"could not decode tool arguments: {exc}") from exc
    if not isinstance(value, dict):
        raise ToolCallParseError("tool arguments must be a JSON object")
    return value


def _call_from_obj(entry: Any) -> ToolCall | None:
    """Build a :class:`ToolCall` from a decoded JSON object."""
    if not isinstance(entry, Mapping):
        raise ToolCallParseError("tool call must be an object")

    # Accept several key spellings; models are inconsistent here.
    name = entry.get("name") or entry.get("tool") or entry.get("function") or entry.get("tool_name")
    if not isinstance(name, str) or not name:
        raise ToolCallParseError("tool call requires a name")

    raw_args = entry.get(
        "arguments", entry.get("args", entry.get("input", entry.get("parameters")))
    )
    arguments: dict[str, Any] = {}
    raw = "{}"
    if isinstance(raw_args, Mapping):
        arguments = dict(raw_args)
        raw = json.dumps(arguments)
    elif isinstance(raw_args, str):
        raw = raw_args
        arguments = _strict_arguments(raw_args)

    elif any(key in entry for key in ("arguments", "args", "input", "parameters")):
        raise ToolCallParseError("tool arguments must be a JSON object")

    call_id = entry.get("id") or entry.get("call_id") or _next_call_id()
    return ToolCall(name=str(name), arguments=arguments, id=str(call_id), raw_arguments=raw)


def spec_default_list(values: list[Any] | None) -> str:
    if not values:
        return ""
    return ", ".join(json.dumps(v) for v in values)


def iter_tool_specs(request: ChatRequest) -> Iterator[ToolSpec]:
    yield from request.tools
