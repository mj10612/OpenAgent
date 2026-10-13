"""Streaming event union.

Every provider adapter normalises its native wire events into these types. The
TUI and the agent loop therefore never import an adapter, which is what keeps
OpenAgent genuinely model-agnostic: adding a provider cannot break the UI.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TypeAlias

from .types import FinishReason, Message, ToolCall, Usage


@dataclass(slots=True)
class StartEvent:
    """Fired once when the request has been accepted."""

    model: str = ""
    provider: str = ""
    created: float = field(default_factory=time.time)


@dataclass(slots=True)
class TextDelta:
    """An increment of assistant-visible text."""

    text: str
    index: int = 0


@dataclass(slots=True)
class ThinkingDelta:
    """An increment of reasoning text, for models that expose it."""

    text: str
    index: int = 0


@dataclass(slots=True)
class ToolCallStart:
    """A model started emitting a tool call.

    Providers that stream arguments incrementally emit ``ToolCallDelta`` events
    until ``ToolCallEnd``.
    """

    index: int
    id: str = ""
    name: str = ""


@dataclass(slots=True)
class ToolCallDelta:
    """A fragment of the JSON arguments of the tool call at ``index``."""

    index: int
    arguments_delta: str = ""


@dataclass(slots=True)
class ToolCallEnd:
    """The tool call at ``index`` is complete."""

    index: int
    call: ToolCall | None = None


@dataclass(slots=True)
class ToolResultEvent:
    """A tool execution completed and produced an output."""

    call_id: str
    tool_name: str
    output: str
    is_error: bool = False


@dataclass(slots=True)
class UsageEvent:
    """Cumulative reported usage for the current provider request.

    Adapters normalize vendor deltas to snapshots. Consumers reconcile fields
    with the terminal DoneEvent instead of adding repeated snapshots. A runner
    may also emit a summary request's usage before the generation StartEvent;
    its final DoneEvent reports the total for the entire agent turn.
    """

    usage: Usage


@dataclass(slots=True)
class DoneEvent:
    """Terminal event of a successful stream."""

    finish_reason: FinishReason = FinishReason.STOP
    message: Message | None = None
    usage: Usage = field(default_factory=Usage)


@dataclass(slots=True)
class ErrorEvent:
    """Terminal event of a failed stream."""

    error: BaseException | str
    retryable: bool = False


StreamEvent: TypeAlias = (
    StartEvent
    | TextDelta
    | ThinkingDelta
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | ToolResultEvent
    | UsageEvent
    | DoneEvent
    | ErrorEvent
)
