"""Tests for AgentRunner: The Core Agent Execution Loop.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from openagent.context.compactor import Compactor
from openagent.context.estimator import TokenEstimator
from openagent.context.messages import MessageManager
from openagent.core.events import (
    DoneEvent,
    ErrorEvent,
    StartEvent,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResultEvent,
    UsageEvent,
)
from openagent.core.provider import ChatProvider, ProviderError
from openagent.core.types import (
    ChatRequest,
    FinishReason,
    Message,
    ToolCall,
    ToolParam,
    Usage,
)
from openagent.runner import AgentRunner
from openagent.session.store import SessionStore
from openagent.tools.base import Tool, ToolResult
from openagent.tools.registry import PermissionAction, ToolRegistry

# =========================================================================== #
# Mock Provider & Tools
# =========================================================================== #


class MockChatProvider(ChatProvider):
    """Configurable mock provider yielding predefined stream sequences."""

    name = "mock-runner-provider"

    def __init__(self, turns: list[list[StreamEvent]] | None = None) -> None:
        self.turns: list[list[StreamEvent]] = list(turns or [])
        self.recorded_requests: list[ChatRequest] = []
        self.default_model = "mock-model"

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        self.recorded_requests.append(request)
        if not self.turns:
            yield StartEvent(model=request.model, provider=self.name)
            yield DoneEvent(
                finish_reason=FinishReason.STOP,
                message=Message.assistant("Default response"),
                usage=Usage(prompt_tokens=10, completion_tokens=5),
            )
            return

        events = self.turns.pop(0)
        for ev in events:
            yield ev


class AddTool(Tool):
    name = "add"
    description = "Add two integers."

    def __init__(self) -> None:
        self.params = [
            ToolParam(name="a", type="integer", description="First number"),
            ToolParam(name="b", type="integer", description="Second number"),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        a = int(kwargs.get("a", 0))
        b = int(kwargs.get("b", 0))
        call_id = kwargs.get("call_id", "")
        return ToolResult(call_id=str(call_id), output=str(a + b))


class EchoTool(Tool):
    name = "echo"
    description = "Echo back input."

    def __init__(self) -> None:
        self.params = [
            ToolParam(name="text", type="string", description="Text to echo"),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        text = str(kwargs.get("text", ""))
        call_id = kwargs.get("call_id", "")
        return ToolResult(call_id=str(call_id), output=f"echo: {text}")


# =========================================================================== #
# Tests
# =========================================================================== #


@pytest.mark.asyncio
async def test_runner_single_turn_streaming() -> None:
    turns: list[list[StreamEvent]] = [
        [
            StartEvent(model="mock-model", provider="mock"),
            ThinkingDelta(text="Thinking about greeting"),
            TextDelta(text="Hello "),
            TextDelta(text="world!"),
            UsageEvent(usage=Usage(prompt_tokens=12, completion_tokens=8)),
            DoneEvent(
                finish_reason=FinishReason.STOP,
                message=Message.assistant("Hello world!", reasoning="Thinking about greeting"),
                usage=Usage(prompt_tokens=12, completion_tokens=8),
            ),
        ]
    ]
    provider = MockChatProvider(turns=turns)

    runner = AgentRunner(provider=provider)
    events: list[StreamEvent] = []
    async for ev in runner.run_turn("Hi"):
        events.append(ev)

    assert len(events) == 6
    assert isinstance(events[0], StartEvent)
    assert isinstance(events[1], ThinkingDelta) and events[1].text == "Thinking about greeting"
    assert isinstance(events[2], TextDelta) and events[2].text == "Hello "
    assert isinstance(events[3], TextDelta) and events[3].text == "world!"
    assert isinstance(events[4], UsageEvent)
    assert isinstance(events[5], DoneEvent)
    assert events[5].finish_reason == FinishReason.STOP
    assert events[5].message is not None
    assert events[5].message.text == "Hello world!"

    # Verify message history contains system, user, assistant
    assert len(runner.messages) == 3
    assert runner.messages[0].role == "system"
    assert runner.messages[1].role == "user"
    assert runner.messages[1].text == "Hi"
    assert runner.messages[2].role == "assistant"
    assert runner.messages[2].text == "Hello world!"
    assert runner.messages[2].reasoning == "Thinking about greeting"


@pytest.mark.asyncio
async def test_runner_run_turn_to_completion() -> None:
    turns: list[list[StreamEvent]] = [
        [
            StartEvent(),
            TextDelta(text="Direct completion text."),
            DoneEvent(
                finish_reason=FinishReason.STOP,
                message=Message.assistant("Direct completion text."),
            ),
        ]
    ]
    provider = MockChatProvider(turns=turns)
    runner = AgentRunner(provider=provider)
    result = await runner.run_turn_to_completion("Hello")
    assert result == "Direct completion text."


@pytest.mark.asyncio
async def test_runner_multi_step_tool_loop() -> None:
    tools = ToolRegistry([AddTool()])

    call = ToolCall(id="call_123", name="add", arguments={"a": 15, "b": 27})
    turn1_events: list[StreamEvent] = [
        StartEvent(),
        ToolCallStart(index=0, id="call_123", name="add"),
        ToolCallDelta(index=0, arguments_delta='{"a": 15, "b": 27}'),
        ToolCallEnd(index=0, call=call),
        UsageEvent(usage=Usage(prompt_tokens=20, completion_tokens=10)),
        DoneEvent(
            finish_reason=FinishReason.TOOL_CALLS,
            message=Message.assistant(tool_calls=[call]),
            usage=Usage(prompt_tokens=20, completion_tokens=10),
        ),
    ]

    turn2_events: list[StreamEvent] = [
        StartEvent(),
        TextDelta(text="The sum is 42."),
        UsageEvent(usage=Usage(prompt_tokens=35, completion_tokens=8)),
        DoneEvent(
            finish_reason=FinishReason.STOP,
            message=Message.assistant("The sum is 42."),
            usage=Usage(prompt_tokens=35, completion_tokens=8),
        ),
    ]

    provider = MockChatProvider(turns=[turn1_events, turn2_events])
    runner = AgentRunner(provider=provider, tools=tools)

    events: list[StreamEvent] = []
    async for ev in runner.run_turn("Calculate 15 + 27"):
        events.append(ev)

    # Check that ToolResultEvent was emitted between turns
    tool_result_events = [ev for ev in events if isinstance(ev, ToolResultEvent)]
    assert len(tool_result_events) == 1
    assert tool_result_events[0].call_id == "call_123"
    assert tool_result_events[0].tool_name == "add"
    assert tool_result_events[0].output == "42"
    assert not tool_result_events[0].is_error

    # Terminal event must be DoneEvent with STOP and total usage
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].finish_reason == FinishReason.STOP
    assert events[-1].message is not None
    assert events[-1].message.text == "The sum is 42."
    assert events[-1].usage.prompt_tokens == 55
    assert events[-1].usage.completion_tokens == 18

    # History: [system, user, assistant_with_call, tool_result, assistant_final]
    assert len(runner.messages) == 5
    assert runner.messages[0].role == "system"
    assert runner.messages[1].role == "user"
    assert runner.messages[2].role == "assistant"
    assert len(runner.messages[2].tool_calls) == 1
    assert runner.messages[3].role == "tool"
    assert runner.messages[3].tool_call_id == "call_123"
    assert runner.messages[3].name == "add"
    assert runner.messages[3].text == "<tool_output>\n42\n</tool_output>"
    assert runner.messages[4].role == "assistant"
    assert runner.messages[4].text == "The sum is 42."


@pytest.mark.asyncio
async def test_runner_parallel_tool_execution() -> None:
    tools = ToolRegistry([AddTool(), EchoTool()])

    call1 = ToolCall(id="c1", name="add", arguments={"a": 3, "b": 4})
    call2 = ToolCall(id="c2", name="echo", arguments={"text": "parallel test"})

    turn1: list[StreamEvent] = [
        StartEvent(),
        ToolCallStart(index=0, id="c1", name="add"),
        ToolCallEnd(index=0, call=call1),
        ToolCallStart(index=1, id="c2", name="echo"),
        ToolCallEnd(index=1, call=call2),
        DoneEvent(
            finish_reason=FinishReason.TOOL_CALLS,
            message=Message.assistant(tool_calls=[call1, call2]),
        ),
    ]

    turn2: list[StreamEvent] = [
        StartEvent(),
        TextDelta(text="Calculated 7 and echoed successfully."),
        DoneEvent(
            finish_reason=FinishReason.STOP,
            message=Message.assistant("Calculated 7 and echoed successfully."),
        ),
    ]

    provider = MockChatProvider(turns=[turn1, turn2])
    runner = AgentRunner(provider=provider, tools=tools)

    events: list[StreamEvent] = []
    async for ev in runner.run_turn("Execute both tools"):
        events.append(ev)

    results = [ev for ev in events if isinstance(ev, ToolResultEvent)]
    assert len(results) == 2
    res_dict = {r.tool_name: r.output for r in results}
    assert res_dict["add"] == "7"
    assert res_dict["echo"] == "echo: parallel test"

    # Messages check: system, user, assistant (2 calls), tool 1, tool 2, assistant final
    assert len(runner.messages) == 6
    assert runner.messages[3].role == "tool"
    assert runner.messages[4].role == "tool"
    assert runner.messages[5].role == "assistant"


@pytest.mark.asyncio
async def test_runner_permission_denial_handling() -> None:
    tools = ToolRegistry([EchoTool()])
    # Deny echo tool
    tools.set_tool_policy("echo", PermissionAction.DENY)

    call = ToolCall(id="deny_call", name="echo", arguments={"text": "forbidden"})
    turn1: list[StreamEvent] = [
        StartEvent(),
        ToolCallEnd(index=0, call=call),
        DoneEvent(
            finish_reason=FinishReason.TOOL_CALLS,
            message=Message.assistant(tool_calls=[call]),
        ),
    ]
    turn2: list[StreamEvent] = [
        StartEvent(),
        TextDelta(text="I cannot run that tool because permission was denied."),
        DoneEvent(
            finish_reason=FinishReason.STOP,
            message=Message.assistant("I cannot run that tool because permission was denied."),
        ),
    ]

    provider = MockChatProvider(turns=[turn1, turn2])
    runner = AgentRunner(provider=provider, tools=tools)

    events: list[StreamEvent] = []
    async for ev in runner.run_turn("Run forbidden echo"):
        events.append(ev)

    # ToolResultEvent must be marked as error
    results = [ev for ev in events if isinstance(ev, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert "Permission denied" in results[0].output

    # Model received denial and responded
    assert runner.messages[3].role == "tool"
    assert runner.messages[3].is_error is True
    assert runner.messages[4].role == "assistant"
    assert "permission was denied" in runner.messages[4].text


@pytest.mark.asyncio
async def test_runner_permission_ask_with_callback() -> None:
    tools = ToolRegistry([EchoTool()])
    tools.set_tool_policy("echo", PermissionAction.ASK)

    call = ToolCall(id="ask_call", name="echo", arguments={"text": "confirm me"})

    # 1. Ask callback returns False
    turn1: list[StreamEvent] = [
        StartEvent(),
        ToolCallEnd(index=0, call=call),
        DoneEvent(
            finish_reason=FinishReason.TOOL_CALLS, message=Message.assistant(tool_calls=[call])
        ),
    ]
    turn2: list[StreamEvent] = [
        StartEvent(),
        TextDelta(text="Cancelled."),
        DoneEvent(finish_reason=FinishReason.STOP, message=Message.assistant("Cancelled.")),
    ]
    provider = MockChatProvider(turns=[turn1, turn2])
    runner = AgentRunner(provider=provider, tools=tools)

    events: list[StreamEvent] = []
    async for ev in runner.run_turn("Confirm test", ask_callback=lambda c: False):
        events.append(ev)

    result_events = [ev for ev in events if isinstance(ev, ToolResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].is_error is True
    assert "user rejected execution" in result_events[0].output

    # 2. Ask callback returns True
    turn1_ok: list[StreamEvent] = [
        StartEvent(),
        ToolCallEnd(index=0, call=call),
        DoneEvent(
            finish_reason=FinishReason.TOOL_CALLS, message=Message.assistant(tool_calls=[call])
        ),
    ]
    turn2_ok: list[StreamEvent] = [
        StartEvent(),
        TextDelta(text="Done."),
        DoneEvent(finish_reason=FinishReason.STOP, message=Message.assistant("Done.")),
    ]
    provider_ok = MockChatProvider(turns=[turn1_ok, turn2_ok])
    runner_ok = AgentRunner(provider=provider_ok, tools=tools)

    events_ok: list[StreamEvent] = []
    async for ev in runner_ok.run_turn("Confirm test ok", ask_callback=lambda c: True):
        events_ok.append(ev)

    result_events_ok = [ev for ev in events_ok if isinstance(ev, ToolResultEvent)]
    assert len(result_events_ok) == 1
    assert result_events_ok[0].is_error is False
    assert result_events_ok[0].output == "echo: confirm me"


@pytest.mark.asyncio
async def test_runner_compaction_integration() -> None:
    compactor = Compactor(keep_recent_groups=1)
    estimator = TokenEstimator()

    # Preload messages that exceed budget
    msg_mgr = MessageManager(estimator=estimator)
    msg_mgr.add_system("System instructions.")
    # Add older turns
    msg_mgr.add_user("User goal: build feature A " + "x" * 200)
    msg_mgr.add_assistant("Plan formulated.")
    msg_mgr.add_user("Step 1 done " + "y" * 200)
    msg_mgr.add_assistant("Acknowledged.")

    # Context budget small enough to trigger compaction
    max_budget = 80

    turns: list[list[StreamEvent]] = [
        [
            StartEvent(),
            TextDelta(text="Response after compaction."),
            DoneEvent(
                finish_reason=FinishReason.STOP,
                message=Message.assistant("Response after compaction."),
            ),
        ]
    ]
    provider = MockChatProvider(turns=turns)

    runner = AgentRunner(
        provider=provider,
        messages=msg_mgr,
        compactor=compactor,
        context_window=max_budget + estimator.request_overhead,
    )

    events: list[StreamEvent] = []
    async for ev in runner.run_turn("Final question"):
        events.append(ev)

    # Verify the request dispatched to provider was compacted
    assert len(provider.recorded_requests) >= 1
    sent_messages = provider.recorded_requests[-1].messages
    # Should contain context summary
    assert any("[Context Summary]" in m.text for m in sent_messages)


@pytest.mark.asyncio
async def test_runner_session_persistence(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)
    turns: list[list[StreamEvent]] = [
        [
            StartEvent(),
            TextDelta(text="Session reply 1"),
            DoneEvent(
                finish_reason=FinishReason.STOP,
                message=Message.assistant("Session reply 1"),
            ),
        ],
        [
            StartEvent(),
            TextDelta(text="Session reply 2"),
            DoneEvent(
                finish_reason=FinishReason.STOP,
                message=Message.assistant("Session reply 2"),
            ),
        ],
    ]
    provider = MockChatProvider(turns=turns)

    runner = AgentRunner(provider=provider, session_store=store)
    assert runner.session_id is None

    # First turn
    async for _ in runner.run_turn("Turn 1 prompt"):
        pass

    session_id = runner.session_id
    assert session_id is not None
    _meta, msgs = store.load_session(session_id)
    assert len(msgs) == 3  # system, user 1, assistant 1
    assert msgs[1].text == "Turn 1 prompt"
    assert msgs[2].text == "Session reply 1"

    # Second turn
    async for _ in runner.run_turn("Turn 2 prompt"):
        pass

    _meta2, msgs2 = store.load_session(session_id)
    assert len(msgs2) == 5  # system, user 1, assistant 1, user 2, assistant 2
    assert msgs2[3].text == "Turn 2 prompt"
    assert msgs2[4].text == "Session reply 2"

    # Resuming with another runner instance
    provider2 = MockChatProvider()
    resumed_runner = AgentRunner(
        provider=provider2,
        session_store=store,
        session_id=session_id,
    )
    assert len(resumed_runner.messages) == 5
    assert resumed_runner.messages[-1].text == "Session reply 2"


@pytest.mark.asyncio
async def test_runner_reset() -> None:
    turns: list[list[StreamEvent]] = [
        [
            StartEvent(),
            TextDelta(text="First turn answer"),
            DoneEvent(
                finish_reason=FinishReason.STOP,
                message=Message.assistant("First turn answer"),
            ),
        ],
        [
            StartEvent(),
            TextDelta(text="Second turn answer"),
            DoneEvent(
                finish_reason=FinishReason.STOP,
                message=Message.assistant("Second turn answer"),
            ),
        ],
    ]
    provider = MockChatProvider(turns=turns)

    runner = AgentRunner(provider=provider)
    await runner.run_turn_to_completion("Hello")
    assert len(runner.messages) == 3

    runner.reset()
    # Reset keeps system prompt and clears conversation
    assert len(runner.messages) == 1
    assert runner.messages[0].role == "system"

    await runner.run_turn_to_completion("Hello again")
    # Fresh conversation: system, new user, new assistant
    assert len(runner.messages) == 3
    assert runner.messages[1].text == "Hello again"
    assert runner.messages[2].text == "Second turn answer"


@pytest.mark.asyncio
async def test_runner_max_iterations_guard() -> None:
    tools = ToolRegistry([EchoTool()])
    call = ToolCall(id="inf_call", name="echo", arguments={"text": "infinite loop"})

    # Provider that perpetually requests tool calls
    turns: list[list[StreamEvent]] = [
        [
            StartEvent(),
            ToolCallEnd(index=0, call=call),
            DoneEvent(
                finish_reason=FinishReason.TOOL_CALLS, message=Message.assistant(tool_calls=[call])
            ),
        ]
        for _ in range(10)
    ]

    provider = MockChatProvider(turns=turns)
    runner = AgentRunner(provider=provider, tools=tools, max_tool_iterations=3)

    events: list[StreamEvent] = []
    async for ev in runner.run_turn("Loop forever"):
        events.append(ev)

    # Must terminate with finish_reason == LENGTH
    terminal = events[-1]
    assert isinstance(terminal, DoneEvent)
    assert terminal.finish_reason == FinishReason.LENGTH


@pytest.mark.asyncio
async def test_runner_error_propagation() -> None:
    turns: list[list[StreamEvent]] = [
        [
            StartEvent(),
            ErrorEvent(error=ProviderError("Rate limit exceeded", status=429, retryable=True)),
        ]
    ]
    provider = MockChatProvider(turns=turns)
    runner = AgentRunner(provider=provider)

    events: list[StreamEvent] = []
    async for ev in runner.run_turn("Fail please"):
        events.append(ev)

    errors = [ev for ev in events if isinstance(ev, ErrorEvent)]
    assert len(errors) == 1
    assert "Rate limit exceeded" in str(errors[0].error)


@pytest.mark.asyncio
async def test_runner_with_user_message_object() -> None:
    turns: list[list[StreamEvent]] = [
        [
            StartEvent(),
            TextDelta(text="Received message object."),
            DoneEvent(
                finish_reason=FinishReason.STOP,
                message=Message.assistant("Received message object."),
            ),
        ]
    ]
    provider = MockChatProvider(turns=turns)
    runner = AgentRunner(provider=provider)
    user_msg = Message.user("Direct Message object")

    async for _ in runner.run_turn(user_msg):
        pass

    assert runner.messages[1] is user_msg
