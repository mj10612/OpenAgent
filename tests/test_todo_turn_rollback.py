import asyncio

import pytest

from openagent.core.events import DoneEvent, ErrorEvent, ToolResultEvent
from openagent.core.provider import ChatProvider, ProviderError
from openagent.core.types import FinishReason, Message, ToolCall
from openagent.runner import AgentRunner
from openagent.session.store import SessionStore
from openagent.tools.agent_tools import TodoTool
from openagent.tools.registry import ToolRegistry


class TodoProvider(ChatProvider):
    model = "fixture"
    context_window = 10000
    default_max_tokens = 100

    def __init__(self, action, outcome):
        self.action = action
        self.outcome = outcome
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, request):
        if request.messages[-1].role == "user":
            arguments = {"action": self.action}
            if self.action == "add":
                arguments["tasks"] = ["Temporary task"]
            elif self.action == "complete":
                arguments["task"] = "1"
            yield DoneEvent(message=Message.assistant(tool_calls=[ToolCall(
                id="todo-call", name="todo", arguments=arguments,
            )]), finish_reason=FinishReason.TOOL_CALLS)
        elif self.outcome == "error":
            yield ErrorEvent(ProviderError("offline failure"))
        elif self.outcome == "exception":
            raise ProviderError("offline exception")
        else:
            if self.outcome == "block":
                self.entered.set()
                await self.release.wait()
            yield DoneEvent(message=Message.assistant("saved"))


def prepared(tmp_path, action="clear", outcome="error"):
    todo = TodoTool()
    provider = TodoProvider(action, outcome)
    store = SessionStore(tmp_path)
    agent = AgentRunner(provider, tools=ToolRegistry([todo]), session_store=store,
                        max_tool_iterations=1 if outcome == "iteration-limit" else 25)
    todo.restore_state({"version": 1, "tasks": [
        {"text": "Kept task", "done": False},
        {"text": "Earlier completed task", "done": True},
    ]})
    agent.messages.append(Message.user("Previous successful request"))
    agent.messages.append(Message.assistant("Ready"))
    agent._persist_session()
    return agent, todo, provider, store


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["add", "complete", "clear"])
@pytest.mark.parametrize("outcome", ["error", "exception", "iteration-limit", "save-error"])
async def test_failed_turn_restores_checklist_before_error_is_observed(
    tmp_path, monkeypatch, action, outcome,
):
    agent, todo, _, store = prepared(tmp_path, action, outcome)
    before = todo.snapshot_state()
    history = agent.messages.copy().messages
    sid = agent.session_id
    path = tmp_path / (sid + ".jsonl")
    archive = path.read_bytes()
    if outcome == "save-error":
        def fail(*args, **kwargs):
            raise OSError("save failed")
        monkeypatch.setattr(store, "save_messages", fail)

    events = []
    async for event in agent.run_turn("Continue"):
        events.append(event)
        if isinstance(event, ErrorEvent):
            assert todo.snapshot_state() == before
            assert agent.messages.messages == history

    assert isinstance(events[-1], ErrorEvent)
    assert todo.snapshot_state() == before
    assert agent.session_id == sid
    assert path.read_bytes() == archive
    assert store.load_metadata(sid).todo_state == before


@pytest.mark.asyncio
async def test_closing_stream_after_todo_result_restores_checklist(tmp_path):
    agent, todo, _, store = prepared(tmp_path, outcome="success")
    before = todo.snapshot_state()
    history = agent.messages.copy().messages
    stream = agent.run_turn("Continue")
    try:
        async for event in stream:
            if isinstance(event, ToolResultEvent):
                assert not event.is_error
                assert todo.snapshot_state() != before
                break
        else:
            pytest.fail("No todo result was emitted")
    finally:
        await stream.aclose()

    assert todo.snapshot_state() == before
    assert agent.messages.messages == history
    assert store.load_metadata(agent.session_id).todo_state == before


@pytest.mark.asyncio
async def test_cancellation_restores_checklist_without_rewinding_another_turn(tmp_path):
    agent, todo, provider, store = prepared(tmp_path, outcome="block")
    before = todo.snapshot_state()
    history = agent.messages.copy().messages

    async def consume():
        return [event async for event in agent.run_turn("Continue")]

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(provider.entered.wait(), timeout=5)
        changed = todo.snapshot_state()
        assert changed != before
        rejected = [event async for event in agent.run_turn("Overlapping request")]
        assert isinstance(rejected[-1], ErrorEvent)
        assert todo.snapshot_state() == changed
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert todo.snapshot_state() == before
    assert agent.messages.messages == history
    assert store.load_metadata(agent.session_id).todo_state == before


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["add", "complete", "clear"])
async def test_completed_turn_keeps_and_persists_checklist_changes(tmp_path, action):
    agent, todo, _, store = prepared(tmp_path, action, outcome="success")
    before = todo.snapshot_state()
    stream = agent.run_turn("Continue")
    try:
        async for event in stream:
            if isinstance(event, DoneEvent):
                break
        else:
            pytest.fail("No successful terminal event was emitted")
    finally:
        await stream.aclose()

    assert todo.snapshot_state() != before
    assert store.load_metadata(agent.session_id).todo_state == todo.snapshot_state()
    assert agent.messages.messages[-1].text == "saved"
