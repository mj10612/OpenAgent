import json

import httpx
import pytest

from openagent.core.events import DoneEvent, ErrorEvent, TextDelta, ToolCallStart
from openagent.core.provider import ProviderError
from openagent.core.types import ChatRequest, Message
from openagent.providers.gemini import GeminiProvider
from openagent.runner import AgentRunner
from openagent.session.store import SessionStore
from openagent.tools.base import Tool, ToolResult
from openagent.tools.registry import ToolRegistry


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("reason", [
    "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS", "MISSING_THOUGHT_SIGNATURE", "FUTURE_FINISH_REASON",
])
async def test_unhandled_finish_is_error_even_after_partial_response(reason, partial):
    chunks = [{"candidates": [{"content": {"parts": [{"text": "partial"}]}}]}] if partial else []
    chunks.append({"candidates": [{"finishReason": reason, "content": {"parts": [
        {"functionCall": {"name": "record", "args": {}}},
    ]}}]})
    body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    response = httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
        provider = GeminiProvider("fixture", client=client)
        events = [event async for event in provider.stream(ChatRequest(
            model="fixture", messages=[Message.user("Hello")],
        ))]

    assert isinstance(events[-1], ErrorEvent)
    assert isinstance(events[-1].error, ProviderError)
    assert events[-1].error.provider == "gemini"
    assert reason in str(events[-1].error)
    assert sum(isinstance(event, ErrorEvent) for event in events) == 1
    assert not any(isinstance(event, (DoneEvent, ToolCallStart)) for event in events)
    assert [event.text for event in events if isinstance(event, TextDelta)] == (
        ["partial"] if partial else []
    )
    assert response.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,expected", [
    ("STOP", "stop"), ("MAX_TOKENS", "length"), ("SAFETY", "content_filter"),
    ("RECITATION", "content_filter"), ("BLOCKLIST", "content_filter"),
    ("PROHIBITED_CONTENT", "content_filter"), ("SPII", "content_filter"),
])
async def test_supported_finish_reasons_keep_existing_classification(reason, expected):
    body = "data: " + json.dumps({"candidates": [{"finishReason": reason}]}) + "\n\n"
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, text=body, headers={"content-type": "text/event-stream"},
    ))) as client:
        provider = GeminiProvider("fixture", client=client)
        events = [event async for event in provider.stream(ChatRequest(
            model="fixture", messages=[Message.user("Hello")],
        ))]

    assert isinstance(events[-1], DoneEvent)
    assert events[-1].finish_reason.value == expected
    assert not any(isinstance(event, ErrorEvent) for event in events)


class RecordingTool(Tool):
    name = "record"
    danger = "none"

    def __init__(self):
        super().__init__()
        self.executions = 0

    async def execute(self, **kwargs):
        self.executions += 1
        return ToolResult(call_id="", output="recorded")


@pytest.mark.asyncio
async def test_unhandled_finish_prevents_execution_of_earlier_streamed_tool_call(tmp_path):
    chunks = [
        {"candidates": [{"content": {"parts": [{"functionCall": {"name": "record", "args": {}}}]}}]},
        {"candidates": [{"finishReason": "TOO_MANY_TOOL_CALLS"}]},
    ]
    body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, text=body, headers={"content-type": "text/event-stream"},
    ))) as client:
        tool = RecordingTool()
        agent = AgentRunner(GeminiProvider("fixture", client=client), tools=ToolRegistry([tool]),
                            session_store=SessionStore(tmp_path), max_tool_iterations=2)
        before = agent.messages.copy().messages

        with pytest.raises(ProviderError, match="TOO_MANY_TOOL_CALLS"):
            await agent.run_turn_to_completion("Hello")

        assert tool.executions == 0
        assert agent.messages.messages == before
        assert agent.session_id is None
        assert not list(tmp_path.glob("*.jsonl"))
