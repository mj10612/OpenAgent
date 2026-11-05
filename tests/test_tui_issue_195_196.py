import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from rich.console import Console

from openagent.core.events import DoneEvent
from openagent.core.provider import ChatProvider
from openagent.core.types import Message, ToolCall, Usage
from openagent.providers.custom import CustomJsonPathProvider
from openagent.runner import AgentRunner
from openagent.tui import app as tui


class InitialProvider(ChatProvider):
    model = "original"
    default_max_tokens = 100
    context_window = 10000

    def __init__(self):
        self.close = AsyncMock()

    async def stream(self, request):
        yield DoneEvent(message=Message.assistant("ok"), usage=Usage())


def scripted_input(monkeypatch, lines):
    answers = iter(lines)

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def __class_getitem__(cls, item):
            return cls

        async def prompt_async(self, prompt, **kwargs):
            try:
                return next(answers)
            except StopIteration:
                raise EOFError from None

    monkeypatch.setattr(tui, "PromptSession", Session)
    output = io.StringIO()
    return Console(file=output, width=120, color_system=None), output


@pytest.mark.asyncio
async def test_model_switch_uses_configured_provider_factory(monkeypatch):
    console, _ = scripted_input(monkeypatch, ["/model custom/new-model", "/exit"])
    original = InitialProvider()
    runner = AgentRunner(original)
    history = runner.messages
    resolved = []

    def configured_provider(model_ref):
        resolved.append(model_ref)
        return CustomJsonPathProvider(
            "https://private.test", model_ref.split("/", 1)[1],
            text_path="output.text", tool_calls_path="output.calls",
            chat_endpoint="/generate", api_key="private-key",
            auth_header="x-api-key", headers={"x-custom": "value"},
        )

    await tui.run_repl(runner, console=console, provider_factory=configured_provider)
    try:
        assert resolved == ["custom/new-model"]
        assert runner.model == "new-model"
        assert runner.provider.base_url == "https://private.test"
        assert runner.provider.text_path == "output.text"
        assert runner.provider.tool_calls_path == "output.calls"
        assert runner.provider.chat_endpoint == "/generate"
        assert runner.provider.transport.extra_headers["x-api-key"] == "private-key"
        assert runner.provider.transport.extra_headers["x-custom"] == "value"
        assert runner.messages is history
        original.close.assert_awaited_once()
    finally:
        await runner.provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_approve,answer,allowed", [(True, None, True), (False, "yes", True), (False, "no", False)])
async def test_approval_preview_only_runs_for_manual_confirmation(monkeypatch, auto_approve, answer, allowed):
    lines = ["perform action"] + ([answer] if answer else []) + ["/exit"]
    console, output = scripted_input(monkeypatch, lines)
    previews = []
    decisions = []
    tool = object()

    async def preview(candidate, arguments):
        previews.append((candidate, arguments))
        if auto_approve:
            raise RuntimeError("Preview must not run in auto mode")
        return "planned file changes"

    class Runner:
        model = "original"
        tools = SimpleNamespace(changes=SimpleNamespace(preview=preview), get=lambda name: tool)

        async def run_turn(self, user_input, ask_callback):
            decisions.append(await ask_callback(ToolCall("write_file", {"path": "a.txt"})))
            yield DoneEvent(message=Message.assistant("ok"), usage=Usage())

    await tui.run_repl(Runner(), console=console, auto_approve=auto_approve)
    assert decisions == [allowed]
    assert len(previews) == (0 if auto_approve else 1)
    assert ("planned file changes" in output.getvalue()) is (not auto_approve)
