import asyncio
import io
from unittest.mock import AsyncMock

import pytest
from rich.console import Console

from openagent.core.events import DoneEvent
from openagent.core.provider import ChatProvider
from openagent.core.types import Message, ToolCall, ToolSpec, Usage
from openagent.providers.custom import CustomJsonPathProvider
from openagent.providers.ollama import OllamaProvider
from openagent.providers.openai_compat import OpenAICompatProvider
from openagent.runner import AgentRunner
from openagent.tools.registry import ToolRegistry
from openagent.tui import app as tui


class FakeProvider(ChatProvider):
    model = "original"
    default_max_tokens = 100
    context_window = 10000

    def __init__(self):
        self.close = AsyncMock()

    async def stream(self, request):
        yield DoneEvent(
            message=Message.assistant("ok"), usage=Usage(prompt_tokens=12, completion_tokens=4)
        )


def setup_input(monkeypatch, lines):
    iterator = iter(lines)

    class Session:
        async def prompt_async(self, prompt):
            try:
                return next(iterator)
            except StopIteration:
                raise EOFError from None

    monkeypatch.setattr(tui, "PromptSession", lambda **kwargs: Session())
    output = io.StringIO()
    return Console(file=output, width=120, color_system=None), output


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [None, 100, 321])
async def test_model_command_swaps_provider_preserving_history(monkeypatch, explicit):
    console, output = setup_input(monkeypatch, ["/model", "/model ollama/llama3.2:7b", "/exit"])
    original = FakeProvider()
    runner = AgentRunner(original, max_tokens=explicit)
    history = runner.messages.messages
    tools = runner.tools
    await tui.run_repl(runner, console=console)
    assert isinstance(runner.provider, OllamaProvider)
    assert runner.model == "llama3.2:7b"
    assert runner.context_window == 32768
    assert runner.max_tokens == (8192 if explicit is None else explicit)
    assert runner.messages.messages == history
    assert runner.tools is tools
    original.close.assert_awaited_once()
    assert "original" in output.getvalue()
    await runner.provider.close()


@pytest.mark.asyncio
async def test_failed_model_switch_keeps_original_provider(monkeypatch):
    console, output = setup_input(monkeypatch, ["/model bedrock/unknown", "/exit"])
    original = FakeProvider()
    runner = AgentRunner(original)
    await tui.run_repl(runner, console=console)
    assert runner.provider is original
    assert runner.model == "original"
    original.close.assert_not_awaited()
    assert "not implemented" in output.getvalue()


@pytest.mark.asyncio
async def test_usage_commands_count_only_terminal_usage(monkeypatch):
    console, output = setup_input(monkeypatch, ["hi", "again", "/tokens", "/usage", "/exit"])
    runner = AgentRunner(FakeProvider())
    await tui.run_repl(runner, console=console)
    text = output.getvalue()
    assert "Prompt: 24" in text
    assert "Completion: 8" in text
    assert "Total: 32" in text
    assert "Context:" in text and "/ 10000" in text


@pytest.mark.asyncio
async def test_tools_table_keeps_untrusted_markup_literal(monkeypatch):
    console, output = setup_input(monkeypatch, ["/tools", "/help", "/exit"])
    runner = AgentRunner(FakeProvider(), tools=ToolRegistry())
    monkeypatch.setattr(
        runner.tools,
        "list_specs",
        lambda: [ToolSpec("[red]name[/red]", "[bold]description[/bold]", danger="execute")],
    )
    await tui.run_repl(runner, console=console)
    text = output.getvalue()
    assert "[red]name[/red]" in text
    assert "[bold]description[/bold]" in text
    assert "execute" in text
    assert "/model" in text and "/tokens" in text and "multiline" in text.lower()


@pytest.mark.asyncio
async def test_multiline_delimiter_collects_one_turn(monkeypatch):
    console, _ = setup_input(
        monkeypatch, ['"""', "first", "/tools", "third", '"""', "plain", "/exit"]
    )
    runner = AgentRunner(FakeProvider())
    await tui.run_repl(runner, console=console)
    users = [m.text for m in runner.messages.messages if m.role == "user"]
    assert users == ["first\n/tools\nthird", "plain"]


@pytest.mark.asyncio
async def test_permission_prompt_cancellation_finishes_input_task(monkeypatch):
    entered = asyncio.Event()
    stopped = asyncio.Event()
    async def prompt(self, message, **kwargs):
        assert kwargs['handle_sigint'] is False
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    monkeypatch.setattr(tui.PromptSession, 'prompt_async', prompt)
    callback = tui.make_ask_callback(console=Console(file=io.StringIO()))
    task = asyncio.create_task(callback(ToolCall('run', {})))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_same_openai_model_switch_keeps_configured_endpoint(monkeypatch):
    console, _ = setup_input(monkeypatch, ['/model gpt-4.1', '/exit'])
    original = OpenAICompatProvider('https://private.test/v1', 'gpt-4o', api_key='private-key', headers={'x-custom':'value'}, extra_body={'custom':True})
    runner = AgentRunner(original)
    await tui.run_repl(runner, console=console)
    assert runner.provider.base_url == 'https://private.test/v1'
    assert runner.provider.transport.extra_headers['Authorization'] == 'Bearer private-key'
    assert runner.provider.transport.extra_headers['x-custom'] == 'value'
    assert runner.provider.extra_body == {'custom':True}
    assert runner.model == runner.provider.model == 'gpt-4.1'
    assert not original.transport._client.is_closed
    await runner.provider.close()


@pytest.mark.asyncio
async def test_same_custom_model_switch_keeps_jsonpath_mapping(monkeypatch):
    console, _ = setup_input(monkeypatch, ['/model custom/new-model', '/exit'])
    original = CustomJsonPathProvider('https://private.test', 'old-model', text_path='output.text', tool_calls_path='output.calls', chat_endpoint='/generate',api_key='private-key')
    runner = AgentRunner(original)
    await tui.run_repl(runner, console=console)
    assert runner.provider.base_url == 'https://private.test'
    assert runner.provider.text_path == 'output.text'
    assert runner.provider.tool_calls_path == 'output.calls'
    assert runner.provider.chat_endpoint == '/generate'
    assert runner.provider.transport.extra_headers['Authorization'] == 'Bearer private-key'
    assert runner.model == runner.provider.model == 'new-model'
    await runner.provider.close()
