"""Tests for Configuration, CLI Entrypoint, and Rich Streaming TUI.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import inspect
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from openagent.cli import main
from openagent.config import (
    OpenAgentConfig,
    load_config,
    save_config,
)
from openagent.core.events import (
    DoneEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallEnd,
    ToolResultEvent,
)
from openagent.core.types import FinishReason, Message, ToolCall, Usage
from openagent.session.store import SessionStore
from openagent.tools.registry import AskCallback, PermissionAction
from openagent.tui.app import TUIApp, make_ask_callback, run_repl


def test_config_defaults() -> None:
    config = OpenAgentConfig()
    assert config.model == "gpt-4o"
    assert config.api_key is None
    assert config.base_url is None
    assert config.temperature is None
    assert config.max_tokens is None
    assert config.max_tool_iterations == 25
    assert not config.auto_approve
    assert isinstance(config.danger_policy, dict)
    assert config.danger_policy.get("write") == PermissionAction.ASK
    assert config.danger_policy.get("none") == PermissionAction.ALLOW
    assert config.workspace == Path.cwd().resolve()


def test_config_from_toml_file(tmp_path: Path) -> None:
    toml_content = """
model = "claude-3-5-sonnet"
base_url = "https://api.anthropic.com/v1"
api_key = "test-anthropic-key"
temperature = 0.3
max_tokens = 4096
max_tool_iterations = 15
auto_approve = true
workspace = "."

[danger_policy]
write = "allow"
execute = "deny"
none = "allow"
network = "ask"

[[mcp_servers]]
name = "test-server"
transport = "stdio"
command = "python"
args = ["-m", "test_server"]
"""
    config_file = tmp_path / "openagent.toml"
    config_file.write_text(toml_content, encoding="utf-8")

    config = load_config(config_path=config_file)
    assert config.model == "claude-3-5-sonnet"
    assert config.base_url == "https://api.anthropic.com/v1"
    assert config.api_key == "test-anthropic-key"
    assert config.temperature == 0.3
    assert config.max_tokens == 4096
    assert config.max_tool_iterations == 15
    assert config.auto_approve is True
    assert config.danger_policy.get("write") == PermissionAction.ALLOW
    assert config.danger_policy.get("execute") == PermissionAction.DENY
    assert len(config.mcp_servers) == 1
    assert config.mcp_servers[0].name == "test-server"
    assert config.mcp_servers[0].command == "python"


def test_config_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAGENT_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("OPENAGENT_BASE_URL", "https://custom.api/v1")
    monkeypatch.setenv("OPENAGENT_API_KEY", "env-api-key")
    monkeypatch.setenv("OPENAGENT_TEMPERATURE", "0.7")
    monkeypatch.setenv("OPENAGENT_MAX_TOKENS", "8192")
    monkeypatch.setenv("OPENAGENT_MAX_TOOL_ITERATIONS", "40")
    monkeypatch.setenv("OPENAGENT_YES", "true")

    config = load_config()
    assert config.model == "gemini-2.0-flash"
    assert config.base_url == "https://custom.api/v1"
    assert config.api_key == "env-api-key"
    assert config.temperature == 0.7
    assert config.max_tokens == 8192
    assert config.max_tool_iterations == 40
    assert config.auto_approve is True


def test_config_hierarchical_merging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    global_dir = tmp_path / "global"
    global_dir.mkdir(parents=True)
    global_file = global_dir / "config.toml"
    global_file.write_text('model = "global-model"\ntemperature = 0.1\nmax_tokens = 1000\n', encoding="utf-8")

    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True)
    project_file = project_dir / "openagent.toml"
    project_file.write_text('model = "project-model"\nmax_tokens = 2000\n', encoding="utf-8")

    monkeypatch.setenv("OPENAGENT_GLOBAL_CONFIG", str(global_file))
    monkeypatch.setenv("OPENAGENT_TEMPERATURE", "0.9")

    # Local project config overrides global config; env var overrides project config; explicit kwargs override all
    config = load_config(
        config_path=project_file,
        max_tool_iterations=50,
    )

    assert config.model == "project-model"  # from project file (overriding global)
    assert config.max_tokens == 2000  # from project file (overriding global 1000)
    assert config.temperature == 0.9  # from env var (overriding global 0.1)
    assert config.max_tool_iterations == 50  # from explicit override


def test_save_and_reload_config(tmp_path: Path) -> None:
    target = tmp_path / "subdir" / "saved_config.toml"
    config = OpenAgentConfig(
        model="custom-model",
        base_url="https://example.com/v1",
        api_key="secret-key",
        temperature=0.4,
        max_tokens=2048,
        max_tool_iterations=10,
        auto_approve=True,
        danger_policy={"write": PermissionAction.ALLOW, "execute": PermissionAction.DENY},
    )

    save_config(config, target)
    assert target.is_file()

    reloaded = load_config(config_path=target)
    assert reloaded.model == "custom-model"
    assert reloaded.base_url == "https://example.com/v1"
    assert reloaded.api_key == "secret-key"
    assert reloaded.temperature == 0.4
    assert reloaded.max_tokens == 2048
    assert reloaded.max_tool_iterations == 10
    assert reloaded.auto_approve is True
    assert reloaded.danger_policy.get("execute") == PermissionAction.DENY


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--help"])
    assert code == 0
    captured = capsys.readouterr()
    assert "OpenAgent" in captured.out or "openagent" in captured.out
    assert "run" in captured.out
    assert "chat" in captured.out
    assert "models" in captured.out
    assert "sessions" in captured.out
    assert "config" in captured.out


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--version"])
    assert code == 0
    captured = capsys.readouterr()
    assert "0.1.0" in captured.out


def test_cli_models_subcommand() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None)
    with patch("openagent.cli.get_console", return_value=console):
        code = main(["models"])
    assert code == 0
    output = buf.getvalue()
    assert "openai" in output
    assert "anthropic" in output
    assert "gemini" in output


def test_cli_config_subcommand(tmp_path: Path) -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None)
    with patch("openagent.cli.get_console", return_value=console):
        code = main(["config", "--model", "test-model-42"])
    assert code == 0
    output = buf.getvalue()
    assert "test-model-42" in output
    assert "Active Configuration" in output or "Configuration" in output


def test_cli_sessions_subcommand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("OPENAGENT_SESSION_DIR", str(sessions_dir))
    store = SessionStore(storage_dir=sessions_dir)
    s_id = store.create_session(model="gpt-4o", title="Test Session CLI")

    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=False, color_system=None)
    with patch("openagent.cli.get_console", return_value=console):
        code = main(["sessions"])
    assert code == 0
    output = buf.getvalue()
    assert s_id[:8] in output
    assert "Test Session CLI" in output


def test_cli_run_subcommand_mocked(tmp_path: Path) -> None:
    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=False, color_system=None)

    async def mock_events(*args, **kwargs):
        yield TextDelta(text="Hello from mocked agent!")
        yield DoneEvent(
            finish_reason=FinishReason.STOP,
            message=Message.assistant("Hello from mocked agent!"),
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )

    with (
        patch("openagent.cli.get_console", return_value=console),
        patch("openagent.runner.AgentRunner.run_turn", side_effect=mock_events),
    ):
        code = main(["run", "Hello world", "--yes", "-w", str(tmp_path)])
        output = buf.getvalue()
    assert code == 0, f"code was {code}, output: {output!r}"
    assert "Hello from mocked agent!" in output


def test_tui_app_render_events() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None)
    app = TUIApp(console=console)

    app.render_event(ThinkingDelta(text="Thinking deeply..."))
    app.render_event(TextDelta(text="Answer to the question."))
    call = ToolCall(id="call_1", name="read_file", arguments={"path": "test.txt"})
    app.render_event(ToolCallEnd(index=0, call=call))
    app.render_event(
        ToolResultEvent(call_id="call_1", tool_name="read_file", output="file contents", is_error=False)
    )
    app.render_event(
        DoneEvent(
            finish_reason=FinishReason.STOP,
            message=Message.assistant("Answer to the question."),
            usage=Usage(prompt_tokens=20, completion_tokens=10),
        )
    )

    output = buf.getvalue()
    assert "Thinking deeply..." in output
    assert "Answer to the question." in output
    assert "read_file" in output
    assert "file contents" in output


async def _invoke_callback(cb: AskCallback, call: ToolCall) -> bool:
    outcome = cb(call)
    if inspect.isawaitable(outcome):
        return bool(await outcome)
    return bool(outcome)


@pytest.mark.asyncio
async def test_tui_ask_callback_auto_approve() -> None:
    cb = make_ask_callback(console=Console(), auto_approve=True)
    call = ToolCall(id="call_1", name="execute_command", arguments={"command": "dir"})
    res = await _invoke_callback(cb, call)
    assert res is True


@pytest.mark.asyncio
async def test_tui_ask_callback_prompt_yes_and_no() -> None:
    # Test answering yes
    with patch("prompt_toolkit.PromptSession.prompt_async", new=AsyncMock(return_value="yes")):
        cb = make_ask_callback(console=Console(), auto_approve=False)
        call = ToolCall(id="call_1", name="execute_command", arguments={"command": "dir"})
        res = await _invoke_callback(cb, call)
        assert res is True

    # Test answering no
    with patch("prompt_toolkit.PromptSession.prompt_async", new=AsyncMock(return_value="no")):
        cb = make_ask_callback(console=Console(), auto_approve=False)
        call = ToolCall(id="call_1", name="execute_command", arguments={"command": "dir"})
        res = await _invoke_callback(cb, call)
        assert res is False


def test_cli_run_without_explicit_subcommand(tmp_path: Path) -> None:
    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=False, color_system=None)

    async def mock_events(*args, **kwargs):
        yield TextDelta(text="Direct prompt executed!")
        yield DoneEvent(
            finish_reason=FinishReason.STOP,
            message=Message.assistant("Direct prompt executed!"),
            usage=Usage(prompt_tokens=5, completion_tokens=5),
        )

    with (
        patch("openagent.cli.get_console", return_value=console),
        patch("openagent.runner.AgentRunner.run_turn", side_effect=mock_events),
    ):
        code = main(["how much is 5 plus 5?", "--yes", "-w", str(tmp_path)])
    assert code == 0
    output = buf.getvalue()
    assert "Direct prompt executed!" in output


def test_cli_chat_subcommand_mocked() -> None:
    with (
        patch("openagent.cli.get_console", return_value=Console()),
        patch("openagent.cli.run_repl", new_callable=AsyncMock) as mock_repl,
    ):
        code = main(["chat", "-m", "claude-3-5-sonnet"])
    assert code == 0
    mock_repl.assert_awaited_once()


def test_config_file_not_found_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(config_path="this_file_does_not_exist_12345.toml")


def test_config_mcp_dict_syntax(tmp_path: Path) -> None:
    toml_content = """
[mcp_servers.sqlite]
transport = "stdio"
command = "uvx"
args = ["mcp-server-sqlite", "--db-path", "test.db"]
"""
    cfg_file = tmp_path / "openagent.toml"
    cfg_file.write_text(toml_content, encoding="utf-8")
    cfg = load_config(config_path=cfg_file)
    assert len(cfg.mcp_servers) == 1
    assert cfg.mcp_servers[0].name == "sqlite"
    assert cfg.mcp_servers[0].command == "uvx"
    assert cfg.mcp_servers[0].args == ["mcp-server-sqlite", "--db-path", "test.db"]


def test_tui_app_tool_error_and_long_output() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None)
    app = TUIApp(console=console)

    long_output = "a" * 2500
    app.render_event(
        ToolResultEvent(call_id="call_err", tool_name="bad_tool", output="Execution failed", is_error=True)
    )
    app.render_event(
        ToolResultEvent(call_id="call_long", tool_name="long_tool", output=long_output, is_error=False)
    )

    output = buf.getvalue()
    assert "Tool Error (bad_tool)" in output
    assert "Execution failed" in output
    assert "Tool Result (long_tool)" in output
    assert "output truncated" in output


@pytest.mark.asyncio
async def test_run_repl_loop() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None)

    runner = MagicMock()
    runner.model = "test-model"
    runner.workspace_root = "."
    runner.session_id = "test-session-1234"

    # Simulate user typing "/clear" then "/exit"
    with (
        patch("builtins.input", side_effect=["/clear", "/exit"]),
        patch("prompt_toolkit.PromptSession.prompt_async", side_effect=["/clear", "/exit"]),
    ):
        await run_repl(runner, console=console)

    runner.reset.assert_called_once()
    output = buf.getvalue()
    assert "OpenAgent" in output
    assert "Context cleared" in output
    assert "Exiting OpenAgent" in output


def test_cli_flags_before_subcommand() -> None:
    from openagent.cli import _build_parser

    parser = _build_parser()
    args1 = parser.parse_args(["--yes", "run", "do task"])
    assert args1.subcommand == "run"
    assert args1.yes is True
    assert args1.prompt == ["do task"]

    args2 = parser.parse_args(["-m", "anthropic/claude-3-5-sonnet", "chat"])
    assert args2.subcommand == "chat"
    assert args2.model == "anthropic/claude-3-5-sonnet"


def test_cli_shorthand_with_leading_flags(tmp_path: Path) -> None:
    captured_args = {}

    async def fake_async_run(args, console):
        captured_args["subcommand"] = args.subcommand
        captured_args["model"] = args.model
        captured_args["yes"] = args.yes
        captured_args["prompt"] = args.prompt
        return 0

    with patch("openagent.cli._async_run", side_effect=fake_async_run):
        code = main(["-m", "custom-model", "-y", "execute this prompt"])
        assert code == 0
        assert captured_args["subcommand"] == "run"
        assert captured_args["model"] == "custom-model"
        assert captured_args["yes"] is True
        assert captured_args["prompt"] == ["execute this prompt"]


def test_save_config_with_mcp_env(tmp_path: Path) -> None:
    from openagent.tools.mcp.client import MCPServerConfig

    cfg_file = tmp_path / "mcp_env_config.toml"
    cfg = OpenAgentConfig(
        model="gpt-4o-mini",
        mcp_servers=[
            MCPServerConfig(
                name="env_server",
                transport="stdio",
                command="node",
                args=["server.js"],
                env={"API_KEY": "secret123", "DEBUG": "1"},
            )
        ],
    )
    save_config(cfg, cfg_file)

    reloaded = load_config(config_path=cfg_file)
    assert len(reloaded.mcp_servers) == 1
    assert reloaded.mcp_servers[0].name == "env_server"
    assert reloaded.mcp_servers[0].env == {"API_KEY": "secret123", "DEBUG": "1"}


