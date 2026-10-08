from __future__ import annotations

import asyncio
import io
import os
import signal
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from openagent.cli import main
from openagent.config import OpenAgentConfig, load_config, save_config
from openagent.core.events import ErrorEvent, ToolCallEnd, ToolResultEvent
from openagent.core.types import ToolCall
from openagent.session.store import SessionStore
from openagent.tools.mcp.client import MCPServerConfig
from openagent.tui.app import TUIApp, run_repl


def test_config_relative_paths_use_config_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = project / "config.toml"
    config.write_text(
        'workspace = "sub"\n[[mcp_servers]]\nname = "local"\ncommand = "server"\ncwd = "tools"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    loaded = load_config(config)
    assert loaded.workspace == (project / "sub").resolve()
    assert Path(loaded.mcp_servers[0].cwd or "") == (project / "tools").resolve()


def test_config_round_trip_unicode_and_mcp_options(tmp_path: Path) -> None:
    instruction = 'emoji: \U0001f600; quotes: "hello"; slash: \\; newline:\nnext'
    server = MCPServerConfig(
        name="remote",
        transport="sse",
        url="https://example.org/sse",
        headers={"X-Key": "value"},
        timeout=12.5,
        read_timeout=99.0,
    )
    config = OpenAgentConfig(extra_instructions=[instruction], mcp_servers=[server])
    path = tmp_path / "config.toml"
    save_config(config, path)
    restored = load_config(path)
    assert restored.extra_instructions == [instruction]
    assert restored.mcp_servers[0] == server


def test_config_mcp_options_round_trip_without_unicode(tmp_path: Path) -> None:
    server = MCPServerConfig(
        name="remote",
        transport="sse",
        url="https://example.org/sse",
        headers={"Authorization": "Bearer test"},
        timeout=12.5,
        read_timeout=99.0,
    )
    path = tmp_path / "config.toml"
    save_config(OpenAgentConfig(mcp_servers=[server]), path)
    assert load_config(path).mcp_servers[0] == server


def test_config_failed_atomic_replace_preserves_original(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('model = "original"\n', encoding="utf-8")
    with (
        patch("openagent.config.os.replace", side_effect=OSError("disk failure")),
        pytest.raises(OSError, match="disk failure"),
    ):
        save_config(OpenAgentConfig(model="changed"), path)
    assert path.read_text(encoding="utf-8") == 'model = "original"\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.toml"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_config_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    save_config(OpenAgentConfig(api_key="test-secret"), path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "event",
    [
        ToolResultEvent(call_id="test", tool_name="[blink]name", output="[/] [bold]literal[/bold]"),
        ToolCallEnd(index=0, call=ToolCall(name="[/]tool", arguments={"value": "[/]literal"})),
        ErrorEvent(error=ValueError("[/]error")),
    ],
)
def test_tui_renders_untrusted_markup_literally(event: object) -> None:
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, width=120)
    TUIApp(console).render_event(event)  # type: ignore[arg-type]
    assert "[/]" in output.getvalue()


def test_sessions_delete_by_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAGENT_SESSION_DIR", str(tmp_path))
    store = SessionStore(tmp_path)
    sid = store.create_session(model="mock")
    assert main(["sessions", "delete", sid[:8], "--yes"]) == 0
    assert not store.list_sessions()


def test_sessions_prune(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAGENT_SESSION_DIR", str(tmp_path))
    store = SessionStore(tmp_path)
    store.create_session(model="mock")
    assert main(["sessions", "prune", "--older-than", "0", "--yes"]) == 0
    assert not store.list_sessions()


def test_sessions_display_untrusted_titles_literally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAGENT_SESSION_DIR", str(tmp_path))
    store = SessionStore(tmp_path)
    sid = store.create_session(model="[/]mock", title="[bold]literal[/bold]")
    output = io.StringIO()
    monkeypatch.setattr("openagent.cli.get_console", lambda: Console(file=output, width=160))
    assert main(["sessions"]) == 0
    assert "[bold]literal[/bold]" in output.getvalue()
    assert "[/]mock" in output.getvalue()
    assert sid in output.getvalue()


@pytest.mark.asyncio
async def test_repl_sigint_cancels_only_active_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = iter(["first", "second", "/exit"])
    seen: list[str] = []

    class Prompt:
        async def prompt_async(self, _: str) -> str:
            return next(prompts)

    class Runner:
        model = "mock"
        workspace_root = "."
        session_id = None

        async def run_turn(self, user_input: str, **kwargs: object):  # type: ignore[no-untyped-def]
            seen.append(user_input)
            if user_input == "first":
                handler = signal.getsignal(signal.SIGINT)
                assert callable(handler)
                handler(signal.SIGINT, None)
                await asyncio.sleep(0)
            if False:
                yield None

    monkeypatch.setattr("openagent.tui.app.PromptSession", lambda **kwargs: Prompt())
    original = signal.getsignal(signal.SIGINT)
    await run_repl(Runner(), console=Console(file=io.StringIO()))
    assert seen == ["first", "second"]
    assert signal.getsignal(signal.SIGINT) == original
