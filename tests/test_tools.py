"""Tests for OpenAgent tool execution system, sandboxed filesystem, shell, and agent tools.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from openagent.core.types import Message, ToolCall, ToolParam, ToolSpec
from openagent.tools.agent_tools import ThinkTool, TodoTool, WebFetchTool
from openagent.tools.base import Tool, ToolResult
from openagent.tools.fs import (
    EditFileTool,
    GlobFindTool,
    GrepSearchTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
    create_fs_tools,
)
from openagent.tools.registry import PermissionAction, ToolRegistry
from openagent.tools.shell import ShellTool, execute_shell

# =========================================================================== #
# Dummy tools for registry testing
# =========================================================================== #


class DummyEchoTool(Tool):
    name = "echo"
    description = "Echo input back"
    danger = "none"

    def __init__(self) -> None:
        self.params = [ToolParam(name="text", type="string", description="Text to echo")]

    async def execute(self, **kwargs: Any) -> ToolResult:
        text = kwargs.get("text", "")
        return ToolResult(call_id=kwargs.get("call_id", ""), output=str(text))


class DummyDangerousWriteTool(Tool):
    name = "dangerous_write"
    description = "A tool with write danger"
    danger = "write"

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(call_id=kwargs.get("call_id", ""), output="wrote data")


class DummyDangerousExecTool(Tool):
    name = "dangerous_exec"
    description = "A tool with execute danger"
    danger = "execute"

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(call_id=kwargs.get("call_id", ""), output="executed code")


# =========================================================================== #
# 1. Base Tool and ToolResult tests
# =========================================================================== #


def test_tool_result_creation_and_to_message() -> None:
    res = ToolResult(call_id="call_123", output="hello world", is_error=False)
    assert res.call_id == "call_123"
    assert res.output == "hello world"
    assert not res.is_error

    msg = res.to_tool_message(name="test_tool")
    assert isinstance(msg, Message)
    assert msg.role == "tool"
    assert msg.tool_call_id == "call_123"
    assert msg.name == "test_tool"
    assert msg.text == "<tool_output>\nhello world\n</tool_output>"
    assert not msg.is_error

    err_res = ToolResult(call_id="call_456", output="file not found", is_error=True)
    err_msg = err_res.to_tool_message(name="read_file")
    assert err_msg.is_error


def test_custom_tool_spec_generation() -> None:
    tool = DummyEchoTool()
    assert tool.name == "echo"
    assert tool.danger == "none"
    assert tool.source == "builtin"
    spec = tool.spec
    assert isinstance(spec, ToolSpec)
    assert spec.name == "echo"
    assert spec.description == "Echo input back"
    assert len(spec.params) == 1
    assert spec.params[0].name == "text"


# =========================================================================== #
# 2. ToolRegistry and Permission tests
# =========================================================================== #


def test_tool_registry_registration_and_lookup() -> None:
    reg = ToolRegistry()
    tool = DummyEchoTool()
    reg.register(tool)

    assert "echo" in reg
    assert len(reg) == 1
    assert reg.get("echo") is tool
    assert reg.get("nonexistent") is None

    specs = reg.list_specs()
    assert len(specs) == 1
    assert specs[0].name == "echo"

    tools = reg.list_tools()
    assert len(tools) == 1
    assert tools[0] is tool

    reg.unregister("echo")
    assert "echo" not in reg
    assert len(reg) == 0


@pytest.mark.asyncio
async def test_tool_registry_default_danger_policies() -> None:
    reg = ToolRegistry()
    echo = DummyEchoTool()  # danger="none" -> ALLOW
    writer = DummyDangerousWriteTool()  # danger="write" -> ASK
    executor = DummyDangerousExecTool()  # danger="execute" -> ASK
    web = WebFetchTool()  # danger="network" -> ASK

    reg.register(echo)
    reg.register(writer)
    reg.register(executor)
    reg.register(web)

    # 1. danger="none" executes without confirmation
    call_echo = ToolCall(id="call_1", name="echo", arguments={"text": "hello"})
    res_echo = await reg.execute_call(call_echo)
    assert not res_echo.is_error
    assert res_echo.output == "hello"

    # 2. danger="write" requires confirmation: fails if no callback provided
    call_write = ToolCall(id="call_2", name="dangerous_write", arguments={})
    res_write = await reg.execute_call(call_write)
    assert res_write.is_error
    assert "Permission denied" in res_write.output

    # 3. danger="write" with ask_callback returning True proceeds
    async def approve_all(call: ToolCall) -> bool:
        return True

    res_write_approved = await reg.execute_call(call_write, ask_callback=approve_all)
    assert not res_write_approved.is_error
    assert res_write_approved.output == "wrote data"

    # 4. danger="write" with ask_callback returning False is rejected
    async def reject_all(call: ToolCall) -> bool:
        return False

    res_write_rejected = await reg.execute_call(call_write, ask_callback=reject_all)
    assert res_write_rejected.is_error
    assert "rejected" in res_write_rejected.output.lower()

    # 5. danger="execute" requires confirmation
    call_exec = ToolCall(id="call_3", name="dangerous_exec", arguments={})
    res_exec = await reg.execute_call(call_exec)
    assert res_exec.is_error


@pytest.mark.asyncio
async def test_tool_registry_custom_policy_override() -> None:
    reg = ToolRegistry()
    echo = DummyEchoTool()
    reg.register(echo)

    # Explicitly deny echo
    reg.set_tool_policy("echo", PermissionAction.DENY)
    call = ToolCall(id="call_1", name="echo", arguments={"text": "hi"})
    res = await reg.execute_call(call)
    assert res.is_error
    assert "blocked by policy" in res.output.lower()

    # Override danger policy for "write" to ALLOW
    writer = DummyDangerousWriteTool()
    reg.register(writer)
    reg.set_danger_policy("write", PermissionAction.ALLOW)
    call_write = ToolCall(id="call_2", name="dangerous_write", arguments={})
    res_write = await reg.execute_call(call_write)
    assert not res_write.is_error
    assert res_write.output == "wrote data"


@pytest.mark.asyncio
async def test_tool_registry_tool_not_found() -> None:
    reg = ToolRegistry()
    call = ToolCall(id="call_99", name="unknown_tool", arguments={})
    res = await reg.execute_call(call)
    assert res.is_error
    assert "Tool not found" in res.output


@pytest.mark.asyncio
async def test_tool_registry_execute_parallel() -> None:
    reg = ToolRegistry()
    reg.register(DummyEchoTool())

    calls = [
        ToolCall(id="c1", name="echo", arguments={"text": "first"}),
        ToolCall(id="c2", name="echo", arguments={"text": "second"}),
        ToolCall(id="c3", name="unknown", arguments={}),
    ]

    results = await reg.execute_parallel(calls)
    assert len(results) == 3
    assert results[0].output == "first"
    assert not results[0].is_error
    assert results[1].output == "second"
    assert not results[1].is_error
    assert results[2].is_error


# =========================================================================== #
# 3. Sandboxed Filesystem tests
# =========================================================================== #


@pytest.fixture
def sandbox_env(tmp_path: Path) -> tuple[Path, dict[str, Tool]]:
    ws = tmp_path / "workspace"
    ws.mkdir()
    tools = {t.name: t for t in create_fs_tools(ws)}
    return ws, tools


def test_fs_tool_classes_direct_instantiation(tmp_path: Path) -> None:
    r = ReadFileTool(tmp_path)
    assert r.name == "read_file"
    w = WriteFileTool(tmp_path)
    assert w.name == "write_file"
    e = EditFileTool(tmp_path)
    assert e.name == "edit_file"
    ld = ListDirectoryTool(tmp_path)
    assert ld.name == "list_directory"
    g = GlobFindTool(tmp_path)
    assert g.name == "glob_find"
    gs = GrepSearchTool(tmp_path)
    assert gs.name == "grep_search"


@pytest.mark.asyncio
async def test_fs_sandboxing_path_traversal_prevention(sandbox_env: tuple[Path, dict[str, Tool]]) -> None:
    ws, tools = sandbox_env
    outside_file = ws.parent / "secret.txt"
    outside_file.write_text("secret_password", encoding="utf-8")

    read_tool = tools["read_file"]
    write_tool = tools["write_file"]
    edit_tool = tools["edit_file"]

    # 1. Traversal via .. escaping workspace
    res = await read_tool.execute(path="../secret.txt")
    assert res.is_error
    assert "outside workspace" in res.output.lower() or "denied" in res.output.lower()

    # 2. Direct absolute path outside workspace
    res2 = await read_tool.execute(path=str(outside_file.resolve()))
    assert res2.is_error
    assert "outside workspace" in res2.output.lower() or "denied" in res2.output.lower()

    # 3. Writing outside workspace
    res3 = await write_tool.execute(path="../hacked.txt", content="payload")
    assert res3.is_error
    assert "outside workspace" in res3.output.lower() or "denied" in res3.output.lower()
    assert not (ws.parent / "hacked.txt").exists()

    # 4. Editing outside workspace
    res4 = await edit_tool.execute(path="../secret.txt", old_str="secret", new_str="exposed")
    assert res4.is_error
    assert outside_file.read_text(encoding="utf-8") == "secret_password"


@pytest.mark.asyncio
async def test_fs_write_and_read_file(sandbox_env: tuple[Path, dict[str, Tool]]) -> None:
    ws, tools = sandbox_env
    read_tool = tools["read_file"]
    write_tool = tools["write_file"]

    # Write creates subdirectories automatically
    res = await write_tool.execute(
        path="src/nested/app.py",
        content="line1\nline2\nline3\nline4\nline5\n",
    )
    assert not res.is_error
    assert (ws / "src" / "nested" / "app.py").exists()

    # Overwrite protection
    res_no_overwrite = await write_tool.execute(
        path="src/nested/app.py",
        content="new content",
        overwrite=False,
    )
    assert res_no_overwrite.is_error
    assert "already exists" in res_no_overwrite.output.lower()

    # Read with 1-indexed offset and limit
    read_res = await read_tool.execute(path="src/nested/app.py", offset=2, limit=2)
    assert not read_res.is_error
    lines = read_res.output.strip().split("\n")
    assert len(lines) == 2
    assert "2: line2" in lines[0]
    assert "3: line3" in lines[1]


@pytest.mark.asyncio
async def test_fs_edit_file_uniqueness(sandbox_env: tuple[Path, dict[str, Tool]]) -> None:
    ws, tools = sandbox_env
    write_tool = tools["write_file"]
    edit_tool = tools["edit_file"]

    file_path = "module.py"
    await write_tool.execute(
        path=file_path,
        content="def foo():\n    return 42\n\ndef bar():\n    return 42\n",
    )

    # 1. Target string not found
    res_not_found = await edit_tool.execute(path=file_path, old_str="def baz()", new_str="pass")
    assert res_not_found.is_error
    assert "not found" in res_not_found.output.lower()

    # 2. Target string appears multiple times (not unique)
    res_ambiguous = await edit_tool.execute(path=file_path, old_str="return 42", new_str="return 100")
    assert res_ambiguous.is_error
    assert "unique" in res_ambiguous.output.lower() or "2 times" in res_ambiguous.output.lower()

    # 3. Unique replacement succeeds
    res_ok = await edit_tool.execute(
        path=file_path,
        old_str="def foo():\n    return 42",
        new_str="def foo():\n    return 999",
    )
    assert not res_ok.is_error
    updated = (ws / file_path).read_text(encoding="utf-8")
    assert "return 999" in updated
    assert "def bar():\n    return 42" in updated


@pytest.mark.asyncio
async def test_fs_directory_and_search_tools(sandbox_env: tuple[Path, dict[str, Tool]]) -> None:
    _ws, tools = sandbox_env
    write_tool = tools["write_file"]
    list_tool = tools["list_directory"]
    glob_tool = tools["glob_find"]
    grep_tool = tools["grep_search"]

    await write_tool.execute(path="pkg/a.py", content="# alpha module\nimport os\n")
    await write_tool.execute(path="pkg/b.py", content="# beta module\nimport sys\n")
    await write_tool.execute(path="docs/readme.txt", content="Documentation file\n")

    # 1. List directory non-recursive
    res_list = await list_tool.execute(path="pkg")
    assert not res_list.is_error
    assert "a.py" in res_list.output
    assert "b.py" in res_list.output

    # 2. Glob find
    res_glob = await glob_tool.execute(pattern="*.py", path="pkg")
    assert not res_glob.is_error
    assert "a.py" in res_glob.output
    assert "b.py" in res_glob.output
    assert "readme.txt" not in res_glob.output

    # 3. Grep search
    res_grep = await grep_tool.execute(query="beta module", path=".")
    assert not res_grep.is_error
    assert "b.py" in res_grep.output
    assert "beta module" in res_grep.output

    res_grep_empty = await grep_tool.execute(query="nonexistent_needle", path=".")
    assert not res_grep_empty.is_error
    assert "no matches" in res_grep_empty.output.lower()


# =========================================================================== #
# 4. Shell execution tests
# =========================================================================== #


@pytest.mark.asyncio
async def test_shell_tool_successful_execution(tmp_path: Path) -> None:
    shell_tool = ShellTool(workspace_root=tmp_path)
    # Simple command that works on both Windows and Unix
    res = await shell_tool.execute(command="python -c \"print('Hello from shell')\"")
    assert not res.is_error
    assert "Hello from shell" in res.output

    # Test convenience helper execute_shell
    res_helper = await execute_shell(
        command="python -c \"print('Hello from helper')\"",
        cwd=tmp_path,
        workspace_root=tmp_path,
    )
    assert not res_helper.is_error
    assert "Hello from helper" in res_helper.output


@pytest.mark.asyncio
async def test_shell_tool_exit_code_failure(tmp_path: Path) -> None:
    shell_tool = ShellTool(workspace_root=tmp_path)
    res = await shell_tool.execute(command="python -c \"import sys; sys.stderr.write('failed badly\\n'); sys.exit(42)\"")
    assert res.is_error
    assert "42" in res.output
    assert "failed badly" in res.output


@pytest.mark.asyncio
async def test_shell_tool_timeout(tmp_path: Path) -> None:
    shell_tool = ShellTool(workspace_root=tmp_path)
    # Command sleeps for 2 seconds with 0.2 second timeout
    res = await shell_tool.execute(
        command="python -c \"import time; time.sleep(2)\"",
        timeout=0.2,
    )
    assert res.is_error
    assert "timed out" in res.output.lower()


@pytest.mark.asyncio
async def test_shell_tool_output_capping(tmp_path: Path) -> None:
    shell_tool = ShellTool(workspace_root=tmp_path)
    # Emit 150KB of data
    res = await shell_tool.execute(
        command="python -c \"import sys; sys.stdout.write('A' * 150000)\"",
    )
    assert not res.is_error
    # 100KB is 102400 bytes, plus truncation note
    assert len(res.output) <= 105000
    assert "truncated" in res.output.lower()


# =========================================================================== #
# 5. Agent Tools tests (Think, Todo, WebFetch)
# =========================================================================== #


@pytest.mark.asyncio
async def test_think_tool() -> None:
    think = ThinkTool()
    res = await think.execute(thought="Analyzing the database schema...")
    assert not res.is_error
    assert "Analyzing the database schema" in res.output


@pytest.mark.asyncio
async def test_todo_tool() -> None:
    todo = TodoTool()

    # Add tasks
    res_add = await todo.execute(action="add", tasks=["Setup repo", "Write tests", "Run CI"])
    assert not res_add.is_error
    assert "Setup repo" in res_add.output
    assert "[ ]" in res_add.output

    # List tasks
    res_list = await todo.execute(action="list")
    assert not res_list.is_error
    assert "Write tests" in res_list.output

    # Complete task
    res_done = await todo.execute(action="complete", task="Setup repo")
    assert not res_done.is_error
    assert "[x] Setup repo" in res_done.output or "[X] Setup repo" in res_done.output.upper()

    # Clear tasks
    res_clear = await todo.execute(action="clear")
    assert not res_clear.is_error
    assert "cleared" in res_clear.output.lower() or "empty" in res_clear.output.lower()


@pytest.mark.asyncio
async def test_web_fetch_tool_success() -> None:
    web = WebFetchTool()
    html_content = "<html><head><title>Test</title></head><body><h1>Hello World</h1><p>Sample paragraph.</p></body></html>"

    mock_resp = httpx.Response(
        status_code=200,
        stream=httpx.ByteStream(html_content.encode()),
        headers={"content-type": "text/html"},
        request=httpx.Request("GET", "https://example.com"),
    )

    with patch.object(httpx.AsyncClient, "send", AsyncMock(return_value=mock_resp)), patch("openagent.tools.agent_tools._public_address", AsyncMock(return_value="93.184.216.34")):
        res = await web.execute(url="https://example.com")
        assert not res.is_error
        assert "Hello World" in res.output
        assert "Sample paragraph." in res.output


@pytest.mark.asyncio
async def test_web_fetch_tool_http_error() -> None:
    web = WebFetchTool()
    mock_resp = httpx.Response(
        status_code=404,
        stream=httpx.ByteStream(b"Not Found"),
        request=httpx.Request("GET", "https://example.com/notfound"),
    )

    with patch.object(httpx.AsyncClient, "send", AsyncMock(return_value=mock_resp)), patch("openagent.tools.agent_tools._public_address", AsyncMock(return_value="93.184.216.34")):
        res = await web.execute(url="https://example.com/notfound")
        assert res.is_error
        assert "404" in res.output


@pytest.mark.asyncio
async def test_fs_symlink_confinement(tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret_file = outside_dir / "secret.txt"
    secret_file.write_text("SUPER_SECRET_TOKEN=xyz123", encoding="utf-8")

    ws = tmp_path / "workspace"
    ws.mkdir()
    symlink_file = ws / "symlink_secret.txt"

    try:
        symlink_file.symlink_to(secret_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported in current environment/permissions")

    tools_map = {t.name: t for t in create_fs_tools(ws)}
    read_tool = tools_map["read_file"]
    grep_tool = tools_map["grep_search"]

    # 1. read_file through external symlink must fail
    res_read = await read_tool.execute(path="symlink_secret.txt")
    assert res_read.is_error
    assert "access denied" in res_read.output.lower() or "outside" in res_read.output.lower()

    # 2. grep_search must ignore the external symlinked file and NOT leak contents
    res_grep = await grep_tool.execute(query="SUPER_SECRET_TOKEN")
    assert "SUPER_SECRET_TOKEN" not in res_grep.output
    assert res_grep.output == "No matches found for query."


@pytest.mark.asyncio
async def test_fs_edit_file_empty_old_str(sandbox_env: tuple[Path, dict[str, Tool]]) -> None:
    _ws, tools = sandbox_env
    edit_tool = tools["edit_file"]
    res = await edit_tool.execute(path="test.txt", old_str="", new_str="something")
    assert res.is_error
    assert "cannot be empty" in res.output.lower()


@pytest.mark.asyncio
async def test_tool_registry_type_error_no_double_call() -> None:
    class FailingTool(Tool):
        name = "fail_type_error"
        description = "Raises TypeError internally"
        call_count = 0

        async def execute(self, **kwargs: Any) -> ToolResult:
            self.call_count += 1
            # Raise an internal TypeError
            _ = None + 1  # type: ignore[operator]
            return ToolResult(call_id=str(kwargs.get("call_id", "")), output="ok")

    tool = FailingTool()
    registry = ToolRegistry()
    registry.register(tool)

    call = ToolCall(id="call_1", name="fail_type_error", arguments={})
    res = await registry.execute_call(call)

    assert res.is_error
    assert "Error executing tool" in res.output
    # Must only be called once, NOT retried!
    assert tool.call_count == 1


@pytest.mark.asyncio
async def test_shell_tool_relative_cwd(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    subdir = ws / "sub"
    subdir.mkdir(parents=True)

    shell = ShellTool(workspace_root=ws)
    res = await shell.execute(command="echo inside_cwd", cwd="sub")
    assert not res.is_error
    assert "inside_cwd" in res.output
