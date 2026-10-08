import asyncio
import os
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import anyio
import httpx
import pytest

from openagent.core.types import ToolCall, ToolParam
from openagent.tools.agent_tools import WebFetchTool
from openagent.tools.base import Tool, ToolResult
from openagent.tools.fs import create_fs_tools
from openagent.tools.mcp.client import MCPClient, MCPServerConfig, MCPTool
from openagent.tools.mcp.manager import MCPManager
from openagent.tools.registry import PermissionAction, ToolRegistry
from openagent.tools.shell import ShellTool, _decode_output, execute_shell


class BadResult(Tool):
    name = "bad"
    description = "bad"
    async def execute(self, **kwargs):
        return None


def test_network_tools_require_confirmation_by_default():
    assert ToolRegistry([WebFetchTool()]).get_policy("web_fetch") == PermissionAction.ASK


@pytest.mark.asyncio
async def test_mcp_reconnect_same_name_revalidates_current_schema():
    class Client:
        def __init__(self, config):
            self.is_connected = False
            self.version = 0
            self.calls = []
        async def connect(self):
            self.is_connected = True
            self.version += 1
        async def list_tools(self):
            schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": self.version}}, "required": ["n"]}
            return [{"name": "write", "inputSchema": schema}]
        async def call_tool(self, name, arguments=None):
            self.calls.append(arguments)
            return {"content": [{"text": "changed"}]}
        async def close(self):
            self.is_connected = False
    manager = MCPManager(configs=[MCPServerConfig(name="s")], client_factory=Client)
    await manager.start()
    client = manager.clients["s"]
    client.is_connected = False
    result = await manager.registry.execute_call(ToolCall(name="s__write", arguments={"n": 1}), lambda _: True)
    assert result.is_error
    assert client.calls == []
    assert "minimum" in result.output.lower() or "less than" in result.output.lower()


@pytest.mark.asyncio
async def test_invalid_result_does_not_abort_registry():
    result = await ToolRegistry([BadResult()]).execute_call(ToolCall(name="bad", arguments={}))
    assert result.is_error and "ToolResult" in result.output


@pytest.mark.asyncio
async def test_arguments_validated_before_execution():
    class Echo(BadResult):
        def __init__(self):
            self.params = [ToolParam(name="n", type="integer", enum=[1, 2])]
        async def execute(self, **kwargs):
            pytest.fail("Invalid arguments reached tool")
    result = await ToolRegistry([Echo()]).execute_call(ToolCall(name="bad", arguments={"n": True}))
    assert result.is_error and "argument" in result.output.lower()


@pytest.mark.asyncio
async def test_mutations_and_confirmations_keep_order():
    events = []
    class Writer(BadResult):
        danger = "write"
        async def execute(self, **kwargs):
            events.append("start" + kwargs["call_id"])
            await asyncio.sleep(0)
            events.append("end" + kwargs["call_id"])
            return ToolResult("", "ok")
    async def ask(call):
        events.append("ask" + call.id)
        await asyncio.sleep(0)
        return True
    await ToolRegistry([Writer()]).execute_parallel([ToolCall(id=str(i), name="bad", arguments={}) for i in range(2)], ask)
    assert events == ["ask0", "start0", "end0", "ask1", "start1", "end1"]


def test_mcp_cannot_shadow_builtin_or_claim_readonly():
    registry = ToolRegistry([BadResult()])
    remote = MCPTool(AsyncMock(), {"name": "bad", "inputSchema": {"type": "object"}, "annotations": {"read_only_hint": True}})
    assert remote.danger == "execute"
    with pytest.raises(ValueError, match="already registered"):
        registry.register(remote)


def test_mcp_schema_is_preserved():
    schema = {"type": "object", "properties": {"o": {"anyOf": [{"type": "null"}, {"type": "object", "properties": {"n": {"type": "number", "minimum": 2}}}]}}, "additionalProperties": False}
    assert MCPTool(AsyncMock(), {"name": "nested", "inputSchema": schema}).spec.schema() == schema


@pytest.mark.asyncio
async def test_fs_preserves_newlines_and_deletes_moves(tmp_path):
    tools = {tool.name: tool for tool in create_fs_tools(tmp_path)}
    file = tmp_path / "a.txt"
    file.write_bytes(b"first\r\nsecond\r\n")
    result = await tools["edit_file"].execute(path="a.txt", old_str="first", new_str="changed")
    assert not result.is_error
    assert file.read_bytes() == b"changed\r\nsecond\r\n"
    result = await tools["write_file"].execute(path="lf.txt", content="a\nb\n")
    assert not result.is_error and (tmp_path / "lf.txt").read_bytes() == b"a\nb\n"
    assert not (await tools["move_file"].execute(source="a.txt", destination="nested/b.txt")).is_error
    assert not (await tools["delete_file"].execute(path="nested/b.txt")).is_error
    assert not (tmp_path / "nested/b.txt").exists()
    assert (await tools["move_file"].execute(source="lf.txt", destination="../escape")).is_error


@pytest.mark.asyncio
async def test_scans_exclude_dependencies_binaries_and_support_single_file(tmp_path):
    tools = {tool.name: tool for tool in create_fs_tools(tmp_path)}
    for name in [".git", ".venv", "node_modules"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "secret").write_text("needle")
    (tmp_path / "binary.dat").write_bytes(b"\x00needle")
    (tmp_path / "code.py").write_text("needle")
    result = await tools["grep_search"].execute(query="needle")
    assert "code.py" in result.output and "secret" not in result.output and "binary" not in result.output
    assert "code.py" in (await tools["grep_search"].execute(query="needle", path="code.py")).output
    listing = await tools["list_directory"].execute(max_items=-1)
    assert listing.output.count("[FILE]") + listing.output.count("[DIR]") <= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [-1, 0, float("inf"), float("nan"), 100000])
async def test_shell_rejects_unbounded_timeout(tmp_path, timeout):
    result = await ShellTool(tmp_path).execute(command="echo test", timeout=timeout)
    assert result.is_error and "timeout" in result.output.lower()


@pytest.mark.asyncio
async def test_shell_helper_root_independent_of_cwd(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    result = await execute_shell("echo bad", cwd=tmp_path, workspace_root=root)
    assert result.is_error and "outside" in result.output


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["http://127.0.0.1", "http://[::1]", "http://169.254.169.254", "http://localhost", "http://10.0.0.1", "http://user:pass@example.com"])
async def test_web_fetch_blocks_private_and_credential_urls(url):
    result = await WebFetchTool().execute(url=url)
    assert result.is_error


@pytest.mark.asyncio
async def test_mcp_transport_failure_invalidates_without_replaying_mutation():
    session = AsyncMock()
    session.call_tool.side_effect = ConnectionResetError("disconnected")
    client = MCPClient(MCPServerConfig(name="s", command="stub"), session=session)
    await client.connect()
    with pytest.raises(ConnectionResetError):
        await client.call_tool("write", {})
    assert not client.is_connected
    assert session.call_tool.await_count == 1


@pytest.mark.asyncio
async def test_client_next_call_reconnects_and_relists():
    session = AsyncMock()
    session.call_tool.side_effect = [ConnectionResetError("lost"), {"content": [{"text": "ok"}]}]
    session.list_tools.return_value = type("Tools", (), {"tools": [], "next_cursor": None})()
    client = MCPClient(MCPServerConfig(name="s", command="stub"), session=session)
    await client.connect()
    with pytest.raises(ConnectionResetError):
        await client.call_tool("echo", {})
    async def reconnect():
        client._session = session
        client._is_connected = True
    client.connect = reconnect
    result = await client.call_tool("echo", {})
    assert result["content"][0]["text"] == "ok"
    assert session.list_tools.await_count == 1


@pytest.mark.asyncio
async def test_manager_refreshes_tools_after_disconnect_on_next_operation():
    class Client:
        def __init__(self, config):
            self.is_connected = False
            self.version = 0
        async def connect(self):
            self.is_connected = True
            self.version += 1
        async def list_tools(self):
            return [{"name": "old" if self.version == 1 else "new", "inputSchema": {"type": "object"}}]
        async def call_tool(self, name, arguments=None):
            return {"content": [{"text": name}]}
        async def close(self):
            self.is_connected = False
    manager = MCPManager(configs=[MCPServerConfig(name="s")], client_factory=Client, prefix_tool_names=False)
    await manager.start()
    client = manager.clients["s"]
    client.is_connected = False
    result = await manager.registry.execute_call(ToolCall(name="old", arguments={}), lambda _: True)
    assert result.is_error and "no longer" in result.output
    assert "new" in manager.registry and "old" not in manager.registry


@pytest.mark.asyncio
async def test_shell_strips_secrets_and_closes_stdin(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("UNRELATED_SECRET", "also-private")
    result = await ShellTool(tmp_path).execute(command='python -c "import os,sys; print(os.getenv(\'OPENAI_API_KEY\'), os.getenv(\'UNRELATED_SECRET\'), repr(sys.stdin.read()))"')
    assert not result.is_error
    assert result.output == "None None ''"


def test_windows_oem_output_fallback(monkeypatch):
    import ctypes
    monkeypatch.setattr(sys, "platform", "win32")
    from types import SimpleNamespace
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(GetOEMCP=lambda: 949)), raising=False)
    assert _decode_output("대한민국".encode("cp949")) == "대한민국"
    assert _decode_output("대한민국".encode()) == "대한민국"


@pytest.mark.asyncio
async def test_web_redirect_private_target_rejected_before_second_request(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda *args, **kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))])
    response = httpx.Response(302, headers={"location": "http://127.0.0.1/secret"}, stream=httpx.ByteStream(b""))
    send = AsyncMock(return_value=response)
    with patch.object(httpx.AsyncClient, "send", send):
        result = await WebFetchTool().execute(url="https://example.com")
    assert result.is_error and "Private" in result.output
    assert send.await_count == 1
    request = send.call_args.args[0]
    assert request.url.host == "93.184.216.34" and request.headers["host"] == "example.com"
    assert request.extensions["sni_hostname"] == "example.com"


@pytest.mark.asyncio
async def test_web_stream_download_limit_stops_reading(monkeypatch):
    class Endless(httpx.AsyncByteStream):
        chunks = 0
        async def __aiter__(self):
            while True:
                self.chunks += 1
                yield b"x" * 65536
    stream = Endless()
    response = httpx.Response(200, stream=stream)
    with patch.object(httpx.AsyncClient, "send", AsyncMock(return_value=response)), patch("openagent.tools.agent_tools._public_address", AsyncMock(return_value="93.184.216.34")):
        result = await WebFetchTool().execute(url="https://example.com", max_chars=1)
    assert result.is_error and "download limit" in result.output
    assert stream.chunks == 33


@pytest.mark.asyncio
async def test_atomic_edit_failure_preserves_original_and_cleans_temp(tmp_path, monkeypatch):
    file = tmp_path / "a"
    file.write_bytes(b"original\r\n")
    def fail(*args):
        raise OSError("disk error")
    monkeypatch.setattr(os, "replace", fail)
    tools = {tool.name: tool for tool in create_fs_tools(tmp_path)}
    result = await tools["edit_file"].execute(path="a", old_str="original", new_str="changed")
    assert result.is_error
    assert file.read_bytes() == b"original\r\n"
    assert list(tmp_path.iterdir()) == [file]


@pytest.mark.asyncio
async def test_multiline_lf_edit_preserves_crlf_and_bom(tmp_path):
    file = tmp_path / "a"
    file.write_bytes(b"\xef\xbb\xbffirst\r\nsecond\r\nlast\r\n")
    tools = {tool.name: tool for tool in create_fs_tools(tmp_path)}
    result = await tools["edit_file"].execute(path="a", old_str="first\nsecond", new_str="changed\nnext")
    assert not result.is_error
    assert file.read_bytes() == b"\xef\xbb\xbfchanged\r\nnext\r\nlast\r\n"


@pytest.mark.asyncio
async def test_mcp_lifecycle_owned_across_registry_child_tasks():
    session = AsyncMock()
    session.list_tools.return_value = type("Tools", (), {"tools": [], "next_cursor": None})()
    session.call_tool.side_effect = [ConnectionResetError("lost"), {"content": [{"text": "ok"}]}]
    @asynccontextmanager
    async def transport(*args, **kwargs):
        async with anyio.create_task_group():
            yield ("read", "write")
    @asynccontextmanager
    async def client_session(*args, **kwargs):
        yield session
    with patch("openagent.tools.mcp.client.stdio_client", transport), patch("openagent.tools.mcp.client.ClientSession", client_session):
        client = MCPClient(MCPServerConfig(name="s", command="stub"))
        await client.connect()
        async def child():
            with pytest.raises(ConnectionResetError):
                await client.call_tool("echo", {})
            return await client.call_tool("echo", {})
        result = await asyncio.create_task(child())
        assert result["content"][0]["text"] == "ok"
        await client.close()


@pytest.mark.asyncio
async def test_real_stdio_mcp_reconnect_across_tasks(tmp_path):
    """Exercise the SDK's actual transport/session cancel scopes without network."""
    server = tmp_path / "server.py"
    server.write_text('''import json, sys
for line in sys.stdin:
    req = json.loads(line)
    if "id" not in req:
        continue
    method = req["method"]
    if method == "initialize":
        result = {"protocolVersion": req["params"]["protocolVersion"], "capabilities": {}, "serverInfo": {"name": "stub", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": "hello"}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}), flush=True)
''', encoding="utf-8")
    client = MCPClient(MCPServerConfig(name="stub", command=sys.executable, args=[str(server)]))
    try:
        await client.connect()
        async def child():
            # Simulate the detected disconnect flag, then exercise a full close,
            # fresh stdio transport, initialize/list, and call from another task.
            client._is_connected = False
            client._reconnect_needed = True
            return await client.call_tool("echo", {})
        result = await asyncio.wait_for(asyncio.create_task(child()), timeout=15)
        assert result.content[0].text == "hello"
    finally:
        await client.close()
