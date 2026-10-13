import asyncio
import errno
import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from openagent.core.types import ToolCall
from openagent.tools.agent_tools import WebFetchTool
from openagent.tools.base import MAX_TOOL_OUTPUT_BYTES
from openagent.tools.fs import create_fs_tools
from openagent.tools.mcp.client import MCPClient, MCPServerConfig, MCPTool
from openagent.tools.mcp.manager import MCPManager
from openagent.tools.registry import ToolRegistry


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["structuredContent", "structured_content"])
async def test_business_call_id_and_structured_mixed_result(field):
    client = AsyncMock()
    client.call_tool.return_value = {"content": [{"text": "summary"}], field: {"answer": 42}, "isError": True}
    tool = MCPTool(client, {"name": "remote", "inputSchema": {"type": "object", "properties": {"call_id": {"type": "string"}}, "required": ["call_id"]}})
    registry = ToolRegistry([tool], danger_policies={"execute": "allow"})
    result = await registry.execute_call(ToolCall(id="local", name="remote", arguments={"call_id": "business"}))
    client.call_tool.assert_awaited_once_with("remote", arguments={"call_id": "business"})
    assert result.call_id == "local" and result.is_error
    assert result.structured_data == {"answer": 42}
    assert "summary" in result.output and "42" in result.output


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [
    {"content": [{"text": "한😀" * 1000000}]},
    {"content": [{"text": "x" * 50000}] * 100},
    {"structuredContent": {"big": "x" * 3000000}},
])
async def test_mcp_output_bounded(content):
    client = AsyncMock()
    client.call_tool.return_value = content
    result = await MCPTool(client, {"name": "remote"}).execute(call_id="local")
    assert len(result.output.encode()) <= MAX_TOOL_OUTPUT_BYTES
    assert "truncated" in result.output.lower()
    assert result.call_id == "local"
    result.output.encode().decode()


@pytest.mark.asyncio
async def test_cancel_discovery_closes_client():
    entered = asyncio.Event()
    client = AsyncMock()
    async def discover():
        entered.set()
        await asyncio.Event().wait()
    client.list_tools.side_effect = discover
    manager = MCPManager(configs=[MCPServerConfig("test")], client_factory=lambda _: client)
    task = asyncio.create_task(manager.start_server("test"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await manager.stop()
    client.close.assert_awaited_once()
    assert not manager.clients and not manager.registry.list_tools()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [errno.ENOTSUP, errno.EPERM])
async def test_exclusive_fallback_preserves_existing(tmp_path, error):
    from openagent.tools.fs import WriteFileTool
    tool = WriteFileTool(tmp_path)
    with patch("openagent.tools.fs.os.link", side_effect=OSError(error, "fixture")):
        first = await tool.execute(path="file", content="한글\r\n", overwrite=False)
        second = await tool.execute(path="file", content="other", overwrite=False)
    assert not first.is_error and second.is_error
    assert (tmp_path / "file").read_bytes() == "한글\r\n".encode()
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("suffix,expected_error", [(b"", False), (b"garbage", True), (gzip.compress(b"tail")[:-3], True)])
async def test_concatenated_gzip(split, suffix, expected_error):
    data = gzip.compress(b"first") + gzip.compress(b"second") + suffix
    response = httpx.Response(200, headers={"content-encoding": "gzip"}, stream=httpx.ByteStream(data))
    if split:
        async def chunks():
            for byte in data:
                yield bytes([byte])
        response.aiter_raw = chunks
    with patch.object(httpx.AsyncClient, "send", AsyncMock(return_value=response)):
        result = await WebFetchTool().execute(url="https://8.8.8.8")
    assert result.is_error == expected_error
    if not expected_error:
        assert result.output == "firstsecond"
    assert response.is_closed


@pytest.mark.asyncio
async def test_checkpoint_preview_undo_and_conflicts(tmp_path):
    original = tmp_path / "original"
    original.write_bytes(b"before\r\n")
    registry = ToolRegistry(create_fs_tools(tmp_path), danger_policies={"write": "allow"})
    preview = await registry.changes.preview(registry.get("edit_file"), {"path": "original", "old_str": "before", "new_str": "after"})
    assert "-before" in preview and "+after" in preview
    assert original.read_bytes() == b"before\r\n"
    await registry.execute_call(ToolCall(name="edit_file", arguments={"path": "original", "old_str": "before", "new_str": "after"}))
    await registry.execute_call(ToolCall(name="move_file", arguments={"source": "original", "destination": "moved"}))
    await registry.execute_call(ToolCall(name="write_file", arguments={"path": "added", "content": "new"}))
    assert "move_file" in registry.changes.describe()
    (tmp_path / "added").write_text("user edited")
    with pytest.raises(ValueError, match="conflict"):
        registry.changes.undo()
    assert not original.exists()
    (tmp_path / "added").write_text("new")
    registry.changes.undo()
    assert original.read_bytes() == b"before\r\n"
    assert not (tmp_path / "moved").exists() and not (tmp_path / "added").exists()


def test_streamable_config_roundtrip():
    config = MCPServerConfig.from_dict({"name": "http", "transport": "streamable_http", "url": "https://example.com/mcp", "headers": {"Authorization": "secret"}, "read_timeout": 4})
    assert MCPServerConfig.from_dict(config.to_dict()) == config


def test_exclusive_publication_race_and_copy_failure(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from openagent.tools.fs import _atomic_write

    def write(value):
        try:
            _atomic_write(tmp_path / "race", value, overwrite=False)
            return True
        except FileExistsError:
            return False

    with patch("openagent.tools.fs.os.link", side_effect=OSError(errno.ENOTSUP, "fixture")), ThreadPoolExecutor(2) as pool:
        assert sum(pool.map(write, ["first", "second"])) == 1
    assert (tmp_path / "race").read_text() in ("first", "second")
    with patch("openagent.tools.fs.os.link", side_effect=OSError(errno.ENOTSUP, "fixture")), patch("openagent.tools.fs.shutil.copyfileobj", side_effect=OSError("disk failure")), pytest.raises(OSError):
        _atomic_write(tmp_path / "failed", "data", overwrite=False)
    assert [path.name for path in tmp_path.iterdir()] == ["race"]


@pytest.mark.asyncio
async def test_gzip_limit_accumulates_across_members():
    data = gzip.compress(b"a" * 1500000) + gzip.compress(b"b" * 1500000)
    response = httpx.Response(200, headers={"content-encoding": "gzip"}, stream=httpx.ByteStream(data))
    with patch.object(httpx.AsyncClient, "send", AsyncMock(return_value=response)):
        result = await WebFetchTool().execute(url="https://8.8.8.8")
    assert result.is_error and "decompressed limit" in result.output
    assert response.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("event_stream", [False, True])
@pytest.mark.parametrize("expire", [False, True])
async def test_streamable_http_real_local_fixture(event_stream, expire):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(request["method"])
            if expire and request["method"] == "tools/call" and seen.count("tools/call") == 1:
                self.send_response(404)
                self.end_headers()
                return
            if "id" not in request:
                self.send_response(202)
                self.end_headers()
                return
            results = {
                "initialize": {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}}, "serverInfo": {"name": "fixture", "version": "1"}},
                "tools/list": {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]},
                "tools/call": {"content": [{"type": "text", "text": "ok"}]},
            }
            body = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": results[request["method"]]})
            if event_stream:
                body = "event: message\ndata: " + body + "\n\n"
            encoded = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if event_stream else "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Mcp-Session-Id", "fixture-session")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            self.send_response(405)
            self.end_headers()

        def do_DELETE(self):
            seen.append("DELETE")
            self.send_response(200)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = MCPClient(MCPServerConfig("fixture", transport="streamable_http", url=f"http://127.0.0.1:{server.server_port}/mcp", timeout=3))
    try:
        await asyncio.wait_for(client.connect(), 5)
        assert (await client.list_tools())[0].name == "echo"
        if expire:
            with pytest.raises(Exception, match="Session terminated"):
                await client.call_tool("echo")
            assert not client.is_connected
        assert (await client.call_tool("echo")).content[0].text == "ok"
        if expire:
            assert seen.count("initialize") == 2
        await client.close()
        assert "DELETE" in seen
    finally:
        await client.close()
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_streamable_redirect_does_not_send_credentials_cross_origin():
    received: list[str | None] = []
    class Destination(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            received.append(self.headers.get("Authorization"))
            self.send_response(500)
            self.end_headers()

    target = ThreadingHTTPServer(("127.0.0.1", 0), Destination)
    class Redirect(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.send_response(307)
            self.send_header("Location", f"http://127.0.0.1:{target.server_port}/mcp")
            self.send_header("Content-Length", "0")
            self.end_headers()

    source = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (target, source)]
    for thread in threads:
        thread.start()
    client = MCPClient(MCPServerConfig("redirect", transport="streamable_http", url=f"http://127.0.0.1:{source.server_port}/mcp", headers={"Authorization": "Bearer secret"}, timeout=2))
    try:
        with pytest.raises(Exception, match=r"[Rr]edirect"):
            await asyncio.wait_for(client.connect(), 4)
        assert received == []
        assert not client.is_connected
    finally:
        await client.close()
        for server in (source, target):
            await asyncio.to_thread(server.shutdown)
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)
