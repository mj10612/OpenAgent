import asyncio
import difflib
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, EmbeddedResource, TextResourceContents

from openagent.core.types import ToolCall
from openagent.tools.base import MAX_TOOL_OUTPUT_BYTES
from openagent.tools.changes import ChangeTracker
from openagent.tools.fs import MAX_FILE_BYTES, WriteFileTool, create_fs_tools
from openagent.tools.mcp.client import MCPServerConfig, MCPTool
from openagent.tools.mcp.manager import MCPManager
from openagent.tools.patch import ApplyPatchTool
from openagent.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_checkpoint_counts_both_snapshots_before_publish(tmp_path):
    registry = ToolRegistry(create_fs_tools(tmp_path), danger_policies={'write': 'allow'})
    registry.changes.max_bytes = 8
    first = await registry.execute_call(ToolCall(name='write_file', arguments={'path': 'a', 'content': 'x' * 8}))
    second = await registry.execute_call(ToolCall(name='write_file', arguments={'path': 'b', 'content': 'y'}))
    assert not first.is_error
    assert second.is_error
    assert not (tmp_path / 'b').exists()
    retained = sum(len(s.data or b'') for turn in registry.changes.turns for entry in turn for side in ('before', 'after') for s in entry.get(side, {}).values())
    assert retained <= 8


@pytest.mark.asyncio
async def test_existing_file_budget_reserves_after_snapshot(tmp_path):
    (tmp_path / 'a').write_bytes(b'1234')
    registry = ToolRegistry(create_fs_tools(tmp_path), danger_policies={'write': 'allow'})
    registry.changes.max_bytes = 7
    result = await registry.execute_call(ToolCall(name='write_file', arguments={'path': 'a', 'content': '5678'}))
    assert result.is_error
    assert (tmp_path / 'a').read_bytes() == b'1234'


@pytest.mark.asyncio
@pytest.mark.parametrize('name,args', [('move_file', {'source': 'large', 'destination': 'moved'}), ('delete_file', {'path': 'large'})])
async def test_large_file_mutations_remain_undoable(tmp_path, name, args):
    source = tmp_path / 'large'
    source.write_bytes(b'x' * (MAX_FILE_BYTES + 1))
    registry = ToolRegistry(create_fs_tools(tmp_path), danger_policies={'write': 'allow'})
    result = await registry.execute_call(ToolCall(name=name, arguments=args))
    assert not result.is_error, result.output
    assert not source.exists()
    registry.changes.undo()
    assert source.stat().st_size == MAX_FILE_BYTES + 1
    assert not (tmp_path / 'moved').exists()


@pytest.mark.asyncio
async def test_oversized_edit_never_publishes(tmp_path):
    (tmp_path / 'a').write_bytes(b'x')
    registry = ToolRegistry(create_fs_tools(tmp_path), danger_policies={'write': 'allow'})
    result = await registry.execute_call(ToolCall(name='edit_file', arguments={'path': 'a', 'old_str': 'x', 'new_str': 'y' * (MAX_FILE_BYTES + 1)}))
    assert result.is_error
    assert (tmp_path / 'a').read_bytes() == b'x'


@pytest.mark.asyncio
async def test_empty_file_creation_preview_shows_existence(tmp_path):
    preview = await ChangeTracker().preview(WriteFileTool(tmp_path), {'path': 'empty', 'content': ''})
    assert 'existence' in preview or 'created' in preview.lower()


@pytest.mark.asyncio
async def test_partial_undo_failure_can_retry(tmp_path, monkeypatch):
    tracker = ChangeTracker()
    tool = WriteFileTool(tmp_path)
    for name in ('a', 'b'):
        (tmp_path / name).write_bytes(b'before')
        before = tracker.capture(tool, {'path': name, 'content': 'after'})
        await tool.execute(path=name, content='after')
        tracker.record(tool, before)
    replace = os.replace
    def fail_second(source, destination):
        if Path(destination).name == 'b':
            raise PermissionError('sharing violation')
        replace(source, destination)
    with monkeypatch.context() as scoped:
        scoped.setattr('openagent.tools.changes.os.replace', fail_second)
        with pytest.raises(PermissionError):
            tracker.undo()
    tracker.undo()
    assert [(tmp_path / name).read_bytes() for name in ('a', 'b')] == [b'before', b'before']


@pytest.mark.asyncio
@pytest.mark.parametrize('separator', ['\v', '\f', '\u2028', '\u0085'])
async def test_patch_preserves_non_lf_line_characters(tmp_path, separator):
    original = f'foo{separator}bar\n'
    (tmp_path / 'a').write_bytes(original.encode())
    diff = ''.join(difflib.unified_diff([original], ['baz\n'], fromfile='a/a', tofile='b/a'))
    result = await ApplyPatchTool(tmp_path).execute(path='a', patch=diff)
    assert not result.is_error, result.output
    assert (tmp_path / 'a').read_bytes() == b'baz\n'


@pytest.mark.asyncio
async def test_stop_during_mcp_discovery_does_not_resurrect():
    entered, release = asyncio.Event(), asyncio.Event()
    client = AsyncMock()
    async def discover():
        entered.set()
        await release.wait()
        return [{'name': 'echo', 'inputSchema': {'type': 'object'}}]
    client.list_tools.side_effect = discover
    manager = MCPManager(configs=[MCPServerConfig('srv')], client_factory=lambda _: client)
    task = asyncio.create_task(manager.start_server('srv'))
    await entered.wait()
    await manager.stop_server('srv')
    release.set()
    await task
    assert not manager.clients and not manager.registry.list_tools()


@pytest.mark.asyncio
async def test_embedded_resource_report_is_rendered_and_bounded():
    client = AsyncMock()
    client.call_tool.return_value = CallToolResult(content=[EmbeddedResource(type='resource', resource=TextResourceContents(uri='file:///report.txt', mimeType='text/plain', text='critical report ' + 'x' * MAX_TOOL_OUTPUT_BYTES))])
    result = await MCPTool(client, {'name': 'report'}).execute_arguments({})
    assert 'critical report' in result.output
    assert 'file:///report.txt' in result.output and 'text/plain' in result.output
    assert len(result.output.encode()) <= MAX_TOOL_OUTPUT_BYTES


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["\n", "\r\n"])
async def test_patch_preserves_crlf_and_no_final_newline(tmp_path, ending):
    (tmp_path / "a").write_bytes(("first\vline" + ending + "last").encode())
    diff = "@@ -1,2 +1,2 @@\n first\vline\n-last\n\\ No newline at end of file\n+updated\n\\ No newline at end of file\n"
    result = await ApplyPatchTool(tmp_path).execute(path="a", patch=diff)
    assert not result.is_error, result.output
    assert (tmp_path / "a").read_bytes() == ("first\vline" + ending + "updated").encode()


@pytest.mark.asyncio
async def test_direct_oversized_edit_rejects_before_publish(tmp_path):
    from openagent.tools.fs import EditFileTool
    (tmp_path / "a").write_bytes(b"x")
    result = await EditFileTool(tmp_path).execute(path="a", old_str="x", new_str="y" * (MAX_FILE_BYTES + 1))
    assert result.is_error and (tmp_path / "a").read_bytes() == b"x"


@pytest.mark.asyncio
async def test_after_capture_failure_restores_published_mutation(tmp_path, monkeypatch):
    tracker = ChangeTracker()
    tool = WriteFileTool(tmp_path)
    (tmp_path / "a").write_bytes(b"before")
    before = tracker.capture(tool, {"path": "a", "content": "after"})
    await tool.execute(path="a", content="after")
    with monkeypatch.context() as scoped:
        scoped.setattr(tracker, "state", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("capture failed")))
        with pytest.raises(OSError, match="capture failed"):
            tracker.record(tool, before)
    assert (tmp_path / "a").read_bytes() == b"before"


@pytest.mark.asyncio
async def test_stop_during_reconnect_closes_reopened_client():
    entered, release = asyncio.Event(), asyncio.Event()
    class Client:
        is_connected = False
        async def close(self):
            self.is_connected = False
        async def connect(self):
            entered.set()
            await release.wait()
            self.is_connected = True
        async def list_tools(self):
            return [{"name": "echo", "inputSchema": {"type": "object"}}]
    client = Client()
    manager = MCPManager(configs=[MCPServerConfig("srv")])
    manager.clients["srv"] = client
    task = asyncio.create_task(manager.ensure_tools("srv"))
    await entered.wait()
    await manager.stop_server("srv")
    release.set()
    await task
    assert not manager.clients and not manager.registry.list_tools()
    assert not client.is_connected


@pytest.mark.asyncio
async def test_disk_checkpoint_cap_rejects_before_large_mutation(tmp_path):
    (tmp_path / "large").write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    registry = ToolRegistry(create_fs_tools(tmp_path), danger_policies={"write": "allow"})
    registry.changes.max_disk_bytes = MAX_FILE_BYTES
    result = await registry.execute_call(ToolCall(name="delete_file", arguments={"path": "large"}))
    assert result.is_error and "disk" in result.output.lower()
    assert (tmp_path / "large").exists()


@pytest.mark.asyncio
async def test_disk_checkpoint_evicts_old_turn_and_closes_backup(tmp_path):
    registry = ToolRegistry(create_fs_tools(tmp_path), danger_policies={"write": "allow"})
    registry.changes.max_disk_bytes = MAX_FILE_BYTES + 1
    for name in ("a", "b"):
        (tmp_path / name).write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    first = await registry.execute_call(ToolCall(name="delete_file", arguments={"path": "a"}))
    assert not first.is_error
    backup = next(iter(registry.changes.turns[0][0]["before"].values())).backup
    registry.changes.begin_turn()
    second = await registry.execute_call(ToolCall(name="delete_file", arguments={"path": "b"}))
    assert not second.is_error
    assert backup.closed
    assert registry.changes._retained_disk_bytes() <= registry.changes.max_disk_bytes
    registry.changes.undo()
    assert (tmp_path / "b").exists() and not (tmp_path / "a").exists()


@pytest.mark.asyncio
async def test_failed_after_capture_and_rollback_never_overwrites_external_edit(tmp_path, monkeypatch):
    tracker = ChangeTracker()
    tool = WriteFileTool(tmp_path)
    (tmp_path / "a").write_bytes(b"before")
    before = tracker.capture(tool, {"path": "a", "content": "after"})
    await tool.execute(path="a", content="after")
    def fail(*args, **kwargs):
        raise OSError("fixture failure")
    with monkeypatch.context() as scoped:
        scoped.setattr(tracker, "state", fail)
        scoped.setattr(tracker, "_restore", fail)
        with pytest.raises(OSError):
            tracker.record(tool, before)
    (tmp_path / "a").write_bytes(b"external edit")
    with pytest.raises(ValueError, match=r"safe|blocked"):
        tracker.undo()
    assert (tmp_path / "a").read_bytes() == b"external edit"
    assert "cannot" in tracker.describe().lower()


@pytest.mark.asyncio
async def test_large_preview_obeys_disk_cap_without_eviction(tmp_path):
    tracker = ChangeTracker()
    tracker.max_disk_bytes = MAX_FILE_BYTES
    (tmp_path / "large").write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    from openagent.tools.fs import DeleteFileTool
    with pytest.raises(ValueError, match="disk"):
        await tracker.preview(DeleteFileTool(tmp_path), {"path": "large"})
    assert (tmp_path / "large").exists()
    assert tracker.turns == [[]]
