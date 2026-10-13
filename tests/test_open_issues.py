"""Regression proofs for the October issue backlog."""

import asyncio
import errno
import gzip
import io
import json
import zlib
from unittest.mock import AsyncMock, patch

import brotli
import httpx
import pytest
from rich.console import Console

from openagent import cli
from openagent.context.compactor import Compactor
from openagent.context.messages import MessageManager
from openagent.core.events import DoneEvent, ErrorEvent, ToolCallStart
from openagent.core.provider import ChatProvider
from openagent.core.router import ProviderRouter
from openagent.core.types import ChatRequest, Message, ToolCall
from openagent.providers.gemini import GeminiProvider
from openagent.providers.openai_compat import OpenAICompatProvider
from openagent.runner import AgentRunner
from openagent.session.store import SessionStore
from openagent.tools.agent_tools import WebFetchTool, _HTMLTextExtractor
from openagent.tools.base import Tool, ToolResult
from openagent.tools.fs import GlobFindTool, GrepSearchTool, MoveFileTool
from openagent.tools.patch import ApplyPatchTool
from openagent.tools.registry import PermissionAction, ToolRegistry
from openagent.tui import app as tui


class Counter(Tool):
    name = "counter"
    description = "count executions"

    def __init__(self):
        self.params = []
        self.current = self.peak = self.count = 0

    async def execute(self, **kwargs):
        self.count += 1
        self.current += 1
        self.peak = max(self.peak, self.current)
        try:
            await asyncio.sleep(0.001)
            return ToolResult(kwargs["call_id"], kwargs["call_id"])
        finally:
            self.current -= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [["duplicate", "duplicate"], ["", "valid"]])
async def test_invalid_tool_batch_never_executes(ids):
    tool = Counter()

    class Provider(ChatProvider):
        async def stream(self, request):
            yield DoneEvent(
                message=Message.assistant(
                    tool_calls=[ToolCall(id=value, name=tool.name) for value in ids]
                )
            )

    runner = AgentRunner(Provider(), tools=ToolRegistry([tool]))
    events = [event async for event in runner.run_turn("work")]
    assert any(isinstance(event, ErrorEvent) for event in events)
    assert tool.count == 0
    assert not any(message.role == "user" for message in runner.messages.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 4, 8])
async def test_tool_batch_has_bounded_workers_and_preserves_order(limit):
    tool = Counter()
    registry = ToolRegistry([tool], max_concurrent_tools=limit)
    calls = [ToolCall(id=str(index), name=tool.name) for index in range(300)]
    results = await registry.execute_parallel(calls)
    assert tool.peak <= limit
    assert tool.count == 300
    assert [result.call_id for result in results] == [call.id for call in calls]


@pytest.mark.asyncio
async def test_tool_batch_cancellation_awaits_workers():
    entered = asyncio.Event()

    class Blocking(Counter):
        async def execute(self, **kwargs):
            self.current += 1
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.current -= 1

    tool = Blocking()
    registry = ToolRegistry([tool], max_concurrent_tools=2)
    calls = [ToolCall(name=tool.name) for _ in range(20)]
    task = asyncio.create_task(registry.execute_parallel(calls))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tool.current == 0


def test_nested_ignored_html_stays_ignored():
    extractor = _HTMLTextExtractor()
    extractor.feed(
        "<p>visible</p><noscript><p>hidden</p>hidden too</noscript><svg><text>secret</text></svg><p>after</p>"
    )
    assert extractor.get_text() == "visible\nafter"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encoding,compress",
    [("gzip", gzip.compress), ("deflate", zlib.compress), ("br", brotli.compress)],
)
async def test_web_fetch_bounded_decompression(encoding, compress):
    async def fetch(data):
        response = httpx.Response(
            200, headers={"content-encoding": encoding}, stream=httpx.ByteStream(compress(data))
        )
        with patch.object(httpx.AsyncClient, "send", AsyncMock(return_value=response)):
            result = await WebFetchTool().execute(url="https://8.8.8.8")
        assert response.is_closed
        return result

    result = await fetch(b"documentation")
    assert not result.is_error and result.output == "documentation"
    result = await fetch(b"X" * (2 * 1024 * 1024 + 1))
    assert result.is_error and "decompressed limit" in result.output


@pytest.mark.asyncio
async def test_grep_searches_non_utf8_files(tmp_path):
    (tmp_path / "legacy.txt").write_bytes(b"needle\xff\n")
    result = await GrepSearchTool(tmp_path).execute(query="needle")
    assert not result.is_error and "legacy.txt:1:" in result.output


@pytest.mark.asyncio
async def test_globs_distinguish_path_depth(tmp_path):
    (tmp_path / "a.py").touch()
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "b.py").touch()
    (tmp_path / "pkg" / "sub" / "c.py").touch()
    tool = GlobFindTool(tmp_path)
    root = await tool.execute(pattern="*.py")
    recursive = await tool.execute(pattern="**/*.py")
    children = await tool.execute(pattern="pkg/*.py")
    assert root.output == "a.py"
    assert len(recursive.output.splitlines()) == 3
    assert children.output.replace("\\", "/") == "pkg/b.py"


@pytest.mark.asyncio
async def test_move_file_cross_device_fallback_and_no_overwrite(tmp_path):
    source = tmp_path / "a.txt"
    source.write_bytes(b"hello\r\n")
    with patch("openagent.tools.fs.os.link", side_effect=OSError(errno.EXDEV, "different drive")):
        result = await MoveFileTool(tmp_path).execute(source="a.txt", destination="folder/b.txt")
    assert not result.is_error
    assert not source.exists()
    assert (tmp_path / "folder/b.txt").read_bytes() == b"hello\r\n"
    source.write_text("keep", encoding="utf-8")
    result = await MoveFileTool(tmp_path).execute(source="a.txt", destination="folder/b.txt")
    assert result.is_error and source.read_text() == "keep"


@pytest.mark.asyncio
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
async def test_patch_multi_hunk_preserves_newlines_and_is_atomic(tmp_path, newline):
    target = tmp_path / "a.txt"
    original = newline.join(["a", "b", "c", "d", "e", ""])
    target.write_bytes(original.encode())
    tool = ApplyPatchTool(tmp_path)
    diff = "--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +1,2 @@\n a\n-b\n+B\n@@ -4,2 +4,2 @@\n d\n-e\n+E\n"
    registry = ToolRegistry([tool])
    assert registry.get_policy(tool.name) == PermissionAction.ASK
    denied = await registry.execute_call(
        ToolCall(name=tool.name, arguments={"path": "a.txt", "patch": diff})
    )
    assert denied.is_error and target.read_bytes() == original.encode()
    result = await tool.execute(path="a.txt", patch=diff.replace("-e", "-wrong"))
    assert result.is_error and target.read_bytes() == original.encode()
    result = await tool.execute(path="a.txt", patch=diff)
    assert not result.is_error
    assert target.read_bytes() == newline.join(["a", "B", "c", "d", "E", ""]).encode()
    result = await tool.execute(path="../outside.txt", patch=diff)
    assert result.is_error


@pytest.mark.asyncio
async def test_openai_start_event_has_id():
    chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "real-id",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ]
                }
            }
        ]
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n"
            )
        )
    ) as client:
        provider = OpenAICompatProvider("https://example.com/v1", "fake", client=client)
        events = [event async for event in provider.stream(ChatRequest(model="fake", messages=[]))]
    start = next(event for event in events if isinstance(event, ToolCallStart))
    assert start.id == "real-id"


@pytest.mark.asyncio
async def test_gemini_signed_parts_survive_stream_session_and_replay(tmp_path):
    parts = [
        {"text": "reason", "thought": True, "thoughtSignature": "thinking-signature"},
        {"functionCall": {"name": "read", "args": {}}, "thoughtSignature": "call-signature"},
        {"functionCall": {"name": "list", "args": {}}},
    ]
    chunks = [
        {"candidates": [{"content": {"parts": parts}}]},
        {"candidates": [{"content": {"parts": [{"thoughtSignature": "last-signature"}]}, "finishReason": "STOP"}]},
    ]
    body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    ) as client:
        provider = GeminiProvider("fake", client=client)
        events = [
            event
            async for event in provider.stream(
                ChatRequest(model="fake", messages=[Message.user("read")])
            )
        ]
        assistant = next(event.message for event in events if isinstance(event, DoneEvent))
        messages = [Message.user("read"), assistant] + [
            Message.tool_result(call.id, call.name, "ok") for call in assistant.tool_calls
        ]
        store = SessionStore(tmp_path)
        sid = store.create_session()
        store.save_messages(sid, messages)
        _, loaded = SessionStore(tmp_path).load_session(sid)
        payload = provider.build_payload(ChatRequest(model="fake", messages=loaded))
        replayed = payload["contents"][1]["parts"]
        assert replayed[0]["thoughtSignature"] == "thinking-signature"
        assert replayed[1]["thoughtSignature"] == "call-signature"
        assert replayed[2]["thoughtSignature"] == "last-signature"


@pytest.mark.asyncio
async def test_bare_model_switch_changes_vendor_and_inspection_is_local(monkeypatch):
    lines = iter(["/model deepseek-chat", "/history", "/messages", "/prompt", "/exit"])

    class Input:
        async def prompt_async(self, *args, **kwargs):
            return next(lines)

    monkeypatch.setattr(tui, "PromptSession", lambda **kwargs: Input())
    output = io.StringIO()
    runner = AgentRunner(ProviderRouter().resolve("gpt-4o", api_key="fake"))
    runner.messages.add_system("workspace [bold]literal instructions[/bold]")
    before = runner.messages.to_dict()
    await tui.run_repl(runner, console=Console(file=output, width=120))
    assert runner.provider.name == "deepseek"
    assert "deepseek" in runner.provider.base_url
    assert runner.messages.to_dict() == before
    assert "Conversation History" in output.getvalue()
    assert runner.messages.system_message.text in output.getvalue()
    await runner.provider.close()


def test_compaction_fits_and_pins_summary_without_growth():
    messages = [Message.system("sys")]
    for index in range(60):
        messages.extend([Message.user(f"goal {index} " + "u" * 200), Message.assistant("a" * 200)])
    compactor = Compactor()
    compacted = compactor.compact_heuristic(messages, 500)
    for _ in range(10):
        assert compactor.estimator.estimate_messages(compacted) <= 500
        window = MessageManager(compacted).window(500)
        assert sum(bool(message.metadata.get("is_summary")) for message in window) == 1
        previous_tokens = compactor.estimator.estimate_messages(compacted)
        compacted = compactor.compact_heuristic(compacted, 500)
        assert compactor.estimator.estimate_messages(compacted) <= previous_tokens


@pytest.mark.asyncio
async def test_long_tool_loop_compacts_through_runner_with_heuristic_fallback():
    class Large(Tool):
        name = "large"
        description = "large result"

        def __init__(self):
            self.params = []

        async def execute(self, **kwargs):
            return ToolResult(kwargs["call_id"], "R" * 1000)

    class Provider(ChatProvider):
        def __init__(self):
            self.turns = self.summaries = 0
            self.requests = []

        async def complete(self, request):
            self.summaries += 1
            raise RuntimeError("summarizer unavailable")

        async def stream(self, request):
            self.requests.append(request)
            self.turns += 1
            message = (
                Message.assistant(tool_calls=[ToolCall(id=str(self.turns), name="large")])
                if self.turns <= 20
                else Message.assistant("done")
            )
            yield DoneEvent(message=message)

    provider = Provider()
    runner = AgentRunner(
        provider,
        tools=ToolRegistry([Large()]),
        messages=MessageManager([Message.system("sys")]),
        context_window=1600,
        max_tokens=100,
        max_tool_iterations=30,
    )
    assert await runner.run_turn_to_completion("active goal") == "done"
    assert provider.summaries > 0
    assert any(any(m.metadata.get("is_summary") for m in req.messages) for req in provider.requests)
    for request in provider.requests:
        assert any(message.text == "active goal" for message in request.messages)
        assert runner.messages.estimator.estimate_request(request) + 100 <= 1600
        assert all(
            MessageManager.valid_tool_group(group)
            for group in MessageManager.get_atomic_groups(request.messages)
        )


def test_session_append_preserves_previous_bytes_and_recovers_torn_batch(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    sid = store.create_session(model="fake")
    messages = [Message.user("first")]
    store.save_messages(sid, messages)
    path = tmp_path / f"{sid}.jsonl"
    before = path.read_bytes()
    monkeypatch.setattr(
        store,
        "load_session",
        lambda *args: pytest.fail("cached incremental save must not reload history"),
    )
    messages.append(Message.assistant("answer"))
    store.save_messages(sid, messages)
    assert path.read_bytes().startswith(before)
    with path.open("ab") as output:
        output.write(b'{"type":"batch","messages":[')
    recovered = SessionStore(tmp_path)
    metadata, loaded = recovered.load_session(sid)
    assert metadata.model == "fake"
    assert [message.text for message in loaded] == ["first", "answer"]
    loaded.append(Message.user("after recovery"))
    recovered.save_messages(sid, loaded)
    assert len(SessionStore(tmp_path).load_session(sid)[1]) == 3
    loaded[0].content = Message.user("revised").content
    recovered.save_messages(sid, loaded)
    assert SessionStore(tmp_path).load_session(sid)[1][0].text == "revised"
    assert recovered.delete_session(sid)
    # Lock identity remains stable for other writers waiting across deletion.
    assert [file.name for file in tmp_path.iterdir()] == [f"{sid}.lock"]


@pytest.mark.parametrize("prompt", ["-weird prompt", "check this", "-m gpt-4o", "run"])
def test_cli_separator_defaults_to_run(monkeypatch, prompt):
    observed = []

    async def run(args, console):
        observed.append(args)
        return 0

    monkeypatch.setattr(cli, "_async_run", run)
    assert cli.main(["--", prompt]) == 0
    assert observed[0].subcommand == "run"
    assert observed[0].prompt == [prompt]


def test_design_module_paths_and_configuration_example(tmp_path, monkeypatch):
    import re
    from pathlib import Path

    from openagent.config import load_config

    root = Path(__file__).resolve().parents[1]
    design = (root / "docs/DESIGN.md").read_text(encoding="utf-8")
    for module in re.findall(r"\| `([^`]+\.py)` \|", design):
        assert (root / "src/openagent" / module).is_file(), module
    example = re.search(r"```toml\n(.*?)```", design, re.DOTALL)[1]
    config_file = tmp_path / "example.toml"
    config_file.write_text(example, encoding="utf-8")
    monkeypatch.setenv("OPENAGENT_GLOBAL_CONFIG", str(tmp_path / "missing.toml"))
    config = load_config(config_path=config_file)
    assert config.model == "custom/internal-model"
    assert config.custom_provider.jsonpath_text == "response.output.text"
    assert config.mcp_servers[0].transport == "stdio"


def test_session_can_export_legacy_format_before_downgrade(tmp_path):
    store = SessionStore(tmp_path)
    sid = store.create_session(model="fake")
    messages = [Message.user("request"), Message.assistant("answer")]
    messages[-1].metadata["gemini_parts"] = [{"text": "answer", "thoughtSignature": "signed"}]
    store.save_messages(sid, messages)
    legacy = SessionStore(tmp_path, incremental=False)
    metadata, loaded = legacy.load_session(sid)
    legacy.save_messages(sid, loaded, metadata=metadata)
    rows = [json.loads(line) for line in (tmp_path / f"{sid}.jsonl").read_text().splitlines()]
    assert [row["type"] for row in rows] == ["metadata", "message", "message"]
    assert rows[0]["model"] == "fake"
    assert [Message.from_dict(row).text for row in rows[1:]] == ["request", "answer"]
    assert rows[-1]["metadata"]["gemini_parts"][0]["thoughtSignature"] == "signed"
