import json
import multiprocessing

import pytest
from test_runner import MockChatProvider

from openagent.context.compactor import Compactor
from openagent.context.messages import MessageManager
from openagent.core.events import DoneEvent, ErrorEvent, TextDelta, UsageEvent
from openagent.core.types import (
    ChatResponse,
    ImagePart,
    Message,
    ThinkingPart,
    ToolCall,
    ToolResultPart,
    Usage,
)
from openagent.runner import AgentRunner
from openagent.session.store import SessionStore


def test_stale_writer_rejected(tmp_path):
    a, b = SessionStore(tmp_path), SessionStore(tmp_path)
    sid = a.create_session()
    a.save_messages(sid, [Message.user("base")])
    _, ma = a.load_session(sid)
    _, mb = b.load_session(sid)
    a.save_messages(sid, [*ma, Message.assistant("A")])
    with pytest.raises(RuntimeError, match="conflict"):
        b.save_messages(sid, [*mb, Message.assistant("B")])
    assert a.load_session(sid)[1][-1].text == "A"


def test_deleted_session_cannot_be_resurrected_by_stale_writer(tmp_path):
    a, b = SessionStore(tmp_path), SessionStore(tmp_path)
    sid = a.create_session()
    a.save_messages(sid, [Message.user("base")])
    _, snapshot = b.load_session(sid)
    assert a.delete_session(sid)
    with pytest.raises(RuntimeError, match="conflict"):
        b.save_messages(sid, [*snapshot, Message.assistant("stale")])
    assert not (tmp_path / f"{sid}.jsonl").exists()


def test_cache_is_bounded_after_active_history_loads(tmp_path):
    store = SessionStore(tmp_path)
    ids = [store.create_session() for _ in range(12)]
    for sid in ids:
        store.save_messages(sid, [Message.user("x" * 600000)])
    fresh = SessionStore(tmp_path)
    for sid in ids:
        fresh.load_session(sid)
    assert len(fresh._cache) <= 8
    assert sum(sum(len(row.encode("utf8")) for row in item[1]) for item in fresh._cache.values()) <= 4 * 1024 * 1024


def _concurrent_writer(directory, sid, ready, results, label, incremental):
    store = SessionStore(directory, incremental=incremental)
    _, messages = store.load_session(sid)
    ready.wait(timeout=10)
    try:
        store.save_messages(sid, [*messages, Message.assistant(label)])
        results.put("saved")
    except RuntimeError:
        results.put("conflict")


@pytest.mark.parametrize("incremental", [False, True])
def test_process_writers_conflict_without_loss(tmp_path, incremental):
    store = SessionStore(tmp_path)
    sid = store.create_session()
    store.save_messages(sid, [Message.user("base")])
    context = multiprocessing.get_context("spawn")
    ready = context.Barrier(2)
    results = context.Queue()
    workers = [context.Process(target=_concurrent_writer, args=(str(tmp_path), sid, ready, results, label, incremental)) for label in ("A", "B")]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=15)
        if worker.is_alive():
            worker.terminate()
            worker.join()
        assert worker.exitcode == 0
    assert sorted(results.get(timeout=2) for _ in workers) == ["conflict", "saved"]
    messages = store.load_session(sid)[1]
    assert len(messages) == 2
    assert messages[-1].text in {"A", "B"}


def test_list_does_not_cache_archive(tmp_path):
    store = SessionStore(tmp_path)
    for i in range(5):
        sid = store.create_session(title=str(i))
        store.save_messages(sid, [Message.user("x" * 100000)])
    fresh = SessionStore(tmp_path)
    assert len(fresh.list_sessions()) == 5
    assert not fresh._cache


@pytest.mark.parametrize("batch", [False, True])
def test_invalid_raw_role_recovered(tmp_path, batch):
    store = SessionStore(tmp_path)
    sid = store.create_session()
    row = {"role": "invalid-role", "content": "corrupt"}
    if batch:
        row = {"type": "batch", "metadata": store.load_session(sid)[0].to_dict(), "messages": [row]}
    with (tmp_path / f"{sid}.jsonl").open("a", encoding="utf8") as out:
        out.write(json.dumps(row) + "\n")
    assert store.load_session(sid)[1] == []


def test_deep_snapshot():
    m = Message.assistant("original", tool_calls=[ToolCall(id="id", name="x", arguments={"nested": {"x": []}}, raw_arguments="raw")])
    m.metadata = {"parts": [{"text": "original"}]}
    snapshot = MessageManager([m]).copy()[0]
    snapshot.tool_calls[0].arguments["nested"]["x"].append(1)
    snapshot.metadata["parts"][0]["text"] = "changed"
    assert m.tool_calls[0].arguments["nested"]["x"] == []
    assert m.metadata["parts"][0]["text"] == "original"
    assert snapshot.tool_calls[0].raw_arguments == "raw"


def test_model_metadata_updated(tmp_path):
    runner = AgentRunner(MockChatProvider(), session_store=SessionStore(tmp_path))
    runner._persist_session()
    runner.model = "changed-model"
    runner._persist_session()
    assert runner.session_store.load_session(runner.session_id)[0].model == "changed-model"


async def test_failed_usage_preserved():
    provider = MockChatProvider([[UsageEvent(Usage(10, 5)), ErrorEvent("fail")]])
    runner = AgentRunner(provider)
    events = [event async for event in runner.run_turn("hello")]
    assert any(isinstance(e, ErrorEvent) for e in events)
    assert runner.usage.total_tokens == 15


async def test_usage_snapshot_reconciliation():
    provider = MockChatProvider([[UsageEvent(Usage(10, 2)), UsageEvent(Usage(10, 5)), DoneEvent(message=Message.assistant("ok"), usage=Usage(10, 5))]])
    runner = AgentRunner(provider)
    events = [event async for event in runner.run_turn("hello")]
    assert events[-1].usage.total_tokens == 15
    assert runner.usage.total_tokens == 15


async def test_cancelled_stream_keeps_reported_usage():
    provider = MockChatProvider([[UsageEvent(Usage(10, 5)), TextDelta("partial")]])
    runner = AgentRunner(provider)
    stream = runner.run_turn("hello")
    assert isinstance(await anext(stream), UsageEvent)
    await stream.aclose()
    assert runner.usage.total_tokens == 15
    assert len(runner.messages) == 1


async def test_nested_provider_mutation_does_not_break_rollback():
    class MutatingProvider(MockChatProvider):
        async def stream(self, request):
            request.messages[-2].metadata["native"][0]["text"] = "changed"
            yield ErrorEvent("fail")

    original = Message.assistant("old")
    original.metadata = {"native": [{"text": "original"}]}
    runner = AgentRunner(MutatingProvider(), messages=MessageManager([Message.system("sys"), original]))
    events = [event async for event in runner.run_turn("hello")]
    assert isinstance(events[-1], ErrorEvent)
    assert runner.messages[1].metadata["native"][0]["text"] == "original"


async def test_missing_done_is_error():
    runner = AgentRunner(MockChatProvider([[TextDelta("partial")]]))
    events = [event async for event in runner.run_turn("hello")]
    assert isinstance(events[-1], ErrorEvent)
    assert len(runner.messages) == 1


async def test_summary_budget_and_usage():
    class Provider(MockChatProvider):
        context_window = 1000

        async def complete(self, request):
            self.recorded_requests.append(request)
            return ChatResponse(Message.assistant("summary"), usage=Usage(20, 10))

    provider = Provider()
    compactor = Compactor()
    await compactor.extract_llm_summary([Message.user("<&" * 10000)], provider)
    for request in provider.recorded_requests:
        assert compactor.estimator.estimate_request(request) + request.max_tokens <= 1000
    assert compactor.usage.total_tokens == 30 if provider.recorded_requests else True


async def test_compaction_and_generation_account_once():
    class Provider(MockChatProvider):
        context_window = 2000

        async def complete(self, request):
            return ChatResponse(Message.assistant("summary"), usage=Usage(20, 10))

    provider = Provider()
    history = MessageManager([Message.system("sys"), Message.user("x" * 3000), Message.assistant("y" * 200)])
    runner = AgentRunner(provider, messages=history, context_window=600, max_tokens=100)
    events = [event async for event in runner.run_turn("latest")]
    assert isinstance(events[-1], DoneEvent)
    assert runner.generation_usage.total_tokens == 15
    assert runner.compaction_usage.total_tokens == 30
    assert runner.usage.total_tokens == events[-1].usage.total_tokens == 45


async def test_compaction_usage_survives_later_hard_failure():
    class FailingCompactor(Compactor):
        async def compact(self, *args, **kwargs):
            self.usage = self.usage + Usage(20, 10)
            raise ValueError("Cannot fit compacted context")

    history = MessageManager([Message.system("sys"), Message.user("x" * 3000), Message.assistant("answer")])
    runner = AgentRunner(MockChatProvider(), messages=history, compactor=FailingCompactor(), context_window=600, max_tokens=100)
    events = [event async for event in runner.run_turn("latest")]
    assert isinstance(events[-1], ErrorEvent)
    assert runner.compaction_usage.total_tokens == runner.usage.total_tokens == 30


@pytest.mark.parametrize("format", ["json", "jsonl"])
def test_portable_roundtrip(tmp_path, format):
    store = SessionStore(tmp_path / "sessions")
    sid = store.create_session(model="model")
    original = Message.user("한글 😀")
    original.metadata = {"native": [{"signature": "signed"}]}
    store.save_messages(sid, [original])
    target = tmp_path / f"export.{format}"
    store.export_session(sid, target, format=format)
    new_sid = store.import_session(target, new_id=True)
    assert new_sid != sid
    meta, messages = store.load_session(new_sid)
    assert meta.model == "model"
    assert messages[0].to_dict() == original.to_dict()
    with pytest.raises(FileExistsError):
        store.import_session(target)


@pytest.mark.parametrize("row", [{"role": "invalid", "content": "x"}, {"role": "tool", "content": "orphan", "tool_call_id": "id"}])
def test_import_rejects_corrupt_schema_or_correspondence(tmp_path, row):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"metadata": {"session_id": "valid"}, "messages": [row]}), encoding="utf8")
    store = SessionStore(tmp_path / "sessions")
    with pytest.raises(ValueError):
        store.import_session(source)
    assert store.list_sessions() == []


def test_import_rejects_traversal(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"metadata": {"session_id": "../escape"}, "messages": []}), encoding="utf8")
    with pytest.raises(ValueError):
        SessionStore(tmp_path / "sessions").import_session(source, new_id=True)


def test_portable_signed_multimodal_tool_payloads(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    sid = store.create_session()
    call = ToolCall(name="tool", id="local", arguments={"nested": [1]}, raw_arguments='{"nested": [1]}')
    assistant = Message.assistant(tool_calls=[call])
    assistant.content = [ThinkingPart("thought", signature="signed"), ImagePart(data="AA==")]
    result = Message.tool_result("local", "tool", "text", structured_data={"answer": 42}, metadata={"source": "mcp"})
    extra = Message.user("next")
    extra.content.append(ToolResultPart("other", "embedded", "content", structured_data={"nested": [1]}, metadata={"native": ["value"]}))
    messages = [assistant, result, extra]
    store.save_messages(sid, messages)
    target = tmp_path / "portable.jsonl"
    store.export_session(sid, target)
    imported = store.import_session(target, new_id=True)
    restored = store.load_session(imported)[1]
    assert [message.to_dict() for message in restored] == [message.to_dict() for message in messages]
    serialized = extra.to_dict()
    serialized["content"][-1]["structured_data"]["nested"].append(2)
    assert extra.content[-1].structured_data == {"nested": [1]}


def test_metadata_scan_latest_batch_and_corruption(tmp_path):
    store = SessionStore(tmp_path)
    sid = store.create_session(model="old")
    store.save_messages(sid, [Message.user("hello")])
    meta = store.load_metadata(sid)
    meta.model = "new"
    store.save_messages(sid, [Message.user("hello"), Message.assistant("answer")], meta)
    # A damaged line must not alter counts or cached history.
    with (tmp_path / f"{sid}.jsonl").open("a", encoding="utf8") as out:
        out.write('{"role":"bad","content":"wrong"}\n')
    fresh = SessionStore(tmp_path)
    metadata = fresh.list_sessions()[0]
    assert metadata.model == "new"
    assert metadata.message_count == 2
    assert not fresh._cache
