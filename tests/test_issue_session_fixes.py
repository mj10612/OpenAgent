import asyncio
import json
import os
import time

import pytest

from openagent.core.events import DoneEvent, ErrorEvent
from openagent.core.provider import ChatProvider
from openagent.core.types import Message, ToolCall
from openagent.runner import AgentRunner
from openagent.session.store import SessionStore


def snapshot(tmp_path, messages, **metadata):
    source = tmp_path / 'input.json'
    source.write_text(json.dumps({'metadata': {'session_id': 'imported', **metadata}, 'messages': messages}), encoding='utf8')
    return source


@pytest.mark.parametrize('format', ['json', 'jsonl', 'markdown'])
@pytest.mark.parametrize('raw', ['{"api_key":"FAKE_SECRET"}', 'invalid FAKE_SECRET'])
def test_redaction_removes_raw_argument_secrets(tmp_path, format, raw):
    store = SessionStore(tmp_path / 'sessions')
    sid = store.create_session()
    store.save_messages(sid, [Message.assistant(tool_calls=[ToolCall('echo', id='c', arguments={'api_key': 'FAKE_SECRET'}, raw_arguments=raw)]), Message.tool_result('c', 'echo', 'ok')])
    target = store.export_session(sid, tmp_path / 'export', format=format, redact=True)
    assert 'FAKE_SECRET' not in target.read_text(encoding='utf8')


@pytest.mark.parametrize('bad_id', [[], {}, None, '', 42])
def test_corrupt_result_ids_preserve_later_history(tmp_path, bad_id):
    store = SessionStore(tmp_path)
    result = Message.tool_result('c', 'echo', 'ok').to_dict()
    result['tool_call_id'] = bad_id
    rows = [{'type': 'metadata', 'session_id': 'damaged'}, Message.assistant(tool_calls=[ToolCall('echo', id='c')]).to_dict(), result, Message.user('later valid').to_dict()]
    (tmp_path / 'damaged.jsonl').write_text('\n'.join(map(json.dumps, rows)), encoding='utf8')
    assert [m.text for m in store.load_session('damaged')[1]] == ['later valid']
    assert store.load_metadata('damaged').message_count == 1
    assert store.cleanup_sessions(older_than_days=1) == 0


@pytest.mark.parametrize('role', ['system', 'user', 'tool'])
def test_import_rejects_calls_on_nonassistant_roles(tmp_path, role):
    row = Message.assistant(tool_calls=[ToolCall('echo', id='c')]).to_dict()
    row['role'] = role
    with pytest.raises(ValueError):
        SessionStore(tmp_path / 'sessions').import_session(snapshot(tmp_path, [row]))


@pytest.mark.parametrize('arguments', ['{"text":', '[]', 'null', '42'])
def test_import_rejects_nonobject_argument_json(tmp_path, arguments):
    row = Message.assistant(tool_calls=[ToolCall('echo', id='c')]).to_dict()
    row['tool_calls'][0]['arguments'] = arguments
    with pytest.raises(ValueError):
        SessionStore(tmp_path / 'sessions').import_session(snapshot(tmp_path, [row, Message.tool_result('c', 'echo', 'ok').to_dict()]))


@pytest.mark.parametrize('field', ['created_at', 'updated_at'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_import_rejects_nonfinite_metadata(tmp_path, field, value):
    with pytest.raises(ValueError):
        SessionStore(tmp_path / 'sessions').import_session(snapshot(tmp_path, [Message.user('new').to_dict()], **{field: value}))


def test_age_cleanup_keeps_invalid_metadata_even_with_old_file_time(tmp_path):
    store = SessionStore(tmp_path)
    path = tmp_path / 'damaged.jsonl'
    path.write_text(json.dumps({'type': 'metadata', 'session_id': 'damaged', 'updated_at': float('nan')}), encoding='utf8')
    old = time.time() - 86400 * 100
    os.utime(path, (old, old))
    assert store.cleanup_sessions(older_than_days=1) == 0
    assert path.exists()


class BlockedProvider(ChatProvider):
    def __init__(self):
        self.ready = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, request):
        if request.messages[-1].text == 'A':
            self.ready.set()
            await self.release.wait()
            yield ErrorEvent(RuntimeError('A fails'))
        else:
            yield DoneEvent(message=Message.assistant('B done'))


@pytest.mark.asyncio
async def test_overlapping_turn_rejected_before_history_mutation():
    provider = BlockedProvider()
    runner = AgentRunner(provider)
    async def consume(text):
        return [event async for event in runner.run_turn(text)]
    task = asyncio.create_task(consume('A'))
    await provider.ready.wait()
    try:
        events = await consume('B')
        assert len(events) == 1 and isinstance(events[0], ErrorEvent)
        assert [m.text for m in runner.messages if m.role != 'system'] == ['A']
    finally:
        provider.release.set()
        await task
    assert any(isinstance(e, DoneEvent) for e in await consume('B'))
    assert [m.text for m in runner.messages if m.role != 'system'] == ['B', 'B done']


@pytest.mark.asyncio
async def test_cancellation_releases_turn_guard():
    provider = BlockedProvider()
    runner = AgentRunner(provider)
    async def consume(text):
        return [event async for event in runner.run_turn(text)]
    task = asyncio.create_task(consume('A'))
    await provider.ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert any(isinstance(e, DoneEvent) for e in await consume('B'))
    assert [m.text for m in runner.messages if m.role != 'system'] == ['B', 'B done']


@pytest.mark.parametrize('bad_id', [[], {}, None, '', 42])
def test_import_rejects_invalid_result_id(tmp_path, bad_id):
    result = Message.tool_result('c', 'echo', 'ok').to_dict()
    result['tool_call_id'] = bad_id
    with pytest.raises(ValueError):
        SessionStore(tmp_path / 'sessions').import_session(snapshot(tmp_path, [Message.assistant(tool_calls=[ToolCall('echo', id='c')]).to_dict(), result]))


def test_valid_string_arguments_and_forgiving_core_parser_are_preserved(tmp_path):
    raw = '{"nested": {"value": 3}}'
    row = Message.assistant(tool_calls=[ToolCall('echo', id='c')]).to_dict()
    row['tool_calls'][0]['arguments'] = raw
    row['tool_calls'][0]['raw_arguments'] = raw
    store = SessionStore(tmp_path / 'sessions')
    sid = store.import_session(snapshot(tmp_path, [row, Message.tool_result('c', 'echo', 'ok').to_dict()]))
    call = store.load_session(sid)[1][0].tool_calls[0]
    assert call.arguments == {'nested': {'value': 3}}
    assert call.raw_arguments == raw
    assert ToolCall.from_dict({'name': 'echo', 'arguments': 'broken'}).arguments == {}
