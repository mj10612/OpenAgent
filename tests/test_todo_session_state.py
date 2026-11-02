import json

import pytest

from openagent.core.types import Message, ToolCall
from openagent.providers.ollama import OllamaProvider
from openagent.runner import AgentRunner
from openagent.session.store import SessionConflictError, SessionStore
from openagent.tools.agent_tools import TodoTool
from openagent.tools.registry import ToolRegistry


def runner(store, sid=None):
    return AgentRunner(OllamaProvider('llava'), tools=ToolRegistry([TodoTool()]),
                       session_store=store, session_id=sid)


async def todo(agent, **arguments):
    result = await agent.tools.execute_call(ToolCall(name='todo', arguments=arguments))
    assert not result.is_error
    return result.output


@pytest.mark.asyncio
@pytest.mark.parametrize('incremental', [True, False])
async def test_todo_resume_reset_and_session_isolation(tmp_path, incremental):
    store = SessionStore(tmp_path, incremental=incremental)
    first = runner(store)
    first.messages.add_user('continue review')
    await todo(first, action='add', tasks=['Review auth', 'Add tests'])
    first._persist_session()
    await todo(first, action='complete', task='1')
    first._persist_session()
    resumed = runner(SessionStore(tmp_path), first.session_id)
    assert await todo(resumed, action='list') == '1. [x] Review auth\n2. [ ] Add tests'
    await todo(resumed, action='add', tasks=['Independent'])
    assert 'Independent' not in await todo(first, action='list')
    assert await todo(runner(store), action='list') == 'Task checklist is empty.'
    resumed.reset()
    assert await todo(resumed, action='list') == 'Task checklist is empty.'
    resumed._persist_session()
    assert resumed.session_id != first.session_id


@pytest.mark.asyncio
@pytest.mark.parametrize('format', ['json', 'jsonl'])
async def test_todo_export_import_new_id_and_redaction(tmp_path, format):
    store = SessionStore(tmp_path / 'sessions')
    source = runner(store)
    source.messages.add_user('review')
    await todo(source, action='add', tasks=['PRIVATE_CHECKLIST'])
    source._persist_session()
    path = store.export_session(source.session_id, tmp_path / ('export.' + format), format=format)
    copied = store.import_session(path, new_id=True)
    resumed = runner(store, copied)
    assert 'PRIVATE_CHECKLIST' in await todo(resumed, action='list')
    await todo(resumed, action='clear')
    resumed._persist_session()
    assert 'PRIVATE_CHECKLIST' in await todo(runner(store, source.session_id), action='list')
    redacted = store.export_session(source.session_id, tmp_path / ('redacted.' + format), format=format, redact=True)
    assert 'PRIVATE_CHECKLIST' not in redacted.read_text(encoding='utf-8')


@pytest.mark.asyncio
@pytest.mark.parametrize('state', [None, {'version': 99, 'tasks': []}, {'version': 1, 'tasks': [{'text': 'bad', 'done': 'false'}]}, []])
async def test_legacy_or_invalid_state_preserves_history(tmp_path, state, caplog):
    document = {'schema_version': 1, 'metadata': {'session_id': 'legacy'},
                'messages': [Message.user('kept conversation').to_dict()]}
    if state is not None:
        document['metadata']['todo_state'] = state
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(document), encoding='utf-8')
    store = SessionStore(tmp_path / 'sessions')
    sid = store.import_session(path)
    agent = runner(store, sid)
    assert agent.messages.messages[-1].text == 'kept conversation'
    assert await todo(agent, action='list') == 'Task checklist is empty.'
    if state is not None:
        assert 'checklist' in caplog.text.lower()


@pytest.mark.asyncio
async def test_todo_updates_obey_stale_writer_conflicts(tmp_path):
    first = runner(SessionStore(tmp_path))
    first.messages.add_user('review')
    await todo(first, action='add', tasks=['one'])
    first._persist_session()
    stale = runner(SessionStore(tmp_path), first.session_id)
    await todo(first, action='add', tasks=['two'])
    first._persist_session()
    await todo(stale, action='clear')
    with pytest.raises(SessionConflictError):
        stale._persist_session()


@pytest.mark.asyncio
async def test_completed_agent_turn_persists_todo_without_history_reconstruction(tmp_path):
    from openagent.core.events import DoneEvent
    from openagent.core.provider import ChatProvider
    from openagent.core.types import FinishReason

    class Provider(ChatProvider):
        model = 'fixture'
        context_window = 10000
        default_max_tokens = 100

        async def stream(self, request):
            if request.messages[-1].role == 'user':
                yield DoneEvent(message=Message.assistant(tool_calls=[ToolCall(
                    name='todo', arguments={'action': 'add', 'tasks': ['Persisted task']})]),
                    finish_reason=FinishReason.TOOL_CALLS)
            else:
                yield DoneEvent(message=Message.assistant('saved'))

    store = SessionStore(tmp_path)
    agent = AgentRunner(Provider(), tools=ToolRegistry([TodoTool()]), session_store=store)
    events = [event async for event in agent.run_turn('plan')]
    assert isinstance(events[-1], DoneEvent)
    # Compacted/replaced history must not determine checklist restoration.
    store.save_messages(agent.session_id, [Message.user('new history')])
    resumed = runner(SessionStore(tmp_path), agent.session_id)
    assert await todo(resumed, action='list') == '1. [ ] Persisted task'
