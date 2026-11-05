import json
from unittest.mock import Mock

import pytest

from openagent import cli
from openagent.config import OpenAgentConfig
from openagent.core.events import DoneEvent, ErrorEvent
from openagent.core.types import Message
from openagent.session.store import SessionStore


@pytest.mark.parametrize('argv', [
    ['--json', 'hello'], ['--jsonl', 'hello'],
    ['--image', 'models', 'hello'], ['--prompt-file', 'notes.txt'],
    ['--model', 'gpt-4o', '--image=shot.png', '--json', 'hello'], ['--json', '-'],
])
def test_shorthand_run_options_are_parsed(argv, monkeypatch):
    seen = []
    async def run(args, console):
        seen.append(args)
        return 0
    monkeypatch.setattr(cli, '_async_run', run)
    assert cli.main(argv) == 0
    assert seen[0].subcommand == 'run'


@pytest.mark.parametrize('action', [[], ['list']])
def test_session_json_is_metadata_only_and_nonmutating(tmp_path, monkeypatch, capsys, action):
    store = SessionStore(tmp_path / 'sessions')
    sid = store.create_session(model='fixture', title='Unicode \uac80\ud1a0')
    meta, _ = store.load_session(sid)
    meta.todo_state = {'version': 1, 'tasks': [{'text': 'PRIVATE', 'done': False}]}
    store.save_messages(sid, [Message.user('PRIVATE')], metadata=meta)
    before = {p.name: p.read_bytes() for p in store.storage_dir.iterdir()}
    monkeypatch.setenv('OPENAGENT_SESSION_DIR', str(store.storage_dir))
    assert cli.main(['sessions', *action, '--json']) == 0
    output = capsys.readouterr().out
    data = json.loads(output)
    assert data['schema_version'] == 1
    assert data['sessions'][0]['session_id'] == sid
    assert data['sessions'][0]['title'] == 'Unicode \uac80\ud1a0'
    assert 'PRIVATE' not in output and 'todo_state' not in output
    assert {p.name: p.read_bytes() for p in store.storage_dir.iterdir()} == before


def test_missing_session_archive_json_has_no_side_effects(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'missing' / 'sessions'
    monkeypatch.setenv('OPENAGENT_SESSION_DIR', str(path))
    assert cli.main(['sessions', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['sessions'] == []
    assert not path.parent.exists()


@pytest.mark.parametrize('action', [['--clear'], ['prune', '--older-than', '1'], ['export', 'id'], ['delete', 'id']])
def test_session_json_rejects_nonlisting_actions(action, capsys):
    assert cli.main(['sessions', *action, '--json']) == 1
    assert json.loads(capsys.readouterr().out)['outcome'] == 'error'


@pytest.mark.parametrize('terminal', ['success', 'error', 'cancelled'])
def test_no_session_save_never_constructs_archive(tmp_path, monkeypatch, capsys, terminal):
    from openagent.providers.ollama import OllamaProvider
    monkeypatch.setenv('OPENAGENT_SESSION_DIR', str(tmp_path / 'missing' / 'sessions'))
    monkeypatch.setattr(cli, '_load_args_config', lambda args: OpenAgentConfig(model='ollama/llava'))
    forbid = Mock(side_effect=AssertionError('session store must not be constructed'))
    monkeypatch.setattr(cli, 'SessionStore', forbid)
    async def stream(self, request):
        if terminal == 'cancelled':
            import asyncio
            raise asyncio.CancelledError
        if terminal == 'error':
            yield ErrorEvent(RuntimeError('fixture error'))
        else:
            yield DoneEvent(message=Message.assistant('ok'))
    monkeypatch.setattr(OllamaProvider, 'stream', stream)
    code = cli.main(['run', 'hello', '--json', '--no-session-save'])
    assert code == {'success': 0, 'error': 1, 'cancelled': 130}[terminal]
    assert json.loads(capsys.readouterr().out)['session'] is None
    forbid.assert_not_called()
    assert not (tmp_path / 'missing').exists()


def test_no_session_save_rejects_configured_resume_before_provider(monkeypatch, capsys):
    monkeypatch.setattr(cli, '_load_args_config', lambda args: OpenAgentConfig(session_id='saved'))
    build = Mock(side_effect=AssertionError('provider must not be built'))
    monkeypatch.setattr(cli, '_build_runner', build)
    assert cli.main(['run', 'hello', '--json', '--no-session-save']) == 1
    assert 'resume' in json.loads(capsys.readouterr().out)['errors'][0]
    build.assert_not_called()


def test_no_session_save_available_in_chat():
    assert cli._build_parser().parse_args(['chat', '--no-session-save']).no_session_save


def test_session_json_sorts_ties_and_reports_corrupt_entries(tmp_path, monkeypatch, capsys, caplog):
    for sid in ('z-last', 'a-first'):
        (tmp_path / (sid + '.jsonl')).write_text(json.dumps({
            'type': 'metadata', 'session_id': sid, 'created_at': 1, 'updated_at': 2,
        }) + '\n', encoding='utf-8')
    (tmp_path / 'broken.jsonl').write_text('{bad json\n', encoding='utf-8')
    monkeypatch.setenv('OPENAGENT_SESSION_DIR', str(tmp_path))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(SessionStore, '_locked', Mock(side_effect=AssertionError('read-only listing must not open locks')))
    assert cli.main(['sessions', '--json']) == 0
    rows = json.loads(capsys.readouterr().out)['sessions']
    assert [row['session_id'] for row in rows if row['session_id'] != 'broken'] == ['a-first', 'z-last']
    assert 'skipped corrupt line' in caplog.text
    assert all(set(row) == {'session_id', 'title', 'model', 'created_at', 'updated_at', 'message_count'} for row in rows)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.asyncio
async def test_custom_model_factory_uses_effective_configuration():
    from openagent.config import CustomProviderConfig
    cfg = OpenAgentConfig(base_url='https://custom.test', api_key='fixture-key',
        custom_provider=CustomProviderConfig(jsonpath_text='answer', chat_endpoint='/generate',
                                             usage_mode='cumulative', auth_scheme='api_key', auth_header='x-key'))
    provider = cli._resolve_repl_provider('custom/new-model', cfg)
    try:
        assert provider.base_url == 'https://custom.test'
        assert provider.model == 'new-model'
        assert provider.text_path == 'answer' and provider.chat_endpoint == '/generate'
        assert provider.usage_mode == 'cumulative'
        assert provider.transport.extra_headers['x-key'] == 'fixture-key'
    finally:
        await provider.close()


def test_custom_factory_does_not_forward_config_to_other_provider(monkeypatch):
    resolve = Mock(return_value=object())
    monkeypatch.setattr(cli.ProviderRouter, 'resolve', resolve)
    cli._resolve_repl_provider('ollama/llava', OpenAgentConfig(base_url='https://private.test', api_key='private'))
    resolve.assert_called_once_with('ollama/llava')
