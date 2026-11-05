import json

import pytest

from openagent.cli import main
from openagent.core.types import Message
from openagent.session.store import SessionConflictError, SessionStore


def saved(store):
    sid = store.create_session(model='fixture', title='before')
    meta, _ = store.load_session(sid)
    meta.todo_state = {'version': 1, 'tasks': [{'text': 'kept task', 'done': True}]}
    store.save_messages(sid, [Message.user('kept conversation')], metadata=meta)
    return sid


def test_rename_preserves_history_state_and_retention_timestamp(tmp_path):
    store = SessionStore(tmp_path)
    sid = saved(store)
    before, messages = store.load_session(sid)
    store.rename_session(sid[:8], '\uac80\ud1a0 [red]literal')
    after, loaded = SessionStore(tmp_path).load_session(sid)
    assert after.title == '\uac80\ud1a0 [red]literal'
    assert {k: v for k, v in after.to_dict().items() if k != 'title'} == {
        k: v for k, v in before.to_dict().items() if k != 'title'}
    assert loaded == messages
    path = store.export_session(sid, tmp_path / 'export.json', format='json')
    copied = store.import_session(path, new_id=True)
    assert store.load_metadata(copied).title == after.title


@pytest.mark.parametrize('title', ['', '  \n'])
def test_blank_title_leaves_archive_unchanged(tmp_path, title):
    store = SessionStore(tmp_path)
    sid = saved(store)
    path = tmp_path / (sid + '.jsonl')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='title'):
        store.rename_session(sid, title)
    assert path.read_bytes() == before


def test_rename_and_save_enforce_stale_writer_conflicts(tmp_path):
    first = SessionStore(tmp_path)
    sid = saved(first)
    second = SessionStore(tmp_path)
    second.load_session(sid)
    first.rename_session(sid, 'new')
    with pytest.raises(SessionConflictError):
        second.rename_session(sid, 'stale')
    with pytest.raises(SessionConflictError):
        second.save_messages(sid, [Message.user('lost update')])


def test_fresh_rename_loads_latest_conversation_under_lock(tmp_path):
    first = SessionStore(tmp_path)
    sid = saved(first)
    first.append_message(sid, Message.assistant('new reply'))
    fresh = SessionStore(tmp_path)
    fresh.rename_session(sid, 'latest')
    assert fresh.load_session(sid)[1][-1].text == 'new reply'


def test_rename_rejects_missing_and_ambiguous_ids(tmp_path):
    store = SessionStore(tmp_path)
    for sid in ['same-one', 'same-two']:
        (tmp_path / (sid + '.jsonl')).write_text(json.dumps({'type': 'metadata', 'session_id': sid}) + '\n')
    with pytest.raises(ValueError, match='ambiguous'):
        store.rename_session('same-', 'new')
    with pytest.raises(FileNotFoundError):
        store.rename_session('missing', 'new')


def test_cli_rename_lists_unicode_title(tmp_path, monkeypatch, capsys):
    store = SessionStore(tmp_path)
    sid = saved(store)
    monkeypatch.setenv('OPENAGENT_SESSION_DIR', str(tmp_path))
    assert main(['sessions', 'rename', sid[:8], '\uac80\ud1a0 [red]literal']) == 0
    assert '[red]literal' in capsys.readouterr().out
    assert main(['sessions', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['sessions'][0]['title'] == '\uac80\ud1a0 [red]literal'


def test_failed_rename_preserves_archive(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    sid = saved(store)
    path = tmp_path / (sid + '.jsonl')
    before = path.read_bytes()
    def fail(*args):
        raise OSError('publish failed')
    monkeypatch.setattr('openagent.session.store.os.replace', fail)
    with pytest.raises(OSError, match='publish failed'):
        store.rename_session(sid, 'new')
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('*.tmp'))
