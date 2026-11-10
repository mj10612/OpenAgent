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


@pytest.mark.parametrize('incremental', [False, True])
def test_rename_preserves_history_state_and_retention_timestamp(tmp_path, incremental):
    store = SessionStore(tmp_path, incremental=incremental)
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


@pytest.mark.parametrize('incremental', [False, True])
@pytest.mark.parametrize('preloaded', [False, True])
@pytest.mark.parametrize('damaged_row', [
    b'{"role":"user","content":"RECOVERY_TARGET"\n',
    b'{"role":"tool","tool_call_id":"missing","content":"orphan"}\n',
    b'{"role":"assistant","content":[],"tool_calls":[{"id":"pending","name":"lookup","arguments":{}}]}\n',
    b'{"type":"batch","metadata":{},"messages":[{"role":"invalid"}]}\n',
    b'{"role":"user","content":"\xff"}\n',
], ids=['corrupt-json', 'orphan-result', 'incomplete-call', 'corrupt-batch', 'invalid-utf8'])
def test_rename_refuses_damaged_archive_without_discarding_bytes(
    tmp_path, incremental, preloaded, damaged_row,
):
    store = SessionStore(tmp_path, incremental=incremental)
    sid = saved(store)
    path = tmp_path / (sid + '.jsonl')
    with path.open('ab') as output:
        output.write(damaged_row)
    before = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns
    fresh = SessionStore(tmp_path, incremental=incremental)
    if preloaded:
        # Reading for recovery is still allowed, but must not make rename destructive.
        assert fresh.load_session(sid)[1][0].text == 'kept conversation'
        assert path.read_bytes() == before

    with pytest.raises(ValueError, match=r'(?i)(damaged|corrupt|utf-8)'):
        fresh.rename_session(sid, 'Only change title')

    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == before_mtime
    assert not list(tmp_path.glob('*.tmp'))
    metadata, recovered = SessionStore(tmp_path).load_session(sid)
    assert metadata.title == 'before'
    assert recovered[0].text == 'kept conversation'


def test_cli_damaged_session_rename_fails_without_rewriting(tmp_path, monkeypatch, capsys):
    store = SessionStore(tmp_path)
    sid = saved(store)
    path = tmp_path / (sid + '.jsonl')
    with path.open('ab') as output:
        output.write(b'{"role":"user","content":"RECOVERY_TARGET"\n')
    before = path.read_bytes()
    monkeypatch.setenv('OPENAGENT_SESSION_DIR', str(tmp_path))

    assert main(['sessions', 'rename', sid, 'new']) != 0
    assert 'damaged' in capsys.readouterr().out.lower()
    assert path.read_bytes() == before


@pytest.mark.parametrize('metadata_source', ['snapshot', 'batch'])
def test_rename_preserves_invalid_checklist_metadata(tmp_path, metadata_source):
    store = SessionStore(tmp_path)
    sid = saved(store)
    path = tmp_path / (sid + '.jsonl')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    metadata = rows[0] if metadata_source == 'snapshot' else rows[-1]['metadata']
    metadata['todo_state'] = ['RECOVERY_TARGET']
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
    before = path.read_bytes()

    with pytest.raises(ValueError, match='damaged'):
        SessionStore(tmp_path).rename_session(sid, 'new')

    assert path.read_bytes() == before
    # Loading still recovers the usable history without writing a repaired archive.
    assert SessionStore(tmp_path).load_session(sid)[1][0].text == 'kept conversation'
    assert path.read_bytes() == before


@pytest.mark.parametrize('incremental', [False, True])
@pytest.mark.parametrize('preloaded', [False, True])
@pytest.mark.parametrize('duplicate_source', ['snapshot', 'batch', 'message', 'content-part'])
def test_rename_refuses_duplicate_json_keys_without_discarding_values(
    tmp_path, incremental, preloaded, duplicate_source,
):
    store = SessionStore(tmp_path, incremental=incremental)
    sid = saved(store)
    path = tmp_path / (sid + '.jsonl')
    if duplicate_source == 'snapshot':
        path.write_bytes(path.read_bytes().replace(
            b'"title": "before"', b'"title":"RECOVERY_TARGET","title":"before"', 1,
        ))
    else:
        rows = {
            'batch': b'{"type":"batch","metadata":{"title":"RECOVERY_TARGET","title":"before"},"messages":[]}\n',
            'message': br'{"role":"user","content":"RECOVERY_TARGET","con\u0074ent":"last value"}' + b'\n',
            'content-part': b'{"role":"user","content":[{"type":"text","text":"RECOVERY_TARGET","text":"last value"}]}\n',
        }
        with path.open('ab') as output:
            output.write(rows[duplicate_source])
    before = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns
    assert b'RECOVERY_TARGET' in before
    fresh = SessionStore(tmp_path)
    if preloaded:
        # Recovery reads may still use the last value, but must preserve the archive.
        assert fresh.load_session(sid)[1][0].text == 'kept conversation'
        assert path.read_bytes() == before

    with pytest.raises(ValueError, match='damaged'):
        fresh.rename_session(sid, 'new')

    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == before_mtime
    assert not list(tmp_path.glob('*.tmp'))
