import io
from unittest.mock import Mock

import pytest
from rich.console import Console

from openagent import cli
from openagent.session.store import SessionStore


@pytest.mark.parametrize('argv', [
    ['prune', '--older-than', '30', '--dry-run'],
    ['--clear', '--dry-run', '--yes'],
])
def test_cli_cleanup_preview_lists_targets_without_confirmation(argv, tmp_path, monkeypatch):
    SessionStore(tmp_path)
    (tmp_path / 'old.jsonl').write_text('{"type":"metadata","session_id":"old","title":"Old review","updated_at":1,"created_at":1}\n')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setenv('OPENAGENT_SESSION_DIR', str(tmp_path))
    confirm = Mock(side_effect=AssertionError('dry-run must not confirm'))
    monkeypatch.setattr(cli.Confirm, 'ask', confirm)
    args = cli._build_parser().parse_args(['sessions', *argv])
    output = io.StringIO()
    assert cli._cmd_sessions(Console(file=output, width=120), args) == 0
    text = output.getvalue()
    assert 'old' in text and 'Old review' in text and '1 session' in text
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    confirm.assert_not_called()


def test_dry_run_rejected_outside_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAGENT_SESSION_DIR', str(tmp_path))
    args = cli._build_parser().parse_args(['sessions', '--dry-run'])
    with pytest.raises(ValueError, match='dry-run'):
        cli._cmd_sessions(Console(file=io.StringIO()), args)


def test_dry_run_does_not_create_missing_storage(tmp_path, monkeypatch):
    directory = tmp_path / 'missing' / 'sessions'
    monkeypatch.setenv('OPENAGENT_SESSION_DIR', str(directory))
    args = cli._build_parser().parse_args(['sessions', '--clear', '--dry-run'])
    assert cli._cmd_sessions(Console(file=io.StringIO()), args) == 0
    assert not directory.parent.exists()
