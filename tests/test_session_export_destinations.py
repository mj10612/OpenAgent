from pathlib import Path

import pytest

from openagent.core.types import Message
from openagent.session.store import SessionStore


@pytest.mark.parametrize("format", ["json", "jsonl", "markdown"])
def test_export_rejects_another_live_session_archive(tmp_path: Path, format: str) -> None:
    sessions = tmp_path / "sessions"
    store = SessionStore(sessions)
    source = store.create_session()
    target = store.create_session()
    store.save_messages(source, [Message.user("source conversation")])
    store.save_messages(target, [Message.user("target history")])
    source_before = (sessions / f"{source}.jsonl").read_bytes()
    target_path = sessions / f"{target}.jsonl"
    target_before = target_path.read_bytes()

    with pytest.raises(ValueError, match="live session archive"):
        store.export_session(source, target_path, format=format)

    assert (sessions / f"{source}.jsonl").read_bytes() == source_before
    assert target_path.read_bytes() == target_before
    assert store.load_session(target)[1][0].text == "target history"


def test_export_rejects_symlink_alias_to_live_session_archive(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    store = SessionStore(sessions)
    source = store.create_session()
    target = store.create_session()
    store.save_messages(source, [Message.user("source conversation")])
    store.save_messages(target, [Message.user("target history")])
    target_path = sessions / f"{target}.jsonl"
    target_before = target_path.read_bytes()
    alias = tmp_path / "export.jsonl"
    try:
        alias.symlink_to(target_path)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="live session archive"):
        store.export_session(source, alias)

    assert target_path.read_bytes() == target_before


def test_export_allows_jsonl_outside_session_storage(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    source = store.create_session()
    store.save_messages(source, [Message.user("portable")])

    target = store.export_session(source, tmp_path / "backup.jsonl")

    assert target == tmp_path / "backup.jsonl"
    assert target.is_file()
