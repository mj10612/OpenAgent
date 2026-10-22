import contextlib
import json

import pytest

from openagent.session.store import SessionStore


def write_session(directory, session_id, updated_at):
    path = directory / f"{session_id}.jsonl"
    path.write_text(json.dumps({
        "type": "metadata", "session_id": session_id,
        "created_at": 1, "updated_at": updated_at, "title": session_id,
    }) + '\n{"role":"user","content":"kept"}\ninvalid tail\n', encoding="utf-8")
    return path


def files(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def test_preview_selects_age_candidates_without_writes_or_recovery(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    monkeypatch.setattr("openagent.session.store.time.time", lambda: 864000)
    write_session(tmp_path, "old", 1)
    write_session(tmp_path, "boundary", 777600)
    write_session(tmp_path, "recent", 864000)
    write_session(tmp_path, "damaged", "invalid")
    (tmp_path / "old.leftover.tmp").write_bytes(b"keep")
    before = files(tmp_path)

    candidates, skipped = store.preview_cleanup_sessions(older_than_days=1)

    assert set(candidates) == {"old"}
    assert candidates["old"].title == "old"
    assert candidates["old"].message_count == 1
    assert skipped == ["damaged"]
    assert files(tmp_path) == before
    assert store.cleanup_sessions(older_than_days=1) == 1
    assert {p.stem for p in tmp_path.glob("*.jsonl")} == {"boundary", "recent", "damaged"}


def test_preview_clear_all_includes_unknown_metadata_and_agrees_with_cleanup(tmp_path):
    store = SessionStore(tmp_path)
    write_session(tmp_path, "valid", 1)
    write_session(tmp_path, "damaged", "invalid")
    before = files(tmp_path)

    candidates, skipped = store.preview_cleanup_sessions()

    assert set(candidates) == {"valid", "damaged"}
    assert candidates["damaged"] is None
    assert skipped == []
    assert files(tmp_path) == before
    assert store.cleanup_sessions() == 2
    assert list(tmp_path.glob("*.jsonl")) == []


@pytest.mark.parametrize("days", [-1, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("method", ["preview_cleanup_sessions", "cleanup_sessions"])
def test_cleanup_rejects_invalid_age_without_creating_files(tmp_path, method, days):
    store = SessionStore(tmp_path)
    write_session(tmp_path, "valid", 1)
    before = files(tmp_path)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        getattr(store, method)(older_than_days=days)
    assert files(tmp_path) == before


def test_cleanup_rechecks_age_after_acquiring_lock(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    monkeypatch.setattr("openagent.session.store.time.time", lambda: 864000)
    path = write_session(tmp_path, "updated", 1)

    @contextlib.contextmanager
    def concurrent_update(session_id):
        write_session(tmp_path, session_id, 864000)
        yield

    assert set(store.preview_cleanup_sessions(older_than_days=1)[0]) == {"updated"}
    monkeypatch.setattr(store, "_locked", concurrent_update)
    assert store.cleanup_sessions(older_than_days=1) == 0
    assert path.exists()


def test_zero_age_preview_and_execution_select_only_past_sessions(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    monkeypatch.setattr("openagent.session.store.time.time", lambda: 864000)
    write_session(tmp_path, "past", 863999)
    write_session(tmp_path, "current", 864000)
    assert set(store.preview_cleanup_sessions(older_than_days=0)[0]) == {"past"}
    assert store.cleanup_sessions(older_than_days=0) == 1
    assert (tmp_path / "current.jsonl").exists()
