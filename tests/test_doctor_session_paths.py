"""Doctor checks the same session destination used by the session store."""

from pathlib import Path

import pytest

from openagent.config import OpenAgentConfig
from openagent.doctor import diagnose
from openagent.session.store import SessionStore


@pytest.fixture
def session_path(request, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.chdir(tmp_path)
    value, destination = {
        "unset": (None, home / ".openagent" / "sessions"),
        "empty": ("", home / ".openagent" / "sessions"),
        "whitespace": (" \t ", home / ".openagent" / "sessions"),
        "tilde": ("~/nested/sessions", home / "nested" / "sessions"),
        "relative": ("relative/sessions", tmp_path / "relative" / "sessions"),
        "absolute": (str(tmp_path / "absolute" / "sessions"), tmp_path / "absolute" / "sessions"),
    }[request.param]
    if value is None:
        monkeypatch.delenv("OPENAGENT_SESSION_DIR", raising=False)
    else:
        monkeypatch.setenv("OPENAGENT_SESSION_DIR", value)
    return value, destination


PATH_CASES = ["unset", "empty", "whitespace", "tilde", "relative", "absolute"]


def session_check():
    return next(check for check in diagnose(OpenAgentConfig()) if check["name"] == "sessions")


@pytest.mark.parametrize("session_path", PATH_CASES, indirect=True)
def test_doctor_checks_missing_destination_without_creating_it(session_path):
    value, destination = session_path
    assert session_check()["status"] == "ok"
    assert not destination.exists()
    assert not destination.parent.exists()
    store = SessionStore(value)
    assert store._storage_dir == destination
    assert destination.is_dir()


@pytest.mark.parametrize("session_path", PATH_CASES, indirect=True)
@pytest.mark.parametrize("blocked_ancestor", [False, True], ids=["target-file", "ancestor-file"])
def test_doctor_rejects_file_blocking_session_destination(session_path, blocked_ancestor):
    _, destination = session_path
    blocker = destination.parent if blocked_ancestor else destination
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("not a directory", encoding="utf-8")
    assert session_check()["status"] == "error"
    assert blocker.read_text(encoding="utf-8") == "not a directory"
