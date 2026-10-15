"""Bounded, conflict-checked checkpoints for builtin filesystem mutations."""

from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, cast

from openagent.tools.base import bounded_output
from openagent.tools.fs import MAX_FILE_BYTES, BaseFSTool, _read_bounded


@dataclass(frozen=True)
class FileState:
    data: bytes | None
    mode: int = 0
    digest: bytes | None = None
    size: int = 0
    backup: BinaryIO | None = field(default=None, compare=False, repr=False)

    @property
    def exists(self) -> bool:
        return self.data is not None or self.digest is not None


class ChangeTracker:
    """Keep twenty turns, with 40 MiB memory and 1 GiB disk snapshot caps.

    Completed oldest turns may be evicted to make room for disk snapshots.
    A mutation exceeding the disk cap is refused before execution.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.turns: list[list[dict[str, Any]]] = [[]]
        self.max_bytes = 20 * MAX_FILE_BYTES
        self.max_disk_bytes = 1024 * 1024 * 1024

    def begin_turn(self) -> None:
        if self.turns[-1]:
            self.turns.append([])
        for turn in self.turns[:-20]:
            self._close_turn(turn)
        self.turns = self.turns[-20:]

    def _paths(self, tool: Any, arguments: dict[str, Any]) -> list[Path]:
        if not isinstance(tool, BaseFSTool) or tool.danger != "write":
            return []
        keys = ("source", "destination") if tool.name == "move_file" else ("path",)
        return [tool._resolve(arguments[key]) for key in keys if key in arguments]

    @staticmethod
    def state(path: Path, *, snapshot: bool = True, disk_limit: int = 1024 * 1024 * 1024) -> FileState:
        if path.is_symlink():
            raise PermissionError("Checkpoint refuses symbolic links")
        if not path.exists():
            return FileState(None)
        mode = path.stat().st_mode & 0o7777
        if path.stat().st_size <= MAX_FILE_BYTES:
            return FileState(_read_bounded(path), mode)
        # Large moves/deletes retain undo on disk, never loading a whole file.
        backup = cast(BinaryIO, tempfile.TemporaryFile()) if snapshot else None  # noqa: SIM115 - owned by checkpoint
        digest = hashlib.sha256()
        copied = 0
        try:
            with path.open("rb") as source:
                while chunk := source.read(64 * 1024):
                    copied += len(chunk)
                    if snapshot and copied > disk_limit:
                        raise ValueError("Checkpoint disk storage limit exceeded during capture")
                    digest.update(chunk)
                    if backup is not None:
                        backup.write(chunk)
            if backup is not None:
                backup.seek(0)
            return FileState(None, mode, digest.digest(), copied, backup)
        except BaseException:
            if backup is not None:
                backup.close()
            raise

    def capture(self, tool: Any, arguments: dict[str, Any]) -> dict[Path, FileState]:
        if not self.enabled:
            return {}
        paths = self._paths(tool, arguments)
        disk_needed = sum(path.stat().st_size for path in paths if path.is_file() and path.stat().st_size > MAX_FILE_BYTES)
        if disk_needed > self.max_disk_bytes:
            raise ValueError("Checkpoint disk storage limit exceeded; mutation was not executed")
        while self._retained_disk_bytes() + disk_needed > self.max_disk_bytes and len(self.turns) > 1:
            self._close_turn(self.turns.pop(0))
        if self._retained_disk_bytes() + disk_needed > self.max_disk_bytes:
            raise ValueError("Checkpoint disk storage limit exceeded; use /undo or disable checkpoints")
        states: dict[Path, FileState] = {}
        disk_remaining = self.max_disk_bytes - self._retained_disk_bytes()
        try:
            for path in paths:
                state = self.state(path, disk_limit=disk_remaining)
                states[path] = state
                if state.backup is not None:
                    disk_remaining -= state.size
        except BaseException:
            for state in states.values():
                if state.backup is not None:
                    state.backup.close()
            raise
        # Catch growth during capture before any tool mutation is allowed.
        if self._retained_disk_bytes() + sum(state.size for state in states.values() if state.backup is not None) > self.max_disk_bytes:
            for state in states.values():
                if state.backup is not None:
                    state.backup.close()
            raise ValueError("Checkpoint disk storage limit exceeded; mutation was not executed")
        size = self._retained_bytes()
        before_size = sum(len(state.data or b"") for state in states.values())
        after_size = before_size
        if tool.name == "write_file":
            after_size = len(str(arguments.get("content", "")).encode("utf-8"))
        elif tool.name == "edit_file" and states:
            state = next(iter(states.values()))
            content = (state.data or b"").decode("utf-8")
            old, new = str(arguments.get("old_str", "")), str(arguments.get("new_str", ""))
            if "\r\n" in content and content.count("\r\n") == content.count("\n"):
                old = old.replace("\r\n", "\n").replace("\n", "\r\n")
                new = new.replace("\r\n", "\n").replace("\n", "\r\n")
            after_size = len(content.replace(old, new, 1).encode("utf-8")) if old else before_size
        elif tool.name == "delete_file":
            after_size = 0
        elif tool.name == "move_file":
            after_size = len(next(iter(states.values())).data or b"") if states else 0
        elif tool.name == "apply_patch":
            # The actual result cannot exceed this conservative upper bound.
            after_size = min(MAX_FILE_BYTES, before_size + len(str(arguments.get("patch", "")).encode("utf-8")))
        if size + before_size + after_size > self.max_bytes:
            raise ValueError("Checkpoint storage limit exceeded; use /undo or disable checkpoints")
        return states

    @staticmethod
    def _close_turn(turn: list[dict[str, Any]]) -> None:
        for entry in turn:
            for side in ("before", "after"):
                for state in entry.get(side, {}).values():
                    if state.backup is not None:
                        state.backup.close()

    def _retained_disk_bytes(self) -> int:
        backups = {id(state.backup): state.size for turn in self.turns for entry in turn
                   for side in ("before", "after") for state in entry.get(side, {}).values()
                   if state.backup is not None}
        return sum(backups.values())

    def _retained_bytes(self) -> int:
        return sum(len(state.data or b"") for turn in self.turns for entry in turn
                   for side in ("before", "after") for state in entry.get(side, {}).values())

    def record(self, tool: Any, before: dict[Path, FileState]) -> None:
        if not self.enabled:
            return
        if before:
            try:
                after = {path: self.state(path, snapshot=False) for path in before}
            except BaseException:
                # Keep restoration data even if after-capture or rollback fails.
                entry = {"tool": tool.name, "before": before, "after": {}, "recovery_blocked": True}
                self.turns[-1].append(entry)
                for path, state in before.items():
                    self._restore(path, state)
                self.turns[-1].remove(entry)
                raise
            if before != after:
                entry = {"tool": tool.name, "before": before, "after": after}
                self.turns[-1].append(entry)
                if self._retained_bytes() > self.max_bytes:
                    # Unexpected result sizes are rolled back; retain a retryable
                    # checkpoint if restoration itself encounters an I/O failure.
                    for path, state in before.items():
                        self._restore(path, state)
                        after[path] = state
                    self.turns[-1].remove(entry)
                    raise ValueError("Checkpoint storage limit exceeded; mutation restored")
        elif getattr(tool, "danger", "none") in ("execute", "network"):
            self.turns[-1].append({"tool": tool.name, "irreversible": True})

    @staticmethod
    def _diff(before: dict[Path, FileState], after: dict[Path, FileState]) -> str:
        rows: list[str] = []
        for path, state in before.items():
            updated = after[path]
            if state == updated:
                continue
            if state.digest is not None or updated.digest is not None:
                rows.append(f"{path}: large file changed (disk-backed undo)\n")
                continue
            rows.extend(difflib.unified_diff(
                (state.data or b"").decode("utf-8", "replace").splitlines(keepends=True),
                (updated.data or b"").decode("utf-8", "replace").splitlines(keepends=True),
                fromfile=str(path) if state.data is not None else "/dev/null",
                tofile=str(path) if updated.data is not None else "/dev/null",
            ))
            if state.data == updated.data or state.exists != updated.exists:
                rows.append(f"{path}: existence/mode changed\n")
        return bounded_output("".join(rows))

    def describe(self) -> str:
        rows: list[str] = []
        for index, turn in enumerate(self.turns, 1):
            for entry in turn:
                rows.append(f"Turn {index}: {entry['tool']}\n")
                if entry.get("recovery_blocked"):
                    rows.append("Checkpoint capture and restoration failed; cannot safely undo these file changes.\n")
                else:
                    rows.append("External side effect; cannot undo.\n" if entry.get("irreversible") else self._diff(entry["before"], entry["after"]))
        return bounded_output("".join(rows)) or "No tracked changes."

    async def preview(self, tool: Any, arguments: dict[str, Any]) -> str:
        paths = self._paths(tool, arguments)
        disk_remaining = self.max_disk_bytes - self._retained_disk_bytes()
        disk_needed = sum(path.stat().st_size for path in paths if path.is_file() and path.stat().st_size > MAX_FILE_BYTES)
        if disk_needed > disk_remaining:
            raise ValueError("Checkpoint disk storage limit exceeded; cannot preview")
        before: dict[Path, FileState] = {}
        try:
            for path in paths:
                state = self.state(path, disk_limit=disk_remaining)
                before[path] = state
                if state.backup is not None:
                    disk_remaining -= state.size
        except BaseException:
            self._close_turn([{"before": before}])
            raise
        if not before:
            return "External side effect; cannot preview or undo."
        with tempfile.TemporaryDirectory(prefix="openagent-preview-") as directory:
            root = Path(directory)
            for path, state in before.items():
                target = root / path.relative_to(tool.workspace_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                if state.exists:
                    self._restore(target, state)
            args = dict(arguments)
            for key in ("path", "source", "destination"):
                if key in args:
                    args[key] = str(tool._resolve(args[key]).relative_to(tool.workspace_root))
            result = await type(tool)(root).execute(**args)
            if result.is_error:
                return result.output
            after = {path: self.state(root / path.relative_to(tool.workspace_root), snapshot=False) for path in before}
            return self._diff(before, after) or "No file changes."

    def undo(self) -> str:
        turn = next((turn for turn in reversed(self.turns) if turn), None)
        if turn is None:
            return "No tracked changes to undo."
        if any(entry.get("recovery_blocked") for entry in turn):
            raise ValueError("Undo blocked: checkpoint capture and restoration failed; cannot safely restore files")
        expected: dict[Path, FileState] = {}
        original: dict[Path, FileState] = {}
        for entry in turn:
            for path, state in entry.get("before", {}).items():
                original.setdefault(path, state)
            expected.update(entry.get("after", {}))
        for path, state in expected.items():
            if path.resolve() != path or self.state(path, snapshot=False) != state:
                raise ValueError(f"Undo conflict: '{path}' was changed after the checkpoint")
        for path, state in original.items():
            self._restore(path, state)
            # Record progress in the final expected state for this path. A later
            # I/O failure can be retried while still rejecting external edits.
            for entry in reversed(turn):
                if path in entry.get("after", {}):
                    entry["after"][path] = state
                    break
        irreversible = any(entry.get("irreversible") for entry in turn)
        self._close_turn(turn)
        turn.clear()
        return "Restored builtin file changes." + (" External side effects remain." if irreversible else "")


    @staticmethod
    def _restore(path: Path, state: FileState) -> None:
        if not state.exists:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                if state.backup is not None:
                    state.backup.seek(0)
                    shutil.copyfileobj(state.backup, stream, 64 * 1024)
                else:
                    stream.write(state.data or b"")
            temporary.chmod(state.mode)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
