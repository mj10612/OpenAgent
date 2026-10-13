"""Bounded, conflict-checked checkpoints for builtin filesystem mutations."""

from __future__ import annotations

import difflib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openagent.tools.base import bounded_output
from openagent.tools.fs import MAX_FILE_BYTES, BaseFSTool, _read_bounded


@dataclass(frozen=True)
class FileState:
    data: bytes | None
    mode: int = 0


class ChangeTracker:
    """Keep the last twenty turns without using git or replacing user edits."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.turns: list[list[dict[str, Any]]] = [[]]
        self.max_bytes = 20 * MAX_FILE_BYTES

    def begin_turn(self) -> None:
        if self.turns[-1]:
            self.turns.append([])
        self.turns = self.turns[-20:]

    def _paths(self, tool: Any, arguments: dict[str, Any]) -> list[Path]:
        if not isinstance(tool, BaseFSTool) or tool.danger != "write":
            return []
        keys = ("source", "destination") if tool.name == "move_file" else ("path",)
        return [tool._resolve(arguments[key]) for key in keys if key in arguments]

    @staticmethod
    def state(path: Path) -> FileState:
        if path.is_symlink():
            raise PermissionError("Checkpoint refuses symbolic links")
        if not path.exists():
            return FileState(None)
        return FileState(_read_bounded(path), path.stat().st_mode & 0o7777)

    def capture(self, tool: Any, arguments: dict[str, Any]) -> dict[Path, FileState]:
        if not self.enabled:
            return {}
        states = {path: self.state(path) for path in self._paths(tool, arguments)}
        size = sum(len(state.data or b"") for turn in self.turns for entry in turn for state in entry.get("before", {}).values())
        if size + sum(len(state.data or b"") for state in states.values()) > self.max_bytes:
            raise ValueError("Checkpoint storage limit exceeded; use /undo or disable checkpoints")
        return states

    def record(self, tool: Any, before: dict[Path, FileState]) -> None:
        if not self.enabled:
            return
        if before:
            after = {path: self.state(path) for path in before}
            if before != after:
                self.turns[-1].append({"tool": tool.name, "before": before, "after": after})
        elif getattr(tool, "danger", "none") in ("execute", "network"):
            self.turns[-1].append({"tool": tool.name, "irreversible": True})

    @staticmethod
    def _diff(before: dict[Path, FileState], after: dict[Path, FileState]) -> str:
        rows: list[str] = []
        for path, state in before.items():
            updated = after[path]
            if state == updated:
                continue
            rows.extend(difflib.unified_diff(
                (state.data or b"").decode("utf-8", "replace").splitlines(keepends=True),
                (updated.data or b"").decode("utf-8", "replace").splitlines(keepends=True),
                fromfile=str(path) if state.data is not None else "/dev/null",
                tofile=str(path) if updated.data is not None else "/dev/null",
            ))
            if state.data == updated.data:
                rows.append(f"{path}: existence/mode changed\n")
        return bounded_output("".join(rows))

    def describe(self) -> str:
        rows: list[str] = []
        for index, turn in enumerate(self.turns, 1):
            for entry in turn:
                rows.append(f"Turn {index}: {entry['tool']}\n")
                rows.append("External side effect; cannot undo.\n" if entry.get("irreversible") else self._diff(entry["before"], entry["after"]))
        return bounded_output("".join(rows)) or "No tracked changes."

    async def preview(self, tool: Any, arguments: dict[str, Any]) -> str:
        before = {path: self.state(path) for path in self._paths(tool, arguments)}
        if not before:
            return "External side effect; cannot preview or undo."
        with tempfile.TemporaryDirectory(prefix="openagent-preview-") as directory:
            root = Path(directory)
            for path, state in before.items():
                target = root / path.relative_to(tool.workspace_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                if state.data is not None:
                    target.write_bytes(state.data)
                    target.chmod(state.mode)
            args = dict(arguments)
            for key in ("path", "source", "destination"):
                if key in args:
                    args[key] = str(tool._resolve(args[key]).relative_to(tool.workspace_root))
            result = await type(tool)(root).execute(**args)
            if result.is_error:
                return result.output
            after = {path: self.state(root / path.relative_to(tool.workspace_root)) for path in before}
            return self._diff(before, after) or "No file changes."

    def undo(self) -> str:
        turn = next((turn for turn in reversed(self.turns) if turn), None)
        if turn is None:
            return "No tracked changes to undo."
        expected: dict[Path, FileState] = {}
        original: dict[Path, FileState] = {}
        for entry in turn:
            for path, state in entry.get("before", {}).items():
                original.setdefault(path, state)
            expected.update(entry.get("after", {}))
        for path, state in expected.items():
            if path.resolve() != path or self.state(path) != state:
                raise ValueError(f"Undo conflict: '{path}' was changed after the checkpoint")
        for path, state in original.items():
            if state.data is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(state.data)
                try:
                    temporary.chmod(state.mode)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        irreversible = any(entry.get("irreversible") for entry in turn)
        turn.clear()
        return "Restored builtin file changes." + (" External side effects remain." if irreversible else "")
