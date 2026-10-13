"""Strict unified-diff editing of existing sandboxed files."""

from __future__ import annotations

import re
from typing import Any

from openagent.core.types import ToolParam
from openagent.tools.base import DangerLevel, ToolResult
from openagent.tools.fs import MAX_FILE_BYTES, BaseFSTool, _atomic_write, _read_bounded


class ApplyPatchTool(BaseFSTool):
    """Apply all hunks only after validating the complete patch."""

    name = "apply_patch"
    description = "Apply a unified diff to one existing workspace file. Exact context is required; all hunks publish atomically."
    danger: DangerLevel = "write"

    def __init__(self, workspace_root) -> None:
        super().__init__(workspace_root)
        self.params = [
            ToolParam("path", "string", "Existing workspace file to patch"),
            ToolParam(
                "patch",
                "string",
                "Unified diff with @@ hunk headers; optional ---/+++ file headers",
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        call_id = str(kwargs.get("call_id", ""))
        try:
            target = self._resolve(kwargs["path"])
            original = _read_bounded(target).decode("utf-8")
            patch = str(kwargs["patch"])
            if len(patch.encode("utf-8")) > MAX_FILE_BYTES:
                raise ValueError("Patch exceeds 2 MB size limit")
            lines = original.splitlines(keepends=True)
            diff = patch.splitlines()
            newline = "\r\n" if "\r\n" in original else "\n"
            output: list[str] = []
            cursor = 0
            index = 0
            hunks = 0
            while index < len(diff):
                row = diff[index]
                if not hunks and row.startswith(("--- ", "+++ ")):
                    name = row[4:].split("\t", 1)[0]
                    if name.startswith(("a/", "b/")):
                        name = name[2:]
                    if self._resolve(name) != target:
                        raise ValueError("Patch header does not match the requested file")
                    index += 1
                    continue
                header = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*", row)
                if not header:
                    raise ValueError("Expected a unified diff hunk header")
                old_start, old_count, new_start, new_count = (
                    int(header[1]),
                    int(header[2] or 1),
                    int(header[3]),
                    int(header[4] or 1),
                )
                position = old_start - 1 if old_count else old_start
                if position < cursor or position > len(lines):
                    raise ValueError("Overlapping or out-of-range patch hunk")
                output.extend(lines[cursor:position])
                cursor = position
                if (new_start - 1 if new_count else new_start) != len(output):
                    raise ValueError("Inconsistent new hunk position")
                index += 1
                removed = added = 0
                previous = ""
                while index < len(diff) and not diff[index].startswith("@@ "):
                    row = diff[index]
                    index += 1
                    if row == "\\ No newline at end of file":
                        if previous not in (" ", "+", "-"):
                            raise ValueError("Unexpected no-newline marker")
                        if previous in (" ", "+"):
                            output[-1] = output[-1].rstrip("\r\n")
                        if previous in (" ", "-") and lines[cursor - 1].endswith("\n"):
                            raise ValueError("No-newline marker conflicts with source")
                        previous = ""
                        continue
                    if not row or row[0] not in (" ", "+", "-"):
                        raise ValueError("Invalid hunk line")
                    kind, text = row[0], row[1:]
                    if kind in (" ", "-"):
                        if cursor >= len(lines) or lines[cursor].rstrip("\r\n") != text:
                            raise ValueError("Patch context does not match source")
                        if kind == " ":
                            output.append(lines[cursor])
                        cursor += 1
                        removed += 1
                    if kind == "+":
                        output.append(text + newline)
                    if kind in (" ", "+"):
                        added += 1
                    previous = kind
                if (removed, added) != (old_count, new_count):
                    raise ValueError("Hunk line counts do not match header")
                hunks += 1
            if not hunks:
                raise ValueError("Patch contains no hunks")
            output.extend(lines[cursor:])
            updated = "".join(output)
            if len(updated.encode("utf-8")) > MAX_FILE_BYTES:
                raise ValueError("Patched file exceeds 2 MB size limit")
            _atomic_write(target, updated)
            return ToolResult(call_id, f"Applied {hunks} hunk(s) atomically.")
        except Exception as exc:
            return ToolResult(call_id, str(exc), True)
