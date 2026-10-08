"""Sandboxed filesystem tools for OpenAgent.

Strictly confines file operations to a configured workspace root directory.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import tempfile
from itertools import islice
from pathlib import Path
from typing import Any

from openagent.core.types import ToolParam
from openagent.tools.base import DangerLevel, Tool, ToolResult

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_SCAN_ITEMS = 10000
MAX_OUTPUT_CHARS = 100000
DEFAULT_IGNORES = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".tox"}


def _read_bounded(target: Path) -> bytes:
    with target.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("File exceeds 2 MB size limit")
    return data


def _walk(root: Path, workspace: Path):
    """Bound traversal and prune dependency directories and symlinks before descent."""
    count = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in DEFAULT_IGNORES and not (Path(directory) / d).is_symlink())
        for name in sorted(dirs + files):
            item = Path(directory) / name
            if item.is_symlink() or not item.resolve().is_relative_to(workspace):
                continue
            yield item
            count += 1
            if count >= MAX_SCAN_ITEMS:
                return


def _atomic_write(target: Path, content: str, overwrite: bool = True) -> None:
    """Write UTF-8 bytes without newline translation; publish only complete content."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            shutil.copymode(target, temporary)
        if overwrite:
            os.replace(temporary, target)
        else:
            # Hard-link publication preserves exclusive-create semantics under races.
            os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resolve_sandboxed_path(workspace_root: Path, path: str | Path) -> Path:
    """Resolve a path relative to workspace_root and ensure it does not escape.

    Raises PermissionError if path resolves outside workspace_root.
    """
    ws_root = workspace_root.resolve()
    target = Path(path)
    if not target.is_absolute():
        target = ws_root / target
    target = target.resolve()

    if not target.is_relative_to(ws_root):
        raise PermissionError(
            f"Access denied: path '{path}' resolves outside workspace root '{ws_root}'"
        )
    return target


class BaseFSTool(Tool):
    """Base class for filesystem tools with sandbox root."""

    def __init__(self, workspace_root: Path | str) -> None:
        super().__init__()
        self.workspace_root = Path(workspace_root).resolve()

    def _resolve(self, path: str | Path) -> Path:
        return resolve_sandboxed_path(self.workspace_root, path)


class ReadFileTool(BaseFSTool):
    """Tool to read files with 1-indexed line viewing."""

    name = "read_file"
    description = "Read lines from a file within the workspace with 1-indexed line viewing."
    danger: DangerLevel = "none"

    def __init__(self, workspace_root: Path | str) -> None:
        super().__init__(workspace_root)
        self.params = [
            ToolParam(name="path", type="string", description="Path to file"),
            ToolParam(
                name="offset",
                type="integer",
                description="1-indexed starting line number",
                required=False,
                default=1,
            ),
            ToolParam(
                name="limit",
                type="integer",
                description="Maximum number of lines to return",
                required=False,
                default=2000,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        call_id = str(kwargs.get("call_id", ""))
        path = str(kwargs.get("path", ""))
        offset = int(kwargs.get("offset", 1))
        limit = max(1, min(int(kwargs.get("limit", 2000)), 2000))

        if not path:
            return ToolResult(call_id=call_id, output="Parameter 'path' is required.", is_error=True)

        try:
            target = self._resolve(path)
        except PermissionError as exc:
            return ToolResult(call_id=call_id, output=str(exc), is_error=True)

        if not target.exists():
            return ToolResult(
                call_id=call_id,
                output=f"File not found: '{path}'",
                is_error=True,
            )
        if not target.is_file():
            return ToolResult(
                call_id=call_id,
                output=f"Path is not a regular file: '{path}'",
                is_error=True,
            )

        try:
            if target.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("File exceeds 2 MB size limit")
            data = _read_bounded(target)
            if b"\x00" in data[:8192]:
                raise ValueError("Binary files cannot be read as text")
            content = data.decode("utf-8", errors="replace")
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                output=f"Error reading file '{path}': {exc}",
                is_error=True,
            )

        lines = content.splitlines()
        start_idx = max(1, offset)
        end_idx = start_idx - 1 + max(1, limit)
        selected_lines = lines[start_idx - 1 : end_idx]

        formatted = [f"{i}: {line}" for i, line in enumerate(selected_lines, start=start_idx)]
        return ToolResult(call_id=call_id, output="\n".join(formatted)[:MAX_OUTPUT_CHARS])


class WriteFileTool(BaseFSTool):
    """Tool to write files within workspace, auto-creating parent directories."""

    name = "write_file"
    description = "Write text content to a file within the workspace."
    danger: DangerLevel = "write"

    def __init__(self, workspace_root: Path | str) -> None:
        super().__init__(workspace_root)
        self.params = [
            ToolParam(name="path", type="string", description="Path to file"),
            ToolParam(name="content", type="string", description="Text content to write"),
            ToolParam(
                name="overwrite",
                type="boolean",
                description="Whether to overwrite existing file",
                required=False,
                default=True,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        call_id = str(kwargs.get("call_id", ""))
        path = str(kwargs.get("path", ""))
        content = str(kwargs.get("content", ""))
        overwrite = bool(kwargs.get("overwrite", True))

        if not path:
            return ToolResult(call_id=call_id, output="Parameter 'path' is required.", is_error=True)

        try:
            target = self._resolve(path)
        except PermissionError as exc:
            return ToolResult(call_id=call_id, output=str(exc), is_error=True)

        if target.exists() and not overwrite:
            return ToolResult(
                call_id=call_id,
                output=f"File already exists: '{path}'. Set overwrite=True to overwrite.",
                is_error=True,
            )
        if target.is_dir():
            return ToolResult(
                call_id=call_id,
                output=f"Target path '{path}' is a directory.",
                is_error=True,
            )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(target, content, overwrite=overwrite)
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                output=f"Error writing file '{path}': {exc}",
                is_error=True,
            )

        return ToolResult(
            call_id=call_id,
            output=f"Successfully wrote {len(content)} characters to {path}",
        )


class EditFileTool(BaseFSTool):
    """Tool for precise string replacement with uniqueness validation."""

    name = "edit_file"
    description = (
        "Replace an exact string in a file with new text. The target string must be unique."
    )
    danger: DangerLevel = "write"

    def __init__(self, workspace_root: Path | str) -> None:
        super().__init__(workspace_root)
        self.params = [
            ToolParam(name="path", type="string", description="Path to file"),
            ToolParam(
                name="old_str",
                type="string",
                description="Exact text to replace (must appear exactly once)",
            ),
            ToolParam(name="new_str", type="string", description="Replacement text"),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        call_id = str(kwargs.get("call_id", ""))
        path = str(kwargs.get("path", ""))
        old_str = str(kwargs.get("old_str", ""))
        new_str = str(kwargs.get("new_str", ""))

        if not path:
            return ToolResult(call_id=call_id, output="Parameter 'path' is required.", is_error=True)
        if not old_str:
            return ToolResult(call_id=call_id, output="Parameter 'old_str' cannot be empty.", is_error=True)

        try:
            target = self._resolve(path)
        except PermissionError as exc:
            return ToolResult(call_id=call_id, output=str(exc), is_error=True)

        if not target.exists():
            return ToolResult(
                call_id=call_id,
                output=f"File not found: '{path}'",
                is_error=True,
            )
        if not target.is_file():
            return ToolResult(
                call_id=call_id,
                output=f"Path is not a regular file: '{path}'",
                is_error=True,
            )

        try:
            content = _read_bounded(target).decode("utf-8")
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                output=f"Error reading file '{path}': {exc}",
                is_error=True,
            )

        crlf_only = "\r\n" in content and content.count("\r\n") == content.count("\n")
        if crlf_only:
            old_str = old_str.replace("\r\n", "\n").replace("\n", "\r\n")
            new_str = new_str.replace("\r\n", "\n").replace("\n", "\r\n")
        occurrences = content.count(old_str)
        if occurrences == 0:
            return ToolResult(
                call_id=call_id,
                output=f"Target string not found in '{path}'.",
                is_error=True,
            )
        if occurrences > 1:
            return ToolResult(
                call_id=call_id,
                output=f"Target string found {occurrences} times in '{path}'. Must be unique.",
                is_error=True,
            )

        new_content = content.replace(old_str, new_str, 1)
        try:
            _atomic_write(target, new_content)
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                output=f"Error writing edited file '{path}': {exc}",
                is_error=True,
            )

        return ToolResult(call_id=call_id, output=f"Successfully edited '{path}'.")


class ListDirectoryTool(BaseFSTool):
    """Tool to list directory contents."""

    name = "list_directory"
    description = "List files and directories in a given path within the workspace."
    danger: DangerLevel = "none"

    def __init__(self, workspace_root: Path | str) -> None:
        super().__init__(workspace_root)
        self.params = [
            ToolParam(
                name="path",
                type="string",
                description="Path to directory",
                required=False,
                default=".",
            ),
            ToolParam(
                name="recursive",
                type="boolean",
                description="Whether to list recursively",
                required=False,
                default=False,
            ),
            ToolParam(
                name="max_items",
                type="integer",
                description="Maximum number of items to return",
                required=False,
                default=200,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        call_id = str(kwargs.get("call_id", ""))
        path = str(kwargs.get("path", "."))
        recursive = bool(kwargs.get("recursive", False))
        max_items = max(1, min(int(kwargs.get("max_items", 200)), 1000))

        try:
            target = self._resolve(path)
        except PermissionError as exc:
            return ToolResult(call_id=call_id, output=str(exc), is_error=True)

        if not target.exists() or not target.is_dir():
            return ToolResult(
                call_id=call_id,
                output=f"Directory not found: '{path}'",
                is_error=True,
            )

        try:
            source = _walk(target, self.workspace_root) if recursive else (p for p in target.iterdir() if not p.is_symlink())
            entries = list(islice(source, max_items + 1))
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                output=f"Error reading directory '{path}': {exc}",
                is_error=True,
            )

        total = len(entries)
        items_slice = entries[:max_items]
        lines = []
        for entry in items_slice:
            try:
                rel = entry.relative_to(target)
                if entry.is_dir():
                    lines.append(f"[DIR]  {rel}/")
                else:
                    lines.append(f"[FILE] {rel} ({entry.stat().st_size} bytes)")
            except Exception:
                continue

        if total > max_items:
            lines.append(f"... (truncated, showing {max_items} items)")

        output = "\n".join(lines) if lines else "(empty directory)"
        return ToolResult(call_id=call_id, output=output)


class GlobFindTool(BaseFSTool):
    """Tool to find files matching a glob pattern within the workspace."""

    name = "glob_find"
    description = "Find files matching a glob pattern within the workspace."
    danger: DangerLevel = "none"

    def __init__(self, workspace_root: Path | str) -> None:
        super().__init__(workspace_root)
        self.params = [
            ToolParam(
                name="pattern",
                type="string",
                description="Glob pattern (e.g. *.py, **/*.md)",
            ),
            ToolParam(
                name="path",
                type="string",
                description="Base path to search from",
                required=False,
                default=".",
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        call_id = str(kwargs.get("call_id", ""))
        pattern = str(kwargs.get("pattern", ""))
        path = str(kwargs.get("path", "."))

        try:
            target = self._resolve(path)
        except PermissionError as exc:
            return ToolResult(call_id=call_id, output=str(exc), is_error=True)

        if not target.exists() or not target.is_dir():
            return ToolResult(
                call_id=call_id,
                output=f"Directory not found: '{path}'",
                is_error=True,
            )

        try:
            if not pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                raise ValueError("Pattern must stay within the search directory")
            matches = [p for p in _walk(target, self.workspace_root) if p.match(pattern) or fnmatch.fnmatch(str(p.relative_to(target)), pattern)]
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                output=f"Invalid glob pattern or error: {exc}",
                is_error=True,
            )

        rel_paths = []
        for m in matches:
            try:
                if m.resolve().is_relative_to(self.workspace_root):
                    rel_paths.append(str(m.relative_to(target)))
            except Exception:
                continue

        if not rel_paths:
            return ToolResult(call_id=call_id, output="No files matched pattern.")
        return ToolResult(call_id=call_id, output="\n".join(rel_paths[:1000])[:MAX_OUTPUT_CHARS])


class GrepSearchTool(BaseFSTool):
    """Tool to search file content matching a query string."""

    name = "grep_search"
    description = "Search for a string pattern across files within the workspace."
    danger: DangerLevel = "none"

    def __init__(self, workspace_root: Path | str) -> None:
        super().__init__(workspace_root)
        self.params = [
            ToolParam(
                name="query",
                type="string",
                description="Text or pattern to search for",
            ),
            ToolParam(
                name="path",
                type="string",
                description="Base path to search from",
                required=False,
                default=".",
            ),
            ToolParam(
                name="case_sensitive",
                type="boolean",
                description="Whether the search is case-sensitive",
                required=False,
                default=True,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        call_id = str(kwargs.get("call_id", ""))
        query = str(kwargs.get("query", ""))
        path = str(kwargs.get("path", "."))
        case_sensitive = bool(kwargs.get("case_sensitive", True))

        try:
            target = self._resolve(path)
        except PermissionError as exc:
            return ToolResult(call_id=call_id, output=str(exc), is_error=True)

        if not target.exists():
            return ToolResult(
                call_id=call_id,
                output=f"Directory not found: '{path}'",
                is_error=True,
            )

        matches: list[str] = []
        target_files = _walk(target, self.workspace_root) if target.is_dir() else [target]

        query_cmp = query if case_sensitive else query.lower()

        for f in target_files:
            if not f.is_file():
                continue
            try:
                if not f.resolve().is_relative_to(self.workspace_root):
                    continue
            except Exception:
                continue
            try:
                if f.stat().st_size > MAX_FILE_BYTES:
                    continue
                data = _read_bounded(f)
                if b"\x00" in data[:8192]:
                    continue
                content = data.decode("utf-8")
            except Exception:
                continue

            for idx, line in enumerate(content.splitlines(), start=1):
                cmp_line = line if case_sensitive else line.lower()
                if query_cmp in cmp_line:
                    try:
                        rel = f.relative_to(self.workspace_root)
                    except ValueError:
                        rel = f
                    matches.append(f"{rel}:{idx}: {line[:2000]}")
                    if len(matches) >= 500:
                        matches.append("... (truncated at 500 matches)")
                        break
            if len(matches) >= 500:
                break

        if not matches:
            return ToolResult(call_id=call_id, output="No matches found for query.")
        return ToolResult(call_id=call_id, output="\n".join(matches)[:MAX_OUTPUT_CHARS])


class DeleteFileTool(BaseFSTool):
    """Delete a regular file within the workspace; never remove directories."""
    name = "delete_file"
    description = "Delete a file within the workspace."
    danger: DangerLevel = "write"

    def __init__(self, workspace_root: Path | str) -> None:
        super().__init__(workspace_root)
        self.params = [ToolParam(name="path", type="string", description="File to delete")]

    async def execute(self, **kwargs: Any) -> ToolResult:
        call_id = str(kwargs.get("call_id", ""))
        try:
            target = self._resolve(kwargs.get("path", ""))
            if not target.is_file():
                raise ValueError("Path is not a regular file")
            target.unlink()
            return ToolResult(call_id, "Successfully deleted file.")
        except Exception as exc:
            return ToolResult(call_id, str(exc), True)


class MoveFileTool(BaseFSTool):
    """Move a regular file, refusing to overwrite any existing destination."""
    name = "move_file"
    description = "Move or rename a file within the workspace without overwriting."
    danger: DangerLevel = "write"

    def __init__(self, workspace_root: Path | str) -> None:
        super().__init__(workspace_root)
        self.params = [ToolParam(name="source", type="string", description="Source file"), ToolParam(name="destination", type="string", description="Destination file")]

    async def execute(self, **kwargs: Any) -> ToolResult:
        call_id = str(kwargs.get("call_id", ""))
        try:
            source = self._resolve(kwargs.get("source", ""))
            destination = self._resolve(kwargs.get("destination", ""))
            if not source.is_file():
                raise ValueError("Source is not a regular file")
            if destination.exists():
                raise FileExistsError("Destination already exists")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            source.unlink()
            return ToolResult(call_id, "Successfully moved file.")
        except Exception as exc:
            return ToolResult(call_id, str(exc), True)


def create_fs_tools(workspace_root: Path | str) -> list[Tool]:
    """Create all standard sandboxed filesystem tools bound to the workspace root."""
    return [
        ReadFileTool(workspace_root),
        WriteFileTool(workspace_root),
        EditFileTool(workspace_root),
        ListDirectoryTool(workspace_root),
        GlobFindTool(workspace_root),
        GrepSearchTool(workspace_root),
        DeleteFileTool(workspace_root),
        MoveFileTool(workspace_root),
    ]
