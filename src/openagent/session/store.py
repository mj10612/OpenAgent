"""JSONL-based session persistence and metadata storage.

Provides atomic file persistence, history serialization with full fidelity for
multi-part content and tool calls, session listing, and session resumption.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import math
import os
import re
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openagent.context.messages import MessageManager
from openagent.core.types import Message

logger = logging.getLogger(__name__)


class SessionConflictError(RuntimeError):
    """The saved session changed after this writer loaded it."""


@dataclass
class SessionMetadata:
    """Metadata describing a saved conversation session."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    model: str = ""
    title: str = ""
    message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a serializable dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": self.model,
            "title": self.title,
            "message_count": self.message_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SessionMetadata:
        """Construct SessionMetadata from a dictionary."""
        created_at = float(data.get("created_at", time.time()))
        updated_at = float(data.get("updated_at", time.time()))
        if not math.isfinite(created_at) or not math.isfinite(updated_at):
            raise ValueError("Session timestamps must be finite")
        return cls(
            session_id=str(data.get("session_id", "")),
            created_at=created_at,
            updated_at=updated_at,
            model=str(data.get("model", "")),
            title=str(data.get("title", "")),
            message_count=int(data.get("message_count", 0)),
        )


class SessionStore:
    """Manages persistence of conversation histories in JSONL format with atomic writes."""

    def __init__(self, storage_dir: str | Path | None = None, *, incremental: bool = True) -> None:
        self.incremental = incremental
        if storage_dir is None or (isinstance(storage_dir, str) and not storage_dir.strip()):
            self._storage_dir = Path.home() / ".openagent" / "sessions"
        else:
            self._storage_dir = Path(storage_dir).expanduser().resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[SessionMetadata, list[str], tuple[int, int]]] = {}
        self._revisions: dict[str, tuple[int, int] | None] = {}

    @contextlib.contextmanager
    def _locked(self, session_id: str):
        self._get_session_path(session_id)
        lock_path = self._storage_dir / f"{session_id}.lock"
        with lock_path.open("a+b") as handle:
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt
                deadline = time.monotonic() + 30
                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"Timed out acquiring session lock for '{session_id}'") from None
                        time.sleep(0.01)
            else:
                fcntl = importlib.import_module("fcntl")
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                handle.seek(0)
                if sys.platform == "win32":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _revision(self, session_id: str) -> tuple[int, int] | None:
        path = self._get_session_path(session_id)
        if not path.exists():
            return None
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns

    def _remember(self, session_id: str, metadata: SessionMetadata, messages: Sequence[Message]) -> None:
        stat = self._get_session_path(session_id).stat()
        self._cache.pop(session_id, None)
        self._cache[session_id] = (
            SessionMetadata.from_dict(metadata.to_dict()),
            [json.dumps(self._serialize_message(message), ensure_ascii=False) for message in messages],
            (stat.st_size, stat.st_mtime_ns),
        )
        self._revisions[session_id] = (stat.st_size, stat.st_mtime_ns)
        # Cached histories are an optimization, bounded independently of archive size.
        while len(self._cache) > 8 or sum(sum(len(row.encode("utf-8")) for row in entry[1]) for entry in self._cache.values()) > 4 * 1024 * 1024:
            self._cache.pop(next(iter(self._cache)))

    def _cached(self, session_id: str) -> tuple[SessionMetadata, list[str], tuple[int, int]] | None:
        cached = self._cache.get(session_id)
        target = self._get_session_path(session_id)
        if cached and target.exists():
            stat = target.stat()
            if cached[2] == (stat.st_size, stat.st_mtime_ns):
                self._cache.pop(session_id)
                self._cache[session_id] = cached
                return cached
        self._cache.pop(session_id, None)
        return None

    @property
    def storage_dir(self) -> Path:
        """Return the base storage directory."""
        return self._storage_dir

    def create_session(self, model: str = "", title: str | None = None) -> str:
        """Create a new session record and return its unique identifier.

        Args:
            model: The primary model identifier associated with this session.
            title: Human-readable title for the session.

        Returns:
            The generated session_id string.
        """
        session_id = uuid.uuid4().hex
        now = time.time()
        effective_title = title if title is not None else f"Session {session_id[:8]}"
        meta = SessionMetadata(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            model=model,
            title=effective_title,
            message_count=0,
        )
        self._atomic_write_session(session_id, meta, [])
        return session_id

    def save_messages(
        self,
        session_id: str,
        messages: Sequence[Message],
        metadata: SessionMetadata | None = None,
    ) -> None:
        """Atomically persist a sequence of messages for a session.

        Args:
            session_id: The session identifier.
            messages: List of Message objects to persist.
            metadata: Optional updated SessionMetadata. If omitted, existing metadata
                      is updated with new message count and updated timestamp.
        """
        with self._locked(session_id):
            revision = self._revision(session_id)
            if session_id in self._revisions and self._revisions[session_id] != revision:
                raise SessionConflictError(f"Session conflict for '{session_id}': reload or fork before saving")
            self._save_messages(session_id, messages, metadata)

    def _save_messages(self, session_id: str, messages: Sequence[Message], metadata: SessionMetadata | None = None) -> None:
        now = time.time()
        cached = self._cached(session_id)
        if metadata is None:
            target = self._get_session_path(session_id)
            if target.exists():
                try:
                    if cached:
                        existing_meta = cached[0]
                    else:
                        existing_meta, _ = self._load_session(session_id)
                        cached = self._cached(session_id)
                    metadata = SessionMetadata(
                        session_id=session_id,
                        created_at=existing_meta.created_at,
                        updated_at=now,
                        model=existing_meta.model,
                        title=existing_meta.title,
                        message_count=len(messages),
                    )
                except Exception:
                    metadata = SessionMetadata(
                        session_id=session_id,
                        created_at=now,
                        updated_at=now,
                        model="",
                        title=f"Session {session_id[:8]}",
                        message_count=len(messages),
                    )
            else:
                metadata = SessionMetadata(
                    session_id=session_id,
                    created_at=now,
                    updated_at=now,
                    model="",
                    title=f"Session {session_id[:8]}",
                    message_count=len(messages),
                )
        else:
            metadata.message_count = len(messages)
            metadata.updated_at = now

        rows = [json.dumps(self._serialize_message(message), ensure_ascii=False) for message in messages]
        if self.incremental and cached and len(rows) >= len(cached[1]) and rows[:len(cached[1])] == cached[1]:
            new_rows = rows[len(cached[1]):]
            if new_rows or metadata.to_dict() != cached[0].to_dict():
                batch = json.dumps({"type": "batch", "metadata": metadata.to_dict(), "messages": [json.loads(row) for row in new_rows]}, ensure_ascii=False) + "\n"
                target = self._get_session_path(session_id)
                original_size = target.stat().st_size
                try:
                    with target.open("ab") as output:
                        output.write(batch.encode("utf-8"))
                        output.flush()
                        os.fsync(output.fileno())
                except BaseException:
                    self._cache.pop(session_id, None)
                    with target.open("r+b") as output:
                        output.truncate(original_size)
                    raise
            self._remember(session_id, metadata, messages)
        else:
            self._atomic_write_session(session_id, metadata, messages)

    def append_message(self, session_id: str, message: Message) -> None:
        """Append a single message to an existing session history atomically.

        Args:
            session_id: The target session identifier.
            message: The Message to append.

        Raises:
            FileNotFoundError: If the session does not exist.
        """
        cached = self._cached(session_id)
        if cached:
            meta = SessionMetadata.from_dict(cached[0].to_dict())
            messages = [self._deserialize_message(json.loads(row)) for row in cached[1]]
        else:
            meta, messages = self.load_session(session_id)
        messages.append(message)
        self.save_messages(session_id, messages, metadata=meta)

    def load_session(self, session_id: str) -> tuple[SessionMetadata, list[Message]]:
        """Load session metadata and message history from disk.

        Args:
            session_id: The session identifier to load.

        Returns:
            Tuple of (SessionMetadata, list of Message).

        Raises:
            FileNotFoundError: If the session file does not exist.
        """
        with self._locked(session_id):
            return self._load_session(session_id)

    def _load_session(self, session_id: str) -> tuple[SessionMetadata, list[Message]]:
        session_path = self._get_session_path(session_id)
        if not session_path.is_file():
            raise FileNotFoundError(f"Session '{session_id}' not found at {session_path}")

        lines: list[str] = []
        with open(session_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)

        if not lines:
            now = time.time()
            meta = SessionMetadata(
                session_id=session_id,
                created_at=now,
                updated_at=now,
                model="",
                title=f"Session {session_id[:8]}",
                message_count=0,
            )
            return meta, []

        stat = session_path.stat()
        self._revisions[session_id] = (stat.st_size, stat.st_mtime_ns)
        meta = SessionMetadata(
            session_id=session_id,
            created_at=stat.st_ctime,
            updated_at=stat.st_mtime,
            title=f"Session {session_id[:8]}",
        )
        messages: list[Message] = []
        corrupt = False
        for lineno, line in enumerate(lines, start=1):
            try:
                raw_obj = json.loads(line)
                if not isinstance(raw_obj, Mapping):
                    raise ValueError("session row must be an object")
                if raw_obj.get("type") == "batch":
                    batch_messages = [self._deserialize_message(row) for row in raw_obj["messages"]]
                    batch_meta = SessionMetadata.from_dict(raw_obj["metadata"])
                    batch_meta.session_id = session_id
                    messages.extend(batch_messages)
                    meta = batch_meta
                elif raw_obj.get("type") == "metadata" or "session_id" in raw_obj:
                    if lineno != 1:
                        raise ValueError("unexpected metadata row")
                    meta = SessionMetadata.from_dict(raw_obj)
                    meta.session_id = session_id
                else:
                    message = self._deserialize_message(raw_obj)
                    if message.role not in {"system", "user", "assistant", "tool"}:
                        raise ValueError("invalid message role")
                    messages.append(message)
            except (TypeError, ValueError, AttributeError, KeyError) as exc:
                corrupt = True
                logger.warning("Session %s: skipped corrupt line %d: %s", session_id, lineno, exc)

        valid_messages = []
        for group in MessageManager.get_atomic_groups(messages):
            if MessageManager.valid_tool_group(group):
                valid_messages.extend(group)
            else:
                corrupt = True
                logger.warning(
                    "Session %s: dropped incomplete or invalid tool correspondence (%d messages)",
                    session_id,
                    len(group),
                )
        messages = valid_messages

        meta.message_count = len(messages)
        if not corrupt:
            self._remember(session_id, meta, messages)
        else:
            self._cache.pop(session_id, None)
        return meta, messages

    def list_sessions(self) -> list[SessionMetadata]:
        """List all stored sessions sorted by updated timestamp descending.

        Returns:
            List of SessionMetadata records.
        """
        sessions: list[SessionMetadata] = []
        for file_path in self._storage_dir.glob("*.jsonl"):
            try:
                metadata = self.load_metadata(file_path.stem)
                sessions.append(metadata)
            except Exception as exc:
                logger.warning("Failed to read session header from %s: %s", file_path, exc)
                continue

        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def load_metadata(self, session_id: str) -> SessionMetadata:
        """Scan metadata and correspondence without retaining history or filling cache."""
        with self._locked(session_id):
            return self._load_metadata(session_id)

    def _load_metadata(self, session_id: str) -> SessionMetadata:
        path = self._get_session_path(session_id)
        stat = path.stat()
        meta = SessionMetadata(session_id, created_at=stat.st_ctime, updated_at=stat.st_mtime)
        count = 0
        pending: list[Message] = []
        invalid_metadata = False

        def lightweight(row: Mapping[str, Any]) -> Message:
            message = self._deserialize_message(row)
            message.content = []
            message.metadata = {}
            message.reasoning = None
            for call in message.tool_calls:
                call.arguments = {}
                call.raw_arguments = ""
            return message

        def accept(message: Message) -> None:
            nonlocal count, pending
            if message.role != "tool" and pending:
                if MessageManager.valid_tool_group(pending):
                    count += len(pending)
                pending = []
            if message.tool_calls or message.role == "tool":
                pending.append(message)
            else:
                count += 1

        with path.open(encoding="utf-8", errors="replace") as source:
            for lineno, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = None
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise ValueError("session row must be an object")
                    if row.get("type") == "batch":
                        decoded = [lightweight(item) for item in row["messages"]]
                        batch_meta = SessionMetadata.from_dict(row["metadata"])
                        for item in decoded:
                            accept(item)
                        meta = batch_meta
                    elif row.get("type") == "metadata" or "session_id" in row:
                        if lineno != 1:
                            raise ValueError("unexpected metadata row")
                        meta = SessionMetadata.from_dict(row)
                    else:
                        accept(lightweight(row))
                except (TypeError, ValueError, AttributeError, KeyError):
                    if isinstance(row, Mapping) and (row.get("type") in {"metadata", "batch"} or "session_id" in row):
                        invalid_metadata = True
                    logger.warning("Session %s: skipped corrupt line %d", session_id, lineno)
        if invalid_metadata:
            raise ValueError("Invalid session age metadata")
        if pending and MessageManager.valid_tool_group(pending):
            count += len(pending)
        meta.session_id = session_id
        meta.message_count = count
        return meta

    def export_session(self, session_id: str, destination: str | Path, *, format: str = "jsonl", redact: bool = False) -> Path:
        """Export full snapshots, or a display-only Markdown transcript.

        Exports include user text and tool output. Redaction masks credential keys
        and removes tool output; it cannot identify every secret in natural text.
        """
        session_id = self.resolve_session_id(session_id)
        metadata, messages = self.load_session(session_id)
        rows = [self._serialize_message(message) for message in messages]
        if redact:
            def mask(value: Any) -> Any:
                if isinstance(value, Mapping):
                    return {key: "[REDACTED]" if re.search(r"secret|password|api.?key|authorization|token", str(key), re.I) else mask(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [mask(item) for item in value]
                return value
            rows = mask(rows)
            for row in rows:
                for call in row.get("tool_calls", []):
                    call["raw_arguments"] = json.dumps(call["arguments"], ensure_ascii=False)
                if row["role"] == "tool":
                    row["content"] = [{"type": "text", "text": "[REDACTED tool output]"}]
        meta = metadata.to_dict()
        meta["type"] = "metadata"
        if format == "json":
            text = json.dumps({"schema_version": 1, "metadata": meta, "messages": rows}, ensure_ascii=False, indent=2) + "\n"
        elif format == "jsonl":
            text = "\n".join(json.dumps(row, ensure_ascii=False) for row in [meta, *rows]) + "\n"
        elif format in {"markdown", "md"}:
            text = f"# {metadata.title}\n\nDisplay-only transcript; use JSON/JSONL to resume.\n\n"
            for row in rows:
                message = self._deserialize_message(row)
                text += f"## {message.role}\n\n{message.text}\n\n"
                if message.tool_calls:
                    text += "```json\n" + json.dumps(row["tool_calls"], ensure_ascii=False, indent=2) + "\n```\n\n"
        else:
            raise ValueError("Export format must be json, jsonl or markdown")
        target = Path(destination).expanduser().resolve()
        if target == self._get_session_path(session_id):
            raise ValueError("Export destination must differ from source session")
        temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def import_session(self, source: str | Path, *, new_id: bool = False) -> str:
        """Strictly import JSON/full or incremental JSONL without overwriting IDs."""
        text = Path(source).read_text(encoding="utf-8")
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            document = None
        if isinstance(document, Mapping) and "messages" in document and "metadata" in document and document.get("type") != "batch":
            if document.get("schema_version", 1) != 1:
                raise ValueError("Unsupported session export schema version")
            rows = [document["metadata"], *document["messages"]]
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not rows or not isinstance(rows[0], Mapping) or "session_id" not in rows[0]:
            raise ValueError("Import requires session metadata")
        metadata = SessionMetadata.from_dict(rows[0])
        self._get_session_path(metadata.session_id)
        messages: list[Message] = []
        for row in rows[1:]:
            if not isinstance(row, Mapping):
                raise ValueError("Invalid session row")
            if row.get("type") == "batch":
                metadata = SessionMetadata.from_dict(row["metadata"])
                self._get_session_path(metadata.session_id)
                if metadata.session_id != rows[0]["session_id"]:
                    raise ValueError("Batch session ID differs from snapshot")
                messages.extend(self._deserialize_message(item) for item in row["messages"])
            else:
                messages.append(self._deserialize_message(row))
        if any(not MessageManager.valid_tool_group(group) for group in MessageManager.get_atomic_groups(messages)):
            raise ValueError("Incomplete or invalid tool correspondence")
        session_id = uuid.uuid4().hex if new_id else metadata.session_id
        metadata.session_id = session_id
        metadata.message_count = len(messages)
        with self._locked(session_id):
            if self._get_session_path(session_id).exists():
                raise FileExistsError(f"Session '{session_id}' already exists; import with a new ID")
            self._atomic_write_session(session_id, metadata, messages)
        return session_id

    def delete_session(self, session_id: str) -> bool:
        """Delete session data and any associated temporary files.

        Args:
            session_id: The session identifier to delete.

        Returns:
            True if session file was found and removed, False otherwise.
        """
        with self._locked(session_id):
            return self._delete_session(session_id)

    def _delete_session(self, session_id: str) -> bool:
        session_path = self._get_session_path(session_id)
        self._cache.pop(session_id, None)
        existed = False
        if session_path.is_file():
            session_path.unlink()
            existed = True

        # Clean up any leftover temporary files for this session
        for tmp_file in self._storage_dir.glob(f"{session_id}.*.tmp"):
            with contextlib.suppress(OSError):
                tmp_file.unlink(missing_ok=True)

        return existed

    def _get_session_path(self, session_id: str) -> Path:
        if (
            not isinstance(session_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", session_id)
            or ".." in session_id
        ):
            raise ValueError(f"Invalid session id: {session_id!r}")
        path = (self._storage_dir / f"{session_id}.jsonl").resolve()
        if not path.is_relative_to(self._storage_dir.resolve()):
            raise ValueError(f"Session path escapes storage directory: {session_id!r}")
        return path

    def resolve_session_id(self, session_id: str) -> str:
        """Resolve an exact ID or unique prefix; never create a missing session."""
        if self._get_session_path(session_id).is_file():
            return session_id
        matches = []
        for path in self._storage_dir.glob("*.jsonl"):
            if path.stem.startswith(session_id):
                self._get_session_path(path.stem)
                matches.append(path.stem)
        if not matches:
            raise FileNotFoundError(f"Session '{session_id}' not found")
        if len(matches) != 1:
            raise ValueError(f"Session prefix '{session_id}' is ambiguous")
        return matches[0]

    def cleanup_sessions(self, *, older_than_days: float | None = None) -> int:
        """Delete all sessions, or those last updated more than N days ago."""
        if older_than_days is not None and (
            not math.isfinite(older_than_days) or older_than_days < 0
        ):
            raise ValueError("Session age must be finite and nonnegative")
        cutoff = time.time() - older_than_days * 86400 if older_than_days is not None else None
        removed = 0
        for path in self._storage_dir.glob("*.jsonl"):
            with self._locked(path.stem):
                if not path.exists():
                    continue
                if cutoff is not None:
                    try:
                        metadata = self._load_metadata(path.stem)
                    except (TypeError, ValueError, AttributeError, KeyError):
                        logger.warning("Session %s: skipped cleanup with invalid age metadata", path.stem)
                        continue
                    if metadata.updated_at >= cutoff:
                        continue
                removed += self._delete_session(path.stem)
        return removed

    def _atomic_write_session(
        self,
        session_id: str,
        metadata: SessionMetadata,
        messages: Sequence[Message],
    ) -> None:
        """Atomically write metadata and messages to disk using a temp file."""
        target_path = self._get_session_path(session_id)
        tmp_path = self._storage_dir / f"{session_id}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                meta_dict = metadata.to_dict()
                meta_dict["type"] = "metadata"
                f.write(json.dumps(meta_dict, ensure_ascii=False) + "\n")
                for msg in messages:
                    msg_dict = self._serialize_message(msg)
                    f.write(json.dumps(msg_dict, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target_path)
            self._remember(session_id, metadata, messages)
        except Exception:
            if tmp_path.exists():
                with contextlib.suppress(OSError):
                    tmp_path.unlink()
            raise

    @staticmethod
    def _serialize_message(msg: Message) -> dict[str, Any]:
        """Serialize a Message with full fidelity including ToolCall raw_arguments."""
        data = msg.to_dict()
        data["type"] = "message"
        if msg.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": dict(tc.arguments),
                    "raw_arguments": tc.raw_arguments,
                }
                for tc in msg.tool_calls
            ]
        return data

    @staticmethod
    def _deserialize_message(data: Mapping[str, Any]) -> Message:
        """Deserialize a dict into a Message restoring ToolCall raw_arguments."""
        payload = dict(data)
        if (
            payload.get("type") == "message"
            and "data" in payload
            and isinstance(payload["data"], Mapping)
        ):
            payload = dict(payload["data"])

        if payload.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("invalid message role")
        content = payload.get("content")
        if not isinstance(content, (str, list)):
            raise ValueError("invalid message content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, Mapping) or part.get("type") not in {"text", "thinking", "image", "tool_result"}:
                    raise ValueError("invalid content part")
                if part.get("type") in {"text", "thinking"} and not isinstance(part.get("text"), str):
                    raise ValueError("invalid text part")
        if "metadata" in payload and not isinstance(payload["metadata"], Mapping):
            raise ValueError("invalid message metadata")
        if "created_at" in payload and (type(payload["created_at"]) not in (int, float) or not math.isfinite(payload["created_at"])):
            raise ValueError("invalid message timestamp")
        calls = payload.get("tool_calls", [])
        if not isinstance(calls, list) or any(not isinstance(call, Mapping) or not isinstance(call.get("id"), str) or not call.get("id") or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments", {}), (str, Mapping)) for call in calls):
            raise ValueError("invalid tool calls")

        role = payload["role"]
        if calls and role != "assistant":
            raise ValueError("tool calls require assistant role")
        result_id = payload.get("tool_call_id")
        if role == "tool":
            if not isinstance(result_id, str) or not result_id:
                raise ValueError("tool result ID must be a nonempty string")
        elif result_id is not None:
            raise ValueError("tool result ID requires tool role")
        for call in calls:
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                parsed = json.loads(arguments)
                if not isinstance(parsed, Mapping):
                    raise ValueError("tool arguments must be a JSON object")

        msg = Message.from_dict(payload)
        raw_tool_calls = payload.get("tool_calls")
        if raw_tool_calls and isinstance(raw_tool_calls, Sequence) and msg.tool_calls:
            for tc, tc_dict in zip(msg.tool_calls, raw_tool_calls, strict=False):
                if isinstance(tc_dict, Mapping) and "raw_arguments" in tc_dict:
                    tc.raw_arguments = str(tc_dict["raw_arguments"])
        return msg
