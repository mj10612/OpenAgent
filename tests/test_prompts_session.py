"""Tests for PromptBuilder and SessionStore.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from openagent.core.types import (
    ImagePart,
    Message,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolParam,
    ToolResultPart,
    ToolSpec,
)
from openagent.prompts import PromptBuilder
from openagent.session import SessionMetadata, SessionStore
from openagent.tools.base import Tool, ToolResult
from openagent.tools.registry import ToolRegistry


class DummyEchoTool(Tool):
    name = "dummy_echo"
    description = "Echoes back input text."
    danger = "none"

    def __init__(self) -> None:
        self.params = [
            ToolParam(
                name="text",
                type="string",
                description="The text to echo back.",
                required=True,
            ),
            ToolParam(
                name="prefix",
                type="string",
                description="Optional prefix to add.",
                required=False,
                default="",
            ),
        ]

    async def execute(self, **kwargs: object) -> ToolResult:
        return ToolResult(call_id="test", output=str(kwargs.get("text", "")))


# =========================================================================== #
# PromptBuilder Tests
# =========================================================================== #


def test_prompt_builder_basic_system_prompt(tmp_path: Path) -> None:
    builder = PromptBuilder()
    prompt = builder.build_system_prompt(workspace_root=tmp_path)

    assert "OpenAgent" in prompt
    assert str(tmp_path.resolve()) in prompt
    assert "Operating System" in prompt
    assert "Guidelines" in prompt


def test_prompt_builder_workspace_resolution(tmp_path: Path) -> None:
    builder = PromptBuilder()
    # Path object
    prompt_path = builder.build_system_prompt(workspace_root=tmp_path)
    assert str(tmp_path.resolve()) in prompt_path

    # String path
    prompt_str = builder.build_system_prompt(workspace_root=str(tmp_path))
    assert str(tmp_path.resolve()) in prompt_str

    # None defaults to current working directory
    prompt_none = builder.build_system_prompt(workspace_root=None)
    assert str(Path.cwd().resolve()) in prompt_none


def test_prompt_builder_with_tools_list() -> None:
    builder = PromptBuilder()
    tool = DummyEchoTool()
    spec = ToolSpec(
        name="custom_calculator",
        description="Evaluates mathematical expressions.",
        params=[
            ToolParam(
                name="expression",
                type="string",
                description="Math expression string.",
                required=True,
            )
        ],
    )

    prompt = builder.build_system_prompt(tools=[tool, spec])

    assert "dummy_echo" in prompt
    assert "Echoes back input text." in prompt
    assert "custom_calculator" in prompt
    assert "Evaluates mathematical expressions." in prompt
    assert "expression" in prompt
    assert "required" in prompt


def test_prompt_builder_with_tool_registry() -> None:
    builder = PromptBuilder()
    registry = ToolRegistry()
    registry.register(DummyEchoTool())

    prompt = builder.build_system_prompt(tools=registry)
    assert "dummy_echo" in prompt
    assert "Echoes back input text." in prompt
    assert "text" in prompt


def test_prompt_builder_extra_instructions() -> None:
    builder = PromptBuilder()

    # Extra instructions as string
    prompt_str = builder.build_system_prompt(
        extra_instructions="Always use tabs instead of spaces."
    )
    assert "Always use tabs instead of spaces." in prompt_str

    # Extra instructions as sequence of strings
    instructions = [
        "Commit messages must follow Conventional Commits.",
        "Never edit generated files directly.",
    ]
    prompt_seq = builder.build_system_prompt(extra_instructions=instructions)
    assert "Commit messages must follow Conventional Commits." in prompt_seq
    assert "Never edit generated files directly." in prompt_seq


def test_prompt_builder_custom_guidelines_and_template() -> None:
    custom_guidelines = ["Custom rule 1: Be fast.", "Custom rule 2: Be reliable."]
    builder = PromptBuilder(
        name="CustomAssistant",
        guidelines=custom_guidelines,
    )
    prompt = builder.build_system_prompt()

    assert "CustomAssistant" in prompt
    assert "Custom rule 1: Be fast." in prompt
    assert "Custom rule 2: Be reliable." in prompt


def test_prompt_builder_custom_template_override(tmp_path: Path) -> None:
    custom_template = (
        "Agent: {name}\nWorkspace: {workspace}\nTools:\n{tools}\nInstructions: {extra_instructions}"
    )
    builder = PromptBuilder(name="TestBot", template=custom_template)
    prompt = builder.build_system_prompt(
        workspace_root=tmp_path,
        tools=[DummyEchoTool()],
        extra_instructions="Be awesome.",
    )

    assert prompt.startswith("Agent: TestBot")
    assert f"Workspace: {tmp_path.resolve()}" in prompt
    assert "dummy_echo" in prompt
    assert "Be awesome." in prompt


# =========================================================================== #
# SessionMetadata Tests
# =========================================================================== #


def test_session_metadata_defaults_and_serialization() -> None:
    now = time.time()
    meta = SessionMetadata(
        session_id="session-123",
        created_at=now,
        updated_at=now,
        model="gpt-4o",
        title="Refactor core",
        message_count=5,
    )

    data = meta.to_dict()
    assert data["session_id"] == "session-123"
    assert data["created_at"] == now
    assert data["updated_at"] == now
    assert data["model"] == "gpt-4o"
    assert data["title"] == "Refactor core"
    assert data["message_count"] == 5

    restored = SessionMetadata.from_dict(data)
    assert restored == meta


# =========================================================================== #
# SessionStore Tests
# =========================================================================== #


def test_session_store_create_and_load_empty(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)
    session_id = store.create_session(model="claude-3-5-sonnet", title="Initial Task")

    assert session_id
    session_file = tmp_path / f"{session_id}.jsonl"
    assert session_file.exists()

    meta, messages = store.load_session(session_id)
    assert meta.session_id == session_id
    assert meta.model == "claude-3-5-sonnet"
    assert meta.title == "Initial Task"
    assert meta.message_count == 0
    assert messages == []


def test_session_store_save_and_load_messages(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)
    session_id = store.create_session(model="gpt-4o")

    messages = [
        Message.system("System prompt instructions"),
        Message.user("Hello, please inspect main.py"),
        Message.assistant("Certainly, I will check the file."),
    ]

    store.save_messages(session_id, messages)

    meta, loaded_messages = store.load_session(session_id)
    assert meta.session_id == session_id
    assert meta.message_count == 3
    assert len(loaded_messages) == 3
    assert loaded_messages[0].role == "system"
    assert loaded_messages[0].text == "System prompt instructions"
    assert loaded_messages[1].role == "user"
    assert loaded_messages[1].text == "Hello, please inspect main.py"
    assert loaded_messages[2].role == "assistant"
    assert loaded_messages[2].text == "Certainly, I will check the file."


def test_session_store_append_message(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)
    session_id = store.create_session(title="Append Test")

    msg1 = Message.user("First message")
    msg2 = Message.assistant("First response")

    store.append_message(session_id, msg1)
    meta1, loaded1 = store.load_session(session_id)
    assert meta1.message_count == 1
    assert len(loaded1) == 1
    assert loaded1[0].text == "First message"

    store.append_message(session_id, msg2)
    meta2, loaded2 = store.load_session(session_id)
    assert meta2.message_count == 2
    assert len(loaded2) == 2
    assert loaded2[1].text == "First response"
    assert meta2.updated_at >= meta1.updated_at


def test_session_store_roundtrip_full_fidelity(tmp_path: Path) -> None:
    """Verify round-trip fidelity for all Message variants and complex parts."""
    store = SessionStore(storage_dir=tmp_path)
    session_id = store.create_session(model="claude-3-5-sonnet")

    t1 = time.time() - 100
    t2 = time.time() - 50
    t3 = time.time()

    # 1. System message
    m_sys = Message.system("System instructions")
    m_sys.created_at = t1

    # 2. User message with multi-part text and image
    m_user = Message(
        role="user",
        content=[
            TextPart(text="Check this architecture diagram"),
            ImagePart(
                data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                mime_type="image/png",
                url="https://example.com/diag.png",
                detail="high",
            ),
        ],
        created_at=t2,
        metadata={"user_id": "u-42"},
    )

    # 3. Assistant message with reasoning, text, and tool calls
    tc1 = ToolCall(
        name="read_file",
        arguments={"path": "src/main.py", "start_line": 1},
        id="call_read_123",
        raw_arguments='{"path": "src/main.py", "start_line": 1}',
    )
    tc2 = ToolCall(
        name="execute_command",
        arguments={"cmd": "pytest"},
        id="call_exec_456",
        raw_arguments='{"cmd": "pytest"}',
    )
    m_assist = Message(
        role="assistant",
        content=[
            ThinkingPart(text="Let me inspect the file first.", signature="sig_abc"),
            TextPart(text="Reading the file now."),
        ],
        tool_calls=[tc1, tc2, ToolCall(id="call_part_789", name="status_check", arguments={})],
        reasoning="Step 1 is inspecting the entrypoint.",
        created_at=t3,
        metadata={"finish_reason": "tool_calls"},
    )

    # 4. Tool result message with error flag and tool_call_id
    m_tool = Message.tool_result(
        call_id="call_read_123",
        name="read_file",
        content="def main(): pass",
        is_error=False,
    )

    # 5. Tool result with is_error=True
    m_tool_err = Message.tool_result(
        call_id="call_exec_456",
        name="execute_command",
        content="Command failed: exit code 1",
        is_error=True,
    )

    # 6. ToolResultPart embedded in assistant or user turn
    m_embedded_part = Message(
        role="tool",
        content=[
            ToolResultPart(
                call_id="call_part_789",
                name="status_check",
                content="System is healthy",
                is_error=False,
            )
        ],
        tool_call_id="call_part_789",
        name="status_check",
    )

    original_messages = [
        m_sys,
        m_user,
        m_assist,
        m_tool,
        m_tool_err,
        m_embedded_part,
    ]

    store.save_messages(session_id, original_messages)
    _, loaded_messages = store.load_session(session_id)

    assert len(loaded_messages) == len(original_messages)

    for i, (orig, loaded) in enumerate(zip(original_messages, loaded_messages, strict=True)):
        assert orig.role == loaded.role, f"Message {i} role mismatch"
        assert orig.tool_call_id == loaded.tool_call_id, f"Message {i} tool_call_id mismatch"
        assert orig.name == loaded.name, f"Message {i} name mismatch"
        assert orig.reasoning == loaded.reasoning, f"Message {i} reasoning mismatch"
        assert orig.metadata == loaded.metadata, f"Message {i} metadata mismatch"
        assert orig.created_at == pytest.approx(loaded.created_at), f"Message {i} created_at mismatch"
        assert orig.content == loaded.content, f"Message {i} content parts mismatch"
        assert orig.tool_calls == loaded.tool_calls, f"Message {i} tool_calls mismatch"
        assert orig == loaded, f"Message {i} not strictly equal"


def test_session_store_list_sessions_sorted(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)

    id1 = store.create_session(title="Session 1")
    time.sleep(0.01)
    id2 = store.create_session(title="Session 2")
    time.sleep(0.01)
    id3 = store.create_session(title="Session 3")

    sessions = store.list_sessions()
    assert len(sessions) == 3
    # Most recently created/updated first
    assert [s.session_id for s in sessions] == [id3, id2, id1]

    # Now update id1 so it becomes the most recently updated
    time.sleep(0.01)
    store.append_message(id1, Message.user("Update session 1"))

    sessions_updated = store.list_sessions()
    assert [s.session_id for s in sessions_updated] == [id1, id3, id2]
    assert sessions_updated[0].message_count == 1


def test_session_store_delete_session(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)
    session_id = store.create_session(title="To be deleted")

    assert (tmp_path / f"{session_id}.jsonl").exists()
    assert store.delete_session(session_id) is True
    assert not (tmp_path / f"{session_id}.jsonl").exists()

    # Deleting again returns False
    assert store.delete_session(session_id) is False

    # Loading deleted session raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        store.load_session(session_id)


def test_session_store_append_to_nonexistent_raises(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        store.append_message("nonexistent-id", Message.user("Hello"))


def test_session_store_load_legacy_raw_jsonl(tmp_path: Path) -> None:
    """Verify loading JSONL files that don't have a metadata header."""
    raw_file = tmp_path / "raw_session.jsonl"
    m1 = Message.user("Raw message 1")
    m2 = Message.assistant("Raw response 2")

    with open(raw_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(m1.to_dict()) + "\n")
        f.write(json.dumps(m2.to_dict()) + "\n")

    store = SessionStore(storage_dir=tmp_path)
    meta, loaded = store.load_session("raw_session")

    assert meta.session_id == "raw_session"
    assert meta.message_count == 2
    assert len(loaded) == 2
    assert loaded[0].text == "Raw message 1"
    assert loaded[1].text == "Raw response 2"
