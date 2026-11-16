import pytest

from openagent.core.types import ToolCall
from openagent.tools.agent_tools import TodoTool
from openagent.tools.registry import ToolRegistry


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["0", "3", "99", "-1", "-0", "+0", "+3", "+99"])
@pytest.mark.parametrize("field", ["task", "tasks"])
async def test_out_of_range_numeric_id_does_not_complete_matching_title(target, field):
    todo = TodoTool()
    registry = ToolRegistry([todo])
    added = await registry.execute_call(ToolCall(id="add", name="todo", arguments={
        "action": "add", "tasks": [f"Review ticket {target}", "Write report"],
    }))
    assert not added.is_error
    before = todo.snapshot_state()

    result = await registry.execute_call(ToolCall(id="complete", name="todo", arguments={
        "action": "complete", field: target if field == "task" else [target],
    }))

    assert result.is_error
    assert "not found" in result.output
    assert todo.snapshot_state() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["task", "tasks"])
@pytest.mark.parametrize("target,expected_done", [
    ("1", [True, False]),
    ("2", [False, True]),
    ("+1", [True, False]),
    ("+2", [False, True]),
    ("TICKET", [True, False]),
    ("ticket-1", [True, False]),
])
async def test_completion_preserves_numeric_ids_and_text_lookup(field, target, expected_done):
    todo = TodoTool()
    registry = ToolRegistry([todo])
    await registry.execute_call(ToolCall(id="add", name="todo", arguments={
        "action": "add", "tasks": ["Review ticket 2 and ticket-1", "Write report"],
    }))

    result = await registry.execute_call(ToolCall(id="complete", name="todo", arguments={
        "action": "complete", field: target if field == "task" else [target],
    }))

    assert not result.is_error
    assert [task["done"] for task in todo.snapshot_state()["tasks"]] == expected_done
