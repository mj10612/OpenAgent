"""Base classes and types for the OpenAgent tool execution system.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import html
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from openagent.core.types import Message, ToolParam, ToolSpec

DangerLevel = Literal["none", "write", "execute", "network"]
ToolSource = Literal["builtin", "mcp"]
MAX_TOOL_OUTPUT_BYTES = 100000


def bounded_output(value: str, limit: int = MAX_TOOL_OUTPUT_BYTES) -> str:
    """Bound stored UTF-8 output without splitting code points."""
    raw = value.encode("utf-8")
    marker = "\n[Tool output truncated]"
    if len(raw) <= limit:
        return value
    return raw[:limit - len(marker.encode())].decode("utf-8", errors="ignore") + marker


@dataclass(slots=True)
class ToolResult:
    """The result of executing a tool call."""

    call_id: str
    output: str
    is_error: bool = False
    structured_data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tool_message(self, name: str | None = None) -> Message:
        """Convert this ToolResult into a canonical tool-role Message."""
        return Message.tool_result(
            call_id=self.call_id,
            name=name or "",
            content=f"<tool_output>\n{bounded_output(html.escape(self.output), MAX_TOOL_OUTPUT_BYTES - 30)}\n</tool_output>",
            is_error=self.is_error,
            structured_data=self.structured_data,
            metadata=self.metadata,
        )


class Tool(ABC):
    """Abstract base class for all OpenAgent tools."""

    name: str
    description: str
    danger: DangerLevel = "none"
    source: ToolSource = "builtin"
    params: list[ToolParam]

    @property
    def spec(self) -> ToolSpec:
        """Generate provider-independent ToolSpec for this tool."""
        return ToolSpec(
            name=getattr(self, "name", ""),
            description=getattr(self, "description", ""),
            params=list(getattr(self, "params", [])),
            danger=getattr(self, "danger", "none"),
            source=getattr(self, "source", "builtin"),
            input_schema=getattr(self, "input_schema", None),
        )

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given keyword arguments."""
        ...
