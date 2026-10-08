"""Tool registry with danger-level permission checks and parallel execution.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from jsonschema import Draft202012Validator

from openagent.core.types import ToolCall, ToolSpec
from openagent.tools.base import Tool, ToolResult


class PermissionAction(StrEnum):
    """Permission action decisions for tool execution."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


DEFAULT_DANGER_POLICIES: dict[str, PermissionAction] = {
    "none": PermissionAction.ALLOW,
    "write": PermissionAction.ASK,
    "execute": PermissionAction.ASK,
    "network": PermissionAction.ASK,
}

AskCallback = Callable[[ToolCall], Awaitable[bool] | bool]


class ToolRegistry:
    """Registry managing available tools, permission policies, and invocation."""

    def __init__(
        self,
        tools: list[Tool] | None = None,
        danger_policies: dict[str, PermissionAction] | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._danger_policies: dict[str, PermissionAction] = dict(DEFAULT_DANGER_POLICIES)
        if danger_policies:
            self._danger_policies.update(danger_policies)
        self._tool_policies: dict[str, PermissionAction] = {}
        self._mutation_lock = asyncio.Lock()
        self._confirmation_lock = asyncio.Lock()

        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        if tool.name in self._tools and self._tools[tool.name] is not tool:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def list_specs(self) -> list[ToolSpec]:
        """Return ToolSpecs for all registered tools."""
        return [tool.spec for tool in self._tools.values()]

    def set_danger_policy(self, danger: str, policy: PermissionAction | str) -> None:
        """Override default permission policy for a danger level."""
        self._danger_policies[str(danger)] = PermissionAction(policy)

    def set_tool_policy(self, tool_name: str, policy: PermissionAction | str) -> None:
        """Override permission policy for a specific tool name."""
        self._tool_policies[tool_name] = PermissionAction(policy)

    def get_policy(self, tool_name: str) -> PermissionAction:
        """Get the effective permission policy for a tool."""
        if tool_name in self._tool_policies:
            return self._tool_policies[tool_name]

        tool = self._tools.get(tool_name)
        if tool:
            danger = str(getattr(tool, "danger", "none"))
            return self._danger_policies.get(danger, PermissionAction.ASK)

        return PermissionAction.DENY

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    async def execute_call(
        self,
        call: ToolCall,
        ask_callback: AskCallback | None = None,
    ) -> ToolResult:
        """Execute a call, serializing state changes and terminal confirmations."""
        tool = self.get(call.name)
        if tool is not None and getattr(tool, "danger", "none") in ("write", "execute"):
            async with self._mutation_lock:
                return await self._execute_call(call, ask_callback)
        return await self._execute_call(call, ask_callback)

    async def _execute_call(self, call: ToolCall, ask_callback: AskCallback | None) -> ToolResult:
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(
                call_id=call.id,
                output=f"Tool not found: '{call.name}'",
                is_error=True,
            )

        try:
            schema = tool.spec.schema()
            Draft202012Validator.check_schema(schema)
            errors = list(Draft202012Validator(schema).iter_errors(call.arguments))
            if errors:
                return ToolResult(call_id=call.id, output=f"Invalid tool arguments: {errors[0].message}", is_error=True)
        except Exception as exc:
            return ToolResult(call_id=call.id, output=f"Invalid tool argument schema: {exc}", is_error=True)

        policy = self.get_policy(call.name)
        if policy == PermissionAction.DENY:
            return ToolResult(
                call_id=call.id,
                output=f"Permission denied: tool '{call.name}' is blocked by policy.",
                is_error=True,
            )

        if policy == PermissionAction.ASK:
            if ask_callback is None:
                return ToolResult(
                    call_id=call.id,
                    output=f"Permission denied: tool '{call.name}' requires confirmation, but no ask_callback was provided.",
                    is_error=True,
                )
            try:
                async with self._confirmation_lock:
                    verdict = ask_callback(call)
                    allowed = await verdict if inspect.isawaitable(verdict) else verdict
            except Exception as exc:
                return ToolResult(
                    call_id=call.id,
                    output=f"Permission error: confirmation check failed for tool '{call.name}': {exc}",
                    is_error=True,
                )
            if not allowed:
                return ToolResult(
                    call_id=call.id,
                    output=f"Permission denied: user rejected execution of tool '{call.name}'.",
                    is_error=True,
                )

        kwargs: dict[str, Any] = dict(call.arguments)
        try:
            sig = inspect.signature(tool.execute)
            if "call_id" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                kwargs["call_id"] = call.id
            res = await tool.execute(**kwargs)
            if not isinstance(res, ToolResult):
                raise TypeError(f"Expected ToolResult, got {type(res).__name__}")
            if not isinstance(res.output, str):
                raise TypeError("ToolResult.output must be a string")
        except Exception as exc:
            return ToolResult(
                call_id=call.id,
                output=f"Error executing tool '{call.name}': {exc}",
                is_error=True,
            )

        if not res.call_id:
            return ToolResult(call_id=call.id, output=res.output, is_error=res.is_error)
        return res

    async def execute_parallel(
        self,
        calls: list[ToolCall],
        ask_callback: AskCallback | None = None,
    ) -> list[ToolResult]:
        """Execute multiple tool calls in parallel with asyncio.gather."""
        tasks = [self.execute_call(call, ask_callback=ask_callback) for call in calls]
        return list(await asyncio.gather(*tasks))
