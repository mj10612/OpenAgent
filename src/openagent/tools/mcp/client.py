"""MCP (Model Context Protocol) client and tool integration.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import anyio
import httpx
import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

from openagent.core.types import ToolParam
from openagent.tools.base import DangerLevel, Tool, ToolResult


def _transport_failed(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, OSError, EOFError, anyio.EndOfStream, anyio.BrokenResourceError, anyio.ClosedResourceError, httpx.TransportError)):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_transport_failed(child) for child in exc.exceptions)
    return type(exc).__name__ in ("McpError", "MCPError") and "connection closed" in str(exc).lower()


@dataclass(slots=True)
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    name: str
    transport: Literal["stdio", "sse"] = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | Path | None = None
    url: str = ""
    headers: dict[str, str] | None = None
    timeout: float = 30.0
    read_timeout: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], name: str = "") -> MCPServerConfig:
        """Create a server configuration from a dictionary."""
        server_name = str(data.get("name") or name)
        transport_raw = str(data.get("transport") or ("sse" if "url" in data and "command" not in data else "stdio")).lower()
        if transport_raw not in ("stdio", "sse"):
            raise ValueError(f"Invalid transport: '{transport_raw}'. Expected 'stdio' or 'sse'.")
        transport: Literal["stdio", "sse"] = "stdio" if transport_raw == "stdio" else "sse"

        env_raw = data.get("env")
        env_dict = {str(k): str(v) for k, v in env_raw.items()} if env_raw is not None else None

        return cls(
            name=server_name,
            transport=transport,
            command=str(data.get("command", "")),
            args=list(data.get("args") or []),
            env=env_dict,
            cwd=data.get("cwd"),
            url=str(data.get("url", "")),
            headers=dict(data["headers"]) if data.get("headers") is not None else None,
            timeout=float(data.get("timeout", 30.0)),
            read_timeout=float(data["read_timeout"]) if data.get("read_timeout") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        result: dict[str, Any] = {
            "name": self.name,
            "transport": self.transport,
        }
        if self.transport == "stdio":
            result["command"] = self.command
            result["args"] = list(self.args)
            if self.env is not None:
                result["env"] = dict(self.env)
            if self.cwd is not None:
                result["cwd"] = str(self.cwd)
        elif self.transport == "sse":
            result["url"] = self.url
            if self.headers is not None:
                result["headers"] = dict(self.headers)
            result["timeout"] = self.timeout

        if self.read_timeout is not None:
            result["read_timeout"] = self.read_timeout
        return result


class MCPClient:
    """Client for connecting to an external MCP server and executing operations."""

    def __init__(
        self,
        config: MCPServerConfig,
        session: ClientSession | None = None,
    ) -> None:
        self.config = config
        self._session: ClientSession | None = session
        self._exit_stack: AsyncExitStack | None = None
        self._is_connected: bool = False
        self._init_result: types.InitializeResult | None = None
        self._reconnect_needed = False
        self._retry_after = 0.0
        self._reconnect_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._transport_task: asyncio.Task[None] | None = None
        self._close_event: asyncio.Event | None = None

    async def _recover_connection(self) -> None:
        async with self._reconnect_lock:
            if self.is_connected or not self._reconnect_needed:
                return
            if time.monotonic() < self._retry_after:
                raise RuntimeError("MCP reconnect temporarily backed off")
            try:
                await self.close()
                await self.connect()
                # A fresh transport may expose a different tool set.
                await self._list_tools_once()
                self._reconnect_needed = False
            except Exception:
                self._retry_after = time.monotonic() + 1.0
                self._reconnect_needed = True
                raise

    @property
    def is_connected(self) -> bool:
        """Whether the client is currently connected and initialized."""
        return self._is_connected and self._session is not None

    @property
    def init_result(self) -> types.InitializeResult | None:
        """The InitializeResult returned by the server upon handshake."""
        return self._init_result

    async def connect(self) -> None:
        """Establish connection and complete the MCP protocol handshake."""
        async with self._lifecycle_lock:
            await self._connect()

    async def _connect(self) -> None:
        if self.is_connected:
            return

        if self._transport_task is not None:
            await self._close_transport()

        if self._session is not None:
            self._init_result = await self._session.initialize()
            self._is_connected = True
            return

        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._close_event = asyncio.Event()
        self._transport_task = asyncio.create_task(self._run_transport(ready, self._close_event))
        try:
            await asyncio.shield(ready)
        except BaseException:
            await self._close_transport()
            raise

    async def _run_transport(self, ready: asyncio.Future[None], stop: asyncio.Event) -> None:
        """Enter and exit every AnyIO cancel scope in one dedicated owner task."""
        stack = AsyncExitStack()
        try:
            if self.config.transport == "stdio":
                params = StdioServerParameters(
                    command=self.config.command,
                    args=list(self.config.args),
                    env=self.config.env,
                    cwd=self.config.cwd,
                )
                read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
            elif self.config.transport == "sse":
                sse_kwargs: dict[str, Any] = {
                    "url": self.config.url,
                    "headers": self.config.headers,
                    "timeout": self.config.timeout,
                }
                if self.config.read_timeout is not None:
                    sse_kwargs["sse_read_timeout"] = self.config.read_timeout
                read_stream, write_stream = await stack.enter_async_context(
                    sse_client(**sse_kwargs)
                )
            else:
                raise ValueError(f"Unsupported transport: {self.config.transport}")

            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=self.config.read_timeout,
                )
            )
            self._init_result = await session.initialize()
            self._session = session
            self._exit_stack = stack
            self._is_connected = True
            ready.set_result(None)
            await stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                self._reconnect_needed = not stop.is_set()
        finally:
            try:
                await stack.aclose()
            finally:
                self._exit_stack = None
                self._session = None
                self._is_connected = False

    async def _close_transport(self) -> None:
        task = self._transport_task
        if task is not None:
            if self._close_event is not None:
                self._close_event.set()
            try:
                await asyncio.shield(task)
            finally:
                self._transport_task = None
                self._close_event = None

    async def list_tools(self) -> list[types.Tool]:
        """Discover tools exposed by the MCP server, exhausting pagination."""
        await self._recover_connection()
        if not self.is_connected or self._session is None:
            raise RuntimeError(f"MCPClient '{self.config.name}' is not connected.")

        try:
            return await self._list_tools_once()
        except Exception as exc:
            if _transport_failed(exc):
                self._is_connected = False
                self._reconnect_needed = True
            raise

    async def _list_tools_once(self) -> list[types.Tool]:
        assert self._session is not None
        result = await self._session.list_tools()
        tools = list(result.tools)
        cursor = result.next_cursor
        seen_cursors: set[str] = set()
        while cursor and cursor not in seen_cursors and len(seen_cursors) < 1000:
            seen_cursors.add(cursor)
            result = await self._session.list_tools(params=types.PaginatedRequestParams(cursor=cursor))
            tools.extend(result.tools)
            cursor = result.next_cursor
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> types.CallToolResult | Any:
        """Call a tool on the MCP server."""
        await self._recover_connection()
        if not self.is_connected or self._session is None:
            raise RuntimeError(f"MCPClient '{self.config.name}' is not connected.")

        try:
            return await self._session.call_tool(name, arguments=arguments or {})
        except Exception as exc:
            # Never replay: a mutating remote call may have completed before
            # the transport failed. The next operation performs recovery.
            if _transport_failed(exc):
                self._is_connected = False
                self._reconnect_needed = True
            raise

    async def close(self) -> None:
        """Cleanly disconnect and release transport resources."""
        async with self._lifecycle_lock:
            await self._close_transport()
            self._reconnect_needed = False
            self._session = None
            self._is_connected = False

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


class MCPTool(Tool):
    """Wraps an MCP-discovered tool into OpenAgent's native Tool interface."""

    def __init__(
        self,
        client: MCPClient,
        tool_def: types.Tool | Mapping[str, Any],
        server_name: str = "",
        name_prefix: bool = False,
        danger: DangerLevel | None = None,
    ) -> None:
        raw_name = getattr(tool_def, "name", None) or (tool_def.get("name", "") if isinstance(tool_def, Mapping) else "")
        self.raw_name: str = str(raw_name)
        self.server_name: str = server_name
        self.name: str = f"{server_name}__{raw_name}" if name_prefix and server_name else self.raw_name
        self.client: MCPClient = client
        self.source = "mcp"

        self.description: str = str(
            getattr(tool_def, "description", None)
            or (tool_def.get("description", "") if isinstance(tool_def, Mapping) else "")
            or ""
        )

        # Remote hints are descriptions, never local authorization.
        self.danger = danger if danger is not None else "execute"

        schema = getattr(tool_def, "input_schema", None)
        if schema is None:
            schema = getattr(tool_def, "inputSchema", None)
        if schema is None and isinstance(tool_def, Mapping):
            schema = tool_def.get("input_schema") or tool_def.get("inputSchema")

        self.params: list[ToolParam] = self._parse_input_schema(schema)
        self.input_schema = copy.deepcopy(dict(schema)) if isinstance(schema, Mapping) else {"type": "object"}
        self.prepare_callback: Callable[[str], Awaitable[MCPTool | None]] | None = None

    @classmethod
    def _parse_input_schema(cls, schema: Any) -> list[ToolParam]:
        """Convert JSON Schema properties into OpenAgent ToolParams."""
        if not isinstance(schema, Mapping):
            return []

        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            return []

        required_raw = schema.get("required") or []
        required_set = set(required_raw) if isinstance(required_raw, (list, tuple, set)) else set()

        params: list[ToolParam] = []
        for prop_name, prop_data in properties.items():
            if not isinstance(prop_data, Mapping):
                prop_data = {}

            p_type = str(prop_data.get("type", "string"))
            p_desc = str(prop_data.get("description", ""))
            p_req = prop_name in required_set
            p_enum = (
                list(prop_data["enum"])
                if "enum" in prop_data and isinstance(prop_data["enum"], (list, tuple))
                else None
            )
            p_default = prop_data.get("default")
            p_items = (
                dict(prop_data["items"])
                if "items" in prop_data and isinstance(prop_data["items"], Mapping)
                else None
            )

            params.append(
                ToolParam(
                    name=str(prop_name),
                    type=p_type,
                    description=p_desc,
                    required=p_req,
                    enum=p_enum,
                    default=p_default,
                    items=p_items,
                )
            )
        return params

    async def execute(self, call_id: str = "", **kwargs: Any) -> ToolResult:
        """Execute the MCP tool via the underlying MCPClient."""
        declared = {p.name for p in self.params}
        arguments = {
            k: v for k, v in kwargs.items()
            if k != "call_id" or "call_id" in declared
        }

        try:
            if self.prepare_callback is not None:
                refreshed = await self.prepare_callback(self.name)
                if refreshed is None:
                    raise RuntimeError("Tool is no longer available after MCP reconnect")
                if refreshed is not self:
                    from jsonschema import Draft202012Validator
                    Draft202012Validator(refreshed.spec.schema()).validate(arguments)
                    return await refreshed.execute(call_id=call_id, **arguments)
            res = await self.client.call_tool(self.raw_name, arguments=arguments)
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                output=f"Error executing MCP tool '{self.name}': {exc}",
                is_error=True,
            )

        output_parts: list[str] = []
        content = getattr(res, "content", None)
        if content is None and isinstance(res, Mapping):
            content = res.get("content")

        if content and isinstance(content, (list, tuple)):
            for item in content:
                if isinstance(item, str):
                    output_parts.append(item)
                elif hasattr(item, "text"):
                    output_parts.append(str(item.text))
                elif isinstance(item, Mapping) and "text" in item:
                    output_parts.append(str(item["text"]))
                elif hasattr(item, "data"):
                    mime = getattr(item, "mime_type", getattr(item, "mimeType", "binary"))
                    output_parts.append(f"[Binary content: {mime}]")
                elif isinstance(item, Mapping) and ("data" in item or "mimeType" in item or "mime_type" in item):
                    mime = item.get("mimeType") or item.get("mime_type") or "binary"
                    output_parts.append(f"[Binary content: {mime}]")
                else:
                    output_parts.append(str(item))

        output = "\n".join(output_parts)
        if not output:
            structured = getattr(res, "structured_content", None)
            if structured is None and isinstance(res, Mapping):
                structured = res.get("structured_content")
            if structured is not None:
                output = json.dumps(structured)

        is_error = bool(
            getattr(res, "is_error", False)
            or getattr(res, "isError", False)
            or (isinstance(res, Mapping) and (res.get("is_error", False) or res.get("isError", False)))
        )
        return ToolResult(call_id=call_id, output=output, is_error=is_error)
