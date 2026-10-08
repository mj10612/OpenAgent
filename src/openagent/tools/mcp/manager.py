"""Lifecycle and registration manager for multiple MCP servers.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from openagent.tools.mcp.client import MCPClient, MCPServerConfig, MCPTool
from openagent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class MCPManager:
    """Orchestrates lifecycle, discovery, and registration of multiple MCP servers."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        configs: Sequence[MCPServerConfig] | None = None,
        prefix_tool_names: bool = True,
        client_factory: Callable[[MCPServerConfig], MCPClient] | None = None,
    ) -> None:
        self.registry: ToolRegistry = registry if registry is not None else ToolRegistry()
        self.prefix_tool_names = prefix_tool_names
        self._client_factory: Callable[[MCPServerConfig], MCPClient] = (
            client_factory if client_factory is not None else MCPClient
        )
        self.configs: dict[str, MCPServerConfig] = {cfg.name: cfg for cfg in configs or []}
        self.clients: dict[str, MCPClient] = {}
        self.tools: dict[str, list[MCPTool]] = {}
        self.failed_servers: dict[str, Exception] = {}
        self._reconnect_locks: dict[str, asyncio.Lock] = {}
        self._retry_after: dict[str, float] = {}

    def _wrap(self, client: MCPClient, definition: Any, name: str) -> MCPTool:
        tool = MCPTool(client, definition, server_name=name, name_prefix=self.prefix_tool_names)
        async def prepare(tool_name: str):
            await self.ensure_tools(name)
            return self.registry.get(tool_name)
        tool.prepare_callback = prepare
        return tool

    async def ensure_tools(self, name: str) -> list[MCPTool]:
        """Recover once on the next operation, refreshing the registry atomically."""
        async with self._reconnect_locks.setdefault(name, asyncio.Lock()):
            client = self.clients.get(name)
            if client is not None and client.is_connected:
                return self.tools.get(name, [])
            if time.monotonic() < self._retry_after.get(name, 0):
                raise RuntimeError(f"MCP server '{name}' reconnect is temporarily backed off")
            if client is None:
                return await self.start_server(name)
            try:
                await client.close()
                await client.connect()
                definitions = await client.list_tools()
                wrapped = [self._wrap(client, definition, name) for definition in definitions]
                old = self.tools.get(name, [])
                old_names = {tool.name for tool in old}
                names = [tool.name for tool in wrapped]
                if len(set(names)) != len(names) or any(tool.name in self.registry and tool.name not in old_names for tool in wrapped):
                    raise ValueError("MCP tools collide with already registered tools")
                for tool in old:
                    if self.registry.get(tool.name) is tool:
                        self.registry.unregister(tool.name)
                for tool in wrapped:
                    self.registry.register(tool)
                self.tools[name] = wrapped
                self.failed_servers.pop(name, None)
                self._retry_after.pop(name, None)
                return wrapped
            except Exception as exc:
                self.failed_servers[name] = exc
                self._retry_after[name] = time.monotonic() + 1.0
                with contextlib.suppress(Exception):
                    await client.close()
                raise

    def add_server(self, config: MCPServerConfig) -> None:
        """Register or update an MCP server configuration."""
        self.configs[config.name] = config

    async def start_server(self, name: str) -> list[MCPTool]:
        """Start a single MCP server, discover tools, and register them."""
        if name not in self.configs:
            raise KeyError(f"No configuration found for server '{name}'.")

        config = self.configs[name]
        if name in self.clients:
            await self.stop_server(name)

        client = self._client_factory(config)
        wrapped_tools: list[MCPTool] = []
        try:
            await client.connect()
            mcp_tool_defs = await client.list_tools()
            for tool_def in mcp_tool_defs:
                tool = self._wrap(client, tool_def, config.name)
                self.registry.register(tool)
                wrapped_tools.append(tool)

            self.clients[name] = client
            self.tools[name] = wrapped_tools
            self.failed_servers.pop(name, None)
            return wrapped_tools
        except Exception as exc:
            self.failed_servers[name] = exc
            logger.warning("Failed to connect to MCP server '%s': %s", name, exc)
            for tool in wrapped_tools:
                self.registry.unregister(tool.name)
            with contextlib.suppress(Exception):
                await client.close()
            raise

    async def stop_server(self, name: str) -> None:
        """Stop an individual MCP server and unregister its tools."""
        registered_tools = self.tools.pop(name, [])
        for tool in registered_tools:
            if self.registry.get(tool.name) is tool:
                self.registry.unregister(tool.name)

        client = self.clients.pop(name, None)
        if client is not None:
            try:
                await client.close()
            except Exception as exc:
                logger.debug("Error while closing MCP client '%s': %s", name, exc)

    async def start(self) -> dict[str, list[MCPTool]]:
        """Start all configured MCP servers with error isolation."""
        discovered: dict[str, list[MCPTool]] = {}
        for name in list(self.configs.keys()):
            try:
                tools = await self.start_server(name)
                discovered[name] = tools
            except Exception:
                # Error is captured in self.failed_servers, allowing other servers to proceed
                continue
        return discovered

    async def stop(self) -> None:
        """Stop all running MCP servers and unregister all their tools."""
        for name in list(self.clients.keys()):
            await self.stop_server(name)
        for name in list(self.tools.keys()):
            await self.stop_server(name)

    async def __aenter__(self) -> MCPManager:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()

    @classmethod
    def load_from_dict(
        cls,
        data: Mapping[str, Any],
        registry: ToolRegistry | None = None,
        prefix_tool_names: bool = True,
        client_factory: Callable[[MCPServerConfig], MCPClient] | None = None,
    ) -> MCPManager:
        """Instantiate an MCPManager from a dictionary or TOML structure."""
        raw_servers = data.get("mcpServers") or data.get("mcp_servers") or data.get("servers")
        configs: list[MCPServerConfig] = []

        if raw_servers is not None:
            if isinstance(raw_servers, Mapping):
                for name, cfg in raw_servers.items():
                    if isinstance(cfg, Mapping):
                        configs.append(MCPServerConfig.from_dict(cfg, name=str(name)))
            elif isinstance(raw_servers, list):
                for cfg in raw_servers:
                    if isinstance(cfg, Mapping):
                        name = str(cfg.get("name") or f"server_{len(configs)}")
                        configs.append(MCPServerConfig.from_dict(cfg, name=name))
        else:
            for name, cfg in data.items():
                if isinstance(cfg, Mapping) and any(
                    k in cfg for k in ("command", "url", "transport")
                ):
                    configs.append(MCPServerConfig.from_dict(cfg, name=str(name)))

        return cls(
            registry=registry,
            configs=configs,
            prefix_tool_names=prefix_tool_names,
            client_factory=client_factory,
        )
