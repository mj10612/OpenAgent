"""Tests for MCP (Model Context Protocol) client and manager integration.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import mcp.types as types
import pytest

from openagent.core.types import ToolCall
from openagent.tools.base import ToolResult
from openagent.tools.mcp.client import MCPClient, MCPServerConfig, MCPTool
from openagent.tools.mcp.manager import MCPManager
from openagent.tools.registry import ToolRegistry

# --------------------------------------------------------------------------- #
# MCPServerConfig tests
# --------------------------------------------------------------------------- #


def test_mcp_server_config_stdio_defaults_and_custom() -> None:
    cfg = MCPServerConfig(
        name="test-server",
        command="python",
        args=["-m", "demo"],
        env={"FOO": "BAR"},
    )
    assert cfg.name == "test-server"
    assert cfg.transport == "stdio"
    assert cfg.command == "python"
    assert cfg.args == ["-m", "demo"]
    assert cfg.env == {"FOO": "BAR"}

    d = cfg.to_dict()
    assert d["name"] == "test-server"
    assert d["transport"] == "stdio"
    assert d["command"] == "python"
    assert d["args"] == ["-m", "demo"]
    assert d["env"] == {"FOO": "BAR"}

    reconstructed = MCPServerConfig.from_dict(d)
    assert reconstructed.name == cfg.name
    assert reconstructed.transport == cfg.transport
    assert reconstructed.command == cfg.command
    assert reconstructed.args == cfg.args
    assert reconstructed.env == cfg.env


def test_mcp_server_config_sse() -> None:
    d: dict[str, Any] = {
        "name": "remote-sse",
        "transport": "sse",
        "url": "http://127.0.0.1:8000/sse",
        "headers": {"Authorization": "Bearer test-token"},
        "timeout": 15.0,
        "read_timeout": 60.0,
    }
    cfg = MCPServerConfig.from_dict(d)
    assert cfg.name == "remote-sse"
    assert cfg.transport == "sse"
    assert cfg.url == "http://127.0.0.1:8000/sse"
    assert cfg.headers == {"Authorization": "Bearer test-token"}
    assert cfg.timeout == 15.0
    assert cfg.read_timeout == 60.0

    serialized = cfg.to_dict()
    assert serialized["url"] == cfg.url
    assert serialized["headers"] == cfg.headers
    assert serialized["timeout"] == 15.0
    assert serialized["read_timeout"] == 60.0


def test_mcp_server_config_infer_sse_and_invalid_transport() -> None:
    cfg = MCPServerConfig.from_dict({"url": "http://example.com/sse"}, name="inferred")
    assert cfg.name == "inferred"
    assert cfg.transport == "sse"

    with pytest.raises(ValueError, match="Invalid transport"):
        MCPServerConfig.from_dict({"transport": "websocket", "name": "bad"})


# --------------------------------------------------------------------------- #
# MCPClient tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mcp_client_handshake_and_connection_state() -> None:
    cfg = MCPServerConfig(name="stdio-srv", command="python")
    mock_session = AsyncMock()
    mock_session.initialize.return_value = MagicMock(server_info=MagicMock(name="test", version="1.0"))

    client = MCPClient(cfg, session=mock_session)
    assert not client.is_connected

    await client.connect()
    assert client.is_connected
    mock_session.initialize.assert_awaited_once()

    await client.close()
    assert not client.is_connected


@pytest.mark.asyncio
async def test_mcp_client_list_tools_with_pagination() -> None:
    cfg = MCPServerConfig(name="tool-srv", command="python")
    mock_session = AsyncMock()

    t1 = types.Tool(
        name="tool_1",
        description="First tool",
        input_schema={"type": "object"},
    )
    t2 = types.Tool(
        name="tool_2",
        description="Second tool",
        input_schema={"type": "object"},
    )

    page1 = types.ListToolsResult(tools=[t1], next_cursor="cursor-page-2")
    page2 = types.ListToolsResult(tools=[t2], next_cursor=None)

    mock_session.list_tools.side_effect = [page1, page2]

    client = MCPClient(cfg, session=mock_session)
    await client.connect()

    tools = await client.list_tools()
    assert len(tools) == 2
    assert tools[0].name == "tool_1"
    assert tools[1].name == "tool_2"
    assert mock_session.list_tools.await_count == 2


@pytest.mark.asyncio
async def test_mcp_client_call_tool() -> None:
    cfg = MCPServerConfig(name="calc-srv", command="python")
    mock_session = AsyncMock()

    expected_res = types.CallToolResult(
        content=[types.TextContent(type="text", text="42")],
        is_error=False,
    )
    mock_session.call_tool.return_value = expected_res

    client = MCPClient(cfg, session=mock_session)
    await client.connect()

    result = await client.call_tool("add", {"a": 20, "b": 22})
    assert result == expected_res
    mock_session.call_tool.assert_awaited_once_with("add", arguments={"a": 20, "b": 22})


@pytest.mark.asyncio
async def test_mcp_client_not_connected_errors() -> None:
    cfg = MCPServerConfig(name="calc-srv", command="python")
    client = MCPClient(cfg)

    with pytest.raises(RuntimeError, match="not connected"):
        await client.list_tools()

    with pytest.raises(RuntimeError, match="not connected"):
        await client.call_tool("tool")


@pytest.mark.asyncio
async def test_mcp_client_stdio_connect_and_context_manager() -> None:
    cfg = MCPServerConfig(
        name="stdio-srv",
        transport="stdio",
        command="test-cmd",
        args=["--flag"],
        env={"TEST": "1"},
    )

    fake_session = AsyncMock()
    fake_session.initialize.return_value = MagicMock()

    fake_stdio_cm = AsyncMock()
    fake_stdio_cm.__aenter__.return_value = ("read_stream", "write_stream")
    fake_stdio_cm.__aexit__.return_value = None

    fake_session_cm = AsyncMock()
    fake_session_cm.__aenter__.return_value = fake_session
    fake_session_cm.__aexit__.return_value = None

    with (
        patch("openagent.tools.mcp.client.stdio_client", return_value=fake_stdio_cm) as mock_stdio,
        patch("openagent.tools.mcp.client.ClientSession", return_value=fake_session_cm) as mock_session_cls,
    ):
        async with MCPClient(cfg) as client:
            assert client.is_connected
            fake_session.initialize.assert_awaited_once()

        mock_stdio.assert_called_once()
        mock_session_cls.assert_called_once_with(
            "read_stream", "write_stream", read_timeout_seconds=None
        )
        fake_stdio_cm.__aexit__.assert_awaited_once()
        fake_session_cm.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_mcp_client_sse_connect() -> None:
    cfg = MCPServerConfig(
        name="sse-srv",
        transport="sse",
        url="http://localhost:8080/sse",
        headers={"X-Test": "val"},
        timeout=10.0,
        read_timeout=30.0,
    )

    fake_session = AsyncMock()
    fake_session.initialize.return_value = MagicMock()

    fake_sse_cm = AsyncMock()
    fake_sse_cm.__aenter__.return_value = ("read_stream", "write_stream")
    fake_sse_cm.__aexit__.return_value = None

    fake_session_cm = AsyncMock()
    fake_session_cm.__aenter__.return_value = fake_session
    fake_session_cm.__aexit__.return_value = None

    with (
        patch("openagent.tools.mcp.client.sse_client", return_value=fake_sse_cm) as mock_sse,
        patch("openagent.tools.mcp.client.ClientSession", return_value=fake_session_cm) as mock_session_cls,
    ):
        client = MCPClient(cfg)
        await client.connect()
        assert client.is_connected
        await client.close()

        mock_sse.assert_called_once_with(
            url="http://localhost:8080/sse",
            headers={"X-Test": "val"},
            timeout=10.0,
            sse_read_timeout=30.0,
        )
        mock_session_cls.assert_called_once_with(
            "read_stream", "write_stream", read_timeout_seconds=30.0
        )


# --------------------------------------------------------------------------- #
# MCPTool tests
# --------------------------------------------------------------------------- #


def test_mcp_tool_schema_mapping_and_spec_generation() -> None:
    mcp_raw_tool = types.Tool(
        name="search_database",
        description="Search entries in the DB",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query or term",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results",
                    "default": 10,
                },
                "category": {
                    "type": "string",
                    "enum": ["docs", "code", "issues"],
                },
            },
            "required": ["query"],
        },
    )

    client = MagicMock(spec=MCPClient)
    tool = MCPTool(client=client, tool_def=mcp_raw_tool, server_name="db_srv")

    assert tool.name == "search_database"
    assert tool.description == "Search entries in the DB"
    assert tool.source == "mcp"
    assert tool.danger == "execute"

    spec = tool.spec
    assert spec.name == "search_database"
    assert spec.source == "mcp"
    assert spec.danger == "execute"
    assert len(spec.params) == 3

    param_map = {p.name: p for p in spec.params}
    assert param_map["query"].required is True
    assert param_map["query"].type == "string"
    assert param_map["limit"].required is False
    assert param_map["limit"].default == 10
    assert param_map["category"].enum == ["docs", "code", "issues"]

    # Verify provider schema translations
    openai_schema = spec.to_openai_schema()
    assert openai_schema["function"]["name"] == "search_database"
    assert "query" in openai_schema["function"]["parameters"]["properties"]
    assert openai_schema["function"]["parameters"]["required"] == ["query"]

    anthropic_schema = spec.to_anthropic_schema()
    assert anthropic_schema["name"] == "search_database"
    assert "query" in anthropic_schema["input_schema"]["properties"]


def test_mcp_tool_danger_inference_from_annotations() -> None:
    client = MagicMock(spec=MCPClient)

    ro_tool_def = types.Tool(
        name="read_only_tool",
        input_schema={"type": "object"},
        annotations=types.ToolAnnotations(read_only_hint=True),
    )
    t1 = MCPTool(client=client, tool_def=ro_tool_def)
    assert t1.danger == "execute"

    dest_tool_def = types.Tool(
        name="destructive_tool",
        input_schema={"type": "object"},
        annotations=types.ToolAnnotations(destructive_hint=True),
    )
    t2 = MCPTool(client=client, tool_def=dest_tool_def)
    assert t2.danger == "execute"

    # Explicit override takes precedence
    t3 = MCPTool(client=client, tool_def=ro_tool_def, danger="network")
    assert t3.danger == "network"


@pytest.mark.asyncio
async def test_mcp_tool_execute_success() -> None:
    client = AsyncMock(spec=MCPClient)
    client.call_tool.return_value = types.CallToolResult(
        content=[types.TextContent(type="text", text="Search finished: 3 items")],
        is_error=False,
    )

    tool_def = types.Tool(
        name="search",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    tool = MCPTool(client=client, tool_def=tool_def)

    res = await tool.execute(call_id="call_999", q="openagent")
    assert isinstance(res, ToolResult)
    assert res.call_id == "call_999"
    assert res.output == "Search finished: 3 items"
    assert not res.is_error

    # Verify call_id was not forwarded to client.call_tool
    client.call_tool.assert_awaited_once_with("search", arguments={"q": "openagent"})


@pytest.mark.asyncio
async def test_mcp_tool_execute_error_propagation_and_exceptions() -> None:
    client = AsyncMock(spec=MCPClient)
    tool_def = types.Tool(name="failing_tool", input_schema={"type": "object"})
    tool = MCPTool(client=client, tool_def=tool_def)

    # MCP error result
    client.call_tool.return_value = types.CallToolResult(
        content=[types.TextContent(type="text", text="Index out of bounds")],
        is_error=True,
    )
    res = await tool.execute(call_id="call_err")
    assert res.is_error is True
    assert "Index out of bounds" in res.output

    # Client exception
    client.call_tool.side_effect = ConnectionResetError("Server disconnected")
    res_exc = await tool.execute(call_id="call_exc")
    assert res_exc.is_error is True
    assert "Server disconnected" in res_exc.output


@pytest.mark.asyncio
async def test_mcp_tool_multipart_content() -> None:
    client = AsyncMock(spec=MCPClient)
    client.call_tool.return_value = types.CallToolResult(
        content=[
            types.TextContent(type="text", text="Part 1: Text summary"),
            types.ImageContent(type="image", data="abc123==", mime_type="image/png"),
            types.TextContent(type="text", text="Part 2: Conclusion"),
        ]
    )
    tool = MCPTool(client=client, tool_def=types.Tool(name="multi", input_schema={"type": "object"}))
    res = await tool.execute()
    assert "Part 1: Text summary" in res.output
    assert "[Binary content: image/png]" in res.output
    assert "Part 2: Conclusion" in res.output


# --------------------------------------------------------------------------- #
# MCPManager tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mcp_manager_multi_server_lifecycle_and_registry() -> None:
    registry = ToolRegistry()

    cfg1 = MCPServerConfig(name="srv1", command="cmd1")
    cfg2 = MCPServerConfig(name="srv2", command="cmd2")

    mock_client1 = AsyncMock(spec=MCPClient)
    mock_client1.connect.return_value = None
    mock_client1.list_tools.return_value = [
        types.Tool(name="tool_a", input_schema={"type": "object"})
    ]
    mock_client1.call_tool.return_value = types.CallToolResult(
        content=[types.TextContent(type="text", text="Result from tool A")],
        is_error=False,
    )

    mock_client2 = AsyncMock(spec=MCPClient)
    mock_client2.connect.return_value = None
    mock_client2.list_tools.return_value = [
        types.Tool(name="tool_b", input_schema={"type": "object"})
    ]

    clients = {"srv1": mock_client1, "srv2": mock_client2}

    def client_factory(cfg: MCPServerConfig) -> MCPClient:
        return clients[cfg.name]

    manager = MCPManager(
        registry=registry,
        prefix_tool_names=False,
        configs=[cfg1, cfg2],
        client_factory=client_factory,
    )

    # Start manager and check tools registered into registry
    registered = await manager.start()
    assert "srv1" in registered
    assert "srv2" in registered
    assert len(registry) == 2
    assert "tool_a" in registry
    assert "tool_b" in registry

    # Execute a tool via the registry (MCP tools have danger="execute" so ASK policy applies)
    call = ToolCall(name="tool_a", arguments={})
    res = await registry.execute_call(call, ask_callback=lambda _: True)
    assert res.output == "Result from tool A"
    assert not res.is_error

    # Stop manager and verify cleanup
    await manager.stop()
    assert len(registry) == 0
    mock_client1.close.assert_awaited_once()
    mock_client2.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_mcp_manager_error_isolation() -> None:
    registry = ToolRegistry()

    cfg_good = MCPServerConfig(name="good_srv", command="good")
    cfg_bad = MCPServerConfig(name="bad_srv", command="bad")

    mock_good = AsyncMock(spec=MCPClient)
    mock_good.connect.return_value = None
    mock_good.list_tools.return_value = [
        types.Tool(name="good_tool", input_schema={"type": "object"})
    ]

    mock_bad = AsyncMock(spec=MCPClient)
    mock_bad.connect.side_effect = TimeoutError("Failed to connect to bad_srv")

    def client_factory(cfg: MCPServerConfig) -> MCPClient:
        if cfg.name == "good_srv":
            return mock_good
        return mock_bad

    manager = MCPManager(
        registry=registry,
        prefix_tool_names=False,
        configs=[cfg_good, cfg_bad],
        client_factory=client_factory,
    )

    await manager.start()

    # Good server's tools are registered
    assert "good_tool" in registry
    assert len(manager.tools.get("good_srv", [])) == 1

    # Bad server error was isolated and recorded
    assert "bad_srv" in manager.failed_servers
    assert isinstance(manager.failed_servers["bad_srv"], TimeoutError)

    await manager.stop()
    assert len(registry) == 0


@pytest.mark.asyncio
async def test_mcp_manager_async_context_manager() -> None:
    registry = ToolRegistry()
    cfg = MCPServerConfig(name="ctx_srv", command="ctx")

    mock_client = AsyncMock(spec=MCPClient)
    mock_client.connect.return_value = None
    mock_client.list_tools.return_value = [
        types.Tool(name="ctx_tool", input_schema={"type": "object"})
    ]

    manager = MCPManager(
        registry=registry,
        prefix_tool_names=False,
        configs=[cfg],
        client_factory=lambda _: mock_client,
    )

    async with manager:
        assert "ctx_tool" in registry

    assert len(registry) == 0
    mock_client.close.assert_awaited_once()


def test_mcp_manager_load_from_dict_variants() -> None:
    # 1. Claude desktop / mcpServers style
    data_desktop = {
        "mcpServers": {
            "fetch": {
                "url": "http://localhost:8000/sse",
            },
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            },
        }
    }
    m1 = MCPManager.load_from_dict(data_desktop)
    assert len(m1.configs) == 2
    assert m1.configs["fetch"].transport == "sse"
    assert m1.configs["filesystem"].transport == "stdio"

    # 2. servers list style
    data_list = {
        "servers": [
            {"name": "s1", "command": "python"},
            {"name": "s2", "url": "https://api.example.com/sse"},
        ]
    }
    m2 = MCPManager.load_from_dict(data_list)
    assert len(m2.configs) == 2
    assert "s1" in m2.configs
    assert "s2" in m2.configs

    # 3. Direct server mapping style
    data_direct = {
        "sqlite": {
            "command": "uvx",
            "args": ["mcp-server-sqlite", "--db-path", "test.db"],
        }
    }
    m3 = MCPManager.load_from_dict(data_direct)
    assert len(m3.configs) == 1
    assert "sqlite" in m3.configs


@pytest.mark.asyncio
async def test_mcp_manager_name_prefixing() -> None:
    registry = ToolRegistry()
    cfg = MCPServerConfig(name="weather", command="dummy")

    mock_client = AsyncMock(spec=MCPClient)
    mock_client.connect.return_value = None
    mock_client.list_tools.return_value = [
        types.Tool(name="get_forecast", input_schema={"type": "object"})
    ]

    manager = MCPManager(
        registry=registry,
        configs=[cfg],
        prefix_tool_names=True,
        client_factory=lambda _: mock_client,
    )

    await manager.start()
    assert "weather__get_forecast" in registry
    assert "get_forecast" not in registry

    await manager.stop()
