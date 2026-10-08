"""Agent utility tools: ThinkTool, TodoTool, and WebFetchTool.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import re
import socket
from html.parser import HTMLParser
from typing import Any

import httpx

from openagent.core.types import ToolParam
from openagent.tools.base import DangerLevel, Tool, ToolResult


class ThinkTool(Tool):
    """Tool for recording internal reasoning and thoughts without affecting state."""

    name = "think"
    description = (
        "Record internal reasoning, planning, or thoughts in a scratchpad without "
        "modifying the environment."
    )
    danger: DangerLevel = "none"

    def __init__(self) -> None:
        super().__init__()
        self.params = [
            ToolParam(name="thought", type="string", description="The thought or reasoning to record")
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        call_id = str(kwargs.get("call_id", ""))
        thought = str(kwargs.get("thought", ""))
        return ToolResult(call_id=call_id, output=f"Recorded thought: {thought}")


class TodoTool(Tool):
    """Tool for managing the agent's task checklist."""

    name = "todo"
    description = (
        "Manage a structured task checklist for multi-step plans. "
        "Supports adding, listing, completing, and clearing tasks."
    )
    danger: DangerLevel = "none"

    def __init__(self) -> None:
        super().__init__()
        self._tasks: list[dict[str, Any]] = []
        self.params = [
            ToolParam(
                name="action",
                type="string",
                description="Action to perform: 'add', 'list', 'complete', 'clear'",
                enum=["add", "list", "complete", "clear"],
            ),
            ToolParam(
                name="tasks",
                type="array",
                description="List of task strings to add",
                required=False,
            ),
            ToolParam(
                name="task",
                type="string",
                description="Task text or number to mark complete",
                required=False,
            ),
        ]

    def _render_checklist(self) -> str:
        if not self._tasks:
            return "Task checklist is empty."
        lines: list[str] = []
        for i, t in enumerate(self._tasks, start=1):
            box = "[x]" if t["done"] else "[ ]"
            lines.append(f"{i}. {box} {t['text']}")
        return "\n".join(lines)

    async def execute(self, **kwargs: Any) -> ToolResult:
        call_id = str(kwargs.get("call_id", ""))
        action = str(kwargs.get("action", ""))
        tasks = kwargs.get("tasks")
        task = kwargs.get("task")

        act = action.lower().strip()

        if act == "add":
            added: list[str] = []
            if isinstance(tasks, list):
                added.extend(str(item) for item in tasks if str(item).strip())
            if task and str(task).strip():
                added.append(str(task).strip())

            if not added:
                return ToolResult(
                    call_id=call_id,
                    output="No tasks provided to add.",
                    is_error=True,
                )

            for t in added:
                self._tasks.append({"text": t, "done": False})
            return ToolResult(
                call_id=call_id,
                output=f"Added {len(added)} task(s).\nCurrent checklist:\n{self._render_checklist()}",
            )

        if act in ("list", "get"):
            return ToolResult(call_id=call_id, output=self._render_checklist())

        if act in ("complete", "done"):
            target_str = str(task or "").strip()
            if not target_str and isinstance(tasks, list) and tasks:
                target_str = str(tasks[0]).strip()

            if not target_str:
                return ToolResult(
                    call_id=call_id,
                    output="No task identifier provided to complete.",
                    is_error=True,
                )

            found = False
            # Check numeric index (1-based)
            if target_str.isdigit():
                idx = int(target_str) - 1
                if 0 <= idx < len(self._tasks):
                    self._tasks[idx]["done"] = True
                    found = True

            if not found:
                target_lower = target_str.lower()
                for item in self._tasks:
                    if target_lower in item["text"].lower():
                        item["done"] = True
                        found = True
                        break

            if not found:
                return ToolResult(
                    call_id=call_id,
                    output=f"Task '{target_str}' not found in checklist.",
                    is_error=True,
                )

            return ToolResult(
                call_id=call_id,
                output=f"Completed task.\nCurrent checklist:\n{self._render_checklist()}",
            )

        if act in ("clear", "reset"):
            self._tasks.clear()
            return ToolResult(call_id=call_id, output="Task checklist cleared.")

        return ToolResult(
            call_id=call_id,
            output=f"Unknown action '{action}'. Valid actions: add, list, complete, clear.",
            is_error=True,
        )


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML parser that extracts human-readable text."""

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._ignore_tags = {"script", "style", "head", "noscript", "svg"}
        self._current_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._current_tag = tag.lower()
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._current_tag == tag.lower():
            self._current_tag = None
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if self._current_tag not in self._ignore_tags:
            self._pieces.append(data)

    def get_text(self) -> str:
        raw = "".join(self._pieces)
        decoded = html.unescape(raw)
        lines = [line.strip() for line in decoded.splitlines()]
        cleaned = "\n".join(line for line in lines if line)
        return re.sub(r"\n{3,}", "\n\n", cleaned)


class WebFetchTool(Tool):
    """Tool to fetch web page content and extract readable text."""

    name = "web_fetch"
    description = "Fetch web page content via HTTP GET and extract readable text."
    danger: DangerLevel = "network"

    def __init__(self, timeout: float = 30.0) -> None:
        super().__init__()
        self.timeout = timeout
        self.params = [
            ToolParam(name="url", type="string", description="HTTP/HTTPS URL to fetch"),
            ToolParam(
                name="max_chars",
                type="integer",
                description="Maximum characters to return (default 50000)",
                required=False,
                default=50000,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        call_id = str(kwargs.get("call_id", ""))
        url = str(kwargs.get("url", ""))
        max_chars = max(1, min(int(kwargs.get("max_chars", 50000)), 100000))

        if not (url.startswith("http://") or url.startswith("https://")):
            return ToolResult(
                call_id=call_id,
                output=f"Invalid URL '{url}'. Must start with http:// or https://",
                is_error=True,
            )

        headers = {
            "Accept-Encoding": "identity",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 OpenAgent/0.1"
            )
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False, trust_env=False) as client:
                current = httpx.URL(url)
                for redirect in range(6):
                    address = await asyncio.wait_for(_public_address(current), timeout=self.timeout)
                    # Pin the validated address. Host and TLS SNI retain the original
                    # hostname, so DNS rebinding cannot change the peer after validation.
                    request_headers = dict(headers)
                    request_headers["Host"] = current.netloc.decode("ascii")
                    pinned = current.copy_with(host=address)
                    request = client.build_request("GET", pinned, headers=request_headers, extensions={"sni_hostname": current.host})
                    response = await client.send(request, stream=True)
                    try:
                        if response.status_code in (301, 302, 303, 307, 308):
                            if redirect == 5 or "location" not in response.headers:
                                raise ValueError("Redirect limit exceeded or missing Location")
                            current = current.join(str(response.headers["location"]))
                            continue
                        if response.headers.get("content-encoding", "identity").lower() not in ("", "identity"):
                            raise ValueError("Compressed responses are not supported")
                        cap = 2 * 1024 * 1024
                        if int(response.headers.get("content-length", "0")) > cap:
                            raise ValueError("Response exceeds 2 MB download limit")
                        body = bytearray()
                        async for chunk in response.aiter_raw():
                            if len(body) + len(chunk) > cap:
                                raise ValueError("Response exceeds 2 MB download limit")
                            body.extend(chunk)
                        response = httpx.Response(response.status_code, headers=response.headers, content=bytes(body), request=request)
                        break
                    finally:
                        await response.aclose()
        except httpx.TimeoutException:
            return ToolResult(
                call_id=call_id,
                output=f"Request to '{url}' timed out after {self.timeout}s.",
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                output=f"Failed to fetch '{url}': {exc}",
                is_error=True,
            )

        if response.status_code >= 400:
            return ToolResult(
                call_id=call_id,
                output=f"HTTP {response.status_code} Error: {response.text[:500]}",
                is_error=True,
            )

        content_type = response.headers.get("content-type", "")
        if "html" in content_type:
            parser = _HTMLTextExtractor()
            parser.feed(response.text)
            text = parser.get_text()
        else:
            text = response.text

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [Content truncated at {max_chars} characters]"

        return ToolResult(call_id=call_id, output=text)


async def _public_address(url: httpx.URL) -> str:
    """Resolve only HTTP(S) URLs whose every address is globally routable."""
    if url.scheme not in ("http", "https") or not url.host or url.username or url.password:
        raise ValueError("Only public HTTP(S) URLs without credentials are permitted")
    if url.host.lower().rstrip(".") == "localhost":
        raise ValueError("Private hosts are not permitted")
    try:
        addresses = [str(ipaddress.ip_address(url.host))]
    except ValueError:
        records = await asyncio.to_thread(socket.getaddrinfo, url.host, url.port or (443 if url.scheme == "https" else 80), type=socket.SOCK_STREAM)
        addresses = list(dict.fromkeys(str(record[4][0]) for record in records))
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("Private, loopback, link-local, or reserved hosts are not permitted")
    return addresses[0]
