"""Safe asynchronous shell execution tool with timeout and output bounding.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import asyncio
import contextlib
import locale
import math
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Any, Literal

from openagent.core.types import ToolParam
from openagent.tools.base import DangerLevel, Tool, ToolResult

MAX_OUTPUT_BYTES = 100 * 1024  # 100 KB cap
MAX_TIMEOUT = 600.0
SAFE_ENV_KEYS = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA", "HOMEDRIVE", "HOMEPATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM"}


def _decode_output(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        encoding = locale.getpreferredencoding(False)
        if sys.platform == "win32":
            import ctypes
            encoding = f"cp{ctypes.windll.kernel32.GetOEMCP()}"
        return data.decode(encoding, errors="replace")


async def _collect_output(proc: asyncio.subprocess.Process) -> tuple[bytes, bytes, bool]:
    """Drain both pipes while retaining at most MAX_OUTPUT_BYTES in total."""
    remaining = MAX_OUTPUT_BYTES
    truncated = False

    async def drain(stream: asyncio.StreamReader | None) -> bytes:
        nonlocal remaining, truncated
        retained = bytearray()
        if stream is None:
            return b""
        while chunk := await stream.read(8192):
            keep = min(remaining, len(chunk))
            retained.extend(chunk[:keep])
            remaining -= keep
            truncated = truncated or keep < len(chunk)
        return bytes(retained)

    stdout, stderr, _ = await asyncio.gather(drain(proc.stdout), drain(proc.stderr), proc.wait())
    return stdout, stderr, truncated


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        if sys.platform == "win32":
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/F",
                "/T",
                "/PID",
                str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
            if proc.returncode is None:
                proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)


def _truncate_output(text: str, cap: int = MAX_OUTPUT_BYTES) -> str:
    """Cap output string to limit and append truncation indicator if exceeded."""
    raw_bytes = text.encode("utf-8", errors="replace")
    if len(raw_bytes) <= cap:
        return text
    truncated = raw_bytes[:cap].decode("utf-8", errors="ignore")
    return f"{truncated}\n... [Output truncated at 100KB]"


class ShellTool(Tool):
    """Executes terminal commands asynchronously with timeout and bounded output."""

    name = "execute_shell"
    description = (
        "Execute a terminal command asynchronously with timeout enforcement and bounded output."
    )
    danger: DangerLevel = "execute"

    def __init__(
        self,
        workspace_root: Path | str | None = None,
        shell_type: Literal["auto", "powershell", "cmd", "bash", "sh"] = "auto",
    ) -> None:
        super().__init__()
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.shell_type = shell_type
        self.params = [
            ToolParam(name="command", type="string", description="Shell command line to execute"),
            ToolParam(
                name="timeout",
                type="number",
                description="Timeout in seconds (default 120.0)",
                required=False,
                default=120.0,
            ),
            ToolParam(
                name="cwd",
                type="string",
                description="Optional working directory for the command",
                required=False,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        call_id = str(kwargs.get("call_id", ""))
        command = str(kwargs.get("command", ""))
        timeout = float(kwargs.get("timeout", 120.0))
        cwd = kwargs.get("cwd")

        if not math.isfinite(timeout) or not 0 < timeout <= MAX_TIMEOUT:
            return ToolResult(call_id, f"Timeout must be finite and between 0 and {MAX_TIMEOUT} seconds.", True)

        if not command:
            return ToolResult(
                call_id=call_id,
                output="Parameter 'command' is required.",
                is_error=True,
            )

        # Determine effective working directory
        work_dir: Path | None = None
        if cwd is not None:
            target_cwd = Path(cwd)
            if not target_cwd.is_absolute() and self.workspace_root is not None:
                work_dir = (self.workspace_root / target_cwd).resolve()
            else:
                work_dir = target_cwd.resolve()
        elif self.workspace_root is not None:
            work_dir = self.workspace_root

        if work_dir and self.workspace_root and not work_dir.is_relative_to(self.workspace_root):
            return ToolResult(
                call_id=call_id,
                output=f"Access denied: working directory '{work_dir}' is outside workspace root '{self.workspace_root}'.",
                is_error=True,
            )

        # Prepare subprocess invocation
        process_options: dict[str, Any] = (
            {"start_new_session": True} if sys.platform != "win32" else {}
        )
        process_options["stdin"] = asyncio.subprocess.DEVNULL
        process_options["env"] = {key: value for key, value in os.environ.items() if key.upper() in SAFE_ENV_KEYS}
        try:
            if self.shell_type == "powershell":
                proc = await asyncio.create_subprocess_exec(
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; $OutputEncoding = [System.Text.Encoding]::UTF8; " + command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(work_dir) if work_dir else None,
                    **process_options,
                )
            elif self.shell_type == "cmd":
                proc = await asyncio.create_subprocess_exec(
                    os.environ.get("COMSPEC", "cmd.exe"),
                    "/c",
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(work_dir) if work_dir else None,
                    **process_options,
                )
            elif self.shell_type in ("bash", "sh"):
                sh_binary = shutil.which(self.shell_type) or "/bin/sh"
                proc = await asyncio.create_subprocess_exec(
                    sh_binary,
                    "-c",
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(work_dir) if work_dir else None,
                    **process_options,
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(work_dir) if work_dir else None,
                    **process_options,
                )
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                output=f"Failed to start command process: {exc}",
                is_error=True,
            )

        # Wait with timeout
        collection = asyncio.create_task(_collect_output(proc))
        try:
            stdout_bytes, stderr_bytes, truncated = await asyncio.wait_for(
                asyncio.shield(collection),
                timeout=timeout,
            )
        except TimeoutError:
            await _kill_process_tree(proc)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(collection, timeout=5)
            return ToolResult(
                call_id=call_id,
                output=f"Command timed out after {timeout} seconds.",
                is_error=True,
            )
        except asyncio.CancelledError:
            await _kill_process_tree(proc)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(collection, timeout=5)
            raise

        stdout_text = _decode_output(stdout_bytes)
        stderr_text = _decode_output(stderr_bytes)

        returncode = proc.returncode if proc.returncode is not None else -1
        is_error = returncode != 0

        if not is_error:
            if stderr_text.strip():
                output = (
                    f"{stdout_text.strip()}\n{stderr_text.strip()}"
                    if stdout_text.strip()
                    else stderr_text.strip()
                )
            else:
                output = stdout_text.strip() or "(command completed with no output)"
        else:
            pieces = [f"Command failed with exit code {returncode}"]
            if stdout_text.strip():
                pieces.append(f"Stdout:\n{stdout_text.strip()}")
            if stderr_text.strip():
                pieces.append(f"Stderr:\n{stderr_text.strip()}")
            output = "\n".join(pieces)

        # Bound total combined output string
        output = _truncate_output(output, cap=MAX_OUTPUT_BYTES)
        if truncated and "[Output truncated at 100KB]" not in output:
            output += "\n... [Output truncated at 100KB]"
        return ToolResult(call_id=call_id, output=output, is_error=is_error)


async def execute_shell(
    command: str,
    timeout: float = 120.0,
    cwd: str | Path | None = None,
    call_id: str = "",
    *,
    workspace_root: str | Path | None = None,
) -> ToolResult:
    """Helper function to execute shell command asynchronously."""
    tool = ShellTool(workspace_root=workspace_root or Path.cwd())
    return await tool.execute(command=command, timeout=timeout, cwd=cwd, call_id=call_id)
