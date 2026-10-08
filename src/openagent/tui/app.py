"""Rich streaming terminal interface and REPL for OpenAgent.

Provides styled thinking blocks, tool syntax highlighting, interactive confirmation
prompts, and full-featured conversation loop.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import asyncio
import json
import signal
from types import FrameType
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from openagent import __version__
from openagent.core.events import (
    DoneEvent,
    ErrorEvent,
    StartEvent,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResultEvent,
    UsageEvent,
)
from openagent.core.router import ModelReference, ProviderRouter
from openagent.core.types import ToolCall, Usage
from openagent.tools.registry import AskCallback


class TUIApp:
    """Manages rich terminal display for streaming agent events and tool execution."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console if console is not None else Console()
        self._in_thinking = False
        self._in_text = False
        self._thinking_buffer: list[str] = []
        self._text_buffer: list[str] = []
        self.usage = Usage()

    def reset_turn(self) -> None:
        """Reset state tracking before starting a new conversation turn."""
        self._in_thinking = False
        self._in_text = False
        self._thinking_buffer.clear()
        self._text_buffer.clear()

    def render_event(self, event: StreamEvent) -> None:
        """Process and render a single streaming agent event."""
        match event:
            case StartEvent():
                self.reset_turn()

            case ThinkingDelta(text=t):
                if not self._in_thinking:
                    self._in_thinking = True
                    self.console.print("\n[dim]Thinking...[/dim]", highlight=False)
                self.console.print(t, style="dim", end="", markup=False, highlight=False)
                self._thinking_buffer.append(t)

            case TextDelta(text=t):
                if self._in_thinking:
                    self.console.print()  # End thinking block newline
                    self._in_thinking = False
                if not self._in_text:
                    self._in_text = True
                self.console.print(t, end="", markup=False, highlight=False)
                self._text_buffer.append(t)

            case ToolCallStart():
                if self._in_thinking or self._in_text:
                    self.console.print()
                    self._in_thinking = False
                    self._in_text = False

            case ToolCallDelta():
                pass

            case ToolCallEnd(call=call):
                if self._in_thinking or self._in_text:
                    self.console.print()
                    self._in_thinking = False
                    self._in_text = False

                if call is not None:
                    name = escape(call.name)
                    args = call.arguments
                    if args:
                        formatted_args = escape(json.dumps(args, indent=2))
                        if "\n" in formatted_args or len(formatted_args) > 60:
                            call_str = f"[bold blue]Tool: {name}[/bold blue] (args:\n[dim]{formatted_args}[/dim])"
                        else:
                            call_str = (
                                f"[bold blue]Tool: {name}[/bold blue] ({escape(json.dumps(args))})"
                            )
                    else:
                        call_str = f"[bold blue]Tool: {name}[/bold blue] ()"
                    self.console.print(call_str, highlight=False)

            case ToolResultEvent(tool_name=tool_name, output=output, is_error=is_error):
                trimmed_output = escape(output)
                if len(output) > 2000:
                    trimmed_output = (
                        escape(output[:2000])
                        + f"\n... [dim](output truncated, {len(output) - 2000} more characters)[/dim]"
                    )

                if is_error:
                    self.console.print(
                        f"[bold red]Tool Error ({escape(tool_name)}):[/bold red]\n{trimmed_output}",
                        highlight=False,
                    )
                else:
                    self.console.print(
                        f"[bold green]Tool Result ({escape(tool_name)}):[/bold green]\n{trimmed_output}",
                        highlight=False,
                    )

            case UsageEvent(usage=usage):
                pass

            case ErrorEvent(error=error):
                if self._in_thinking or self._in_text:
                    self.console.print()
                    self._in_thinking = False
                    self._in_text = False
                self.console.print(
                    f"\n[bold red]Error:[/bold red] {escape(str(error))}", highlight=False
                )

            case DoneEvent(usage=usage):
                self.usage = self.usage + usage
                if self._in_thinking or self._in_text:
                    self.console.print()
                    self._in_thinking = False
                    self._in_text = False

                if usage and usage.total_tokens > 0:
                    self.console.print(
                        f"[dim]Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}[/dim]",
                        highlight=False,
                    )

            case _:
                pass


async def _run_repl_turn(runner: Any, user_input: str, ask_cb: AskCallback, app: TUIApp) -> None:
    """Cancel the current turn on SIGINT while keeping the prompt task alive."""

    async def consume() -> None:
        async for event in runner.run_turn(user_input, ask_callback=ask_cb):
            app.render_event(event)

    task = asyncio.create_task(consume())
    previous = signal.getsignal(signal.SIGINT)
    interrupted = False
    installed = False
    loop = asyncio.get_running_loop()

    def interrupt(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        interrupted = True
        loop.call_soon_threadsafe(task.cancel)

    try:
        try:
            signal.signal(signal.SIGINT, interrupt)
            installed = True
        except ValueError:
            pass  # Embedded REPLs may run outside the main thread.
        try:
            await task
        except asyncio.CancelledError:
            if not interrupted:
                raise
            app.console.print("\n[yellow]Turn cancelled by user.[/yellow]")
    finally:
        if installed:
            signal.signal(signal.SIGINT, previous)


def make_ask_callback(console: Console | None = None, auto_approve: bool = False) -> AskCallback:
    """Create an interactive ask_callback confirming tool execution with the user."""
    active_console = console if console is not None else Console()

    async def _ask_callback(tool_call: ToolCall) -> bool:
        if auto_approve:
            return True

        args_str = json.dumps(tool_call.arguments, indent=2) if tool_call.arguments else "{}"
        active_console.print(
            f"\n[bold yellow]⚠️  Permission Request:[/bold yellow] Tool [cyan]{escape(tool_call.name)}[/cyan]"
        )
        if tool_call.arguments:
            active_console.print(f"[dim]Arguments:\n{escape(args_str)}[/dim]")

        try:
            answer: str = await PromptSession[str](output=None if active_console.is_terminal else DummyOutput()).prompt_async(
                "Allow tool execution? [y/N] ", handle_sigint=False
            )
            return answer.strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    return _ask_callback


async def run_repl(
    runner: Any,
    console: Console | None = None,
    auto_approve: bool = False,
) -> None:
    """Interactive REPL chat loop."""
    active_console = console if console is not None else Console()

    model_name = getattr(runner, "model", "default")
    workspace = getattr(runner, "workspace_root", ".")
    session_id = getattr(runner, "session_id", None) or "new"

    header_text = (
        f"[bold cyan]OpenAgent[/bold cyan] [dim]v{__version__}[/dim]\n"
        f"Model: [bold green]{escape(str(model_name))}[/bold green] | "
        f"Workspace: [dim]{escape(str(workspace))}[/dim] | "
        f"Session: [dim]{escape(session_id)}[/dim]\n"
        f"[dim]Commands: /exit to quit, /clear to reset context, /help for help[/dim]"
    )
    active_console.print(Panel(header_text, border_style="blue", expand=False))

    try:
        session: PromptSession[str] | None = PromptSession(history=InMemoryHistory())
    except Exception:
        session = None

    app = TUIApp(console=active_console)
    ask_cb = make_ask_callback(console=active_console, auto_approve=auto_approve)

    while True:
        try:
            if session is not None:
                user_input = await session.prompt_async("\nopenagent> ")
            else:
                user_input = await asyncio.to_thread(input, "\nopenagent> ")
            multiline = user_input.strip() in ('"""', "'''")
            if multiline:
                delimiter = user_input.strip()
                lines = []
                while True:
                    line = (
                        await session.prompt_async("... ")
                        if session is not None
                        else await asyncio.to_thread(input, "... ")
                    )
                    if line.strip() == delimiter:
                        break
                    lines.append(line)
                user_input = "\n".join(lines)
        except (KeyboardInterrupt, EOFError):
            active_console.print("\n[yellow]Exiting OpenAgent. Goodbye![/yellow]")
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        if not multiline and stripped.lower() in ("/exit", "/quit", "exit", "quit"):
            active_console.print("[yellow]Exiting OpenAgent. Goodbye![/yellow]")
            break

        if not multiline and stripped.lower() == "/clear":
            runner.reset()
            active_console.print("[green]✔ Context cleared and session reset.[/green]")
            continue

        if not multiline and stripped.lower() == "/help":
            active_console.print(
                "[bold]OpenAgent Help:[/bold]\n"
                "  /clear  - Clear conversation history and reset session\n"
                "  /exit   - Exit the interactive REPL session\n"
                "  /quit   - Exit the interactive REPL session\n"
                "  /model [name] - Display or switch the active provider/model\n"
                "  /tokens, /usage - Show cumulative REPL usage and context utilization\n"
                "  /tools - List available tools and danger levels\n"
                "  Multiline: enter triple quotes on their own line, then close with the same delimiter.\n"
            )
            continue

        if not multiline and stripped.split(maxsplit=1)[0].lower() == "/model":
            model_ref = stripped.partition(" ")[2].strip()
            if not model_ref:
                active_console.print(f"Model: {runner.model}", markup=False)
                continue
            try:
                new_provider = ProviderRouter().resolve(model_ref)
            except Exception as exc:
                active_console.print(f"Could not switch model: {exc}", markup=False)
                continue
            old_provider = runner.provider
            ref = ModelReference.parse(model_ref)
            same_provider = type(new_provider) is type(old_provider) and (
                ref.provider_hint is None
                or new_provider.name == old_provider.name
                or ref.provider_hint == old_provider.name
            )
            if same_provider:
                # Retain the configured gateway, auth headers and custom mappings.
                target_model = getattr(new_provider, "model", model_ref)
                await new_provider.close()
                old_provider.model = target_model
                runner.model = target_model
                active_console.print(f"Model: {runner.model} ({old_provider.name})", markup=False)
                continue
            explicit_max = getattr(runner, "explicit_max_tokens", None)
            runner.provider = new_provider
            runner.model = getattr(new_provider, "model", model_ref)
            runner.context_window = getattr(
                new_provider, "context_window", new_provider.default_context_window
            )
            runner.max_tokens = (
                explicit_max if explicit_max is not None else new_provider.default_max_tokens
            )
            active_console.print(f"Model: {runner.model} ({new_provider.name})", markup=False)
            try:
                await old_provider.close()
            except Exception as exc:
                active_console.print(f"Previous provider cleanup failed: {exc}", markup=False)
            continue

        if not multiline and stripped.lower() in ("/tokens", "/usage"):
            usage = app.usage
            context_tokens = runner.messages.total_tokens()
            context_window = runner.context_window
            percent = context_tokens / context_window * 100 if context_window else 0
            active_console.print(
                f"Prompt: {usage.prompt_tokens} | Completion: {usage.completion_tokens} | Total: {usage.total_tokens}\n"
                f"Context: {context_tokens} / {context_window} tokens ({percent:.1f}%)",
                markup=False,
            )
            continue

        if not multiline and stripped.lower() == "/tools":
            table = Table(title="Available Tools")
            table.add_column("Name")
            table.add_column("Description")
            table.add_column("Danger")
            for spec in runner.tools.list_specs():
                table.add_row(Text(spec.name), Text(spec.description), Text(spec.danger))
            active_console.print(table)
            continue

        try:
            await _run_repl_turn(runner, user_input, ask_cb, app)
        except KeyboardInterrupt:
            active_console.print("\n[yellow]Turn cancelled by user.[/yellow]")
        except Exception as exc:
            active_console.print(f"\n[bold red]✖ Unexpected error: {escape(str(exc))}[/bold red]")
