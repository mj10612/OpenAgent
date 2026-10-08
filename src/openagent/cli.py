"""Command-line interface entrypoint for OpenAgent.

Provides subcommands `run`, `chat`, `models`, `sessions`, and `config`
with hierarchical configuration and rich terminal output.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections.abc import Sequence

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from openagent import __version__
from openagent.config import OpenAgentConfig, get_config_locations, load_config
from openagent.core.events import ErrorEvent
from openagent.core.presets import all_presets
from openagent.core.router import ProviderRouter
from openagent.runner import AgentRunner
from openagent.session.store import SessionStore
from openagent.tools.agent_tools import ThinkTool, TodoTool, WebFetchTool
from openagent.tools.fs import create_fs_tools
from openagent.tools.mcp.manager import MCPManager
from openagent.tools.registry import PermissionAction, ToolRegistry
from openagent.tools.shell import ShellTool
from openagent.tui.app import TUIApp, make_ask_callback, run_repl

_console: Console | None = None


def get_console() -> Console:
    """Return the global rich Console instance."""
    global _console
    if _console is None:
        _console = Console()
    return _console


class _OpenAgentArgumentParser(argparse.ArgumentParser):
    """Custom parser ensuring root default attributes are preserved when using subparsers."""

    def parse_args(  # type: ignore[override]
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        if namespace is None:
            namespace = argparse.Namespace(
                model=None,
                base_url=None,
                api_key=None,
                workspace=None,
                resume=None,
                yes=False,
                config=None,
            )
        return super().parse_args(args=args, namespace=namespace)


def _build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser with subcommands and global flags."""
    common_parser = _OpenAgentArgumentParser(add_help=False)
    common_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=argparse.SUPPRESS,
        help="Model identifier or provider-prefixed reference (e.g. gpt-4o, anthropic/claude-3-5-sonnet)",
    )
    common_parser.add_argument(
        "--base-url",
        type=str,
        default=argparse.SUPPRESS,
        help="Custom API base URL",
    )
    common_parser.add_argument(
        "--api-key",
        type=str,
        default=argparse.SUPPRESS,
        help="Custom API key credential",
    )
    common_parser.add_argument(
        "--workspace",
        "-w",
        type=str,
        default=argparse.SUPPRESS,
        help="Workspace root directory (defaults to current directory)",
    )
    common_parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default=argparse.SUPPRESS,
        help="Resume an existing conversation session by ID",
    )
    common_parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Automatically approve all tool executions without confirmation prompts",
    )
    common_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=argparse.SUPPRESS,
        help="Path to custom TOML configuration file",
    )

    root_parser = _OpenAgentArgumentParser(
        prog="openagent",
        description="OpenAgent: A model-agnostic AI coding agent for the terminal.",
        parents=[common_parser],
    )
    root_parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"openagent {__version__}",
    )

    subparsers = root_parser.add_subparsers(dest="subcommand", title="subcommands")

    # run subcommand
    run_parser = subparsers.add_parser(
        "run",
        parents=[common_parser],
        help="Execute a prompt with real-time streaming output",
    )
    run_parser.add_argument(
        "prompt",
        nargs="+",
        help="Instruction or query to execute",
    )

    # chat subcommand
    subparsers.add_parser(
        "chat",
        parents=[common_parser],
        help="Start an interactive REPL chat session",
    )

    # models subcommand
    subparsers.add_parser(
        "models",
        parents=[common_parser],
        help="List available provider presets and model options",
    )

    # sessions subcommand
    sessions_parser = subparsers.add_parser(
        "sessions",
        parents=[common_parser],
        help="List past conversation sessions and resumption commands",
    )
    sessions_parser.add_argument(
        "session_action", nargs="?", choices=["list", "delete", "rm", "prune"], default="list"
    )
    sessions_parser.add_argument("session_target", nargs="?")
    sessions_parser.add_argument("--older-than", type=float, dest="older_than_days")
    sessions_parser.add_argument("--clear", action="store_true")

    # config subcommand
    subparsers.add_parser(
        "config",
        parents=[common_parser],
        help="Display active configuration and config file locations",
    )

    return root_parser


def _cmd_models(console: Console) -> int:
    presets = all_presets()
    table = Table(
        title="Available Provider Presets",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Preset Name", style="bold cyan")
    table.add_column("Kind")
    table.add_column("Default Model")
    table.add_column("Base URL")
    table.add_column("Auth Scheme")
    table.add_column("Context Window", justify="right")

    for name, preset in sorted(presets.items()):
        table.add_row(
            name,
            preset.kind,
            preset.default_model or "-",
            preset.base_url,
            preset.auth.scheme,
            f"{preset.context_window:,}",
        )
    console.print(table)
    return 0


def _cmd_sessions(console: Console, args: argparse.Namespace | None = None) -> int:
    session_dir = os.environ.get("OPENAGENT_SESSION_DIR")
    store = SessionStore(storage_dir=session_dir)
    action = getattr(args, "session_action", "list")
    clear = getattr(args, "clear", False)
    if action in ("delete", "rm") or action == "prune" or clear:
        target = getattr(args, "session_target", None)
        if action in ("delete", "rm") and not target:
            raise ValueError("Session ID is required: openagent sessions delete <id>")
        older_than = getattr(args, "older_than_days", None)
        if action == "prune" and older_than is None and not clear:
            raise ValueError("Use sessions prune --older-than <days> or sessions --clear")
        resolved = store.resolve_session_id(str(target)) if action in ("delete", "rm") else None
        if not getattr(args, "yes", False) and not Confirm.ask(
            "Delete selected sessions?", console=console, default=False
        ):
            console.print("Cancelled.")
            return 0
        if resolved is not None:
            if not store.delete_session(resolved):
                raise FileNotFoundError(f"Session '{resolved}' was not found")
            console.print(f"Deleted session {resolved}", markup=False)
        else:
            count = store.cleanup_sessions(older_than_days=None if clear else older_than)
            console.print(f"Deleted {count} session(s).", markup=False)
        return 0
    sessions = store.list_sessions()

    if not sessions:
        console.print("[dim]No saved sessions found.[/dim]")
        return 0

    table = Table(
        title="Saved Conversation Sessions",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Session ID", style="bold cyan")
    table.add_column("Created")
    table.add_column("Updated")
    table.add_column("Model")
    table.add_column("Messages", justify="right")
    table.add_column("Title")

    for s in sessions:
        created_dt = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.created_at))
        updated_dt = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.updated_at))
        table.add_row(
            s.session_id,
            created_dt,
            updated_dt,
            escape(s.model) if s.model else "-",
            str(s.message_count),
            escape(s.title) if s.title else "-",
        )
    console.print(table)
    console.print(
        "\n[dim]Resume a session with: openagent --resume <session_id> or openagent run --resume <session_id> <prompt>[/dim]"
    )
    return 0


def _cmd_config(console: Console, args: argparse.Namespace) -> int:
    cfg = load_config(
        config_path=args.config,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        workspace=args.workspace,
        auto_approve=args.yes if args.yes else None,
        session_id=args.resume,
    )

    console.print(Panel("[bold cyan]Active Configuration[/bold cyan]", border_style="cyan"))

    files_table = Table(title="Configuration Files", show_header=True, header_style="bold")
    files_table.add_column("Scope")
    files_table.add_column("Path")
    files_table.add_column("Status")

    for scope, path, exists in get_config_locations():
        status = "[green]Found[/green]" if exists else "[dim]Not Found[/dim]"
        files_table.add_row(scope, str(path), status)
    if cfg.config_path:
        files_table.add_row("Explicit", str(cfg.config_path), "[green]Active[/green]")

    console.print(files_table)

    settings_table = Table(title="Runtime Settings", show_header=True, header_style="bold")
    settings_table.add_column("Setting", style="bold")
    settings_table.add_column("Value")

    settings_table.add_row("Model", cfg.model)
    settings_table.add_row("Base URL", cfg.base_url or "[dim](default for provider)[/dim]")
    settings_table.add_row("API Key", "********" if cfg.api_key else "[dim](from env/preset)[/dim]")
    settings_table.add_row(
        "Temperature",
        str(cfg.temperature) if cfg.temperature is not None else "[dim](default)[/dim]",
    )
    settings_table.add_row(
        "Top P", str(cfg.top_p) if cfg.top_p is not None else "[dim](default)[/dim]"
    )
    settings_table.add_row(
        "Max Tokens",
        str(cfg.max_tokens) if cfg.max_tokens is not None else "[dim](default)[/dim]",
    )
    settings_table.add_row("Max Tool Iterations", str(cfg.max_tool_iterations))
    settings_table.add_row("Auto Approve Tools", str(cfg.auto_approve))
    settings_table.add_row("Workspace", str(cfg.workspace))

    danger_summary = ", ".join(
        f"{k}={v.value if isinstance(v, PermissionAction) else v}"
        for k, v in cfg.danger_policy.items()
    )
    settings_table.add_row("Danger Policy", danger_summary)
    settings_table.add_row("MCP Servers", str(len(cfg.mcp_servers)))

    console.print(settings_table)
    return 0


def _build_runner(
    cfg: OpenAgentConfig,
    console: Console,
) -> tuple[AgentRunner, MCPManager | None]:
    """Construct an initialized AgentRunner from configuration."""
    router = ProviderRouter()
    model_ref = cfg.model
    provider_kwargs = {}
    if cfg.custom_provider is not None:
        model_ref = (
            cfg.model if cfg.model.startswith(("custom/", "custom:")) else f"custom/{cfg.model}"
        )
        provider_kwargs = cfg.custom_provider.provider_kwargs()
    provider = router.resolve(
        model_ref,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        **provider_kwargs,
    )

    registry = ToolRegistry(danger_policies=cfg.danger_policy)
    for tool in create_fs_tools(workspace_root=cfg.workspace):
        registry.register(tool)
    registry.register(ShellTool(workspace_root=cfg.workspace))
    registry.register(ThinkTool())
    registry.register(TodoTool())
    registry.register(WebFetchTool())

    mcp_manager: MCPManager | None = None
    if cfg.mcp_servers:
        mcp_manager = MCPManager(registry=registry, configs=cfg.mcp_servers)

    session_dir = os.environ.get("OPENAGENT_SESSION_DIR")
    session_store = SessionStore(storage_dir=session_dir)

    runner = AgentRunner(
        provider=provider,
        tools=registry,
        session_store=session_store,
        session_id=cfg.session_id,
        max_tool_iterations=cfg.max_tool_iterations,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
        context_window=cfg.context_window,
        workspace_root=cfg.workspace,
        extra_instructions=cfg.extra_instructions,
    )
    return runner, mcp_manager


async def _async_run(args: argparse.Namespace, console: Console) -> int:
    """Execute run or chat subcommands asynchronously."""
    cfg = load_config(
        config_path=args.config,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        workspace=args.workspace,
        auto_approve=args.yes if args.yes else None,
        session_id=args.resume,
    )

    runner, mcp_manager = _build_runner(cfg, console)

    try:
        if mcp_manager is not None:
            await mcp_manager.start()

        if args.subcommand == "run":
            prompt = " ".join(args.prompt) if isinstance(args.prompt, list) else str(args.prompt)
            app = TUIApp(console=console)
            ask_cb = make_ask_callback(console=console, auto_approve=cfg.auto_approve)
            failed = False
            async for event in runner.run_turn(prompt, ask_callback=ask_cb):
                app.render_event(event)
                failed = failed or isinstance(event, ErrorEvent)
            return 1 if failed else 0
        else:
            # Interactive chat REPL
            await run_repl(runner, console=console, auto_approve=cfg.auto_approve)
            return 0
    finally:
        if mcp_manager is not None:
            await mcp_manager.stop()


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line application entrypoint."""
    if argv is None:
        argv = sys.argv[1:]

    args_list = list(argv)

    # If no subcommand was provided but a prompt positional argument is present, default to 'run'
    known_subcommands = {"run", "chat", "models", "sessions", "config"}
    val_options = {
        "--model",
        "-m",
        "--base-url",
        "--api-key",
        "--workspace",
        "-w",
        "--resume",
        "-r",
        "--config",
        "-c",
    }

    has_subcommand = False
    first_positional_idx: int | None = None
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if arg in ("--help", "-h", "--version", "-v"):
            has_subcommand = True
            break
        if arg in val_options:
            i += 2
            continue
        if any(arg.startswith(f"{opt}=") for opt in val_options):
            i += 1
            continue
        if arg.startswith("-"):
            i += 1
            continue
        if arg in known_subcommands:
            has_subcommand = True
            break
        if first_positional_idx is None:
            first_positional_idx = i
            break
        i += 1

    if not has_subcommand and first_positional_idx is not None:
        args_list.insert(first_positional_idx, "run")

    parser = _build_parser()
    try:
        args = parser.parse_args(args_list)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    console = get_console()

    # If no subcommand specified, default to interactive chat
    if not args.subcommand:
        args.subcommand = "chat"

    try:
        if args.subcommand == "models":
            return _cmd_models(console)
        elif args.subcommand == "sessions":
            return _cmd_sessions(console, args)
        elif args.subcommand == "config":
            return _cmd_config(console, args)
        elif args.subcommand in ("run", "chat"):
            return asyncio.run(_async_run(args, console))
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        return 130
    except Exception as exc:
        console.print(f"\n[bold red]Error:[/bold red] {escape(str(exc))}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
