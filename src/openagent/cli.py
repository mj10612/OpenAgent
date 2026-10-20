"""Command-line interface entrypoint for OpenAgent.

Provides subcommands `run`, `chat`, `models`, `sessions`, and `config`
with hierarchical configuration and rich terminal output.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from openagent import __version__
from openagent.config import OpenAgentConfig, get_config_locations, load_config
from openagent.core.events import DoneEvent, ErrorEvent, TextDelta, ToolResultEvent
from openagent.core.presets import all_presets
from openagent.core.router import ProviderRouter
from openagent.core.types import FinishReason, Message, Usage
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


class _ToolPolicyAction(argparse.Action):
    """Collect repeated policies across the shared root/subparser action."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.values: list[str] = []

    def __call__(self, parser, namespace, values, option_string=None):
        self.values.append(values)
        setattr(namespace, self.dest, list(self.values))


class _OpenAgentArgumentParser(argparse.ArgumentParser):
    """Custom parser ensuring root default attributes are preserved when using subparsers."""

    def parse_args(  # type: ignore[override]
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        for action in self._actions:
            if isinstance(action, _ToolPolicyAction):
                action.values.clear()
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
    common_parser.add_argument('--tool-policy', action=_ToolPolicyAction, default=argparse.SUPPRESS, metavar='NAME=ACTION', help='Exact tool allow/ask/deny override; repeatable')

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
    output = run_parser.add_mutually_exclusive_group()
    output.add_argument('--json', action='store_true', help='Print one versioned JSON result')
    output.add_argument('--jsonl', action='store_true', help='Print versioned JSON events and a terminal result')
    run_parser.add_argument('--image', action='append', default=[], help='Attach local PNG/JPEG/GIF/WebP, max 5 MiB each, four images')

    # chat subcommand
    subparsers.add_parser(
        "chat",
        parents=[common_parser],
        help="Start an interactive REPL chat session",
    )

    # models subcommand
    models_parser = subparsers.add_parser(
        "models",
        parents=[common_parser],
        help="List available provider presets and model options",
    )
    models_parser.add_argument('--discover', action='store_true', help='Discover models from the configured provider')
    models_parser.add_argument('--refresh', action='store_true', help='Refresh cached model metadata')
    models_parser.add_argument('--json', action='store_true')

    # sessions subcommand
    sessions_parser = subparsers.add_parser(
        "sessions",
        parents=[common_parser],
        help="List past conversation sessions and resumption commands",
    )
    sessions_parser.add_argument(
        "session_action", nargs="?", choices=["list", "delete", "rm", "prune", "export", "import", "convert"], default="list"
    )
    sessions_parser.add_argument("session_target", nargs="?")
    sessions_parser.add_argument("--older-than", type=float, dest="older_than_days")
    sessions_parser.add_argument("--clear", action="store_true")
    sessions_parser.add_argument('--output', '-o')
    sessions_parser.add_argument('--format', choices=['json', 'jsonl', 'markdown'], default='jsonl')
    sessions_parser.add_argument('--redact', action='store_true')
    sessions_parser.add_argument('--new-id', action='store_true')

    # config subcommand
    subparsers.add_parser(
        "config",
        parents=[common_parser],
        help="Display active configuration and config file locations",
    )
    doctor_parser = subparsers.add_parser('doctor', parents=[common_parser], help='Offline provider, MCP and platform readiness checks (Windows/Linux)')
    doctor_parser.add_argument('--json', action='store_true')
    doctor_parser.add_argument('--network', action='store_true', help='Explicitly discover models only at the configured provider endpoint')

    return root_parser


def _tool_policy_args(args: argparse.Namespace) -> dict[str, PermissionAction] | None:
    policies: dict[str, PermissionAction] = {}
    for value in getattr(args, 'tool_policy', []):
        name, separator, action = value.partition('=')
        if not separator or not name or name in policies:
            raise ValueError('Tool policy requires unique NAME=allow|ask|deny entries')
        try:
            policies[name] = PermissionAction(action)
        except ValueError:
            raise ValueError(f'Invalid tool policy action for {name}') from None
    return policies or None


def _load_args_config(args: argparse.Namespace) -> OpenAgentConfig:
    return load_config(config_path=args.config, model=args.model, base_url=args.base_url,
                       api_key=args.api_key, workspace=args.workspace,
                       auto_approve=args.yes if args.yes else None, session_id=args.resume,
                       tool_policy=_tool_policy_args(args))


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
    if action in ('export', 'import', 'convert'):
        assert args is not None
        target = getattr(args, 'session_target', None)
        if not target:
            raise ValueError('Session ID or input path is required')
        if action == 'export':
            if not args.output:
                raise ValueError('sessions export requires --output PATH')
            store.export_session(store.resolve_session_id(target), args.output, format=args.format, redact=args.redact)
            console.print('Session exported.', markup=False)
        elif action == 'import':
            imported = store.import_session(target, new_id=args.new_id)
            console.print(f'Imported session {imported}', markup=False)
        else:
            if not args.output:
                raise ValueError('sessions convert requires --output PATH')
            # Validate and migrate without modifying the live session archive.
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                temporary = SessionStore(storage_dir=directory)
                imported = temporary.import_session(target, new_id=False)
                temporary.export_session(imported, args.output, format=args.format, redact=args.redact)
            console.print('Session converted.', markup=False)
        return 0
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
    cfg = _load_args_config(args)

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
    settings_table.add_row('Tool Policy', ', '.join(f'{k}={v.value}' for k, v in cfg.tool_policy.items()))
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

    registry = ToolRegistry(danger_policies=cfg.danger_policy, checkpoint_enabled=cfg.checkpoint_enabled)
    for name, action in cfg.tool_policy.items():
        registry.set_tool_policy(name, action)
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
    cfg = _load_args_config(args)

    runner, mcp_manager = _build_runner(cfg, console)

    try:
        if mcp_manager is not None:
            await mcp_manager.start()
        for name in cfg.tool_policy:
            if runner.tools.get(name) is None:
                Console(stderr=True).print(f'Warning: configured tool is not registered: {name}', markup=False)

        if args.subcommand == "run":
            prompt: str | Message = " ".join(args.prompt) if isinstance(args.prompt, list) else str(args.prompt)
            if getattr(args, 'image', []):
                from openagent.attachments import load_images
                prompt = Message.user(str(prompt), images=load_images(args.image, runner.provider))
            machine = getattr(args, 'json', False) or getattr(args, 'jsonl', False)
            app = TUIApp(console=console)
            ask_cb = make_ask_callback(console=Console(stderr=True) if machine else console, auto_approve=cfg.auto_approve)
            original_ask = ask_cb
            async def preview_ask(call):
                tool = runner.tools.get(call.name)
                if tool is not None and not cfg.auto_approve:
                    preview = await runner.tools.changes.preview(tool, call.arguments)
                    Console(stderr=True).print(preview, markup=False)
                decision = original_ask(call)
                import inspect
                return bool(await decision) if inspect.isawaitable(decision) else bool(decision)
            ask_cb = preview_ask
            if machine and not sys.stdin.isatty() and not cfg.auto_approve:
                async def deny_ask(call):
                    return False
                ask_cb = deny_ask
            failed = False
            errors = []
            assistant = ''
            final_usage = Usage()
            outcome = 'success'
            completed = False
            finish_reason = None
            sequence = 0
            try:
                async for event in runner.run_turn(prompt, ask_callback=ask_cb):
                    sequence += 1
                    if getattr(args, 'jsonl', False):
                        _emit_json(_redact({'schema_version': 1, 'sequence': sequence, 'type': type(event).__name__, 'event': _json_data(event)}, cfg))
                    elif not machine:
                        app.render_event(event)
                    if isinstance(event, TextDelta):
                        assistant += event.text
                    if isinstance(event, DoneEvent):
                        completed = True
                        finish_reason = event.finish_reason.value
                        final_usage = final_usage + event.usage
                        if event.message:
                            assistant = event.message.text
                        if event.finish_reason != FinishReason.STOP and not failed:
                            failed = True
                            outcome = ('filtered' if event.finish_reason == FinishReason.CONTENT_FILTER
                                       else 'incomplete')
                            errors.append(f'Provider finished with {finish_reason}')
                    if isinstance(event, ErrorEvent):
                        failed = True
                        errors.append(str(event.error))
                        outcome = 'iteration_limit' if 'iteration' in str(event.error).lower() else 'error'
                    if isinstance(event, ToolResultEvent) and event.is_error:
                        failed = True
                        errors.append(f'Tool {event.tool_name} failed')
                        outcome = 'partial_failure'
            except (asyncio.CancelledError, KeyboardInterrupt):
                failed = True
                outcome = 'cancelled'
            except Exception as exc:
                failed = True
                outcome = 'error'
                errors.append(str(exc))
            if not completed and not failed:
                failed = True
                outcome = 'error'
                errors.append('Turn ended without a completed response')
            code = 130 if outcome == 'cancelled' else 1 if failed else 0
            if machine:
                usage = getattr(runner, 'usage', final_usage)
                result = {'schema_version': 1, 'type': 'result', 'outcome': outcome, 'exit_code': code,
                          'finish_reason': finish_reason, 'assistant': assistant,
                          'usage': _json_data(usage), 'model': runner.model,
                          'session': runner.session_id, 'changes': runner.tools.changes.describe(), 'errors': errors}
                if getattr(args, 'jsonl', False):
                    result['sequence'] = sequence + 1
                _emit_json(_redact(result, cfg))
            return code
        else:
            # Interactive chat REPL
            await run_repl(runner, console=console, auto_approve=cfg.auto_approve)
            return 0
    finally:
        try:
            if mcp_manager is not None:
                await mcp_manager.stop()
        except Exception:
            Console(stderr=True).print('Warning: MCP cleanup failed.', markup=False)
        finally:
            try:
                await runner.provider.close()
            except Exception:
                Console(stderr=True).print('Warning: provider cleanup failed.', markup=False)


def _json_data(value):
    if isinstance(value, BaseException):
        return str(value)
    if is_dataclass(value):
        return {key: _json_data(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _json_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_data(item) for item in value]
    return value


def _emit_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _redact(value, cfg: OpenAgentConfig):
    credential_suffixes = ('API_KEY', 'ACCESS_TOKEN', 'AUTH_TOKEN')
    secrets = [cfg.api_key] if cfg.api_key else []
    secrets.extend(v for k, v in os.environ.items() if k.endswith(credential_suffixes) and v)
    for server in cfg.mcp_servers:
        secrets.extend((server.headers or {}).values())
        secrets.extend(v for k, v in (server.env or {}).items() if k.endswith(credential_suffixes) and v)
    def clean(item):
        if isinstance(item, str):
            for secret in secrets:
                item = item.replace(secret, '[REDACTED]')
            return item
        if isinstance(item, dict):
            return {key: clean(data) for key, data in item.items()}
        if isinstance(item, list):
            return [clean(data) for data in item]
        return item
    return clean(value)


async def _discover(args):
    from openagent.core.model_metadata import discover_models
    cfg = _load_args_config(args)
    kwargs = cfg.custom_provider.provider_kwargs() if cfg.custom_provider else {}
    ref = f'custom/{cfg.model}' if cfg.custom_provider and not cfg.model.startswith(('custom/', 'custom:')) else cfg.model
    provider = ProviderRouter().resolve(ref, api_key=cfg.api_key, base_url=cfg.base_url, **kwargs)
    try:
        return await discover_models(provider, refresh=args.refresh if hasattr(args, 'refresh') else True)
    finally:
        await provider.close()


def _cmd_doctor(args, console):
    from openagent.doctor import diagnose
    checks = diagnose(_load_args_config(args))
    if args.network:
        try:
            models = asyncio.run(_discover(args))
            checks.append({'name': 'network', 'status': 'ok', 'detail': f'{len(models)} models discovered', 'required': False})
        except Exception:
            checks.append({'name': 'network', 'status': 'warning', 'detail': 'Model discovery unavailable', 'required': False})
    code = int(any(c['status'] == 'error' and c['required'] for c in checks))
    if args.json:
        _emit_json({'schema_version': 1, 'checks': checks, 'exit_code': code})
    else:
        for check in checks:
            console.print(f"{check['status']}: {check['name']}: {check['detail']}", markup=False)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line application entrypoint."""
    if argv is None:
        argv = sys.argv[1:]

    args_list = list(argv)

    # If no subcommand was provided but a prompt positional argument is present, default to 'run'
    known_subcommands = {"run", "chat", "models", "sessions", "config", "doctor"}
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
        "--tool-policy",
    }

    has_subcommand = False
    first_positional_idx: int | None = None
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if arg == "--":
            if i + 1 < len(args_list):
                first_positional_idx = i
            break
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
            if args.discover or args.refresh:
                models = asyncio.run(_discover(args))
                if args.json:
                    _emit_json({'schema_version': 1, 'models': [_json_data(model) for model in models]})
                else:
                    table = Table(title='Discovered Models')
                    for column in ('Model', 'Context', 'Output', 'Vision', 'Tools', 'Source'):
                        table.add_column(column)
                    for model in models:
                        source = 'discovered' if model.context_window is not None or model.max_output is not None else 'heuristic'
                        table.add_row(model.id, str(model.context_window or 'unknown'), str(model.max_output or 'unknown'), str(model.supports_vision), str(model.supports_tools), source)
                    console.print(table)
                return 0
            if args.json:
                _emit_json({'schema_version': 1, 'presets': [_json_data(preset) for _, preset in sorted(all_presets().items())]})
                return 0
            return _cmd_models(console)
        elif args.subcommand == 'doctor':
            return _cmd_doctor(args, console)
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
        if getattr(args, 'json', False) or getattr(args, 'jsonl', False):
            _emit_json({'schema_version': 1, 'type': 'result', 'outcome': 'error', 'exit_code': 1, 'errors': [str(exc)]})
            return 1
        console.print(f"\n[bold red]Error:[/bold red] {escape(str(exc))}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
