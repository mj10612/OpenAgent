"""Offline installation checks, without executing tools or making model requests."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

from openagent.config import OpenAgentConfig


def diagnose(cfg: OpenAgentConfig) -> list[dict[str, Any]]:
    from openagent.core.router import ModelReference, ProviderRouter

    checks: list[dict[str, Any]] = []
    def add(name: str, status: str, detail: str, required: bool = True) -> None:
        checks.append({'name': name, 'status': status, 'detail': detail, 'required': required})

    ref = ModelReference.parse(cfg.model)
    router = ProviderRouter()
    preset = router._find_preset(ref.provider_hint or ref.model_name)
    if preset is None:
        family = 'anthropic' if ref.model_name.startswith('claude') else 'gemini' if ref.model_name.startswith('gemini') else 'ollama' if ':' in ref.model_name else 'openai'
        preset = router._find_preset(family)
    custom = cfg.custom_provider is not None or ref.provider_hint == 'custom'
    auth = cfg.custom_provider.auth_scheme if cfg.custom_provider else preset.auth.scheme if preset else 'bearer'
    credential = bool(cfg.api_key or (os.environ.get('CUSTOM_API_KEY') if custom else preset.auth.resolve_key() if preset else os.environ.get('OPENAI_API_KEY')))
    add('provider', 'ok', f'Model: {cfg.model}')
    add('credential', 'ok' if auth == 'none' or credential else 'error', 'Configured' if credential else 'Not required' if auth == 'none' else 'Required credential missing')
    add('runtime', 'ok', f'Python {sys.version_info.major}.{sys.version_info.minor}')
    shell = shutil.which('powershell' if os.name == 'nt' else 'sh')
    add('shell', 'ok' if shell else 'error', 'Available' if shell else 'Required shell missing')
    for server in cfg.mcp_servers:
        if server.transport == 'stdio':
            executable = shutil.which(server.command)
            add(f'mcp:{server.name}:command', 'ok' if executable else 'error', 'Available' if executable else 'Executable missing')
            if server.cwd:
                add(f'mcp:{server.name}:cwd', 'ok' if Path(server.cwd).is_dir() else 'error', 'Directory exists' if Path(server.cwd).is_dir() else 'Working directory missing')
        else:
            add(f'mcp:{server.name}:remote', 'warning', 'Remote health unchecked in offline mode', False)
    directory = Path(os.environ.get('OPENAGENT_SESSION_DIR', str(Path.home() / '.openagent' / 'sessions')))
    ancestor = directory
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    writable = ancestor.is_dir() and os.access(ancestor, os.W_OK) and bool(ancestor.stat().st_mode & 0o222)
    add('sessions', 'ok' if writable else 'error', 'Storage parent writable' if writable else 'Storage parent is not writable')
    return checks
