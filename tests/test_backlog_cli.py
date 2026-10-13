import json

import pytest

from openagent.cli import main
from openagent.config import OpenAgentConfig, load_config, save_config


@pytest.mark.parametrize('kwargs', [
    {'auto_approve': 'false'}, {'max_tool_iterations': 0},
    {'max_tokens': True}, {'temperature': float('nan')},
    {'top_p': 2}, {'danger_policy': {'execute': 'denny'}},
    {'danger_policy': {'unknown': 'allow'}},
])
def test_invalid_runtime_config_rejected(kwargs):
    with pytest.raises(ValueError):
        OpenAgentConfig(**kwargs)


def test_tool_policy_roundtrip_and_partial_override(tmp_path):
    path = tmp_path / 'config.toml'
    save_config(OpenAgentConfig(tool_policy={'write_file': 'allow', 'delete_file': 'deny'}), path)
    cfg = load_config(path, tool_policy={'write_file': 'ask'})
    assert cfg.tool_policy['write_file'].value == 'ask'
    assert cfg.tool_policy['delete_file'].value == 'deny'


def test_invalid_env_is_not_silently_ignored(monkeypatch):
    monkeypatch.setenv('OPENAGENT_MAX_TOKENS', 'bad')
    with pytest.raises(ValueError, match='OPENAGENT_MAX_TOKENS'):
        load_config()


def test_unknown_config_key_reports_source(tmp_path):
    path = tmp_path / 'invalid.toml'
    path.write_text('auto_aprove = true\n')
    with pytest.raises(ValueError, match='auto_aprove'):
        load_config(path)


def test_doctor_json_offline_missing_executable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('OPENAGENT_SESSION_DIR', str(tmp_path / 'sessions'))
    path = tmp_path / 'config.toml'
    path.write_text('model="ollama/llama3.2"\n[[mcp_servers]]\nname="bad"\ncommand="nonexistent-openagent-executable"\n')
    assert main(['doctor', '--json', '--config', str(path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert any(c['status'] == 'error' for c in payload['checks'])


def test_images_reject_invalid_mime(tmp_path):
    from openagent.attachments import load_images
    path = tmp_path / 'fake.png'
    path.write_text('not an image')
    with pytest.raises(ValueError, match='image'):
        load_images([str(path)], type('Provider', (), {'supports_vision': True, 'model': 'gpt-4o'})())


@pytest.mark.asyncio
@pytest.mark.parametrize('target', ['ollama/llava', 'gpt-4.1'])
async def test_model_switch_preserves_explicit_context(monkeypatch, target):
    from test_repl_commands import setup_input

    from openagent.providers.openai_compat import OpenAICompatProvider
    from openagent.runner import AgentRunner
    from openagent.tui.app import run_repl
    provider = OpenAICompatProvider('https://private.test/v1', 'gpt-4o')
    runner = AgentRunner(provider, context_window=7777, max_tokens=100)
    console, _ = setup_input(monkeypatch, [f'/model {target}', '/exit'])
    try:
        await run_repl(runner, console=console)
        assert runner.context_window == 7777
        assert runner.max_tokens == 100
    finally:
        await runner.provider.close()


@pytest.mark.parametrize('mode', ['--json', '--jsonl'])
def test_machine_run_valid_stdout_and_provider_cleanup(monkeypatch, tmp_path, capsys, mode):
    from unittest.mock import AsyncMock

    from openagent.core.events import DoneEvent, TextDelta
    from openagent.core.types import Message, Usage
    from openagent.providers.openai_compat import OpenAICompatProvider
    from openagent.runner import AgentRunner
    provider = OpenAICompatProvider('https://unused.test', 'gpt-4o')
    close = AsyncMock(wraps=provider.close)
    provider.close = close
    runner = AgentRunner(provider)
    async def turn(*args, **kwargs):
        yield TextDelta('한글\n[red]😊')
        yield DoneEvent(message=Message.assistant('한글\n[red]😊'), usage=Usage(3, 4))
    runner.run_turn = turn
    monkeypatch.setattr('openagent.cli._build_runner', lambda cfg, console: (runner, None))
    assert main(['run', 'hello', mode, '-w', str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    payloads = [json.loads(line) for line in lines]
    assert payloads[-1]['assistant'] == '한글\n[red]😊'
    assert payloads[-1]['outcome'] == 'success'
    assert len(lines) == (1 if mode == '--json' else 3)
    close.assert_awaited_once()


@pytest.mark.parametrize('events, expected, code', [('error', 'error', 1), ('limit', 'iteration_limit', 1), ('tool', 'partial_failure', 1), ('cancel', 'cancelled', 130)])
def test_machine_failure_outcomes(monkeypatch, capsys, events, expected, code):
    import asyncio

    from openagent.core.events import DoneEvent, ErrorEvent, ToolResultEvent
    from openagent.core.types import Message
    from openagent.providers.openai_compat import OpenAICompatProvider
    from openagent.runner import AgentRunner
    provider = OpenAICompatProvider('https://unused.test', 'gpt-4o')
    runner = AgentRunner(provider)
    async def turn(*args, **kwargs):
        if events == 'cancel':
            raise asyncio.CancelledError
        if events in ('error', 'limit'):
            yield ErrorEvent('maximum tool iterations reached' if events == 'limit' else 'failed')
        else:
            yield ToolResultEvent('a', 'write_file', 'bad', is_error=True)
            yield DoneEvent(message=Message.assistant('partial'))
    monkeypatch.setattr(runner, 'run_turn', turn)
    monkeypatch.setattr('openagent.cli._build_runner', lambda cfg, console: (runner, None))
    assert main(['run', 'hello', '--json']) == code
    assert json.loads(capsys.readouterr().out)['outcome'] == expected


def test_images_supported_content_and_limits(tmp_path):
    import base64

    from openagent.attachments import load_images
    provider = type('Provider', (), {'supports_vision': True, 'model': 'gpt-4o'})()
    image = tmp_path / '한글 image.png'
    data = b'\x89PNG\r\n\x1a\n' + b'content'
    image.write_bytes(data)
    result = load_images([str(image)], provider)
    assert result[0].mime_type == 'image/png'
    assert base64.b64decode(result[0].data) == data
    with pytest.raises(ValueError, match=r'four|4'):
        load_images([str(image)] * 5, provider)
    image.write_bytes(b'\x89PNG\r\n\x1a\n' + b'x' * (5 * 1024 * 1024))
    with pytest.raises(ValueError, match='5 MiB'):
        load_images([str(image)], provider)


def test_tool_policy_applies_and_deny_survives_yes(tmp_path):
    from rich.console import Console

    from openagent.cli import _build_runner
    from openagent.tools.registry import PermissionAction
    cfg = OpenAgentConfig(workspace=tmp_path, auto_approve=True, tool_policy={'write_file': 'allow', 'delete_file': 'deny'})
    runner, _ = _build_runner(cfg, Console())
    assert runner.tools.get_policy('write_file') == PermissionAction.ALLOW
    assert runner.tools.get_policy('delete_file') == PermissionAction.DENY


def test_doctor_missing_key_and_invalid_cwd_are_secret_free(tmp_path, monkeypatch):
    from openagent.doctor import diagnose
    from openagent.tools.mcp.client import MCPServerConfig
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    cfg = OpenAgentConfig(model='gpt-4o', mcp_servers=[MCPServerConfig(name='test', command='python', cwd=str(tmp_path / 'absent'), env={'SECRET': 'do-not-print'})])
    checks = diagnose(cfg)
    assert any(c['name'] == 'credential' and c['status'] == 'error' for c in checks)
    assert any(c['name'].endswith(':cwd') and c['status'] == 'error' for c in checks)
    assert 'do-not-print' not in json.dumps(checks)


def test_session_cli_export_import_convert_and_markdown(tmp_path, monkeypatch):
    from openagent.core.types import Message
    from openagent.session.store import SessionStore
    archive = tmp_path / 'sessions'
    monkeypatch.setenv('OPENAGENT_SESSION_DIR', str(archive))
    store = SessionStore(archive)
    sid = store.create_session(model='fake')
    store.save_messages(sid, [Message.user('한글'), Message.assistant('answer')])
    exported = tmp_path / 'export.jsonl'
    converted = tmp_path / 'converted.json'
    markdown = tmp_path / 'transcript.md'
    assert main(['sessions', 'export', sid, '--output', str(exported)]) == 0
    assert main(['sessions', 'convert', str(exported), '--format', 'json', '--output', str(converted)]) == 0
    assert json.loads(converted.read_text(encoding='utf8'))['metadata']['session_id'] == sid
    assert main(['sessions', 'import', str(converted), '--new-id']) == 0
    assert len(store.list_sessions()) == 2
    assert main(['sessions', 'export', sid, '--format', 'markdown', '--output', str(markdown)]) == 0
    assert '한글' in markdown.read_text(encoding='utf8')


def test_machine_redacts_configured_credentials():
    from openagent.cli import _redact
    cfg = OpenAgentConfig(api_key='secret-example')
    assert _redact({'errors': ['failed secret-example'], 'assistant': 'secret-example'}, cfg) == {'errors': ['failed [REDACTED]'], 'assistant': '[REDACTED]'}


@pytest.mark.parametrize('table', ['mcp_servers="bad"', '[[mcp_servers]]\nname="test"\ncommand="python"\ntimeout="5"', '[[mcp_servers]]\nname="test"\nargs=[5]'])
def test_invalid_mcp_settings_fail_before_execution(tmp_path, table):
    path = tmp_path / 'invalid.toml'
    path.write_text(table)
    with pytest.raises(ValueError):
        load_config(path)
