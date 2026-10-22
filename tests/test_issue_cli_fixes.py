import json

import pytest

from openagent.cli import _build_parser, _tool_policy_args, main
from openagent.config import OpenAgentConfig, load_config, save_config
from openagent.doctor import diagnose
from openagent.tools.registry import PermissionAction


@pytest.mark.parametrize('short_first', [True, False])
def test_overlapping_credentials_redacted_from_original_text(monkeypatch, short_first):
    from openagent.cli import _redact
    from openagent.tools.mcp.client import MCPServerConfig

    monkeypatch.setenv('AUDIT_API_KEY', 'audit-prefix-private-suffix')
    cfg = OpenAgentConfig(
        api_key='audit-prefix' if short_first else 'audit-prefix-private-suffix',
        mcp_servers=[MCPServerConfig(name='fixture', command='fake',
            env={'SERVICE_API_KEY': 'audit-prefix', 'EMPTY_API_KEY': ''})],
    )
    assert _redact({'nested': ['audit-prefix-private-suffix audit-prefix']}, cfg) == {
        'nested': ['[REDACTED] [REDACTED]'],
    }


def test_mixed_policy_positions_preserve_deny_and_reset_between_parses():
    parser = _build_parser()
    args = parser.parse_args([
        '--tool-policy', 'delete_file=deny', 'run', 'hello',
        '--tool-policy', 'write_file=allow', '--yes',
    ])
    assert _tool_policy_args(args) == {
        'delete_file': PermissionAction.DENY, 'write_file': PermissionAction.ALLOW,
    }
    assert _tool_policy_args(parser.parse_args(['run', 'hello'])) is None


def test_mixed_policy_positions_reject_duplicate_tool():
    args = _build_parser().parse_args([
        '--tool-policy=delete_file=deny', 'run', 'hello',
        '--tool-policy', 'delete_file=allow',
    ])
    with pytest.raises(ValueError, match='unique'):
        _tool_policy_args(args)


def test_doctor_uses_deepseek_auth_for_inferred_family(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'fake-key')
    checks = diagnose(OpenAgentConfig(model='deepseek-special'))
    assert next(c for c in checks if c['name'] == 'credential')['status'] == 'ok'


@pytest.mark.parametrize('model', ['bedrock/model', 'azure/model', 'unknown/model'])
def test_doctor_rejects_unroutable_providers(model):
    checks = diagnose(OpenAgentConfig(model=model))
    assert next(c for c in checks if c['name'] == 'provider')['status'] == 'error'


def test_models_json_default_presets(capsys):
    assert main(['models', '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['schema_version'] == 1
    assert any(preset['name'] == 'openai' for preset in result['presets'])


def test_config_preserves_resume_target(tmp_path, monkeypatch):
    monkeypatch.setattr('openagent.config.find_global_config_path', lambda: None)
    monkeypatch.setattr('openagent.config.find_project_config_path', lambda: None)
    path = tmp_path / 'config.toml'
    save_config(OpenAgentConfig(session_id='resume-me'), path)
    assert load_config(path).session_id == 'resume-me'


@pytest.mark.parametrize('key', ['API_KEY', 'SERVICE_API_KEY', 'ACCESS_TOKEN', 'AUTH_TOKEN'])
def test_machine_redacts_mcp_env_credentials(key):
    from openagent.cli import _redact
    from openagent.tools.mcp.client import MCPServerConfig

    cfg = OpenAgentConfig(mcp_servers=[MCPServerConfig(
        name='fixture', command='fake',
        env={key: 'fake-mcp-credential', 'MODE': 'development', 'EMPTY_API_KEY': ''},
    )])
    payload = {'events': [{'output': 'failed fake-mcp-credential'}], 'mode': 'development'}
    assert _redact(payload, cfg) == {
        'events': [{'output': 'failed [REDACTED]'}], 'mode': 'development',
    }


@pytest.mark.parametrize('mode', ['--json', '--jsonl'])
def test_machine_output_masks_mcp_env_credentials_in_errors(mode, monkeypatch, capsys):
    from openagent.core.events import ErrorEvent, TextDelta, ToolResultEvent
    from openagent.providers.openai_compat import OpenAICompatProvider
    from openagent.runner import AgentRunner
    from openagent.tools.mcp.client import MCPServerConfig

    credential = 'fake-mcp-env-token'
    cfg = OpenAgentConfig(mcp_servers=[MCPServerConfig(
        name='fixture', command='fake', env={'SERVICE_API_KEY': credential},
    )])
    provider = OpenAICompatProvider('https://unused.test', 'gpt-4o')
    runner = AgentRunner(provider)

    async def turn(*args, **kwargs):
        yield TextDelta(f'diagnostic {credential}')
        yield ToolResultEvent('call', 'fixture__tool', f'failed {credential}', is_error=True)
        yield ErrorEvent(RuntimeError(f'server error {credential}'))

    monkeypatch.setattr(runner, 'run_turn', turn)
    monkeypatch.setattr('openagent.cli._load_args_config', lambda args: cfg)
    monkeypatch.setattr('openagent.cli._build_runner', lambda cfg, console: (runner, None))
    assert main(['run', 'fixture', mode]) == 1
    output = capsys.readouterr().out
    assert credential not in output
    rows = [json.loads(line) for line in output.splitlines()]
    assert rows[-1]['errors'][-1] == 'server error [REDACTED]'
    assert rows[-1]['assistant'] == 'diagnostic [REDACTED]'
    if mode == '--jsonl':
        assert next(row for row in rows if row['type'] == 'ToolResultEvent')['event']['output'] == 'failed [REDACTED]'
