import json

import pytest

from openagent import cli
from openagent.config import OpenAgentConfig, save_config
from openagent.tools.mcp.client import MCPServerConfig


@pytest.mark.parametrize('mode', ['--json', '--jsonl'])
@pytest.mark.parametrize('stage', ['attachment', 'runner', 'mcp'])
def test_startup_errors_redact_loaded_configuration(mode, stage, tmp_path, monkeypatch, capsys):
    secret = 'fake-configured-secret'
    path = tmp_path / 'config.toml'
    save_config(OpenAgentConfig(model='gpt-4o', api_key=secret,
        mcp_servers=[MCPServerConfig(name='fixture', command='fake',
            env={'SERVICE_API_KEY': 'fake-mcp-secret'})]), path)
    if stage == 'runner':
        def fail(*args):
            raise RuntimeError(f'initialization {secret} fake-mcp-secret')
        monkeypatch.setattr(cli, '_build_runner', fail)
    elif stage == 'mcp':
        async def fail_start(*args):
            raise RuntimeError(f'mcp startup {secret} fake-mcp-secret')
        monkeypatch.setattr('openagent.tools.mcp.manager.MCPManager.start', fail_start)
    else:
        # Skip real MCP startup so attachment validation is reached without any process/network.
        async def no_start(*args):
            pass
        monkeypatch.setattr('openagent.tools.mcp.manager.MCPManager.start', no_start)
    args = ['run', 'hello', mode, '--config', str(path)]
    if stage == 'attachment':
        args += ['--image', str(tmp_path / secret)]
    assert cli.main(args) == 1
    output = capsys.readouterr().out
    assert secret not in output and 'fake-mcp-secret' not in output
    result = json.loads(output)
    assert result['outcome'] == 'error' and result['exit_code'] == 1
    assert '[REDACTED]' in result['errors'][0]


@pytest.mark.parametrize('mode', ['--json', '--jsonl'])
@pytest.mark.parametrize('stage', ['prompt', 'config'])
def test_errors_before_loaded_config_redact_cli_and_environment(mode, stage, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('AUDIT_API_KEY', 'fake-environment-secret')
    secret = 'fake-cli-secret'
    args = ['run', mode, '--api-key', secret]
    if stage == 'prompt':
        args += ['--prompt-file', str(tmp_path / (secret + '-fake-environment-secret'))]
    else:
        def fail(*args, **kwargs):
            raise ValueError(f'bad config {secret} fake-environment-secret')
        monkeypatch.setattr(cli, 'load_config', fail)
        args += ['hello']
    assert cli.main(args) == 1
    output = capsys.readouterr().out
    assert secret not in output and 'fake-environment-secret' not in output
    assert json.loads(output)['outcome'] == 'error'
