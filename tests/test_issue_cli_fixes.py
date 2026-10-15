import json

import pytest

from openagent.cli import _build_parser, _tool_policy_args, main
from openagent.config import OpenAgentConfig, load_config, save_config
from openagent.doctor import diagnose
from openagent.tools.registry import PermissionAction


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
