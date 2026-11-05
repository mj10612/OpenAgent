import pytest

from openagent.config import OpenAgentConfig, load_config
from openagent.tools.mcp.client import MCPServerConfig
from openagent.tools.mcp.manager import MCPManager


def configs():
    return [MCPServerConfig(name='duplicate', command='first'),
            MCPServerConfig(name='duplicate', command='second')]


def test_effective_configuration_rejects_duplicate_server_names():
    with pytest.raises(ValueError, match=r'[Dd]uplicate.*duplicate'):
        OpenAgentConfig(mcp_servers=configs())


def test_manager_rejects_duplicate_names_before_creating_clients():
    def factory(config):
        pytest.fail('duplicate configuration must fail before creating clients')

    with pytest.raises(ValueError, match=r'[Dd]uplicate.*duplicate'):
        MCPManager(configs=configs(), client_factory=factory)


@pytest.mark.parametrize('text', [
    '[[mcp_servers]]\nname="duplicate"\ncommand="first"\n'
    '[[mcp_servers]]\nname="duplicate"\ncommand="second"\n',
    '[mcp_servers.first]\nname="duplicate"\ncommand="first"\n'
    '[mcp_servers.second]\nname="duplicate"\ncommand="second"\n',
])
def test_toml_duplicate_names_rejected(tmp_path, monkeypatch, text):
    monkeypatch.setattr('openagent.config.find_global_config_path', lambda: None)
    monkeypatch.setattr('openagent.config.find_project_config_path', lambda: None)
    path = tmp_path / 'config.toml'
    path.write_text(text, encoding='utf-8')
    with pytest.raises(ValueError, match=r'[Dd]uplicate.*duplicate'):
        load_config(path)


def test_unique_names_preserve_order_and_server_configuration():
    servers = [MCPServerConfig(name='second', command='two'),
               MCPServerConfig(name='first', command='one')]
    cfg = OpenAgentConfig(mcp_servers=servers)
    manager = MCPManager(configs=cfg.mcp_servers)
    assert list(manager.configs) == ['second', 'first']
    assert manager.configs['second'] is servers[0]
    assert manager.configs['first'] is servers[1]


def test_config_layers_can_replace_same_named_server(tmp_path, monkeypatch):
    global_path = tmp_path / 'global.toml'
    project_path = tmp_path / 'project.toml'
    global_path.write_text('[mcp_servers.shared]\ncommand="global"\n', encoding='utf-8')
    project_path.write_text('[mcp_servers.shared]\ncommand="project"\n', encoding='utf-8')
    monkeypatch.setattr('openagent.config.find_global_config_path', lambda: global_path)
    monkeypatch.setattr('openagent.config.find_project_config_path', lambda: project_path)
    cfg = load_config()
    assert len(cfg.mcp_servers) == 1
    assert cfg.mcp_servers[0].command == 'project'
