import io
import json
from unittest.mock import Mock

import pytest

from openagent import cli


def read_prompt(argv):
    return cli._read_run_prompt(cli._build_parser().parse_args(['run', *argv]))


def test_positional_prompt_unchanged():
    assert read_prompt(['hello', 'world']) == 'hello world'


@pytest.mark.parametrize('source', ['stdin', 'file'])
def test_multiline_utf8_prompt_preserved(source, tmp_path, monkeypatch):
    text = 'first\n\uac80\ud1a0\nlast\n'
    if source == 'stdin':
        monkeypatch.setattr(cli.sys, 'stdin', io.TextIOWrapper(io.BytesIO(text.encode()), encoding='ascii'))
        assert read_prompt(['-']) == text
    else:
        path = tmp_path / 'prompt.txt'
        path.write_bytes(text.encode())
        assert read_prompt(['--prompt-file', str(path)]) == text


@pytest.mark.parametrize('data', [b'', b' \n', b'\xff', b'x' * (2 * 1024 * 1024 + 1)], ids=['empty', 'blank', 'encoding', 'oversized'])
def test_bad_file_prompt_fails_before_provider(data, tmp_path, monkeypatch, capsys):
    path = tmp_path / 'prompt.txt'
    path.write_bytes(data)
    build = Mock(side_effect=AssertionError('provider must not be built'))
    monkeypatch.setattr(cli, '_build_runner', build)
    assert cli.main(['run', '--prompt-file', str(path), '--json']) == 1
    assert json.loads(capsys.readouterr().out)['outcome'] == 'error'
    build.assert_not_called()


@pytest.mark.parametrize('argv', [[], ['-', 'extra'], ['hello', '--prompt-file', 'x'], ['-', '--prompt-file', 'x']])
def test_conflicting_or_missing_prompt_rejected(argv):
    with pytest.raises(ValueError):
        read_prompt(argv)


def test_unreadable_prompt_file(tmp_path):
    with pytest.raises(OSError):
        read_prompt(['--prompt-file', str(tmp_path / 'missing')])


def test_invalid_stdin_utf8_rejected(monkeypatch):
    monkeypatch.setattr(cli.sys, 'stdin', io.TextIOWrapper(io.BytesIO(b'\xff')))
    with pytest.raises(ValueError):
        read_prompt(['-'])


@pytest.mark.parametrize('mode', [[], ['--json'], ['--jsonl']])
@pytest.mark.parametrize('source', ['stdin', 'file'])
def test_stdin_reaches_runner_without_approving_ask_tools(monkeypatch, capsys, mode, source, tmp_path):
    from openagent.config import OpenAgentConfig
    from openagent.core.events import DoneEvent
    from openagent.core.types import Message, ToolCall
    from openagent.providers.openai_compat import OpenAICompatProvider
    from openagent.runner import AgentRunner

    runner = AgentRunner(OpenAICompatProvider('https://unused.test', 'gpt-4o'))
    async def turn(prompt, ask_callback):
        assert prompt == 'first\nsecond\n'
        assert not await ask_callback(ToolCall(name='shell', arguments={'command': 'echo unsafe'}))
        yield DoneEvent(message=Message.assistant('ok'))

    monkeypatch.setattr(runner, 'run_turn', turn)
    monkeypatch.setattr(cli.sys, 'stdin', io.StringIO('first\nsecond\n'))
    monkeypatch.setattr(cli, '_load_args_config', lambda args: OpenAgentConfig())
    monkeypatch.setattr(cli, '_build_runner', lambda cfg, console: (runner, None))
    path = tmp_path / 'prompt.txt'
    path.write_bytes(b'first\nsecond\n')
    source_args = ['-'] if source == 'stdin' else ['--prompt-file', str(path)]
    assert cli.main(['run', *source_args, *mode]) == 0
    output = capsys.readouterr().out
    if mode:
        assert json.loads(output.splitlines()[-1])['outcome'] == 'success'
