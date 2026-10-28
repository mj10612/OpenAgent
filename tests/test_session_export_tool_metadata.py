import json

import pytest

from openagent.core.types import Message, ToolCall
from openagent.session.store import SessionStore


@pytest.mark.parametrize('format', ['json', 'jsonl', 'markdown'])
def test_redacted_export_removes_all_tool_payloads(format, tmp_path):
    store = SessionStore(tmp_path / 'sessions')
    sid = store.create_session()
    messages = [
        Message.user('review'),
        Message.assistant(tool_calls=[ToolCall(id='call1', name='fixture', arguments={})]),
        Message.tool_result('call1', 'fixture', 'PRIVATE_TEXT', is_error=True,
            structured_data={'rows': [{'value': 'PRIVATE_STRUCTURED'}]},
            metadata={'raw_response': {'content': 'PRIVATE_RAW'}, 'extra': ['PRIVATE_EXTRA']}),
    ]
    store.save_messages(sid, messages)
    original = (tmp_path / 'sessions' / (sid + '.jsonl')).read_bytes()
    output = store.export_session(sid, tmp_path / ('redacted.' + format), format=format, redact=True)
    text = output.read_text(encoding='utf-8')
    for marker in ('PRIVATE_TEXT', 'PRIVATE_STRUCTURED', 'PRIVATE_RAW', 'PRIVATE_EXTRA'):
        assert marker not in text
    assert '[REDACTED tool output]' in text
    if format != 'markdown':
        rows = json.loads(text)['messages'] if format == 'json' else [json.loads(line) for line in text.splitlines()][1:]
        tool = rows[-1]
        assert tool['metadata'] == {'is_error': True}
        assert tool['tool_call_id'] == 'call1' and tool['name'] == 'fixture'
    assert (tmp_path / 'sessions' / (sid + '.jsonl')).read_bytes() == original
    assert store.load_session(sid)[1][-1].metadata == messages[-1].metadata


@pytest.mark.parametrize('format', ['json', 'jsonl'])
def test_unredacted_export_preserves_structured_tool_results(format, tmp_path):
    store = SessionStore(tmp_path / 'sessions')
    sid = store.create_session()
    result = Message.tool_result('call1', 'fixture', 'PRIVATE_TEXT',
        structured_data={'rows': [{'value': 'PRIVATE_STRUCTURED'}]},
        metadata={'raw_response': {'content': 'PRIVATE_RAW'}})
    store.save_messages(sid, [Message.user('review'),
        Message.assistant(tool_calls=[ToolCall(id='call1', name='fixture', arguments={})]), result])
    output = store.export_session(sid, tmp_path / ('full.' + format), format=format)
    text = output.read_text(encoding='utf-8')
    rows = json.loads(text)['messages'] if format == 'json' else [json.loads(line) for line in text.splitlines()][1:]
    assert rows[-1]['metadata'] == result.metadata
    assert rows[-1]['content'][0]['text'] == 'PRIVATE_TEXT'
