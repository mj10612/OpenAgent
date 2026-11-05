from unittest.mock import AsyncMock

import pytest

from openagent.context.compactor import Compactor
from openagent.context.messages import MessageManager
from openagent.core.types import ChatResponse, Message, ToolCall, Usage
from openagent.providers.ollama import OllamaProvider


def history():
    return [
        Message.user('NEVER change billing. ' + 'old history ' * 1000),
        Message.assistant('findings ' * 1000),
        Message.user('Continue current task'),
        Message.assistant(tool_calls=[ToolCall(id='kept-call', name='fixture', arguments={})]),
        Message.tool_result('kept-call', 'fixture', 'recent result'),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['length', 'content_filter', 'error', 'unknown', 'tool_calls'])
async def test_incomplete_summary_uses_bounded_fallback_preserving_context(reason, caplog):
    provider = OllamaProvider('fixture')
    provider.context_window = 100000
    provider.complete = AsyncMock(return_value=ChatResponse(
        message=Message.assistant('UNSAFE_PARTIAL_SUMMARY'),
        finish_reason=reason, usage=Usage(prompt_tokens=50, completion_tokens=20),
    ))
    compactor = Compactor(keep_recent_groups=2)
    messages = history()
    result = await compactor.compact(messages, 250, provider=provider)
    try:
        assert 'UNSAFE_PARTIAL_SUMMARY' not in '\n'.join(message.text for message in result)
        assert any('NEVER change billing' in message.text for message in result)
        assert result[-3:] == messages[-3:]
        assert all(MessageManager.valid_tool_group(group) for group in MessageManager.get_atomic_groups(result))
        assert compactor.estimator.estimate_messages(result) <= 250
        assert compactor.usage == Usage(prompt_tokens=50, completion_tokens=20)
        assert reason in caplog.text
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_completed_summary_is_used_and_usage_counted_once():
    provider = OllamaProvider('fixture')
    provider.context_window = 100000
    provider.complete = AsyncMock(return_value=ChatResponse(
        message=Message.assistant('COMPLETE_SUMMARY'), finish_reason='stop',
        usage=Usage(prompt_tokens=50, completion_tokens=20),
    ))
    compactor = Compactor(keep_recent_groups=2)
    try:
        result = await compactor.compact(history(), 250, provider=provider)
        assert any('COMPLETE_SUMMARY' in message.text for message in result)
        assert compactor.usage.total_tokens == 70
        provider.complete.assert_awaited_once()
    finally:
        await provider.close()
