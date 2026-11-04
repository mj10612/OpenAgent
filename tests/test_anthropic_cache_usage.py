import json

import httpx
import pytest

from openagent.core.events import DoneEvent, UsageEvent
from openagent.core.types import ChatRequest, Message
from openagent.providers.anthropic import AnthropicProvider


@pytest.mark.asyncio
@pytest.mark.parametrize(('reads', 'writes'), [(0, 0), (1000, 0), (0, 200), (1000, 200)])
@pytest.mark.parametrize('mode', ['stream', 'complete', 'runner'])
async def test_cached_input_contributes_to_total_usage(reads, writes, mode):
    chunks = [
        {'type': 'message_start', 'message': {'usage': {
            'input_tokens': 10, 'cache_read_input_tokens': reads,
            'cache_creation_input_tokens': writes, 'output_tokens': 0,
        }}},
        {'type': 'message_delta', 'delta': {}, 'usage': {'output_tokens': 2}},
        {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 5}},
        {'type': 'message_stop'},
    ]
    body = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks)
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, text=body, headers={'content-type': 'text/event-stream'}))
    async with httpx.AsyncClient(transport=transport) as client:
        provider = AnthropicProvider('fixture', client=client)
        request = ChatRequest(model='fixture', messages=[Message.user('hello')])
        factor = 1
        if mode == 'complete':
            usage = (await provider.complete(request)).usage
        elif mode == 'runner':
            from openagent.runner import AgentRunner
            agent = AgentRunner(provider)
            for _ in range(2):
                events = [event async for event in agent.run_turn('hello')]
                assert isinstance(events[-1], DoneEvent)
                assert events[-1].usage.prompt_tokens == 10 + reads + writes
            usage = agent.usage
            factor = 2
        else:
            events = [event async for event in provider.stream(request)]
            done = next(event for event in events if isinstance(event, DoneEvent))
            usage = done.usage
            snapshots = [event.usage for event in events if isinstance(event, UsageEvent)]
            assert snapshots[-1] == usage
        assert usage.prompt_tokens == factor * (10 + reads + writes)
        assert usage.completion_tokens == factor * 5
        assert usage.total_tokens == factor * (15 + reads + writes)
        assert usage.cached_tokens == factor * reads
