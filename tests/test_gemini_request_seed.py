import json
from copy import deepcopy

import httpx
import pytest

from openagent.core.events import DoneEvent
from openagent.core.types import ChatRequest, Message
from openagent.providers.gemini import GeminiProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("extra_body", [
    {},
    {"generationConfig": None},
    {"generationConfig": {"seed": 7, "topK": 20}},
])
async def test_request_seed_reaches_gemini_http_payload(seed, extra_body):
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=(
            'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},'
            '"finishReason":"STOP"}]}\n\n'
        ))

    before = deepcopy(extra_body)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = GeminiProvider("fixture", extra_body=extra_body, client=client)
        request = ChatRequest(model="fixture", messages=[Message.user("Hello")], seed=seed)
        events = [event async for event in provider.stream(request)]
        assert isinstance(events[-1], DoneEvent)
        assert len(bodies) == 1
        assert bodies[0]["generationConfig"] == {
            **(before.get("generationConfig") or {}), "seed": seed,
        }
        assert provider.extra_body == before
        following = provider.build_payload(ChatRequest(
            model="fixture", messages=[Message.user("Next request")],
        ))
        if "generationConfig" in before:
            assert following["generationConfig"] == before["generationConfig"]
        else:
            assert "seed" not in following.get("generationConfig", {})


@pytest.mark.asyncio
async def test_seed_preserves_other_request_generation_options():
    provider = GeminiProvider("fixture")
    try:
        payload = provider.build_payload(ChatRequest(
            model="fixture", messages=[Message.user("Hello")], seed=0,
            temperature=0.5, top_p=0.8, max_tokens=128, stop=["END"],
            response_format="json_object",
        ))
        assert payload["generationConfig"] == {
            "seed": 0, "temperature": 0.5, "topP": 0.8, "maxOutputTokens": 128,
            "stopSequences": ["END"], "responseMimeType": "application/json",
        }
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_unspecified_seed_does_not_create_generation_config():
    provider = GeminiProvider("fixture")
    try:
        payload = provider.build_payload(ChatRequest(
            model="fixture", messages=[Message.user("Hello")],
        ))
        assert "generationConfig" not in payload
    finally:
        await provider.close()
