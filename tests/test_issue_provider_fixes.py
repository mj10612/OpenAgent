import json

import httpx
import pytest

from openagent.core.events import DoneEvent, ErrorEvent, UsageEvent
from openagent.core.provider import ProviderError
from openagent.core.types import ChatRequest, Message
from openagent.providers.anthropic import AnthropicProvider
from openagent.providers.custom import CustomJsonPathProvider
from openagent.providers.gemini import GeminiProvider
from openagent.providers.ollama import OllamaProvider
from openagent.providers.openai_compat import OpenAICompatProvider


def request(**kwargs):
    return ChatRequest(model="test", messages=[Message.user("hi")], **kwargs)


async def events(cls, chunks, **kwargs):
    body = "".join("data: " + (c if isinstance(c, str) else json.dumps(c)) + "\n\n" for c in chunks)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        )
    ) as client:
        provider = cls(model="test", client=client, **kwargs)
        return [event async for event in provider.stream(request())]


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_openai_inband_error_never_succeeds(partial):
    chunks = [{"choices": [{"delta": {"content": "partial"}}]}] if partial else []
    result = await events(
        OpenAICompatProvider,
        [*chunks, {"error": {"message": "quota exhausted"}}, "[DONE]"],
        base_url="https://example.test",
    )
    assert len([e for e in result if isinstance(e, ErrorEvent)]) == 1
    assert "quota exhausted" in str(next(e.error for e in result if isinstance(e, ErrorEvent)))
    assert not any(isinstance(e, DoneEvent) for e in result)


@pytest.mark.asyncio
async def test_openai_alternative_text_and_finish_are_isolated():
    result = await events(
        OpenAICompatProvider,
        [
            {
                "choices": [
                    {"index": 1, "delta": {"content": "B"}, "finish_reason": "length"},
                    {"index": 0, "delta": {"content": "A"}, "finish_reason": "stop"},
                ]
            },
            "[DONE]",
        ],
        base_url="https://example.test",
        extra_body={"n": 2},
    )
    done = next(e for e in result if isinstance(e, DoneEvent))
    assert done.message.text == "A"
    assert done.finish_reason.value == "stop"


@pytest.mark.asyncio
async def test_openai_alternative_tool_indexes_are_isolated():
    choices = [
        {
            "index": i,
            "delta": {
                "content": text,
                "reasoning_content": text,
                "tool_calls": [
                    {"index": 0, "id": text, "function": {"name": text, "arguments": "{}"}}
                ],
            },
            "finish_reason": "tool_calls",
        }
        for i, text in enumerate(["A", "B"])
    ]
    result = await events(
        OpenAICompatProvider,
        [{"choices": choices}, "[DONE]"],
        base_url="https://example.test",
        extra_body={"n": 2},
    )
    done = next(e for e in result if isinstance(e, DoneEvent))
    assert done.message.text == done.message.reasoning == "A"
    assert [(c.name, c.id) for c in done.message.tool_calls] == [("A", "A")]


@pytest.mark.asyncio
async def test_gemini_usage_preserves_partial_snapshots():
    result = await events(
        GeminiProvider,
        [
            {
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 3,
                    "thoughtsTokenCount": 20,
                    "cachedContentTokenCount": 7,
                }
            },
            {
                "usageMetadata": {"candidatesTokenCount": 4},
                "candidates": [{"finishReason": "STOP"}],
            },
        ],
    )
    usage = next(e.usage for e in result if isinstance(e, DoneEvent))
    initial = next(e.usage for e in result if isinstance(e, UsageEvent))
    assert (initial.total_tokens, initial.reasoning_tokens, initial.cached_tokens) == (33, 20, 7)
    assert (
        usage.total_tokens,
        usage.completion_tokens,
        usage.reasoning_tokens,
        usage.cached_tokens,
    ) == (34, 24, 20, 7)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cls,cursor", [(GeminiProvider, "pageToken"), (AnthropicProvider, "after_id")]
)
async def test_discovery_follows_pages_and_metadata(cls, cursor):
    def handler(r):
        second = cursor in r.url.params
        assert not second or r.url.params[cursor] == "first"
        if cls is GeminiProvider:
            return httpx.Response(
                200,
                json={
                    "models": [{"name": "models/second" if second else "models/first"}],
                    **({} if second else {"nextPageToken": "first"}),
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "second" if second else "first",
                        "max_input_tokens": 200000,
                        "max_tokens": 64000,
                        "context_window": 1,
                        "max_output_tokens": 2,
                    }
                ],
                "has_more": not second,
                "last_id": "first",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        models = await cls("test", client=client).list_models()
    assert [m.id for m in models] == ["first", "second"]
    if cls is AnthropicProvider:
        assert (models[0].context_window, models[0].max_output) == (200000, 64000)


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [GeminiProvider, AnthropicProvider])
async def test_discovery_repeated_cursor_fails(cls):
    body = (
        {"models": [], "nextPageToken": "again"}
        if cls is GeminiProvider
        else {"data": [], "has_more": True, "last_id": "again"}
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    ) as client:
        with pytest.raises(ProviderError, match="cursor"):
            await cls("test", client=client).list_models()


@pytest.mark.asyncio
async def test_payload_defaults_and_overrides():
    custom = CustomJsonPathProvider("https://example.test", "test")
    custom.default_max_tokens = 256
    assert custom.build_payload(request())["max_tokens"] == 256
    assert custom.build_payload(request(max_tokens=128))["max_tokens"] == 128
    await custom.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("context_window", [None, 65536])
async def test_ollama_context_and_options_are_consistent(context_window):
    provider = OllamaProvider(
        "test", context_window=context_window, extra_body={"options": {"num_ctx": 8192, "seed": 7}}
    )
    payload = provider.build_payload(request(temperature=0.5))
    assert payload["options"] == {"num_ctx": context_window or 32768, "seed": 7, "temperature": 0.5}
    await provider.close()


@pytest.mark.asyncio
async def test_gemini_json_mode():
    provider = GeminiProvider("test")
    assert (
        provider.build_payload(request(response_format="json_object"))["generationConfig"][
            "responseMimeType"
        ]
        == "application/json"
    )
    assert "responseMimeType" not in provider.build_payload(request()).get("generationConfig", {})
    await provider.close()
