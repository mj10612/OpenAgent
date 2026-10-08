import json
from unittest.mock import AsyncMock

import httpx
import pytest

from openagent.core.events import DoneEvent, ErrorEvent, ThinkingDelta, ToolCallEnd, UsageEvent
from openagent.core.presets import ProviderPreset, find_preset
from openagent.core.provider import ChatProvider, ProviderError
from openagent.core.router import ModelReference, ProviderRouter
from openagent.core.types import (
    ChatRequest,
    Message,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolProtocol,
    ToolSpec,
    Usage,
    iter_text_chunks,
)
from openagent.providers.anthropic import AnthropicProvider
from openagent.providers.custom import CustomJsonPathProvider
from openagent.providers.gemini import GeminiProvider
from openagent.providers.ollama import OllamaProvider
from openagent.providers.openai_compat import OpenAICompatProvider
from openagent.utils.http import HttpTransport


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, body):
        self.body = body

    async def __aiter__(self):
        yield self.body.encode()


def response(chunks, sse=True):
    body = "".join(
        "data: " + json.dumps(c) + "\n\n" if sse else json.dumps(c) + "\n" for c in chunks
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream" if sse else "application/x-ndjson"},
        stream=BytesStream(body),
    )


@pytest.mark.asyncio
async def test_complete_usage_is_authoritative_once():
    class Provider(ChatProvider):
        async def stream(self, request):
            usage = Usage(prompt_tokens=12, completion_tokens=4)
            yield UsageEvent(usage=usage)
            yield DoneEvent(message=Message.assistant("ok"), usage=usage)

    result = await Provider().complete(ChatRequest(model="m", messages=[]))
    assert result.usage.total_tokens == 16


@pytest.mark.asyncio
async def test_parallel_openai_arguments_are_independent():
    p = OpenAICompatProvider("http://localhost", "m")
    p.transport.stream_post = AsyncMock(
        return_value=response(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "a",
                                        "function": {"name": "a", "arguments": '{"path":"a"}'},
                                    },
                                    {
                                        "index": 1,
                                        "id": "b",
                                        "function": {"name": "b", "arguments": '{"city":"Paris"}'},
                                    },
                                ]
                            }
                        }
                    ]
                }
            ]
        )
    )
    events = [e async for e in p.stream(ChatRequest(model="m", messages=[]))]
    assert [e.call.arguments for e in events if isinstance(e, ToolCallEnd)] == [
        {"path": "a"},
        {"city": "Paris"},
    ]
    await p.close()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://api.openai.com", "https://api.openai.com/v1"),
        ("https://api.cohere.com/v2", "https://api.cohere.com/v2"),
        ("https://api.deepinfra.com/v1/openai", "https://api.deepinfra.com/v1/openai"),
    ],
)
def test_api_root_normalization(raw, expected):
    assert OpenAICompatProvider._normalise_base_url(raw) == expected


def test_bare_ollama_tag_and_preset_lookup():
    ref = ModelReference.parse("llama3.2:7b")
    assert (ref.provider_hint, ref.model_name) == (None, "llama3.2:7b")
    assert isinstance(ProviderRouter().resolve(ref), OllamaProvider)
    assert find_preset("openai_compat").name == "openai_compat"


@pytest.mark.parametrize("name", ["azure", "bedrock"])
def test_unsupported_native_provider_rejected(name):
    with pytest.raises(ValueError, match="not implemented"):
        ProviderRouter().resolve(name + "/m")


def test_preset_defaults_and_protocol_apply():
    preset = ProviderPreset(
        "test",
        "http://localhost/v1",
        max_output=123,
        body_defaults={"custom": True},
        protocol=ToolProtocol.TEXT,
    )
    p = ProviderRouter({"test": preset}).resolve("test/m")
    payload = p.build_payload(ChatRequest(model="m", messages=[]), stream=True)
    assert payload["max_tokens"] == 123
    assert payload["custom"] is True
    assert p.tool_protocol == ToolProtocol.TEXT


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_text_chunk_size_rejected(size):
    with pytest.raises(ValueError):
        next(iter_text_chunks([TextPart("abc")], chunk_size=size))


def test_full_nested_tool_schema_survives():
    schema = {
        "type": "object",
        "$defs": {"Item": {"type": "object"}},
        "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
        "additionalProperties": False,
    }
    spec = ToolSpec("nested", "nested", input_schema=schema)
    assert spec.to_openai_schema()["function"]["parameters"] == schema
    assert spec.to_anthropic_schema()["input_schema"] == schema
    assert spec.to_gemini_schema()["parametersJsonSchema"] == schema


def test_gemini_groups_parallel_results():
    p = GeminiProvider("m")
    contents = p.build_payload(
        ChatRequest(
            model="m",
            messages=[
                Message.user("go"),
                Message.assistant("calling"),
                Message.tool_result("a", "a", "x"),
                Message.tool_result("b", "b", "y"),
            ],
        )
    )["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert len(contents[-1]["parts"]) == 2


@pytest.mark.asyncio
async def test_ollama_thinking_preserved():
    p = OllamaProvider("m")
    p.transport.stream_post = AsyncMock(
        return_value=response(
            [{"message": {"thinking": "reason", "content": "answer"}, "done": True}], False
        )
    )
    events = [e async for e in p.stream(ChatRequest(model="m", messages=[]))]
    assert any(isinstance(e, ThinkingDelta) and e.text == "reason" for e in events)
    assert events[-1].message.reasoning == "reason"
    await p.close()


@pytest.mark.asyncio
async def test_anthropic_signed_thinking_replayed_once():
    p = AnthropicProvider("m")
    p._transport.stream_post = AsyncMock(
        return_value=response(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "reason"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "signed"},
                },
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "answer"},
                },
            ]
        )
    )
    events = [e async for e in p.stream(ChatRequest(model="m", messages=[]))]
    message = events[-1].message
    assert isinstance(message.content[0], ThinkingPart)
    assert message.content[0].signature == "signed"
    blocks = p.build_payload(ChatRequest(model="m", messages=[message]))["messages"][0]["content"]
    assert blocks == [
        {"type": "thinking", "thinking": "reason", "signature": "signed"},
        {"type": "text", "text": "answer"},
    ]
    await p.close()


def test_custom_history_keeps_correlations():
    p = CustomJsonPathProvider("http://localhost", "m")
    call = ToolCall("read", {"path": "a"}, id="a")
    msgs = p.build_payload(
        ChatRequest(
            model="m",
            messages=[Message.assistant(tool_calls=[call]), Message.tool_result("a", "read", "x")],
        )
    )["messages"]
    assert msgs[0]["tool_calls"][0]["id"] == "a"
    assert json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"]) == {"path": "a"}
    assert msgs[1]["tool_call_id"] == "a"


@pytest.mark.asyncio
async def test_custom_fragmented_arguments():
    p = CustomJsonPathProvider(
        "http://localhost", "m", tool_calls_path="choices[0].delta.tool_calls[*]"
    )
    p.transport.stream_post = AsyncMock(
        return_value=response(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "a",
                                        "function": {"name": "read", "arguments": '{"pa'},
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": 'th":"a"}'}}]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ]
        )
    )
    events = [e async for e in p.stream(ChatRequest(model="m", messages=[]))]
    assert events[-1].message.tool_calls[0].arguments == {"path": "a"}
    await p.close()


@pytest.mark.asyncio
async def test_stream_transport_maps_network_failure():
    async def handler(request):
        raise httpx.ConnectError("offline")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        p = OpenAICompatProvider("http://localhost", "m", client=client)
        events = [e async for e in p.stream(ChatRequest(model="m", messages=[]))]
        assert len(events) == 1 and isinstance(events[0], ErrorEvent)
        assert isinstance(events[0].error, ProviderError)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["post_json", "stream_post", "get_json"])
async def test_redirect_never_forwards_credentials(method):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(302, headers={"location": "https://evil.test/"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        transport = HttpTransport(
            "https://origin.test", provider="test", headers={"x-api-key": "secret"}, client=client
        )
        with pytest.raises(ProviderError, match="redirect"):
            await getattr(transport, method)("/chat", {})
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_retry_after_populated():
    def handler(request):
        return httpx.Response(
            429, headers={"retry-after": "12"}, json={"error": {"message": "limited"}}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = HttpTransport("https://origin.test", provider="test", client=client)
        with pytest.raises(ProviderError) as caught:
            await transport.stream_post("/chat", {})
    assert caught.value.retry_after == 12


@pytest.mark.asyncio
async def test_ollama_inline_thinking_split_across_chunks():
    p = OllamaProvider("m")
    p.transport.stream_post = AsyncMock(
        return_value=response(
            [
                {"message": {"content": "<thi"}, "done": False},
                {"message": {"content": "nk>reason</th"}, "done": False},
                {"message": {"content": "ink>answer"}, "done": True},
            ],
            False,
        )
    )
    events = [e async for e in p.stream(ChatRequest(model="m", messages=[]))]
    assert events[-1].message.reasoning == "reason"
    assert events[-1].message.text == "answer"
    await p.close()


@pytest.mark.asyncio
async def test_anthropic_interleaved_tools_preserve_block_order():
    p = AnthropicProvider("m")
    chunks = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "a", "name": "read", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"path":"a"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "thinking_delta", "thinking": "reason"},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "signature_delta", "signature": "signed"},
        },
    ]
    p._transport.stream_post = AsyncMock(return_value=response(chunks))
    events = [e async for e in p.stream(ChatRequest(model="m", messages=[]))]
    message = Message.from_dict(events[-1].message.to_dict())
    blocks = p.build_payload(ChatRequest(model="m", messages=[message]))["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["tool_use", "thinking"]
    await p.close()


@pytest.mark.asyncio
async def test_custom_malformed_final_arguments_are_terminal_error():
    p = CustomJsonPathProvider(
        "http://localhost", "m", tool_calls_path="choices[0].delta.tool_calls[*]"
    )
    p.transport.stream_post = AsyncMock(
        return_value=response(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"name": "read", "arguments": "broken"},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        )
    )
    events = [e async for e in p.stream(ChatRequest(model="m", messages=[]))]
    assert isinstance(events[-1], ErrorEvent)
    assert not any(isinstance(e, DoneEvent) for e in events)
    await p.close()


def test_grok_alias_has_one_canonical_preset():
    from openagent.core.presets import all_presets

    assert find_preset("grok") is find_preset("xai")
    assert "grok" not in all_presets()


def test_heuristic_provider_uses_preset_output_defaults():
    p = ProviderRouter().resolve("gpt-4o")
    assert (
        p.build_payload(ChatRequest(model="gpt-4o", messages=[]), stream=False)["max_tokens"]
        == 16384
    )
