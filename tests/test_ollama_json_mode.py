from typing import Any

import pytest

from openagent.core.types import ChatRequest, Message
from openagent.providers.ollama import OllamaProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_body", [{}, {"format": ""}, {"format": {"type": "object"}}])
async def test_json_object_request_enables_json_even_with_extra_format(
    extra_body: dict[str, Any],
) -> None:
    provider = OllamaProvider("test", extra_body=extra_body)
    try:
        payload = provider.build_payload(ChatRequest(
            model="test",
            messages=[Message.user("Return a JSON object")],
            response_format="json_object",
        ))
        assert payload.get("format") == "json"
        assert provider.extra_body == extra_body
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_body", [{}, {"format": "json"}, {"format": {"type": "object"}}])
async def test_default_text_mode_preserves_extra_format(extra_body: dict[str, Any]) -> None:
    provider = OllamaProvider("test", extra_body=extra_body)
    try:
        payload = provider.build_payload(ChatRequest(
            model="test", messages=[Message.user("Hello")],
        ))
        if "format" in extra_body:
            assert payload["format"] == extra_body["format"]
        else:
            assert "format" not in payload
    finally:
        await provider.close()
