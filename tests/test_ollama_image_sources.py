import json
from pathlib import Path

import httpx
import pytest

from openagent.core.events import ErrorEvent
from openagent.core.provider import ProviderError
from openagent.core.types import ChatRequest, ImagePart, Message
from openagent.providers.ollama import OllamaProvider
from openagent.session.store import SessionStore


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "Describe the image"])
@pytest.mark.parametrize("data", [None, ""])
async def test_url_image_stream_fails_before_sending_request(text: str, data: str | None) -> None:
    sent: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"message": {"content": "answer"}, "done": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = OllamaProvider("fixture", client=client)
        request = ChatRequest(model="fixture", messages=[Message.user(text, images=[
            ImagePart(data=data, url="https://example.test/image.png"),
        ])])
        events = [event async for event in provider.stream(request)]

    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert isinstance(events[0].error, ProviderError)
    assert events[0].error.provider == "ollama"
    assert "base64" in str(events[0].error)
    assert sent == []


@pytest.mark.asyncio
async def test_imported_url_image_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "session.json"
    source.write_text(json.dumps({
        "schema_version": 1,
        "metadata": {"session_id": "url-image"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe"},
            {"type": "image", "url": "https://example.test/image.png", "mime_type": "image/png"},
        ]}],
    }), encoding="utf-8")
    store = SessionStore(tmp_path / "sessions")
    sid = store.import_session(source)
    provider = OllamaProvider("fixture")
    try:
        with pytest.raises(ProviderError, match=r"(?i)image.*base64"):
            provider.build_payload(ChatRequest(model="fixture", messages=store.load_session(sid)[1]))
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_mixed_image_sources_are_rejected_without_dropping_url_image() -> None:
    provider = OllamaProvider("fixture")
    try:
        request = ChatRequest(model="fixture", messages=[Message.user("Describe both", images=[
            ImagePart(data="aGVsbG8="), ImagePart(url="https://example.test/image.png"),
        ])])
        with pytest.raises(ProviderError, match=r"(?i)image.*base64"):
            provider.build_payload(request)
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [None, "https://example.test/unused.png"])
async def test_inline_images_preserved_even_with_optional_url(url: str | None) -> None:
    provider = OllamaProvider("fixture")
    try:
        request = ChatRequest(model="fixture", messages=[Message.user("Describe both", images=[
            ImagePart(data="aGVsbG8=", url=url), ImagePart(data="d29ybGQ="),
        ])])
        assert provider.build_payload(request)["messages"] == [{
            "role": "user", "content": "Describe both", "images": ["aGVsbG8=", "d29ybGQ="],
        }]
    finally:
        await provider.close()
