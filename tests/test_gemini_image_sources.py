import json
from unittest.mock import AsyncMock

import pytest

from openagent.core.provider import ProviderError
from openagent.core.types import ChatRequest, ImagePart, Message
from openagent.providers.gemini import GeminiProvider
from openagent.session.store import SessionStore


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['', 'Describe the image'])
async def test_url_image_rejected_before_transport(text):
    provider = GeminiProvider('fixture')
    provider.transport.stream_post = AsyncMock(side_effect=AssertionError('must not send text-only request'))
    request = ChatRequest(model='fixture', messages=[Message.user(text, images=[
        ImagePart(url='https://example.test/image.png')])])
    try:
        with pytest.raises(ProviderError, match=r'(?i)image.*(URL|base64)'):
            _ = [event async for event in provider.stream(request)]
        provider.transport.stream_post.assert_not_awaited()
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_imported_url_image_is_not_silently_dropped(tmp_path):
    source = tmp_path / 'session.json'
    source.write_text(json.dumps({'schema_version': 1, 'metadata': {'session_id': 'url-image'},
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Describe'},
            {'type': 'image', 'url': 'https://example.test/image.png', 'mime_type': 'image/png'},
        ]}]}), encoding='utf-8')
    store = SessionStore(tmp_path / 'sessions')
    sid = store.import_session(source)
    provider = GeminiProvider('fixture')
    try:
        with pytest.raises(ProviderError, match=r'(?i)image'):
            provider.build_payload(ChatRequest(model='fixture', messages=store.load_session(sid)[1]))
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_inline_image_preserved_even_with_optional_url():
    provider = GeminiProvider('fixture')
    try:
        request = ChatRequest(model='fixture', messages=[Message.user('Describe', images=[
            ImagePart(data='aGVsbG8=', mime_type='image/png', url='https://example.test/unused')])])
        assert provider.build_payload(request)['contents'][0]['parts'] == [
            {'inline_data': {'mime_type': 'image/png', 'data': 'aGVsbG8='}}, {'text': 'Describe'},
        ]
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_mixed_image_sources_do_not_silently_drop_one():
    provider = GeminiProvider('fixture')
    try:
        request = ChatRequest(model='fixture', messages=[Message.user('Describe both', images=[
            ImagePart(data='aGVsbG8='), ImagePart(url='https://example.test/image.png')])])
        with pytest.raises(ProviderError, match=r'(?i)image'):
            provider.build_payload(request)
    finally:
        await provider.close()
