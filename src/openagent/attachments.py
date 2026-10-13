"""Validated local image attachments; remote URLs are intentionally unsupported."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from openagent.core.types import ImagePart

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGES = 4


def load_images(paths: list[str], provider: Any) -> list[ImagePart]:
    if not paths:
        return []
    if not provider.supports_vision:
        raise ValueError('Selected provider does not support image attachments')
    model = str(getattr(provider, 'model', '')).lower()
    capability = getattr(provider, 'model_supports_vision', None)
    if capability is False or (capability is None and not any(x in model for x in ('gpt-4o', 'gpt-4.1', 'gpt-5', 'claude', 'gemini', 'vision', 'llava', 'pixtral', 'qwen2.5-vl', 'qwen3-vl'))):
        raise ValueError('Image capability is unknown or unsupported for this model; select a vision model')
    if len(paths) > MAX_IMAGES:
        raise ValueError(f'At most {MAX_IMAGES} image attachments are supported')
    images = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError('Image attachment exceeds 5 MiB')
        with path.open('rb') as stream:
            data = stream.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError('Image attachment exceeds 5 MiB')
        mime = None
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            mime = 'image/png'
        elif data.startswith(b'\xff\xd8\xff'):
            mime = 'image/jpeg'
        elif data.startswith((b'GIF87a', b'GIF89a')):
            mime = 'image/gif'
        elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
            mime = 'image/webp'
        if mime is None:
            raise ValueError('Unsupported or invalid image MIME; use PNG, JPEG, GIF or WebP')
        if getattr(provider, 'name', '') == 'ollama' and mime == 'image/gif':
            raise ValueError('Ollama image attachments require PNG, JPEG or WebP')
        images.append(ImagePart(data=base64.b64encode(data).decode('ascii'), mime_type=mime))
    return images
