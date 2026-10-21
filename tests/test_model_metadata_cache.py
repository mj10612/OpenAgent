import json
import time
from unittest.mock import AsyncMock

import pytest

from openagent.core.model_metadata import cached_models, discover_models
from openagent.core.provider import ProviderError
from openagent.core.types import ModelInfo
from openagent.providers.openai_compat import OpenAICompatProvider


@pytest.mark.parametrize(
    "change",
    [
        {"context_window": "4096"},
        {"max_output": True},
        {"supports_tools": "yes"},
        {"aliases": [123]},
        {"id": 123},
    ],
)
def test_invalid_cached_model_is_a_miss(tmp_path, change):
    provider = OpenAICompatProvider("http://localhost", "model")
    model = {"id": "model", "provider": provider.name, "context_window": 4096}
    model.update(change)
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"updated": time.time(), "models": [model]}))

    assert cached_models(provider, cache_path=path) is None


async def test_discovery_survives_cache_write_failure_and_preserves_old_cache(tmp_path, monkeypatch):
    provider = OpenAICompatProvider("http://localhost", "model")
    provider.list_models = AsyncMock(return_value=[ModelInfo("model", provider.name, 4096)])
    path = tmp_path / "models.json"
    path.write_text("previous cache")

    def fail_replace(*args):
        raise PermissionError("read-only cache")

    monkeypatch.setattr("openagent.core.model_metadata.os.replace", fail_replace)
    models = await discover_models(provider, cache_path=path, refresh=True)

    assert models[0].context_window == 4096
    assert path.read_text() == "previous cache"
    assert list(tmp_path.iterdir()) == [path]


async def test_invalid_discovery_is_not_cached(tmp_path):
    provider = OpenAICompatProvider("http://localhost", "model")
    provider.list_models = AsyncMock(return_value=[ModelInfo("model", provider.name, "4096")])
    path = tmp_path / "models.json"

    with pytest.raises(ProviderError, match="invalid metadata"):
        await discover_models(provider, cache_path=path, refresh=True)
    assert not path.exists()
