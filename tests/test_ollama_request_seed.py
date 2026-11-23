from typing import Any

import pytest

from openagent.core.types import ChatRequest, Message
from openagent.providers.ollama import OllamaProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("extra_options", [{}, {"seed": 7, "top_k": 20}])
async def test_request_seed_overrides_provider_default(seed: int, extra_options: dict[str, Any]) -> None:
    provider = OllamaProvider("fixture", extra_body={"options": extra_options})
    try:
        payload = provider.build_payload(ChatRequest(
            model="fixture", messages=[Message.user("Hello")], seed=seed, temperature=0.5,
        ))
        assert payload["options"]["seed"] == seed
        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["num_ctx"] == 32768
        if "top_k" in extra_options:
            assert payload["options"]["top_k"] == 20
        # Per-request overrides must not become defaults for the next request.
        assert provider.extra_body == {"options": extra_options}
        next_payload = provider.build_payload(ChatRequest(
            model="fixture", messages=[Message.user("Hello again")],
        ))
        if "seed" in extra_options:
            assert next_payload["options"]["seed"] == 7
        else:
            assert "seed" not in next_payload["options"]
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_unspecified_seed_does_not_add_sampling_seed() -> None:
    provider = OllamaProvider("fixture")
    try:
        payload = provider.build_payload(ChatRequest(
            model="fixture", messages=[Message.user("Hello")],
        ))
        assert "seed" not in payload["options"]
    finally:
        await provider.close()
