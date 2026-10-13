"""Credential-free model discovery cache and offline budget metadata."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from .provider import ChatProvider, ProviderError
from .types import ModelInfo

DEFAULT_TTL = 3600.0


def _cache_path(provider: ChatProvider, cache_path: Path | None = None) -> Path:
    if cache_path is not None:
        return Path(cache_path)
    identity = provider.name + "|" + str(getattr(provider, "base_url", ""))
    key = hashlib.sha256(identity.encode()).hexdigest()
    directory = Path(
        os.environ.get("OPENAGENT_MODEL_CACHE", str(Path.home() / ".openagent" / "model-cache"))
    )
    return directory / (key + ".json")


def cached_models(
    provider: ChatProvider, *, cache_path: Path | None = None, ttl: float = DEFAULT_TTL
) -> list[ModelInfo] | None:
    try:
        data = json.loads(_cache_path(provider, cache_path).read_text(encoding="utf-8"))
        if time.time() - data["updated"] > ttl:
            return None
        return [ModelInfo(**item) for item in data["models"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


async def discover_models(
    provider: ChatProvider,
    *,
    refresh: bool = False,
    cache_path: Path | None = None,
    ttl: float = DEFAULT_TTL,
) -> list[ModelInfo]:
    """Refresh only when requested or stale; propagate discovery failures."""
    if not refresh:
        models = cached_models(provider, cache_path=cache_path, ttl=ttl)
        if models is not None:
            return models
    if type(provider).list_models is ChatProvider.list_models or provider.name == "custom":
        raise ProviderError("model discovery is unsupported", provider=provider.name)
    models = await provider.list_models()
    path = _cache_path(provider, cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only allowlisted ModelInfo fields; never persist transports or credentials.
    path.write_text(
        json.dumps(
            {"updated": time.time(), "models": [asdict(model) for model in models]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return models


def apply_model_metadata(
    provider: ChatProvider,
    *,
    context_override: int | None = None,
    output_override: int | None = None,
) -> ChatProvider:
    source = "preset"
    # Heuristic hints are explicitly labelled, never advertised as discovery.
    from ..providers.openai_compat import CONTEXT_HINTS

    name = str(getattr(provider, "model", "")).lower()
    for hint in sorted(CONTEXT_HINTS, key=len, reverse=True):
        if hint in name and provider.name != "ollama":
            provider.context_window = CONTEXT_HINTS[hint]
            source = "heuristic"
            break
    for model in cached_models(provider) or []:
        if model.id != getattr(provider, "model", ""):
            continue
        if model.context_window:
            provider.context_window = model.context_window
        if model.max_output:
            provider.default_max_tokens = model.max_output
        provider.model_supports_vision = model.supports_vision
        provider.model_supports_tools = model.supports_tools
        provider.model_supports_reasoning = model.supports_reasoning
        source = "discovered"
        break
    if context_override is not None:
        provider.context_window = context_override
        source = "override"
    if output_override is not None:
        provider.default_max_tokens = output_override
        source = "override"
    provider.model_metadata_source = source
    return provider


def metadata_source(provider: ChatProvider, model: ModelInfo | None = None) -> str:
    if model is not None:
        return "discovered"
    return getattr(provider, "model_metadata_source", "preset")
