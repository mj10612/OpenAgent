"""Credential-free model discovery cache and offline budget metadata."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from .provider import ChatProvider, ProviderError
from .types import ModelInfo

DEFAULT_TTL = 3600.0
logger = logging.getLogger(__name__)


def _valid_model(model: ModelInfo) -> bool:
    return (
        isinstance(model.id, str)
        and bool(model.id)
        and isinstance(model.provider, str)
        and bool(model.provider)
        and all(
            value is None or (type(value) is int and value > 0)
            for value in (model.context_window, model.max_output)
        )
        and all(
            value is None or type(value) is bool
            for value in (model.supports_tools, model.supports_vision, model.supports_reasoning)
        )
        and isinstance(model.aliases, list)
        and all(isinstance(alias, str) for alias in model.aliases)
    )


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
        updated = data["updated"]
        if type(updated) not in (int, float) or not math.isfinite(updated):
            return None
        if time.time() - updated > ttl:
            return None
        if not isinstance(data["models"], list):
            return None
        models = [ModelInfo(**item) for item in data["models"]]
        return models if all(_valid_model(model) for model in models) else None
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
    if not isinstance(models, list) or not all(
        isinstance(model, ModelInfo) and _valid_model(model) for model in models
    ):
        raise ProviderError("model discovery returned invalid metadata", provider=provider.name)
    path = _cache_path(provider, cache_path)
    temporary_path: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Only allowlisted ModelInfo fields; never persist transports or credentials.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            json.dump(
                {"updated": time.time(), "models": [asdict(model) for model in models]},
                temporary,
                ensure_ascii=False,
            )
        os.replace(temporary_path, path)
    except OSError as exc:
        logger.warning("Could not save model discovery cache: %s", exc)
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("Could not remove temporary model cache: %s", exc)
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
