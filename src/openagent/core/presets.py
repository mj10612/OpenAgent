"""Built-in provider presets.

A preset is pure configuration: base URL, auth header shape and a couple of
capability hints. Adding a service that already speaks the OpenAI wire format
therefore requires **no code at all** — just an entry here (or, better, just a
user-level config entry, since user config overrides these).

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

from .types import ToolProtocol


@dataclass(frozen=True, slots=True)
class AuthScheme:
    """How to attach credentials to a request.

    ``bearer``   -> ``Authorization: Bearer <key>``
    ``x-api-key`` -> ``x-api-key: <key>``          (Anthropic)
    ``query``    -> ``?key=<key>``                 (Google)
    ``header``   -> an arbitrary named header      (Azure: api-key)
    ``none``     -> no credential at all           (local servers)
    """

    scheme: str = "bearer"
    header: str = ""
    env: tuple[str, ...] = ()

    def resolve_key(self, explicit: str | None = None) -> str | None:
        if explicit:
            return explicit
        for name in self.env:
            value = os.environ.get(name)
            if value:
                return value
        return None


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    """Declarative description of an inference endpoint."""

    name: str
    base_url: str
    kind: str = "openai-compat"
    auth: AuthScheme = field(default_factory=AuthScheme)
    protocol: ToolProtocol = ToolProtocol.JSON_SCHEMA
    context_window: int = 128_000
    max_output: int = 8192
    supports_vision: bool = False
    supports_reasoning: bool = False
    default_model: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    notes: str = ""
    #: Overrides merged into every outgoing request body.
    body_defaults: dict[str, Any] = field(default_factory=dict)

    def with_overrides(self, **kwargs: Any) -> ProviderPreset:
        return replace(self, **kwargs)


def _p(name: str, base_url: str, **kw: Any) -> ProviderPreset:
    return ProviderPreset(name=name, base_url=base_url, **kw)


# --------------------------------------------------------------------------- #
# Cloud providers
# --------------------------------------------------------------------------- #

BUILTIN_PRESETS: dict[str, ProviderPreset] = {
    p.name: p
    for p in (
        _p(
            "openai",
            "https://api.openai.com/v1",
            auth=AuthScheme("bearer", env=("OPENAI_API_KEY",)),
            context_window=128_000,
            max_output=16_384,
            supports_vision=True,
            supports_reasoning=True,
            default_model="gpt-4o-mini",
            notes="GPT-4o, o-series, gpt-5 family. `o*`/`gpt-5*` are reasoning models.",
        ),
        _p(
            "anthropic",
            "https://api.anthropic.com",
            kind="anthropic",
            auth=AuthScheme("x-api-key", env=("ANTHROPIC_API_KEY",)),
            protocol=ToolProtocol.NATIVE,
            context_window=200_000,
            max_output=8192,
            supports_vision=True,
            supports_reasoning=True,
            default_model="claude-sonnet-4-5",
            notes="Native Messages API with extended thinking.",
        ),
        _p(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta",
            kind="gemini",
            auth=AuthScheme("query", env=("GEMINI_API_KEY", "GOOGLE_API_KEY")),
            protocol=ToolProtocol.NATIVE,
            context_window=1_000_000,
            max_output=8192,
            supports_vision=True,
            default_model="gemini-2.0-flash",
            notes="Google Generative Language API.",
        ),
        _p(
            "azure",
            "https://YOUR-RESOURCE.openai.azure.com",
            kind="azure",
            auth=AuthScheme("header", header="api-key", env=("AZURE_OPENAI_API_KEY",)),
            context_window=128_000,
            max_output=16_384,
            supports_vision=True,
            default_model="gpt-4o-mini",
            notes="Set `deployment` in config; it replaces the model in the URL path.",
        ),
        _p(
            "bedrock",
            "https://bedrock-runtime.{region}.amazonaws.com",
            kind="bedrock",
            auth=AuthScheme("none"),
            protocol=ToolProtocol.JSON_SCHEMA,
            supports_vision=True,
            default_model="",
            notes="AWS Bedrock. Requires boto3 credentials from the environment.",
        ),
        _p(
            "deepseek",
            "https://api.deepseek.com/v1",
            auth=AuthScheme("bearer", env=("DEEPSEEK_API_KEY",)),
            context_window=64_000,
            supports_reasoning=True,
            default_model="deepseek-chat",
            notes="deepseek-chat / deepseek-reasoner.",
        ),
        _p(
            "groq",
            "https://api.groq.com/openai/v1",
            auth=AuthScheme("bearer", env=("GROQ_API_KEY",)),
            context_window=128_000,
            max_output=32_768,
            default_model="llama-3.3-70b-versatile",
            notes="Very fast inference on open-weight models.",
        ),
        _p(
            "xai",
            "https://api.x.ai/v1",
            auth=AuthScheme("bearer", env=("XAI_API_KEY",)),
            context_window=131_072,
            supports_vision=True,
            default_model="grok-2-latest",
            aliases=("grok", "x-ai"),
        ),
        _p(
            "mistral",
            "https://api.mistral.ai/v1",
            auth=AuthScheme("bearer", env=("MISTRAL_API_KEY",)),
            context_window=32_000,
            supports_vision=True,
            default_model="mistral-large-latest",
            aliases=("mistralai",),
        ),
        _p(
            "cohere",
            "https://api.cohere.com/v2",
            auth=AuthScheme("bearer", env=("COHERE_API_KEY",)),
            context_window=128_000,
            default_model="command-r-plus",
        ),
        _p(
            "together",
            "https://api.together.xyz/v1",
            auth=AuthScheme("bearer", env=("TOGETHER_API_KEY",)),
            context_window=128_000,
            supports_vision=True,
            default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        ),
        _p(
            "fireworks",
            "https://api.fireworks.ai/inference/v1",
            auth=AuthScheme("bearer", env=("FIREWORKS_API_KEY",)),
            context_window=131_072,
            supports_vision=True,
            default_model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        ),
        _p(
            "openrouter",
            "https://openrouter.ai/api/v1",
            auth=AuthScheme("bearer", env=("OPENROUTER_API_KEY",)),
            context_window=200_000,
            supports_vision=True,
            supports_reasoning=True,
            default_model="anthropic/claude-sonnet-4.5",
            headers={"HTTP-Referer": "https://github.com/openagent/openagent"},
            notes="Aggregator with hundreds of models and a single key.",
        ),
        _p(
            "perplexity",
            "https://api.perplexity.ai",
            auth=AuthScheme("bearer", env=("PERPLEXITY_API_KEY",)),
            context_window=128_000,
            supports_vision=True,
            default_model="sonar-pro",
        ),
        _p(
            "cerebras",
            "https://api.cerebras.ai/v1",
            auth=AuthScheme("bearer", env=("CEREBRAS_API_KEY",)),
            context_window=32_768,
            default_model="llama3.1-8b",
        ),
        _p(
            "sambanova",
            "https://api.sambanova.ai/v1",
            auth=AuthScheme("bearer", env=("SAMBANOVA_API_KEY",)),
            context_window=32_768,
            default_model="Meta-Llama-3.3-70B-Instruct",
        ),
        _p(
            "nvidia",
            "https://integrate.api.nvidia.com/v1",
            auth=AuthScheme("bearer", env=("NVIDIA_API_KEY",)),
            context_window=131_072,
            supports_vision=True,
            default_model="meta/llama-3.3-70b-instruct",
            aliases=("nim", "nvidia-nim"),
        ),
        _p(
            "nebius",
            "https://api.studio.nebius.ai/v1",
            auth=AuthScheme("bearer", env=("NEBIUS_API_KEY",)),
            context_window=131_072,
            default_model="meta-llama-3.1-70b-instruct",
        ),
        _p(
            "baseten",
            "https://inference.baseten.co/v1",
            auth=AuthScheme("bearer", env=("BASETEN_API_KEY",)),
            context_window=131_072,
            default_model="meta-llama/Llama-3.3-70B-Instruct",
        ),
        _p(
            "hyperbolic",
            "https://api.hyperbolic.xyz/v1",
            auth=AuthScheme("bearer", env=("HYPERBOLIC_API_KEY",)),
            context_window=32_768,
            default_model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        ),
        _p(
            "novita",
            "https://api.novita.ai/v3/openai",
            auth=AuthScheme("bearer", env=("NOVITA_API_KEY",)),
            context_window=32_768,
            default_model="meta-llama/llama-3.3-70b-instruct",
        ),
        _p(
            "sakana",
            "https://api.saka.ai/v1",
            auth=AuthScheme("bearer", env=("SAKANA_API_KEY",)),
            context_window=32_768,
            default_model="sakana-ai-scientist",
        ),
        _p(
            "deepinfra",
            "https://api.deepinfra.com/v1/openai",
            auth=AuthScheme("bearer", env=("DEEPINFRA_API_KEY",)),
            context_window=131_072,
            supports_vision=True,
            default_model="meta-llama/Llama-3.3-70B-Instruct",
        ),
        _p(
            "friendliai",
            "https://api.friendli.ai/serverless/v1",
            auth=AuthScheme("bearer", env=("FRIENDLI_API_KEY",)),
            context_window=131_072,
            supports_vision=True,
            default_model="meta-llama-3.3-70b-instruct",
        ),
        _p(
            "llamacpp",
            "http://localhost:8080/v1",
            auth=AuthScheme("none"),
            context_window=32_768,
            default_model="local-model",
            aliases=("llama.cpp", "llama-cpp"),
            notes="llama.cpp server.",
        ),
        _p(
            "lmstudio",
            "http://localhost:1234/v1",
            auth=AuthScheme("none"),
            context_window=32_768,
            default_model="local-model",
            aliases=("lm-studio",),
        ),
        _p(
            "vllm",
            "http://localhost:8000/v1",
            auth=AuthScheme("none"),
            context_window=32_768,
            default_model="local-model",
            notes="vLLM OpenAI-compatible server.",
        ),
        _p(
            "openai_compat",
            "http://localhost:8000/v1",
            auth=AuthScheme("none"),
            context_window=32_768,
            default_model="local-model",
            aliases=("openai-compatible", "compat", "generic"),
            notes="Generic fallback for any OpenAI-compatible endpoint.",
        ),
    )
}


# Local runtimes that speak their own protocol.
LOCAL_PRESETS: dict[str, ProviderPreset] = {
    p.name: p
    for p in (
        _p(
            "ollama",
            "http://localhost:11434/v1",
            auth=AuthScheme("none"),
            context_window=32_768,
            default_model="qwen2.5-coder:7b",
            aliases=("ollama-native",),
            notes="Ollama's OpenAI-compatible surface.",
        ),
    )
}


def all_presets() -> dict[str, ProviderPreset]:
    """Every built-in preset, merged."""
    return {**BUILTIN_PRESETS, **LOCAL_PRESETS}


PRESETS: dict[str, ProviderPreset] = all_presets()


def find_preset(name: str) -> ProviderPreset | None:
    """Look up a preset by name, alias, or conventional spelling."""
    from .router import ProviderRouter

    return ProviderRouter()._find_preset(name)


def preset_names() -> list[str]:
    return sorted(all_presets())


def describe_preset(preset: ProviderPreset) -> str:
    """One-line human summary used by ``openagent models``."""
    flags = []
    if preset.supports_vision:
        flags.append("vision")
    if preset.supports_reasoning:
        flags.append("reasoning")
    flags.append(preset.protocol.value)
    return f"{preset.name:<12} {preset.base_url:<44} {preset.default_model:<28} [{','.join(flags)}]"
