"""Provider router and model reference parsing.

Resolves model names, aliases, and provider-prefixed references (e.g.
`gpt-4o`, `claude-3-5-sonnet`, `gemini-1.5-pro`, `ollama/llama3.2`, `custom/my-model`)
into configured ChatProvider instances.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .presets import ProviderPreset, all_presets, find_preset
from .provider import ChatProvider


@dataclass(slots=True, frozen=True)
class ModelReference:
    """Parsed model specification containing an optional provider hint and model name."""

    provider_hint: str | None
    model_name: str
    raw: str

    @classmethod
    def parse(cls, ref: str) -> ModelReference:
        raw = ref.strip()
        # Prioritize '/' so namespaces like 'ollama/llama3.2:latest' or 'openrouter/anthropic/claude'
        # are correctly split into provider_hint='ollama' and model_name='llama3.2:latest'
        if "/" in raw:
            hint, model = raw.split("/", 1)
            return cls(provider_hint=hint.strip() or None, model_name=model.strip(), raw=raw)
        if ":" in raw and (
            find_preset(raw.split(":", 1)[0]) is not None or raw.split(":", 1)[0] == "custom"
        ):
            hint, model = raw.split(":", 1)
            return cls(provider_hint=hint.strip() or None, model_name=model.strip(), raw=raw)
        return cls(provider_hint=None, model_name=raw, raw=raw)


class ProviderRouter:
    """Routes model references to instantiated ChatProvider implementations."""

    def __init__(self, presets: Mapping[str, ProviderPreset] | None = None) -> None:
        self._presets = dict(presets if presets is not None else all_presets())

    def _find_preset(self, name: str) -> ProviderPreset | None:
        key = name.strip().lower().replace("_", "-")
        for table_key, preset in self._presets.items():
            names = (table_key, preset.name, *preset.aliases)
            candidates = (key, key.rstrip("s"), f"{key}s")
            if any(
                candidate == n.lower().replace("_", "-") for n in names for candidate in candidates
            ):
                return preset
        return None

    def resolve(
        self,
        model_ref: str | ModelReference,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_headers: dict[str, str] | Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> ChatProvider:
        """Resolve a model name or reference into a ChatProvider instance."""
        ref = ModelReference.parse(model_ref) if isinstance(model_ref, str) else model_ref

        # 1. If provider hint is explicitly 'custom'
        if ref.provider_hint == "custom":
            from ..providers.custom import CustomJsonPathProvider

            return CustomJsonPathProvider(
                base_url=base_url or "http://localhost:8000",
                model=ref.model_name,
                api_key=api_key or os.environ.get("CUSTOM_API_KEY"),
                headers=extra_headers,
                **kwargs,
            )

        # 2. If provider hint matches a known preset
        if ref.provider_hint:
            preset = self._find_preset(ref.provider_hint)
            if preset is not None:
                return self._instantiate_preset(
                    preset,
                    model=ref.model_name,
                    api_key=api_key,
                    base_url=base_url,
                    extra_headers=extra_headers,
                    **kwargs,
                )
            if base_url is None:
                raise ValueError(
                    f"Unknown provider prefix {ref.provider_hint!r}; configure an explicit base_url"
                )
            from ..providers.openai_compat import OpenAICompatProvider

            return OpenAICompatProvider(
                base_url=base_url,
                model=ref.model_name,
                api_key=api_key,
                provider_name=ref.provider_hint,
                headers=extra_headers,
                **kwargs,
            )

        if ":" in ref.model_name:
            from ..providers.ollama import OllamaProvider

            return OllamaProvider(
                model=ref.model_name,
                base_url=base_url or "http://localhost:11434",
                api_key=api_key,
                extra_headers=extra_headers,
                **kwargs,
            )

        # 3. No provider hint: check if model_name matches a preset name / alias directly
        preset = self._find_preset(ref.model_name)
        if preset is not None:
            model = preset.default_model or ref.model_name
            return self._instantiate_preset(
                preset,
                model=model,
                api_key=api_key,
                base_url=base_url,
                extra_headers=extra_headers,
                **kwargs,
            )

        # 4. Check if any preset has default_model == ref.model_name
        model_lower = ref.model_name.lower()
        for p in self._presets.values():
            if p.default_model and p.default_model.lower() == model_lower:
                return self._instantiate_preset(
                    p,
                    model=ref.model_name,
                    api_key=api_key,
                    base_url=base_url,
                    extra_headers=extra_headers,
                    **kwargs,
                )

        # Route inferred model families through the same preset configuration path.
        family = None
        for prefixes, name in (
            (("claude",), "anthropic"),
            (("gemini",), "gemini"),
            (("gpt-", "o1", "o3", "o4", "chatgpt"), "openai"),
            (("deepseek",), "deepseek"),
        ):
            if model_lower.startswith(prefixes):
                family = self._find_preset(name)
                break
        if family is not None:
            return self._instantiate_preset(
                family,
                model=ref.model_name,
                api_key=api_key,
                base_url=base_url,
                extra_headers=extra_headers,
                **kwargs,
            )

        # 6. Fallback to OpenAICompatProvider
        from ..providers.openai_compat import OpenAICompatProvider

        key = api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAICompatProvider(
            base_url=base_url or "https://api.openai.com/v1",
            model=ref.model_name,
            api_key=key,
            headers=extra_headers,
            **kwargs,
        )

    def _instantiate_preset(
        self,
        preset: ProviderPreset,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None,
        extra_headers: dict[str, str] | Mapping[str, str] | None,
        **kwargs: Any,
    ) -> ChatProvider:
        if preset.kind in ("azure", "bedrock"):
            raise ValueError(
                f"Provider {preset.name!r} is not implemented; use a configured OpenAI-compatible endpoint"
            )
        key = preset.auth.resolve_key(api_key) if preset.auth.scheme != "none" else None
        kwargs.setdefault("extra_body", preset.body_defaults)
        effective_base_url = base_url or preset.base_url
        merged_headers = {**preset.headers, **dict(extra_headers or {})}
        provider: ChatProvider

        if preset.kind == "anthropic" or preset.name == "anthropic":
            from ..providers.anthropic import AnthropicProvider

            provider = AnthropicProvider(
                model=model,
                base_url=effective_base_url,
                api_key=key,
                extra_headers=merged_headers,
                context_window=preset.context_window,
                **kwargs,
            )
            provider.default_max_tokens = preset.max_output
            provider.tool_protocol = preset.protocol
            return provider

        if preset.kind == "gemini" or preset.name == "gemini":
            from ..providers.gemini import GeminiProvider

            provider = GeminiProvider(
                model=model,
                base_url=effective_base_url,
                api_key=key,
                extra_headers=merged_headers,
                context_window=preset.context_window,
                **kwargs,
            )
            provider.default_max_tokens = preset.max_output
            provider.tool_protocol = preset.protocol
            return provider

        if preset.kind == "ollama" or preset.name == "ollama":
            from ..providers.ollama import OllamaProvider

            provider = OllamaProvider(
                model=model,
                base_url=effective_base_url,
                api_key=key,
                extra_headers=merged_headers,
                context_window=preset.context_window,
                **kwargs,
            )
            provider.default_max_tokens = preset.max_output
            provider.tool_protocol = preset.protocol
            return provider

        # Default to OpenAICompatProvider
        from ..providers.openai_compat import OpenAICompatProvider

        provider = OpenAICompatProvider(
            base_url=effective_base_url,
            model=model,
            api_key=key,
            provider_name=preset.name,
            headers=merged_headers,
            auth_header=preset.auth.header
            or ("authorization" if preset.auth.scheme == "bearer" else preset.auth.scheme),
            context_window=preset.context_window,
            **kwargs,
        )

        provider.default_max_tokens = preset.max_output
        provider.tool_protocol = preset.protocol
        return provider
