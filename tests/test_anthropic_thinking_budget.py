from typing import Any

import pytest

from openagent.core.types import ChatRequest, Message, ThinkingBudget, ToolSpec
from openagent.providers.anthropic import AnthropicProvider


def request(max_tokens: int | None = 2048, budget: Any = None, **kwargs: Any) -> ChatRequest:
    return ChatRequest(
        model="fixture", messages=[Message.user("hello")], max_tokens=max_tokens,
        thinking=ThinkingBudget(enabled=True, budget_tokens=budget), **kwargs,
    )


@pytest.mark.parametrize(("maximum", "expected"), [(1025, 1024), (2048, 2047), (4096, 4095), (8192, 4096), (None, 4096)])
def test_default_budget_fits_output_limit(maximum: int | None, expected: int) -> None:
    payload = AnthropicProvider("fixture").build_payload(request(maximum))
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": expected}


def test_default_budget_uses_provider_default() -> None:
    provider = AnthropicProvider("fixture")
    provider.default_max_tokens = 2048
    assert provider.build_payload(request(None))["thinking"]["budget_tokens"] == 2047


@pytest.mark.parametrize("budget", [0, -1, 1023, 2048, 4096, True, 1024.5, "1024"])
def test_invalid_explicit_budget_fails_locally(budget: Any) -> None:
    with pytest.raises(ValueError, match="budget_tokens"):
        AnthropicProvider("fixture").build_payload(request(budget=budget))


@pytest.mark.parametrize("maximum", [0, 1, 1024])
def test_default_budget_requires_room_for_minimum(maximum: int) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        AnthropicProvider("fixture").build_payload(request(maximum))


def test_explicit_budget_preserved_and_sampling_removed() -> None:
    payload = AnthropicProvider("fixture").build_payload(
        request(budget=1024, temperature=0.5, top_p=0.9)
    )
    assert payload["thinking"]["budget_tokens"] == 1024
    assert "temperature" not in payload
    assert "top_p" not in payload


@pytest.mark.parametrize(("beta", "tools", "choice", "allowed"), [
    ("interleaved-thinking-2025-05-14", True, "auto", True),
    ("other, interleaved-thinking-2025-05-14", True, "auto", True),
    ("interleaved-thinking-2025-05-14", False, "auto", False),
    ("interleaved-thinking-2025-05-14", True, "none", False),
    ("other", True, "auto", False),
])
def test_interleaved_budget_exception_requires_beta_and_tools(
    beta: str, tools: bool, choice: Any, allowed: bool,
) -> None:
    provider = AnthropicProvider("fixture", extra_headers={"Anthropic-Beta": beta})
    req = request(budget=4096, tools=[ToolSpec("lookup", "Lookup")] if tools else [], tool_choice=choice)
    if allowed:
        assert provider.build_payload(req)["thinking"]["budget_tokens"] == 4096
    else:
        with pytest.raises(ValueError, match="budget_tokens"):
            provider.build_payload(req)


def test_extra_body_effective_limit_controls_default() -> None:
    payload = AnthropicProvider("fixture", extra_body={"max_tokens": 1500}).build_payload(request())
    assert payload["thinking"]["budget_tokens"] == 1499


@pytest.mark.parametrize("extra", [
    {"max_tokens": 1024},
    {"thinking": {"type": "enabled", "budget_tokens": 2048}},
    {"thinking": {"type": "enabled", "budget_tokens": 0}},
    {"thinking": {"type": "enabled"}},
])
def test_extra_body_cannot_bypass_manual_budget_validation(extra: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        AnthropicProvider("fixture", extra_body=extra).build_payload(request())


def test_advanced_extra_body_thinking_modes_are_preserved() -> None:
    provider = AnthropicProvider("fixture", extra_body={"thinking": {"type": "adaptive"}})
    assert provider.build_payload(request())["thinking"] == {"type": "adaptive"}


def test_disabled_thinking_unaffected() -> None:
    req = request(100, budget=0, temperature=0.5)
    req.thinking.enabled = False
    payload = AnthropicProvider("fixture").build_payload(req)
    assert "thinking" not in payload
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 100
