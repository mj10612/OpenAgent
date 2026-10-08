"""HTTP transport with retry, backoff and error mapping.

One class handles all providers because 95% of an adapter's HTTP behaviour is
identical; only payload construction differs, and that lives in subclasses.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any, Self

import httpx

from ..core.provider import (
    AuthenticationError,
    ContextWindowError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)
from .sse import ErrorInfo, SSEvent, iter_sse, parse_json_payload

#: Status codes worth retrying: transient server and rate-limit conditions.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 522, 524})

#: Substrings that identify a context-window overflow in an error message.
CONTEXT_MARKERS = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "too many tokens",
    "prompt is too long",
    "reduce the length",
    "exceeds the maximum",
    "input is too long",
    "string too long",
    "token limit",
)

AUTH_MARKERS = ("unauthorized", "invalid api key", "invalid_api_key", "authentication", "forbidden")
NOT_FOUND_MARKERS = ("model not found", "does not exist", "unknown model", "model_not_found")


def classify(
    status: int, info: ErrorInfo, *, provider: str, retry_after: float | None = None
) -> ProviderError:
    """Map an HTTP status plus error body onto a typed exception."""
    message = info.message or f"HTTP {status}"
    haystack = message.lower()

    if status in (401, 403) or any(m in haystack for m in AUTH_MARKERS):
        return AuthenticationError(message, status=status, provider=provider)
    if status == 404 and any(m in haystack for m in NOT_FOUND_MARKERS):
        return ModelNotFoundError(message, status=status, provider=provider)
    if any(m in haystack for m in CONTEXT_MARKERS):
        return ContextWindowError(message, status=status, provider=provider)
    if status == 429:
        return RateLimitError(message, status=status, provider=provider, retry_after=retry_after)
    retryable = status in RETRYABLE_STATUS
    return ProviderError(message, status=status, retryable=retryable, provider=provider)


class HttpTransport:
    """Thin, provider-neutral HTTP client.

    Deliberately *not* a thin wrapper: it owns retry policy, header assembly and
    the mapping from wire errors to typed exceptions, all of which every adapter
    would otherwise reimplement inconsistently.
    """

    def __init__(
        self,
        base_url: str,
        *,
        provider: str,
        headers: Mapping[str, str] | None = None,
        timeout: float = 600.0,
        connect_timeout: float = 15.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.extra_headers = dict(headers or {})
        self.max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            follow_redirects=False,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        )

    # -- lifecycle --------------------------------------------------------- #

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- requests ---------------------------------------------------------- #

    async def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """POST a JSON body and return the decoded response.

        Retries transient failures with exponential backoff and full jitter,
        because synchronous backoff across many concurrent agents amplifies
        exactly the thundering herd it is meant to dampen.
        """
        merged = {**self.extra_headers, "content-type": "application/json", **(headers or {})}
        url = path if path.startswith(("http://", "https://")) else f"{self.base_url}{path}"
        body = dict(payload)
        last_error: ProviderError | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(
                    url, json=body, headers=merged, follow_redirects=False
                )
            except httpx.TimeoutException as exc:
                last_error = ProviderError(
                    f"request timed out: {exc}", retryable=True, provider=self.provider
                )
            except httpx.HTTPError as exc:
                last_error = ProviderError(
                    f"network error: {exc}", retryable=True, provider=self.provider
                )
            else:
                self._reject_redirect(response)
                if response.status_code < 400:
                    return parse_json_payload(response.text, provider=self.provider)

                info = ErrorInfo.from_body(_safe_json(response.text), provider=self.provider)
                error = classify(
                    response.status_code,
                    info,
                    provider=self.provider,
                    retry_after=_retry_after(response.headers.get("retry-after")),
                )
                if not error.retryable or attempt >= self.max_retries:
                    raise error
                last_error = error
                if isinstance(error, RateLimitError) and error.retry_after is not None:
                    await asyncio.sleep(min(error.retry_after, 60.0))
                    continue

            await asyncio.sleep(_backoff(attempt))

        assert last_error is not None  # loop always assigns before sleeping
        raise last_error

    async def stream_post(
        self,
        path: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """POST and yield the still-open response for incremental decoding.

        No retry here: by the time the stream is open we may already have
        delivered tokens to the user, so silently restarting would duplicate
        output. Callers decide whether a mid-stream failure is recoverable.
        """
        merged = {
            **self.extra_headers,
            "content-type": "application/json",
            "accept": "text/event-stream",
            "cache-control": "no-cache",
            **(headers or {}),
        }
        url = path if path.startswith(("http://", "https://")) else f"{self.base_url}{path}"
        req = self._client.build_request("POST", url, json=dict(payload), headers=merged)
        try:
            response = await self._client.send(req, stream=True, follow_redirects=False)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"request timed out: {exc}", retryable=True, provider=self.provider
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"network error: {exc}", retryable=True, provider=self.provider
            ) from exc
        if 300 <= response.status_code < 400:
            await response.aclose()
            self._reject_redirect(response)

        if response.status_code >= 400:
            # The body is still unreadable while streaming, so buffer it first.
            try:
                raw = (await response.aread()).decode("utf-8", "replace")
            finally:
                await response.aclose()
            info = ErrorInfo.from_body(_safe_json(raw), provider=self.provider)
            raise classify(
                response.status_code,
                info,
                provider=self.provider,
                retry_after=_retry_after(response.headers.get("retry-after")),
            )

        return response

    async def post_stream(
        self,
        path: str,
        json: Mapping[str, Any] | None = None,
        *,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """Stream POST a request, returning the open response object."""
        body = json if json is not None else (payload or {})
        return await self.stream_post(path, body, headers=headers)

    async def get_json(
        self,
        path: str,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        merged = {**self.extra_headers, **(headers or {})}
        url = path if path.startswith(("http://", "https://")) else f"{self.base_url}{path}"
        response = await self._client.get(url, headers=merged, follow_redirects=False)
        self._reject_redirect(response)
        if response.status_code >= 400:
            info = ErrorInfo.from_body(_safe_json(response.text), provider=self.provider)
            raise classify(response.status_code, info, provider=self.provider)
        return parse_json_payload(response.text, provider=self.provider)

    async def stream_sse(
        self,
        path: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[SSEvent]:
        """Convenience: POST a streaming request and decode it into events."""
        response = await self.stream_post(path, payload, headers)
        try:
            async for event in iter_sse(response):
                yield event
        finally:
            await response.aclose()

    def _reject_redirect(self, response: httpx.Response) -> None:
        if 300 <= response.status_code < 400:
            raise ProviderError(
                f"HTTP {response.status_code} redirect to {response.headers.get('location', '<none>')!r}; provider base URLs must point at the API endpoint directly",
                status=response.status_code,
                provider=self.provider,
            )


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            delay = (date - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, delay) if math.isfinite(delay) else None


def _backoff(attempt: int, *, base: float = 0.8, cap: float = 30.0) -> float:
    """Exponential backoff with full jitter."""
    window = min(cap, base * (2**attempt))
    return random.uniform(0, window)


def _safe_json(text: str) -> Any:
    """Parse JSON, falling back to the raw string; never raises."""
    try:
        return parse_json_payload(text)
    except ProviderError:
        return text
