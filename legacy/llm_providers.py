"""Pluggable LLM provider behind the OpenRouter text-generation calls.

legacy.query_engine and core.budget_narrator drive an LLMProvider through two
methods, generate_sql(prompt) and generate_json(prompt), instead of importing
openai directly. Provider selection (get_provider): OPENROUTER_API_KEY set ->
OpenRouterProvider (openai SDK, base_url pointed at OpenRouter, currently
configured for qwen/qwen3.6-27b). Without OPENROUTER_API_KEY, get_provider()
raises RuntimeError. This is the only LLM provider in this codebase.

OpenRouterProvider implements a retry/backoff policy (_RETRY_STATUS,
_MAX_ATTEMPTS, _BASE_DELAY_S) and raises DailyQuotaExhausted on a
non-retryable daily-quota 429. The daily-quota markers below
(_DAILY_QUOTA_MARKERS) are a historically observed quota-exhaustion wording
("PerDay" / "RPD" quota IDs), not yet confirmed against a live OpenRouter
daily-quota response, so an OpenRouter 429 that doesn't match these markers
is retried like any other transient error and eventually raised as a plain
openai.APIStatusError rather than DailyQuotaExhausted. Adjust
_DAILY_QUOTA_MARKERS once real OpenRouter quota-error text is seen.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Protocol

import openai

logger = logging.getLogger(__name__)

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 5
_BASE_DELAY_S = 2.0
_DAILY_QUOTA_MARKERS = ("PerDay", "per_day", "RPD")

# Some OpenRouter models (observed with Qwen3.6) run an integrated
# reasoning/"thinking" pass that consumes output tokens before the visible
# answer is written, and can exhaust a small budget leaving message.content
# empty. _OPENROUTER_REASONING_EXTRA_BODY asks OpenRouter's unified request
# format to disable that pass; it's not confirmed to be honored by every
# model/provider route, so the token budget is also raised (6000) as a
# second line of defense — belt and suspenders rather than a verified single
# fix, since this hasn't been checked against a live OpenRouter account.
_OPENROUTER_MAX_OUTPUT_TOKENS = 6000
_OPENROUTER_REASONING_EXTRA_BODY = {"reasoning": {"enabled": False}}

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_DEFAULT_MODEL = "qwen/qwen3.6-27b"


class DailyQuotaExhausted(Exception):
    """Raised when the active provider's daily API quota is exhausted (not retryable)."""


class LLMProvider(Protocol):
    def generate_sql(self, prompt: str) -> str: ...
    def generate_json(self, prompt: str) -> str: ...


def _is_daily_quota_error(status_code: int | None, message: str) -> bool:
    """True if a 429 is a per-day quota exhaustion (vs. a per-minute rate limit)."""
    if status_code != 429:
        return False
    return any(marker in message for marker in _DAILY_QUOTA_MARKERS)


def _log_call(model: str, started_at: float, response, attempt: int) -> None:
    """Log a successful LLM API call: timestamp, tokens_used, latency_ms —
    per the project's logging requirement. tokens_used is None (not guessed)
    when response.usage.total_tokens isn't populated (OpenRouter's Qwen
    route may or may not include it)."""
    latency_ms = (time.perf_counter() - started_at) * 1000
    usage = getattr(response, "usage", None)
    tokens_used = getattr(usage, "total_tokens", None) if usage is not None else None
    logger.info(
        "LLM call succeeded timestamp=%s model=%s attempt=%d tokens_used=%s latency_ms=%.1f",
        datetime.now(timezone.utc).isoformat(), model, attempt + 1, tokens_used, latency_ms,
    )


class OpenRouterProvider:
    """openai SDK client pointed at OpenRouter."""

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self._model = model or os.environ.get("OPENROUTER_MODEL", _OPENROUTER_DEFAULT_MODEL)
        # max_retries=0: retry/backoff is handled by _complete, not the SDK.
        self._client = openai.OpenAI(
            api_key=api_key, base_url=_OPENROUTER_BASE_URL, max_retries=0
        )

    def _complete(self, prompt: str) -> str:
        for attempt in range(_MAX_ATTEMPTS):
            started_at = time.perf_counter()
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=_OPENROUTER_MAX_OUTPUT_TOKENS,
                    extra_body=_OPENROUTER_REASONING_EXTRA_BODY,
                )
                content = response.choices[0].message.content or ""
                if not content.strip():
                    logger.warning(
                        "OpenRouter model=%s returned empty message.content "
                        "(reasoning mode may have consumed the output budget)",
                        self._model,
                    )
                _log_call(self._model, started_at, response, attempt)
                return content
            except openai.APIStatusError as exc:
                latency_ms = (time.perf_counter() - started_at) * 1000
                if _is_daily_quota_error(exc.status_code, str(exc)):
                    logger.info(
                        "LLM call failed timestamp=%s model=%s attempt=%d "
                        "latency_ms=%.1f outcome=daily_quota_exhausted",
                        datetime.now(timezone.utc).isoformat(),
                        self._model, attempt + 1, latency_ms,
                    )
                    raise DailyQuotaExhausted(str(exc)) from exc
                transient = exc.status_code in _RETRY_STATUS
                if not transient or attempt == _MAX_ATTEMPTS - 1:
                    logger.info(
                        "LLM call failed timestamp=%s model=%s attempt=%d "
                        "latency_ms=%.1f outcome=%s",
                        datetime.now(timezone.utc).isoformat(), self._model,
                        attempt + 1, latency_ms,
                        "non_retryable" if not transient else "retries_exhausted",
                    )
                    raise
                logger.info(
                    "LLM call retrying timestamp=%s model=%s attempt=%d latency_ms=%.1f",
                    datetime.now(timezone.utc).isoformat(), self._model, attempt + 1, latency_ms,
                )
                time.sleep(_BASE_DELAY_S * (2**attempt))
            except (openai.APIConnectionError, openai.APITimeoutError):
                # Network reset/timeout, not a status response — no status
                # code to inspect, just retry with the same backoff.
                latency_ms = (time.perf_counter() - started_at) * 1000
                if attempt == _MAX_ATTEMPTS - 1:
                    logger.info(
                        "LLM call failed timestamp=%s model=%s attempt=%d "
                        "latency_ms=%.1f outcome=connection_error",
                        datetime.now(timezone.utc).isoformat(),
                        self._model, attempt + 1, latency_ms,
                    )
                    raise
                logger.info(
                    "LLM call retrying timestamp=%s model=%s attempt=%d latency_ms=%.1f",
                    datetime.now(timezone.utc).isoformat(), self._model, attempt + 1, latency_ms,
                )
                time.sleep(_BASE_DELAY_S * (2**attempt))
        raise RuntimeError("retry loop exited without a result")

    def generate_sql(self, prompt: str) -> str:
        return self._complete(prompt)

    def generate_json(self, prompt: str) -> str:
        return self._complete(prompt)


def get_provider() -> LLMProvider:
    """Select the active provider, or fail with a clear message.

    OPENROUTER_API_KEY set -> OpenRouter. Otherwise -> RuntimeError.
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return OpenRouterProvider(openrouter_key)

    raise RuntimeError(
        "No LLM provider configured. Set OPENROUTER_API_KEY, e.g.:\n"
        "    export OPENROUTER_API_KEY=your_key_here"
    )
