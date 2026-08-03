"""Pluggable LLM provider behind the OpenRouter / internal-endpoint text-
generation calls.

legacy.query_engine and core.budget_narrator drive an LLMProvider through two
methods, generate_sql(prompt) and generate_json(prompt), instead of importing
openai directly. Provider selection (get_provider), in order:
  1. LOCAL_LLM_BASE_URL set -> OpenRouterProvider pointed at that internal
     OpenAI-compatible endpoint (vLLM/TGI etc — the company's own serving stack),
     using LOCAL_LLM_MODEL and (optional) LOCAL_LLM_API_KEY.
  2. OPENROUTER_API_KEY set -> OpenRouterProvider pointed at OpenRouter,
     currently configured for qwen/qwen3.6-27b.
  3. Neither set -> RuntimeError.
Both routes share the exact same class/retry logic — only base_url, model,
and api_key differ; a local deployment target is not a parallel
implementation, just a different set of constructor arguments.

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
# model/provider route, so the token budget is also raised as a second line
# of defense — belt and suspenders rather than a verified single fix.
#
# Confirmed against a live OpenRouter account (2026-07-31) that actual
# completion usage for this app's prompts (JSON step responses, narrative
# summaries) never exceeded ~600 tokens — 6000 was pure headroom, not a real
# requirement. OpenRouter pre-flight-checks whether the account could afford
# max_tokens in the worst case and rejects the call with a 402 if not
# ("requested up to 6000 tokens, but can only afford 3624"), even though the
# actual response would have used a fraction of that. A lower cap clears
# that gate on a near-empty balance without truncating any real response.
# Overridable via OPENROUTER_MAX_OUTPUT_TOKENS so this can be tuned against
# a specific account balance without a code change.
_OPENROUTER_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENROUTER_MAX_OUTPUT_TOKENS", "2000"))
_OPENROUTER_REASONING_EXTRA_BODY = {"reasoning": {"enabled": False}}

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Temporary: qwen/qwen3.6-27b costs real credits and the account is at ~$0.
# Re-checked 2026-08-02 (OpenRouter's public free-model catalog, 15 models,
# unchanged from the 2026-07-27 snapshot): still no free Qwen model (the free
# tier -- qwen3-coder, qwen3-next-80b -- was delisted in late July and hasn't
# come back) and DeepSeek has never had a free model on OpenRouter.
# nemotron-3-super-120b-a12b (120B total / 12B active) is the default again,
# reverted from nemotron-3-ultra-550b-a55b. Ultra was picked partly to fix
# "badly formatted answer" complaints -- but those turned out to be a bug in
# our own deterministic code (multi-row answers built as plain-text joins
# instead of markdown tables, fixed separately in analysis.py), not model
# quality. Real-world testing since the swap shows Ultra noticeably slower
# with no confirmed quality benefit for this app, so there's no longer a
# reason to pay its latency cost by default -- ~4.5x the active parameters
# (55B vs 12B) on a free/shared/rate-limited endpoint.
# Ultra is still one env var away for anyone who wants to trade speed for the
# larger model:
#   export OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
# Revert to a paid model (or point LOCAL_LLM_BASE_URL at a local one) once
# credits are added; swap back is this one line.
_OPENROUTER_DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# Many internal OpenAI-compatible servers (vLLM/TGI) accept any non-empty
# api_key string, or don't check it at all -- LOCAL_LLM_API_KEY is therefore
# optional, defaulting to this placeholder rather than requiring one.
_LOCAL_LLM_DEFAULT_API_KEY = "local"


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
    per PROJECT_NOTES.md's logging requirement. tokens_used is None (not guessed)
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
    """openai SDK client pointed at any OpenAI-compatible chat-completions
    endpoint -- OpenRouter by default, or an internal server (base_url
    overridden, e.g. the company's own vLLM/TGI deployment via LOCAL_LLM_BASE_URL).
    The retry/backoff/logging policy is identical either way; only
    base_url/model/api_key differ, so this stays one implementation rather
    than a parallel class per target."""

    def __init__(self, api_key: str, model: str | None = None, base_url: str = _OPENROUTER_BASE_URL) -> None:
        self._model = model or os.environ.get("OPENROUTER_MODEL", _OPENROUTER_DEFAULT_MODEL)
        # max_retries=0: retry/backoff is handled by _complete, not the SDK.
        self._client = openai.OpenAI(
            api_key=api_key, base_url=base_url, max_retries=0
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
                        "LLM provider model=%s returned empty message.content "
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

    Precedence: LOCAL_LLM_BASE_URL set -> that internal OpenAI-compatible
    endpoint (the deployment target, so it wins even if OPENROUTER_API_KEY
    is also set). Otherwise OPENROUTER_API_KEY -> OpenRouter. Otherwise ->
    RuntimeError.
    """
    local_base_url = os.environ.get("LOCAL_LLM_BASE_URL")
    if local_base_url:
        local_model = os.environ.get("LOCAL_LLM_MODEL")
        if not local_model:
            raise RuntimeError(
                "LOCAL_LLM_BASE_URL is set but LOCAL_LLM_MODEL is not. Set both, e.g.:\n"
                "    export LOCAL_LLM_BASE_URL=http://<llm-host>:<port>/v1\n"
                "    export LOCAL_LLM_MODEL=<served-checkpoint-name>"
            )
        local_api_key = os.environ.get("LOCAL_LLM_API_KEY") or _LOCAL_LLM_DEFAULT_API_KEY
        return OpenRouterProvider(local_api_key, model=local_model, base_url=local_base_url)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return OpenRouterProvider(openrouter_key)

    raise RuntimeError(
        "No LLM provider configured. Set LOCAL_LLM_BASE_URL (+ LOCAL_LLM_MODEL) "
        "to point at an internal OpenAI-compatible endpoint, or OPENROUTER_API_KEY "
        "for OpenRouter, e.g.:\n"
        "    export LOCAL_LLM_BASE_URL=http://<llm-host>:<port>/v1\n"
        "    export LOCAL_LLM_MODEL=<served-checkpoint-name>\n"
        "or:\n"
        "    export OPENROUTER_API_KEY=your_key_here"
    )
