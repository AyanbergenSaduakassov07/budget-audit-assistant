"""Pluggable LLM provider behind the Gemini/OpenRouter text-generation calls.

legacy.query_engine and core.budget_narrator drive an LLMProvider through two
methods, generate_sql(prompt) and generate_json(prompt), instead of importing
google.genai (or openai) directly. Provider selection (get_provider):
OPENROUTER_API_KEY set -> OpenRouterProvider (openai SDK, base_url pointed at
OpenRouter). Gemini never activates implicitly (per ACM Digital VP
directive) -- GeminiProvider only constructs when ALLOW_GEMINI=1 is ALSO set
explicitly alongside GEMINI_API_KEY; GEMINI_API_KEY being present is no
longer sufficient by itself. Without OPENROUTER_API_KEY and without
ALLOW_GEMINI=1, get_provider() raises RuntimeError even if GEMINI_API_KEY
is set.

Both providers implement the same retry/backoff policy (_RETRY_STATUS,
_MAX_ATTEMPTS, _BASE_DELAY_S) and raise DailyQuotaExhausted on a
non-retryable daily-quota 429, so callers don't need to know which provider
is active. The daily-quota markers below are Gemini's documented wording
("PerDay" / "RPD" quota IDs); OpenRouter's equivalent has not been verified
against a live daily-quota response, so an OpenRouter 429 that doesn't match
these markers is retried like any other transient error and eventually
raised as a plain openai.APIStatusError rather than DailyQuotaExhausted.
Adjust _DAILY_QUOTA_MARKERS once real OpenRouter quota-error text is seen.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Protocol

import openai
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 5
_BASE_DELAY_S = 2.0
_DAILY_QUOTA_MARKERS = ("PerDay", "per_day", "RPD")

# Caps output length so a call reserves credit against a small, predictable
# ceiling instead of the model's full output limit (e.g. 65536 tokens on some
# OpenRouter models) — uncapped calls fail with 402 on accounts without a
# large balance even though actual responses (SQL, short Russian answers) are
# a few hundred tokens at most.
_GEMINI_MAX_OUTPUT_TOKENS = 2048

# Some OpenRouter models (observed with Qwen3.6) run an integrated
# reasoning/"thinking" pass that consumes output tokens before the visible
# answer is written, and can exhaust a small budget leaving message.content
# empty. _OPENROUTER_REASONING_EXTRA_BODY asks OpenRouter's unified request
# format to disable that pass; it's not confirmed to be honored by every
# model/provider route, so the token budget is also raised (6000, vs.
# Gemini's 2048) as a second line of defense — belt and suspenders rather
# than a verified single fix, since this hasn't been checked against a live
# OpenRouter account.
_OPENROUTER_MAX_OUTPUT_TOKENS = 6000
_OPENROUTER_REASONING_EXTRA_BODY = {"reasoning": {"enabled": False}}

_GEMINI_MODEL = "gemini-flash-lite-latest"
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


class GeminiProvider:
    """Google Gemini via google-genai. Default provider; behavior unchanged."""

    def __init__(self, api_key: str) -> None:
        self._client = genai.Client(api_key=api_key)

    def _complete(self, prompt: str) -> str:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = self._client.models.generate_content(
                    model=_GEMINI_MODEL,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=_GEMINI_MAX_OUTPUT_TOKENS
                    ),
                )
                return response.text or ""
            except genai_errors.APIError as exc:
                if _is_daily_quota_error(exc.code, str(exc)):
                    raise DailyQuotaExhausted(str(exc)) from exc
                transient = exc.code in _RETRY_STATUS
                if not transient or attempt == _MAX_ATTEMPTS - 1:
                    raise
                time.sleep(_BASE_DELAY_S * (2**attempt))
        raise RuntimeError("retry loop exited without a result")

    def generate_sql(self, prompt: str) -> str:
        return self._complete(prompt)

    def generate_json(self, prompt: str) -> str:
        return self._complete(prompt)


class OpenRouterProvider:
    """openai SDK client pointed at OpenRouter; same retry/quota contract as Gemini."""

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self._model = model or os.environ.get("OPENROUTER_MODEL", _OPENROUTER_DEFAULT_MODEL)
        # max_retries=0: retry/backoff is handled by _complete, not the SDK,
        # so both providers go through one identifiable retry policy.
        self._client = openai.OpenAI(
            api_key=api_key, base_url=_OPENROUTER_BASE_URL, max_retries=0
        )

    def _complete(self, prompt: str) -> str:
        for attempt in range(_MAX_ATTEMPTS):
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
                return content
            except openai.APIStatusError as exc:
                if _is_daily_quota_error(exc.status_code, str(exc)):
                    raise DailyQuotaExhausted(str(exc)) from exc
                transient = exc.status_code in _RETRY_STATUS
                if not transient or attempt == _MAX_ATTEMPTS - 1:
                    raise
                time.sleep(_BASE_DELAY_S * (2**attempt))
            except (openai.APIConnectionError, openai.APITimeoutError):
                # Network reset/timeout, not a status response — no status
                # code to inspect, just retry with the same backoff.
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
                time.sleep(_BASE_DELAY_S * (2**attempt))
        raise RuntimeError("retry loop exited without a result")

    def generate_sql(self, prompt: str) -> str:
        return self._complete(prompt)

    def generate_json(self, prompt: str) -> str:
        return self._complete(prompt)


def get_provider() -> LLMProvider:
    """Select the active provider, or fail with a clear message.

    OPENROUTER_API_KEY set -> OpenRouter. Otherwise, GEMINI_API_KEY alone is
    NOT enough -- Gemini must never activate implicitly (per ACM Digital VP
    directive). GeminiProvider only constructs when ALLOW_GEMINI=1 is ALSO
    set explicitly. Neither OPENROUTER_API_KEY nor an explicitly-allowed
    Gemini -> RuntimeError.
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return OpenRouterProvider(openrouter_key)

    gemini_key = os.environ.get("GEMINI_API_KEY")
    allow_gemini = os.environ.get("ALLOW_GEMINI") == "1"
    if gemini_key and allow_gemini:
        return GeminiProvider(gemini_key)

    raise RuntimeError(
        "No LLM provider configured. Set OPENROUTER_API_KEY (OpenRouter). "
        "Gemini is not used unless explicitly allowed: set GEMINI_API_KEY "
        "AND ALLOW_GEMINI=1, e.g.:\n"
        "    export OPENROUTER_API_KEY=your_key_here\n"
        "or:\n"
        "    export GEMINI_API_KEY=your_key_here\n"
        "    export ALLOW_GEMINI=1"
    )
