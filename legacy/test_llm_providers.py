"""Tests for legacy.llm_providers.OpenRouterProvider — no live API calls.

Covers two failure modes found testing against real Qwen3.6-27B responses:
empty message.content (reasoning mode exhausting the output budget) and an
uncaught network error crashing the retry loop instead of retrying it.
"""

import contextlib
import logging

import httpx
import openai
import pytest

from legacy.llm_providers import (
    DailyQuotaExhausted,
    OpenRouterProvider,
    _OPENROUTER_MAX_OUTPUT_TOKENS,
    _OPENROUTER_REASONING_EXTRA_BODY,
    _daily_quota_limit_source,
    _is_daily_quota_error,
    get_provider,
)


def _provider(client) -> OpenRouterProvider:
    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    provider._model = "test-model"
    provider._client = client
    return provider


def _completion(content: str, usage_total_tokens: int | None = None):
    class _Msg:
        pass

    class _Choice:
        pass

    class _Usage:
        pass

    class _Response:
        pass

    msg = _Msg()
    msg.content = content
    choice = _Choice()
    choice.message = msg
    response = _Response()
    response.choices = [choice]
    if usage_total_tokens is not None:
        usage = _Usage()
        usage.total_tokens = usage_total_tokens
        response.usage = usage
    return response


@contextlib.contextmanager
def _captured_logs():
    """Attach a temporary handler to legacy.llm_providers' logger (and force
    its level to INFO -- the default WARNING would otherwise filter out
    logger.info() calls before any handler ever sees them); yields the list
    of captured message strings (mutated in place as records arrive)."""
    caught = []

    class _Handler(logging.Handler):
        def emit(self, record):
            caught.append(record.getMessage())

    logger = logging.getLogger("legacy.llm_providers")
    handler = _Handler()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield caught
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _connection_error() -> openai.APIConnectionError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return openai.APIConnectionError(message="Connection reset", request=request)


# --- empty message.content -------------------------------------------------------
def test_complete_logs_warning_on_empty_content(monkeypatch):
    """An empty message.content (e.g. reasoning ate the whole budget) must be
    surfaced as a warning, not silently returned as if nothing happened."""

    class _FakeClient:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            self.kwargs = kwargs
            return _completion("")

    fake_client = _FakeClient()
    provider = _provider(fake_client)

    caught = []
    import logging

    class _Handler(logging.Handler):
        def emit(self, record):
            caught.append(record.getMessage())

    logger = logging.getLogger("legacy.llm_providers")
    handler = _Handler()
    logger.addHandler(handler)
    try:
        result = provider._complete("prompt")
    finally:
        logger.removeHandler(handler)

    assert result == ""
    assert any("empty message.content" in msg for msg in caught)


def test_complete_passes_max_tokens_and_disables_reasoning():
    """max_tokens must be capped (not the model's full output ceiling), and
    reasoning must be requested off via OpenRouter's extra_body — the two
    mitigations for Qwen3.6 exhausting its budget on hidden reasoning."""

    class _FakeClient:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return _completion("ok")

    fake_client = _FakeClient()
    provider = _provider(fake_client)

    assert provider._complete("prompt") == "ok"
    assert fake_client.kwargs["max_tokens"] == _OPENROUTER_MAX_OUTPUT_TOKENS
    assert fake_client.kwargs["extra_body"] == _OPENROUTER_REASONING_EXTRA_BODY


# --- call logging (PROJECT_NOTES.md: every LLM API call gets timestamp/tokens/latency) --
def test_complete_logs_call_details_on_success():
    """A successful call must log a timestamp, the model, the attempt number,
    tokens_used, and latency_ms."""

    class _FakeClient:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            return _completion("ok")

    provider = _provider(_FakeClient())

    with _captured_logs() as caught:
        assert provider._complete("prompt") == "ok"

    (msg,) = [m for m in caught if m.startswith("LLM call succeeded")]
    assert "timestamp=" in msg
    assert "model=test-model" in msg
    assert "attempt=1" in msg
    assert "latency_ms=" in msg
    # response.usage wasn't set on the fake response -> logged as None, not guessed.
    assert "tokens_used=None" in msg


def test_complete_logs_actual_token_count_when_response_provides_it():
    """When response.usage.total_tokens IS populated, log the real number."""

    class _FakeClient:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            return _completion("ok", usage_total_tokens=123)

    provider = _provider(_FakeClient())

    with _captured_logs() as caught:
        assert provider._complete("prompt") == "ok"

    (msg,) = [m for m in caught if m.startswith("LLM call succeeded")]
    assert "tokens_used=123" in msg


def test_complete_logs_each_retry_attempt(monkeypatch):
    """Every retried attempt must be logged, not just the final success."""

    class _FakeClient:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise _connection_error()
            return _completion("recovered")

    provider = _provider(_FakeClient())
    monkeypatch.setattr("legacy.llm_providers.time.sleep", lambda *_a, **_k: None)

    with _captured_logs() as caught:
        assert provider._complete("prompt") == "recovered"

    retry_msgs = [m for m in caught if m.startswith("LLM call retrying")]
    assert len(retry_msgs) == 2  # attempts 1 and 2 failed, attempt 3 succeeded
    assert "attempt=1" in retry_msgs[0]
    assert "attempt=2" in retry_msgs[1]
    assert any(m.startswith("LLM call succeeded") for m in caught)


def test_complete_logs_final_failure(monkeypatch):
    """Exhausting all retries must log a final failure with attempt/latency,
    even though no token count is available."""

    class _FakeClient:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            raise _connection_error()

    provider = _provider(_FakeClient())
    monkeypatch.setattr("legacy.llm_providers.time.sleep", lambda *_a, **_k: None)

    with _captured_logs() as caught:
        with pytest.raises(openai.APIConnectionError):
            provider._complete("prompt")

    (msg,) = [m for m in caught if m.startswith("LLM call failed")]
    assert "outcome=connection_error" in msg
    assert "latency_ms=" in msg


# --- retry on network errors ------------------------------------------------------
def test_complete_retries_on_connection_error_then_succeeds(monkeypatch):
    """A transient APIConnectionError (network reset/timeout) must be retried
    with the same backoff as a status error, not crash the call outright."""

    class _FakeClient:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise _connection_error()
            return _completion("recovered")

    fake_client = _FakeClient()
    provider = _provider(fake_client)

    monkeypatch.setattr("legacy.llm_providers.time.sleep", lambda *_a, **_k: None)

    assert provider._complete("prompt") == "recovered"
    assert fake_client.calls == 2


def test_complete_gives_up_after_max_attempts_of_connection_errors(monkeypatch):
    class _FakeClient:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise _connection_error()

    fake_client = _FakeClient()
    provider = _provider(fake_client)

    monkeypatch.setattr("legacy.llm_providers.time.sleep", lambda *_a, **_k: None)

    with pytest.raises(openai.APIConnectionError):
        provider._complete("prompt")


# --- daily-quota 429 detection ----------------------------------------------------
# The real OpenRouter daily-quota response, captured live 2026-08-03. The old
# check matched str(exc) against ("PerDay", "per_day", "RPD") -- none of which
# appear anywhere in this text -- so a genuine quota-out fell through to the
# transient-429 path: 5 attempts with exponential backoff (~62s) on a request
# that could not succeed, then a plain APIStatusError the caller showed as a
# generic "не нашёл точного ответа".
_REAL_DAILY_QUOTA_BODY = {
    "error": {
        "message": (
            "Rate limit exceeded: free-models-per-day. Add 10 credits to "
            "unlock 1000 free model requests per day"
        ),
        "code": 429,
        "metadata": {
            "headers": {
                "X-RateLimit-Limit": "50",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1785801600000",
            },
            "limit_source": "openrouter_free_tier_daily",
            "provider_name": None,
        },
        "user_id": "user_3EmUkUQsYGGE8fNJQnshpTypYz4",
    }
}


def _status_error(status_code: int, body, message: str = "Error") -> openai.APIStatusError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status_code, request=request, json=body)
    return openai.APIStatusError(message, response=response, body=body)


def test_real_openrouter_daily_quota_body_is_detected():
    exc = _status_error(429, _REAL_DAILY_QUOTA_BODY)
    # None of the legacy message markers appear in this response...
    assert "PerDay" not in str(exc) and "RPD" not in str(exc)
    # ...but metadata.limit_source identifies it structurally.
    assert _daily_quota_limit_source(exc) == "openrouter_free_tier_daily"
    assert _is_daily_quota_error(exc) is True


def test_real_daily_quota_raises_immediately_with_zero_retries(monkeypatch):
    """The whole point of the fix: no backoff on a request that cannot
    succeed until the daily reset."""

    class _FakeClient:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise _status_error(429, _REAL_DAILY_QUOTA_BODY)

    fake_client = _FakeClient()
    provider = _provider(fake_client)

    slept = []
    monkeypatch.setattr("legacy.llm_providers.time.sleep", lambda s: slept.append(s))

    with pytest.raises(DailyQuotaExhausted):
        provider._complete("prompt")

    assert fake_client.calls == 1  # first attempt, not 5
    assert slept == []             # no exponential backoff burned


def test_per_minute_429_is_still_retried(monkeypatch):
    """A structured per-minute limit_source means retrying IS correct --
    the fix must not turn every 429 into a hard stop."""
    body = {
        "error": {
            "message": "Rate limit exceeded: free-models-per-min",
            "code": 429,
            "metadata": {"limit_source": "openrouter_free_tier_per_minute"},
        }
    }
    exc = _status_error(429, body)
    assert _is_daily_quota_error(exc) is False

    class _FakeClient:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise _status_error(429, body)
            return _completion("recovered")

    fake_client = _FakeClient()
    provider = _provider(fake_client)
    monkeypatch.setattr("legacy.llm_providers.time.sleep", lambda *_a, **_k: None)

    assert provider._complete("prompt") == "recovered"
    assert fake_client.calls == 2


def test_message_markers_still_used_without_a_json_body():
    """Non-JSON error responses (openai leaves .body as None or raw text)
    fall back to the message markers -- LOCAL_LLM_BASE_URL can point at any
    OpenAI-compatible endpoint, which need not use OpenRouter's shape."""
    request = httpx.Request("POST", "https://internal/chat/completions")
    response = httpx.Response(429, request=request, text="RPD quota exceeded")
    exc = openai.APIStatusError("RPD quota exceeded", response=response, body=None)
    assert _daily_quota_limit_source(exc) == ""
    assert _is_daily_quota_error(exc) is True

    plain = openai.APIStatusError(
        "Too Many Requests",
        response=httpx.Response(429, request=request, text="slow down"),
        body=None,
    )
    assert _is_daily_quota_error(plain) is False


def test_non_429_status_is_never_a_quota_error():
    assert _is_daily_quota_error(_status_error(500, _REAL_DAILY_QUOTA_BODY)) is False


def test_daily_quota_limit_source_tolerates_malformed_bodies():
    """A JSON body of an unexpected shape must not raise -- it just means no
    structured signal, so the message markers decide."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    for body in (None, [], "text", {}, {"error": "boom"}, {"error": {"metadata": None}}):
        exc = openai.APIStatusError(
            "Error", response=httpx.Response(429, request=request, text="x"), body=body
        )
        assert _daily_quota_limit_source(exc) == ""


# --- get_provider() selection: OpenRouter or an internal endpoint ---------------
def _clear_provider_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)


def test_get_provider_selects_openrouter_when_key_set(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")
    assert isinstance(get_provider(), OpenRouterProvider)


def test_get_provider_raises_when_nothing_configured(monkeypatch):
    _clear_provider_env(monkeypatch)
    with pytest.raises(RuntimeError):
        get_provider()


def test_get_provider_local_base_url_takes_precedence_over_openrouter(monkeypatch):
    """LOCAL_LLM_BASE_URL is the deployment target -- it must win even when
    OPENROUTER_API_KEY is also set."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://llm-host:8000/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "Qwen3-32B")

    provider = get_provider()
    assert isinstance(provider, OpenRouterProvider)
    assert provider._model == "Qwen3-32B"
    assert str(provider._client.base_url) == "http://llm-host:8000/v1/"


def test_get_provider_local_llm_api_key_optional(monkeypatch):
    """LOCAL_LLM_API_KEY unset must still construct a working client, using
    the "local" placeholder -- many internal OpenAI-compatible servers don't
    check the key at all."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://llm-host:8000/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "Qwen3-32B")

    provider = get_provider()
    assert isinstance(provider, OpenRouterProvider)
    assert provider._client.api_key == "local"


def test_get_provider_local_base_url_without_model_raises(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://llm-host:8000/v1")
    with pytest.raises(RuntimeError, match="LOCAL_LLM_MODEL"):
        get_provider()
