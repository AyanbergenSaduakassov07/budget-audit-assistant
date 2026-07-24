"""Tests for legacy.llm_providers.OpenRouterProvider — no live API calls.

Covers two failure modes found testing against real Qwen3.6-27B responses:
empty message.content (reasoning mode exhausting the output budget) and an
uncaught network error crashing the retry loop instead of retrying it.
"""

import httpx
import openai
import pytest

from legacy.llm_providers import (
    DailyQuotaExhausted,
    OpenRouterProvider,
    _OPENROUTER_MAX_OUTPUT_TOKENS,
    _OPENROUTER_REASONING_EXTRA_BODY,
)


def _provider(client) -> OpenRouterProvider:
    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    provider._model = "test-model"
    provider._client = client
    return provider


def _completion(content: str):
    class _Msg:
        pass

    class _Choice:
        pass

    class _Response:
        pass

    msg = _Msg()
    msg.content = content
    choice = _Choice()
    choice.message = msg
    response = _Response()
    response.choices = [choice]
    return response


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
