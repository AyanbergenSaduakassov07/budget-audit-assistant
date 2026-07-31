"""Shared pytest configuration.

api-marked tests (legacy/test_query_engine.py) call the live OpenRouter API
and cost real quota/credits -- they're deliberately not mocked, since their
whole point is validating real model behavior end to end, not code paths.

Without this hook, running bare `pytest` in any environment with no provider
configured (a fresh clone, CI without secrets) makes those two tests ERROR
with "No LLM provider configured" instead of skipping cleanly. This hook
checks once at collection time and skips api-marked tests only when no
provider is configured, so `pytest` is always green out of the box, while
anyone who *does* have OPENROUTER_API_KEY / LOCAL_LLM_BASE_URL set still gets
the real integration run exactly as the tests intend.
"""
import pytest

from legacy.llm_providers import get_provider


def pytest_collection_modifyitems(config, items):
    try:
        get_provider()
        return  # a provider IS configured -- let api tests run for real
    except RuntimeError:
        pass
    skip_api = pytest.mark.skip(
        reason="no LLM provider configured (set OPENROUTER_API_KEY or "
               "LOCAL_LLM_BASE_URL to run live-API tests)"
    )
    for item in items:
        if "api" in item.keywords:
            item.add_marker(skip_api)
