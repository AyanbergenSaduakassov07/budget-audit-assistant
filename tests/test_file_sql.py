"""Tests for core.budget_auditor.file_sql — the Gemini text-to-SQL fallback
over the current file's data.

Live Gemini calls are mocked; the focus is the plumbing (table build, schema
grounding, SELECT execution) and the never-raises / graceful-degradation
contract. A real end-to-end call is exercised by scripts, not here (it costs
quota and is non-deterministic).
"""

import core.budget_auditor.file_sql as file_sql
from core.budget_auditor.file_sql import (
    _schema_description,
    _split_columns,
    answer_over_file,
)
from tests.test_budget_auditor import _no_target_year_workbook, _run_pipeline


def _subsidiary_b(tmp_path):
    _, _, _, combined = _run_pipeline(_no_target_year_workbook(tmp_path / "subb.xlsx"))
    return combined


# --- schema / table build -------------------------------------------------------
def test_split_columns_finds_money_and_dims(tmp_path):
    combined = _subsidiary_b(tmp_path)
    dims, money = _split_columns(combined)
    assert "plan_2027" in money
    assert all(c.startswith(("plan_", "fact_", "forecast_")) for c in money)
    assert "branch" in dims and "cost_type" in dims
    # flag/delta helper columns must NOT leak into the SQL surface
    assert "high_change_flag" not in dims + money
    assert "delta_pct" not in dims + money


def test_schema_description_grounds_the_model(tmp_path):
    combined = _subsidiary_b(tmp_path)
    dims, money = _split_columns(combined)
    schema = _schema_description(combined, {"unit": 1000}, dims, money)
    assert "budget_items" in schema
    assert "plan_2027" in schema
    assert "тыс. тенге" in schema
    # The anti-`year`/`subtotal` note that counters query_engine's warehouse
    # few-shot examples must be present.
    assert "НЕТ единой колонки 'year'" in schema
    assert "OPEX" in schema  # exact cost_type value listed


# --- graceful degradation (never raises, returns None) --------------------------
def test_returns_none_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert answer_over_file("сравни года", _subsidiary_b(tmp_path), {"unit": 1000}) is None


def test_returns_none_on_empty_or_no_money(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    assert answer_over_file("что угодно", None, {}) is None
    assert answer_over_file("что угодно", pd.DataFrame(), {}) is None
    # A frame with dims but no plan_/fact_/forecast_ column -> nothing to query.
    no_money = pd.DataFrame({"branch": ["ГО"], "cost_type": ["OPEX"]})
    assert answer_over_file("что угодно", no_money, {}) is None


def test_returns_none_on_quota_exhausted(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr(file_sql, "_get_client", lambda: object())

    def _boom(*a, **k):
        raise file_sql.DailyQuotaExhausted("PerDay limit")

    monkeypatch.setattr(file_sql, "_generate_sql", _boom)
    assert answer_over_file("сравни года", _subsidiary_b(tmp_path), {"unit": 1000}) is None


def test_returns_none_on_bad_sql(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr(file_sql, "_get_client", lambda: object())
    # Model returns a non-SELECT statement -> warehouse.run_sql rejects it ->
    # exception swallowed -> None (never surfaced as an error).
    monkeypatch.setattr(file_sql, "_generate_sql", lambda *a, **k: "DROP TABLE budget_items")
    assert answer_over_file("сравни года", _subsidiary_b(tmp_path), {"unit": 1000}) is None


# --- success path (mocked model), real SQLite execution -------------------------
def test_success_path_runs_generated_sql(tmp_path, monkeypatch):
    combined = _subsidiary_b(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr(file_sql, "_get_client", lambda: object())
    # A real, valid SELECT over the per-file table — actually executed by
    # warehouse.run_sql against the temp DB the fallback builds.
    monkeypatch.setattr(file_sql, "_generate_sql",
                        lambda *a, **k: "SELECT SUM(plan_2027) AS total FROM budget_items")

    captured = {}

    def _phrase(client, question, sql, result_text, empty):
        captured["sql"] = sql
        captured["result_text"] = result_text
        captured["empty"] = empty
        return f"Итог: {result_text}"

    monkeypatch.setattr(file_sql, "_phrase_answer", _phrase)

    answer = answer_over_file("сколько всего на 2027", combined, {"unit": 1000})
    assert answer is not None
    assert captured["empty"] is False
    assert "9800" in captured["result_text"].replace(" ", "")  # 9000 + 800


def test_returns_none_on_empty_provider_answer(tmp_path, monkeypatch):
    """A real OpenRouterProvider whose model returns an empty message.content
    (observed with Qwen3.6's reasoning mode exhausting the output budget)
    must degrade to None like any other failure — never surface a blank
    string as a real answer to app.py."""
    from legacy.llm_providers import OpenRouterProvider

    class _EmptyMessage:
        content = ""

    class _EmptyChoice:
        message = _EmptyMessage()

    class _EmptyResponse:
        choices = [_EmptyChoice()]

    class _FakeOpenAIClient:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            return _EmptyResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    provider._model = "test-model"
    provider._client = _FakeOpenAIClient()

    monkeypatch.setattr(file_sql, "_get_client", lambda: provider)
    # SQL generation succeeds with a valid SELECT; only the answer-phrasing
    # call (also routed through the same provider) returns empty content.
    monkeypatch.setattr(file_sql, "_generate_sql",
                        lambda *a, **k: "SELECT SUM(plan_2027) AS total FROM budget_items")

    combined = _subsidiary_b(tmp_path)
    assert answer_over_file("сколько всего на 2027", combined, {"unit": 1000}) is None
