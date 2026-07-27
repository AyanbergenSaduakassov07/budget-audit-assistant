"""Tests for core.budget_auditor.file_sql — the Gemini text-to-SQL fallback
over the current file's data.

Live Gemini calls are mocked; the focus is the plumbing (table build, schema
grounding, SELECT execution) and the never-raises / graceful-degradation
contract. A real end-to-end call is exercised by scripts, not here (it costs
quota and is non-deterministic).
"""

import os

import openpyxl

from legacy.query_engine import _SQL_EXAMPLES

import core.budget_auditor.file_sql as file_sql
from core.budget_auditor.file_sql import (
    _BUDGET_ITEMS_SQL_EXAMPLES,
    _load_temp_db,
    _schema_description,
    _split_columns,
    _split_flag_columns,
    answer_over_file,
)
from legacy import warehouse
from tests.test_budget_auditor import (
    _full_form_workbook,
    _no_target_year_workbook,
    _run_pipeline,
    _write_headers,
    _write_meta,
)


def _subsidiary_b(tmp_path):
    _, _, _, combined = _run_pipeline(_no_target_year_workbook(tmp_path / "subb.xlsx"))
    return combined


def _quarterly_flag_workbook(path):
    """A quarterly OPEX sheet with three rows: a genuine critical mismatch, a
    row whose quarterly sum is off by only floating-point noise (well under
    the 1% tolerance and must NOT be flagged), and a clean exact-match row."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws)
    hdr = 9
    _write_headers(ws, hdr, {
        9: "Контрагент", 10: "Обоснование", 11: "План за 2026 год",
        12: "1 кв 2026", 13: "2 кв 2026", 14: "3 кв 2026", 15: "4 кв 2026",
    })
    rows = [
        # (item, annual, q1, q2, q3, q4)
        ("Критично", 100_000, 23_500, 23_500, 23_500, 23_500),  # 6% off -> critical
        # quarters sum to 100_000.00000000001 -- pure float addition noise,
        # nowhere near the 1% tolerance -> must NOT be flagged.
        ("Шум округления", 100_000, 25_000.000000001, 25_000, 25_000, 25_000),
        ("Точно", 100_000, 25_000, 25_000, 25_000, 25_000),  # exact -> clean
    ]
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-OP-01")
    ws.cell(row=r, column=4, value="Услуги")
    for item, annual, *quarters in rows:
        r += 1
        ws.cell(row=r, column=7, value="Услуги")
        ws.cell(row=r, column=8, value=item)
        ws.cell(row=r, column=9, value="Вендор")
        ws.cell(row=r, column=10, value="обоснование есть")
        ws.cell(row=r, column=11, value=annual)
        for i, q in enumerate(quarters):
            ws.cell(row=r, column=12 + i, value=q)
    ws.cell(row=r + 1, column=3, value="Итого")
    ws.cell(row=r + 1, column=11, value=300_000)
    wb.save(path)
    return path


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
    flags = _split_flag_columns(combined)
    schema = _schema_description(combined, {"unit": 1000}, dims, money, flags)
    assert "budget_items" in schema
    assert "plan_2027" in schema
    assert "тыс. тенге" in schema
    # The anti-`year`/`subtotal` note that counters query_engine's warehouse
    # few-shot examples must be present.
    assert "НЕТ единой колонки 'year'" in schema
    assert "OPEX" in schema  # exact cost_type value listed


def test_flag_columns_exposed_in_schema_and_filterable(tmp_path):
    """Pre-computed anomaly flags must reach the SQL layer instead of being
    discarded: the schema text must mention quarterly_mismatch_flag (telling
    the model to filter on it rather than recompute it), and a query filtering
    on it must return only the genuinely flagged row — NOT the row whose
    quarterly/annual gap is pure floating-point noise well under the 1%
    tolerance."""
    _, _, _, combined = _run_pipeline(_quarterly_flag_workbook(tmp_path / "quarters.xlsx"))

    dims, money = _split_columns(combined)
    flags = _split_flag_columns(combined)
    assert "quarterly_mismatch_flag" in flags
    assert "quarterly_severity" in flags
    assert "quarterly_gap_pct" in flags

    schema = _schema_description(combined, {"unit": 1000}, dims, money, flags)
    assert "quarterly_mismatch_flag" in schema
    assert "quarterly_severity" in schema
    assert "не пересчитывайте" in schema.lower()

    db_path = _load_temp_db(combined, dims, money, flags)
    try:
        df = warehouse.run_sql(
            "SELECT item_name FROM budget_items WHERE quarterly_mismatch_flag = 1",
            db_path,
        )
        assert set(df["item_name"]) == {"Критично"}
    finally:
        os.unlink(db_path)


def _no_earlier_year_workbook(path):
    """Campaign year 2027, plan_2027 + plan_2028 only (Bug D shape): no year
    before the campaign year exists anywhere in the file, so "compared to
    last year" has nothing valid to compare against. Includes an Обоснование
    column so this test isolates the year-gap warning from the
    justification-gap warning."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws, campaign_year=2027)
    hdr = 9
    _write_headers(ws, hdr, {
        9: "Контрагент", 10: "Обоснование",
        11: "План за 2027 год", 12: "План за 2028 год",
    })
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-OP-01")
    ws.cell(row=r, column=4, value="Связь")
    r += 1
    ws.cell(row=r, column=7, value="Услуги связи")
    ws.cell(row=r, column=8, value="Интернет")
    ws.cell(row=r, column=9, value="Телеком Плюс")
    ws.cell(row=r, column=10, value="обоснование есть")
    ws.cell(row=r, column=11, value=5000)
    ws.cell(row=r, column=12, value=5500)
    ws.cell(row=r + 1, column=3, value="Итого")
    ws.cell(row=r + 1, column=11, value=5000)
    wb.save(path)
    return path


# --- structural facts (Bug D / Bug C schema half) --------------------------------
def test_schema_states_no_earlier_year_when_none_present(tmp_path):
    """Bug D: a file with only years >= the campaign year (e.g. 2027-2028,
    nothing from 2025/2026) must tell the model comparison to a prior year
    is impossible, so it doesn't substitute a later year as "last year"."""
    _, _, _, combined = _run_pipeline(_no_earlier_year_workbook(tmp_path / "no_earlier.xlsx"))
    dims, money = _split_columns(combined)
    flags = _split_flag_columns(combined)
    schema = _schema_description(combined, {"unit": 1000}, dims, money, flags)
    assert "Более ранних лет" in schema
    assert "сравнение невозможно" in schema.lower()


def test_schema_states_no_justification_column_when_absent(tmp_path):
    """Bug C schema half: a file with no Обоснование column at all must
    state that all items lack it structurally, not as a filtering result."""
    combined = _subsidiary_b(tmp_path)
    dims, money = _split_columns(combined)
    flags = _split_flag_columns(combined)
    assert "justification" not in dims  # confirms the fixture's actual shape
    schema = _schema_description(combined, {"unit": 1000}, dims, money, flags)
    assert "НЕТ колонки Обоснование" in schema
    assert "структурный пробел" in schema


def test_schema_omits_both_warnings_when_neither_gap_applies(tmp_path):
    """A file that DOES have both an earlier year and a justification column
    must carry neither structural warning -- they're conditional on the gap
    actually applying, not boilerplate always appended."""
    _, _, _, combined = _run_pipeline(_full_form_workbook(tmp_path / "full_form.xlsx"))
    dims, money = _split_columns(combined)
    flags = _split_flag_columns(combined)
    assert "justification" in dims  # confirms the fixture's actual shape
    schema = _schema_description(combined, {"unit": 1000}, dims, money, flags)
    assert "Более ранних лет" not in schema
    assert "НЕТ колонки Обоснование" not in schema


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

    def _phrase(client, question, sql, result_text, empty, history=None):
        captured["sql"] = sql
        captured["result_text"] = result_text
        captured["empty"] = empty
        return f"Итог: {result_text}"

    monkeypatch.setattr(file_sql, "_phrase_answer", _phrase)

    answer = answer_over_file("сколько всего на 2027", combined, {"unit": 1000})
    assert answer is not None
    assert captured["empty"] is False
    assert "9800" in captured["result_text"].replace(" ", "")  # 9000 + 800


# --- bounded self-correction retry after a failing first-attempt SQL ------------
def test_first_attempt_success_skips_retry(tmp_path, monkeypatch):
    """A valid first-attempt SQL must not trigger the error-feedback retry."""
    combined = _subsidiary_b(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr(file_sql, "_get_client", lambda: object())
    monkeypatch.setattr(file_sql, "_generate_sql",
                        lambda *a, **k: "SELECT SUM(plan_2027) AS total FROM budget_items")

    def _boom(*a, **k):
        raise AssertionError("retry must not be attempted on a successful first attempt")

    monkeypatch.setattr(file_sql, "_generate_sql_with_error_feedback", _boom)
    monkeypatch.setattr(file_sql, "_phrase_answer",
                        lambda client, question, sql, result_text, empty, history=None: "Итог найден.")

    answer = answer_over_file("сколько всего на 2027", combined, {"unit": 1000})
    assert answer == "Итог найден."


def test_retry_recovers_from_failing_first_attempt_sql(tmp_path, monkeypatch, caplog):
    """First-attempt SQL references a nonexistent column and fails to execute;
    the corrected retry SQL is valid and its result is used for the answer."""
    combined = _subsidiary_b(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr(file_sql, "_get_client", lambda: object())
    monkeypatch.setattr(file_sql, "_generate_sql",
                        lambda *a, **k: "SELECT SUM(no_such_column) FROM budget_items")

    captured = {}

    def _feedback(client, question, schema_description, failed_sql, error_message, history=None):
        captured["failed_sql"] = failed_sql
        captured["error_message"] = error_message
        return "SELECT SUM(plan_2027) AS total FROM budget_items"

    monkeypatch.setattr(file_sql, "_generate_sql_with_error_feedback", _feedback)

    phrased = {}

    def _phrase(client, question, sql, result_text, empty, history=None):
        phrased["sql"] = sql
        phrased["result_text"] = result_text
        return f"Итог: {result_text}"

    monkeypatch.setattr(file_sql, "_phrase_answer", _phrase)

    with caplog.at_level("INFO"):
        answer = answer_over_file("сколько всего на 2027", combined, {"unit": 1000})

    assert answer is not None
    assert captured["failed_sql"] == "SELECT SUM(no_such_column) FROM budget_items"
    assert "no_such_column" in captured["error_message"] or captured["error_message"]
    assert phrased["sql"] == "SELECT SUM(plan_2027) AS total FROM budget_items"
    assert "9800" in phrased["result_text"].replace(" ", "")  # 9000 + 800
    assert "self-correction succeeded" in caplog.text
    assert "SELECT SUM(no_such_column) FROM budget_items" in caplog.text
    assert "SELECT SUM(plan_2027) AS total FROM budget_items" in caplog.text


def test_retry_also_fails_returns_none_without_raising(tmp_path, monkeypatch, caplog):
    """Both the first attempt and the corrected retry reference bad columns:
    answer_over_file() must return None, never raise, and log both SQLs."""
    combined = _subsidiary_b(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr(file_sql, "_get_client", lambda: object())
    monkeypatch.setattr(file_sql, "_generate_sql",
                        lambda *a, **k: "SELECT SUM(no_such_column) FROM budget_items")
    monkeypatch.setattr(
        file_sql, "_generate_sql_with_error_feedback",
        lambda *a, **k: "SELECT SUM(still_bad_column) FROM budget_items",
    )

    with caplog.at_level("INFO"):
        answer = answer_over_file("сколько всего на 2027", combined, {"unit": 1000})

    assert answer is None
    assert "retry also failed" in caplog.text
    assert "SELECT SUM(no_such_column) FROM budget_items" in caplog.text
    assert "SELECT SUM(still_bad_column) FROM budget_items" in caplog.text


# --- compound-question classification / decomposition / synthesis --------------
def test_classified_simple_takes_single_sql_path_unchanged(tmp_path, monkeypatch):
    """A question the model classifies "simple" must take the existing
    single-SQL-generation path: one generate_sql call, one generate_json call
    for the classification, one more for _phrase_answer — no decomposition or
    synthesis call is ever made."""
    combined = _subsidiary_b(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    json_prompts = []

    class _FakeProvider:
        def generate_sql(self, prompt):
            return "SELECT SUM(plan_2027) AS total FROM budget_items"

        def generate_json(self, prompt):
            json_prompts.append(prompt)
            if "simple или compound" in prompt:
                return "simple"
            return "Итог: план 2027 = 9 800."

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    answer = answer_over_file("сколько всего плана на 2027 год", combined, {"unit": 1000})

    assert answer == "Итог: план 2027 = 9 800."
    assert len(json_prompts) == 2  # classify + phrase_answer only
    assert not any("Раздели составной вопрос" in p for p in json_prompts)


def test_compound_question_runs_subqueries_and_synthesizes(tmp_path, monkeypatch):
    """A question classified "compound" must be decomposed into sub-questions
    (each answered via the normal single-SQL path against the real temp DB),
    then combined by synthesize_answer() into the final answer."""
    combined = _subsidiary_b(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    sql_prompts = []
    json_prompts = []

    class _FakeProvider:
        def generate_sql(self, prompt):
            sql_prompts.append(prompt)
            return "SELECT SUM(plan_2027) AS total FROM budget_items"

        def generate_json(self, prompt):
            json_prompts.append(prompt)
            if "simple или compound" in prompt:
                return "compound"
            if "Раздели составной вопрос" in prompt:
                return (
                    "1. Сколько плана на 2027 год у ГО?\n"
                    "2. Сколько плана на 2027 год у остальных подразделений?"
                )
            if "исходный составной вопрос" in prompt:
                return "Итоговый ответ: суммарно 9 800 KZT."
            raise AssertionError(f"unexpected generate_json prompt: {prompt!r}")

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    answer = answer_over_file("сравни план ГО и остальных подразделений на 2027 год", combined, {"unit": 1000})

    assert answer == "Итоговый ответ: суммарно 9 800 KZT."
    # classify + decompose + synthesize = 3 generate_json calls; _phrase_answer
    # is never reached on the compound path.
    assert len(json_prompts) == 3
    # one generate_sql call per sub-question
    assert len(sql_prompts) == 2


def test_compound_capped_at_max_subquestions(tmp_path, monkeypatch):
    """Even if decompose_question returns more than 4 sub-questions, only 4
    are actually run."""
    combined = _subsidiary_b(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    sql_prompts = []

    class _FakeProvider:
        def generate_sql(self, prompt):
            sql_prompts.append(prompt)
            return "SELECT SUM(plan_2027) AS total FROM budget_items"

        def generate_json(self, prompt):
            if "simple или compound" in prompt:
                return "compound"
            if "Раздели составной вопрос" in prompt:
                return "\n".join(f"{i}. Подвопрос номер {i}?" for i in range(1, 7))  # 6 items
            if "исходный составной вопрос" in prompt:
                return "Итог по всем подвопросам."
            raise AssertionError(f"unexpected generate_json prompt: {prompt!r}")

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    answer = answer_over_file("составной вопрос с шестью частями", combined, {"unit": 1000})

    assert answer == "Итог по всем подвопросам."
    assert len(sql_prompts) == 4  # capped, not 6


def test_classification_failure_falls_back_to_simple_path(tmp_path, monkeypatch, caplog):
    """If classify_question() itself raises (e.g. a transient provider error),
    the question must still be answered via the simple single-SQL path rather
    than aborting the whole call."""
    combined = _subsidiary_b(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    class _FakeProvider:
        def generate_sql(self, prompt):
            return "SELECT SUM(plan_2027) AS total FROM budget_items"

        def generate_json(self, prompt):
            if "simple или compound" in prompt:
                raise RuntimeError("classification service unavailable")
            return "Итог: план 2027 = 9 800."

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    with caplog.at_level("INFO"):
        answer = answer_over_file("сколько всего плана на 2027 год", combined, {"unit": 1000})

    assert answer == "Итог: план 2027 = 9 800."
    assert "classification failed" in caplog.text
    assert "treating as simple" in caplog.text


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


# --- few-shot examples: real budget_items schema, not the synthetic warehouse ---
def test_generated_sql_prompt_uses_budget_items_examples_not_synthetic(tmp_path, monkeypatch):
    """answer_over_file() must ground SQL generation in _BUDGET_ITEMS_SQL_EXAMPLES,
    never in query_engine's default _SQL_EXAMPLES (written against the unrelated
    synthetic `procurements` schema: subtotal/subsidiary/category/year)."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    captured = {}

    class _FakeProvider:
        def generate_sql(self, prompt):
            captured["sql_prompt"] = prompt
            return "SELECT SUM(plan_2027) AS total FROM budget_items"

        def generate_json(self, prompt):
            return "Итог найден."

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    combined = _subsidiary_b(tmp_path)
    answer = answer_over_file("сколько всего на 2027", combined, {"unit": 1000})

    assert answer is not None
    sql_prompt = captured["sql_prompt"]

    # A budget_items example must be present verbatim.
    sample_question, sample_sql = _BUDGET_ITEMS_SQL_EXAMPLES[0]
    assert sample_question in sql_prompt
    assert sample_sql in sql_prompt

    # None of the synthetic-warehouse examples (procurements/subtotal/subsidiary)
    # may leak into a file-level prompt.
    for _, sql in _SQL_EXAMPLES:
        assert sql not in sql_prompt
    assert "procurements" not in sql_prompt


# --- conversation history threaded into the SQL/answer prompts ------------------
def test_history_threaded_into_generated_sql_prompt(tmp_path, monkeypatch):
    """A 2-turn history where the second question only makes sense given the
    first ('раздели планы на opex/capex по всем годам' then just 'а за все
    года') -- the prior turn must reach the real _generate_sql()/_phrase_answer()
    prompts, not be silently dropped."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    captured = {}

    class _FakeProvider:
        def generate_sql(self, prompt):
            captured["sql_prompt"] = prompt
            return "SELECT SUM(plan_2027) AS total FROM budget_items"

        def generate_json(self, prompt):
            captured["json_prompt"] = prompt
            return "Итог найден."

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    history = [
        (
            "раздели планы на opex/capex по годам",
            "Разбивка OPEX/CAPEX по годам:\n2027 (План): OPEX = 9 000 тыс., CAPEX = 800 тыс.",
        ),
    ]

    combined = _subsidiary_b(tmp_path)
    answer = answer_over_file("а за все года", combined, {"unit": 1000}, history=history)

    assert answer is not None

    sql_prompt = captured["sql_prompt"]
    assert "Предыдущие вопросы и ответы" in sql_prompt
    assert "раздели планы на opex/capex по годам" in sql_prompt
    assert "9 000 тыс." in sql_prompt  # the prior answer text, not just the question
    assert "а за все года" in sql_prompt  # the current question is still present

    json_prompt = captured["json_prompt"]
    assert "Предыдущие вопросы и ответы" in json_prompt
    assert "раздели планы на opex/capex по годам" in json_prompt


def test_no_history_omits_prior_turns_block(tmp_path, monkeypatch):
    """Without a history argument (the default), the prompt must be identical
    in shape to before this change -- no empty 'Предыдущие вопросы' block."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    captured = {}

    class _FakeProvider:
        def generate_sql(self, prompt):
            captured["sql_prompt"] = prompt
            return "SELECT SUM(plan_2027) AS total FROM budget_items"

        def generate_json(self, prompt):
            return "Итог найден."

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    combined = _subsidiary_b(tmp_path)
    answer = answer_over_file("сколько всего на 2027", combined, {"unit": 1000})

    assert answer is not None
    assert "Предыдущие вопросы" not in captured["sql_prompt"]
