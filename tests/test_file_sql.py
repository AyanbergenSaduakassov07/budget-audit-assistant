"""Tests for core.budget_auditor.file_sql — the LLM text-to-SQL fallback
over the current file's data.

Live LLM calls are mocked; the focus is the plumbing (table build, schema
grounding, SELECT execution) and the never-raises / graceful-degradation
contract. A real end-to-end call is exercised by scripts, not here (it costs
quota and is non-deterministic).
"""

import json
import os

import openpyxl

from legacy.llm_providers import DAILY_QUOTA_MESSAGE
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


# --- workbook metadata in the prompt --------------------------------------------
# Live bug: parser.py reads company/campaign_year/currency/unit/version off the
# form's header, but none of it reached the model -- only the table's rows did.
# So "какая это компания" had nothing to ground on and came back with a
# confidently invented, unrelated company name.
def test_schema_description_includes_workbook_metadata(tmp_path):
    combined = _subsidiary_b(tmp_path)
    dims, money = _split_columns(combined)
    flags = _split_flag_columns(combined)
    meta = {
        "company": "ТОО Тест-Дочка",
        "campaign_year": 2027,
        "currency": "KZT",
        "unit": 1000,
        "version": "план",
    }
    schema = _schema_description(combined, meta, dims, money, flags)

    assert "Компания: ТОО Тест-Дочка" in schema
    assert "Год кампании: 2027" in schema
    assert "Валюта: KZT" in schema
    assert "Единица измерения: тыс. тенге" in schema
    assert "Версия формы: план" in schema
    # ...and the instruction not to derive these from the table's rows.
    assert "не придумывайте" in schema


def test_schema_description_metadata_survives_the_real_parsed_meta(tmp_path):
    """The parsed meta dict, not a hand-built one -- guards against a key
    rename in parser.py silently emptying the block."""
    meta, _, _, combined = _run_pipeline(_full_form_workbook(tmp_path / "meta.xlsx"))
    dims, money = _split_columns(combined)
    flags = _split_flag_columns(combined)
    schema = _schema_description(combined, meta, dims, money, flags)

    assert meta["company"] in schema
    assert f"Год кампании: {meta['campaign_year']}" in schema


def test_schema_description_marks_missing_metadata_as_missing(tmp_path):
    """A field the form doesn't carry is spelled out as missing rather than
    omitted -- a blank line is an invitation to guess."""
    combined = _subsidiary_b(tmp_path)
    dims, money = _split_columns(combined)
    flags = _split_flag_columns(combined)
    schema = _schema_description(combined, {"unit": 1000}, dims, money, flags)

    assert "Компания: не указано в форме" in schema
    assert "Валюта: не указано в форме" in schema
    assert "None" not in schema  # never a raw Python None leaking into the prompt


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


def _final(answer: str) -> str:
    return json.dumps({"action": "final", "answer": answer}, ensure_ascii=False)


def _query(sql: str, reason: str = "нужны данные") -> str:
    return json.dumps({"action": "query", "sql": sql, "reason": reason}, ensure_ascii=False)


def _complete() -> str:
    return json.dumps({"complete": True, "missing": ""})


_SELF_CHECK_MARKER = "Ты проверяешь черновой ответ"


# --- graceful degradation (never raises, returns None) --------------------------
def test_returns_none_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert answer_over_file("сравни года", _subsidiary_b(tmp_path), {"unit": 1000}) is None


def test_returns_none_on_empty_or_no_money(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    assert answer_over_file("что угодно", None, {}) is None
    assert answer_over_file("что угодно", pd.DataFrame(), {}) is None
    # A frame with dims but no plan_/fact_/forecast_ column -> nothing to query.
    no_money = pd.DataFrame({"branch": ["ГО"], "cost_type": ["OPEX"]})
    assert answer_over_file("что угодно", no_money, {}) is None


def test_quota_exhausted_surfaces_a_distinct_message(tmp_path, monkeypatch):
    """Quota exhaustion must NOT return None like every other failure: the
    caller's generic not-found reply ("не нашёл точного ответа") made an
    out-of-quota day indistinguishable from a real reasoning failure."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    class _FakeProvider:
        def generate_json(self, prompt):
            raise file_sql.DailyQuotaExhausted("free-models-per-day")

        def generate_sql(self, prompt):
            raise file_sql.DailyQuotaExhausted("free-models-per-day")

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())
    reply = answer_over_file("сравни года", _subsidiary_b(tmp_path), {"unit": 1000})
    assert reply == DAILY_QUOTA_MESSAGE
    assert "лимит" in reply and "позже" in reply


def test_returns_none_when_step_loop_and_forced_synthesis_both_fail(tmp_path, monkeypatch, caplog):
    """A step's SQL (and its retry) both fail to execute; the model then
    never manages to finalize before the step cap, and forced synthesis
    ALSO produces nothing -- answer_over_file() must return None, never
    raise, and the retry failure must be logged."""
    combined = _subsidiary_b(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    class _FakeProvider:
        def generate_sql(self, prompt):
            return "SELECT SUM(still_bad_column) FROM budget_items"

        def generate_json(self, prompt):
            if _SELF_CHECK_MARKER in prompt:
                raise AssertionError("self-check must not run without a draft answer")
            if "исходный составной вопрос" in prompt or "Подвопрос" in prompt:
                return ""  # forced synthesis also yields nothing usable
            return _query("DROP TABLE budget_items")  # every step proposes a non-SELECT

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    with caplog.at_level("INFO"):
        answer = answer_over_file("сколько всего на 2027", combined, {"unit": 1000})

    assert answer is None
    assert "retry also failed" in caplog.text


# --- success path (mocked model), real SQLite execution -------------------------
def test_simple_question_resolves_in_one_query_step(tmp_path, monkeypatch):
    """A simple question must still resolve with exactly ONE query executed:
    step 1 proposes+runs it, step 2 reviews the real result and finalizes,
    then the self-check confirms completeness -- three generate_json calls
    total, same shape as before this change (classify+generate_sql+phrase
    also used to be three calls), and generate_sql (the retry path) is never
    touched since nothing failed."""
    combined = _subsidiary_b(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    json_calls = []

    class _FakeProvider:
        def generate_sql(self, prompt):
            raise AssertionError("no retry should be needed on a valid first query")

        def generate_json(self, prompt):
            json_calls.append(prompt)
            n = len(json_calls)
            if n == 1:
                assert "Уже выполненные шаги" not in prompt
                return _query("SELECT SUM(plan_2027) AS total FROM budget_items", "узнать план на 2027")
            if n == 2:
                assert "9800" in prompt.replace(" ", "") or "9 800" in prompt
                return _final("Итог: план 2027 = 9 800.")
            if n == 3:
                assert _SELF_CHECK_MARKER in prompt
                return _complete()
            raise AssertionError(f"unexpected extra generate_json call #{n}: {prompt!r}")

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    answer = answer_over_file("сколько всего плана на 2027 год", combined, {"unit": 1000})

    assert answer == "Итог: план 2027 = 9 800."
    assert len(json_calls) == 3


# --- bounded self-correction retry after a failing step SQL --------------------
def test_first_attempt_success_skips_retry(tmp_path, monkeypatch):
    """A valid first-attempt SQL must not trigger the error-feedback retry."""
    combined = _subsidiary_b(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    def _boom(*a, **k):
        raise AssertionError("retry must not be attempted on a successful first attempt")

    class _FakeProvider:
        generate_sql = staticmethod(_boom)

        def generate_json(self, prompt):
            if _SELF_CHECK_MARKER in prompt:
                return _complete()
            if "Уже выполненные шаги" not in prompt:
                return _query("SELECT SUM(plan_2027) AS total FROM budget_items")
            return _final("Итог найден.")

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    answer = answer_over_file("сколько всего на 2027", combined, {"unit": 1000})
    assert answer == "Итог найден."


def test_retry_recovers_from_failing_first_attempt_sql(tmp_path, monkeypatch, caplog):
    """First-attempt SQL references a nonexistent column and fails to execute;
    the corrected retry SQL is valid and its result is used for the answer."""
    combined = _subsidiary_b(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    captured = {}

    def _feedback(client, question, schema_description, failed_sql, error_message, history=None):
        captured["failed_sql"] = failed_sql
        captured["error_message"] = error_message
        return "SELECT SUM(plan_2027) AS total FROM budget_items"

    monkeypatch.setattr(file_sql, "_generate_sql_with_error_feedback", _feedback)

    class _FakeProvider:
        def generate_sql(self, prompt):
            raise AssertionError("generate_sql is not used by _generate_sql_with_error_feedback here")

        def generate_json(self, prompt):
            if _SELF_CHECK_MARKER in prompt:
                return _complete()
            if "Уже выполненные шаги" not in prompt:
                return _query("SELECT SUM(no_such_column) FROM budget_items")
            assert "9800" in prompt.replace(" ", "") or "9 800" in prompt
            return _final("Итог: 9 800.")

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    with caplog.at_level("INFO"):
        answer = answer_over_file("сколько всего на 2027", combined, {"unit": 1000})

    assert answer == "Итог: 9 800."
    assert captured["failed_sql"] == "SELECT SUM(no_such_column) FROM budget_items"
    assert captured["error_message"]
    assert "self-correction succeeded" in caplog.text
    assert "SELECT SUM(no_such_column) FROM budget_items" in caplog.text
    assert "SELECT SUM(plan_2027) AS total FROM budget_items" in caplog.text


def test_retry_also_fails_recorded_and_model_recovers_next_step(tmp_path, monkeypatch, caplog):
    """A step's SQL (and its retry) both fail to execute -- the failure is
    recorded in the scratchpad as an explicit note (never raised), and the
    model gets a chance to see it and finalize anyway on the very next
    step, instead of the whole answer aborting."""
    combined = _subsidiary_b(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.setattr(
        file_sql, "_generate_sql_with_error_feedback",
        lambda *a, **k: "SELECT SUM(still_bad_column) FROM budget_items",
    )

    json_calls = []

    class _FakeProvider:
        def generate_sql(self, prompt):
            raise AssertionError("retry SQL comes from _generate_sql_with_error_feedback, not generate_sql")

        def generate_json(self, prompt):
            json_calls.append(prompt)
            n = len(json_calls)
            if n == 1:
                return _query("SELECT SUM(no_such_column) FROM budget_items")
            if n == 2:
                assert "ошибка выполнения запроса" in prompt
                return _final("Не удалось получить данные из файла.")
            if n == 3:
                return _complete()
            raise AssertionError(f"unexpected extra call #{n}")

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    with caplog.at_level("INFO"):
        answer = answer_over_file("сколько всего на 2027", combined, {"unit": 1000})

    assert answer == "Не удалось получить данные из файла."
    assert "retry also failed" in caplog.text


# --- the bounded step loop reacting to an earlier result (new capability) ------
def _multi_branch_quarterly_workbook(path):
    """Two branches (ГО, АФ): ГО has a critical quarterly mismatch AND a
    supplier with no justification; АФ is clean on both counts. A query for
    "the branch with the worst quarterly mismatch" and a follow-up query
    scoped to just THAT branch's suppliers must return genuinely different,
    verifiable results per branch -- exercising the step loop actually using
    an early result to shape what it queries next, not just running two
    independent, pre-planned queries."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ГО"
    _write_meta(ws, campaign_year=2027)
    hdr = 9
    _write_headers(ws, hdr, {
        9: "Контрагент", 10: "Обоснование", 11: "План за 2027 год",
        12: "1 кв 2027", 13: "2 кв 2027", 14: "3 кв 2027", 15: "4 кв 2027",
    })
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-OP-01")
    ws.cell(row=r, column=4, value="Услуги")
    r += 1
    ws.cell(row=r, column=7, value="Услуги")
    ws.cell(row=r, column=8, value="Связь")
    ws.cell(row=r, column=9, value="Телеком Плюс")
    # No обоснование -> missing_justification_flag will be True.
    ws.cell(row=r, column=11, value=100_000)
    # Quarters sum to 70 000 vs. a 100 000 annual plan -> 30% gap -> critical.
    ws.cell(row=r, column=12, value=17_500)
    ws.cell(row=r, column=13, value=17_500)
    ws.cell(row=r, column=14, value=17_500)
    ws.cell(row=r, column=15, value=17_500)

    ws2 = wb.create_sheet("АФ")
    _write_headers(ws2, hdr, {
        9: "Контрагент", 10: "Обоснование", 11: "План за 2027 год",
        12: "1 кв 2027", 13: "2 кв 2027", 14: "3 кв 2027", 15: "4 кв 2027",
    })
    r = hdr + 1
    ws2.cell(row=r, column=3, value="IT-OP-02")
    ws2.cell(row=r, column=4, value="Оборудование")
    r += 1
    ws2.cell(row=r, column=7, value="Оборудование")
    ws2.cell(row=r, column=8, value="Серверы")
    ws2.cell(row=r, column=9, value="Dell")
    ws2.cell(row=r, column=10, value="обоснование есть")
    ws2.cell(row=r, column=11, value=50_000)
    # Exact match -> clean, never flagged.
    for c in (12, 13, 14, 15):
        ws2.cell(row=r, column=c, value=12_500)

    wb.save(path)
    return path


def test_step_loop_uses_first_result_to_shape_second_query(tmp_path, monkeypatch):
    """"Find the branch with the worst quarterly mismatch, then check if its
    top supplier is also in the missing-justification list" -- the second
    query can only be written correctly once the first query's actual
    result (branch "ГО") is known; a fixed up-front decomposition could
    never parameterize a sub-question on a sibling sub-question's result."""
    _, _, _, combined = _run_pipeline(_multi_branch_quarterly_workbook(tmp_path / "multi.xlsx"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    json_calls = []

    class _FakeProvider:
        def generate_sql(self, prompt):
            raise AssertionError("no retry should be needed in this scenario")

        def generate_json(self, prompt):
            json_calls.append(prompt)
            n = len(json_calls)
            if n == 1:
                assert "Уже выполненные шаги" not in prompt
                return _query(
                    "SELECT branch, quarterly_gap_pct FROM budget_items "
                    "WHERE quarterly_mismatch_flag = 1 ORDER BY quarterly_gap_pct DESC LIMIT 1",
                    "найти филиал с наибольшим расхождением по кварталам",
                )
            if n == 2:
                # Step 2 must see step 1's REAL result (branch "ГО") to know
                # which branch to scope the follow-up query to.
                assert "ГО" in prompt
                return _query(
                    "SELECT counterparty FROM budget_items "
                    "WHERE branch = 'ГО' AND missing_justification_flag = 1",
                    "проверить, есть ли у этого филиала поставщики без обоснования",
                )
            if n == 3:
                assert "Телеком Плюс" in prompt
                return _final(
                    "Филиал ГО имеет наибольшее расхождение по кварталам; "
                    "его поставщик Телеком Плюс также без обоснования."
                )
            if n == 4:
                return _complete()
            raise AssertionError(f"unexpected extra call #{n}")

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    answer = answer_over_file(
        "найди филиал с самым большим расхождением по кварталам, и проверь, "
        "есть ли у его основного поставщика обоснование",
        combined, {"unit": 1000},
    )

    assert answer == (
        "Филиал ГО имеет наибольшее расхождение по кварталам; "
        "его поставщик Телеком Плюс также без обоснования."
    )
    assert len(json_calls) == 4


def test_self_check_triggers_one_more_step_when_incomplete(tmp_path, monkeypatch):
    """The self-check flags a gap in the draft answer (steps remain under
    the cap) -> the model gets exactly one more query step to fill it, and
    the improved (forced-synthesis) answer, not the incomplete draft, is
    what's returned."""
    combined = _subsidiary_b(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    json_calls = []

    class _FakeProvider:
        def generate_sql(self, prompt):
            raise AssertionError("no retry needed in this scenario")

        def generate_json(self, prompt):
            json_calls.append(prompt)
            n = len(json_calls)
            if n == 1:
                return _query("SELECT SUM(plan_2027) AS total FROM budget_items", "план на 2027")
            if n == 2:
                return _final("План на 2027 год: 9 800.")  # draft -- missing the fact_2025 half
            if n == 3:
                assert _SELF_CHECK_MARKER in prompt
                return json.dumps({"complete": False, "missing": "не указан факт за 2025 год"})
            if n == 4:
                # The one extra step: propose the missing query.
                assert "не указан факт за 2025 год" in prompt
                return _query("SELECT SUM(fact_2025) AS total FROM budget_items", "факт за 2025")
            if n == 5:
                # Budget exhausted (5 calls == _MAX_STEPS + 1 already spent on
                # the main loop + self-check's one extra) -- forced synthesis
                # takes over instead of a 6th call; this call must be
                # synthesize_answer's own prompt, not another next_step.
                assert "исходный составной вопрос" in prompt
                return "План на 2027 год: 9 800 тыс.; факт за 2025 год: 8 500 тыс."
            raise AssertionError(f"unexpected extra call #{n}: {prompt!r}")

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    answer = answer_over_file("сколько плана на 2027 и факта за 2025", combined, {"unit": 1000})

    assert answer == "План на 2027 год: 9 800 тыс.; факт за 2025 год: 8 500 тыс."


def test_step_loop_respects_max_steps_cap_and_forces_synthesis(tmp_path, monkeypatch):
    """If the model never chooses "final" at all, the main loop stops after
    exactly _MAX_STEPS next_step() calls (each proposing a valid query) and
    forces one synthesis call over everything gathered -- never loops
    forever, never raises."""
    combined = _subsidiary_b(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    json_calls = []

    class _FakeProvider:
        def generate_sql(self, prompt):
            raise AssertionError("no retry needed -- every proposed query is valid")

        def generate_json(self, prompt):
            json_calls.append(prompt)
            n = len(json_calls)
            if n <= file_sql._MAX_STEPS:
                return _query("SELECT SUM(plan_2027) AS total FROM budget_items", f"запрос номер {n}")
            if n == file_sql._MAX_STEPS + 1:
                assert "исходный составной вопрос" in prompt
                return "Итог по всем шагам: 9 800."
            raise AssertionError(f"unexpected extra call #{n}")

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    answer = answer_over_file("бесконечно уточняемый вопрос", combined, {"unit": 1000})

    assert answer == "Итог по всем шагам: 9 800."
    # Exactly _MAX_STEPS next_step calls (all "query") + one forced-synthesis
    # call -- the self-check is never reached since the budget is spent.
    assert len(json_calls) == file_sql._MAX_STEPS + 1


def test_returns_none_on_empty_provider_answer(tmp_path, monkeypatch):
    """A real OpenRouterProvider whose model returns an empty message.content
    (observed with Qwen3.6's reasoning mode exhausting the output budget)
    must degrade to None like any other failure — never surface a blank
    string as a real answer to app.py. An empty response to next_step's very
    first call means nothing was ever queried, so forced synthesis (over an
    empty scratchpad) also yields nothing -- None, not a crash."""
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

    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    provider = OpenRouterProvider.__new__(OpenRouterProvider)
    provider._model = "test-model"
    provider._client = _FakeOpenAIClient()

    monkeypatch.setattr(file_sql, "_get_client", lambda: provider)

    combined = _subsidiary_b(tmp_path)
    assert answer_over_file("сколько всего на 2027", combined, {"unit": 1000}) is None


# --- few-shot examples: real budget_items schema, not the synthetic warehouse ---
def test_generated_sql_prompt_uses_budget_items_examples_not_synthetic(tmp_path, monkeypatch):
    """Every next_step() prompt must ground SQL proposals in
    _BUDGET_ITEMS_SQL_EXAMPLES, never in query_engine's default _SQL_EXAMPLES
    (written against the unrelated synthetic `procurements` schema:
    subtotal/subsidiary/category/year)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    captured = {}

    class _FakeProvider:
        def generate_sql(self, prompt):
            raise AssertionError("no retry needed in this scenario")

        def generate_json(self, prompt):
            if _SELF_CHECK_MARKER in prompt:
                return _complete()
            if "Уже выполненные шаги" not in prompt:
                captured["step_prompt"] = prompt
                return _query("SELECT SUM(plan_2027) AS total FROM budget_items")
            return _final("Итог найден.")

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    combined = _subsidiary_b(tmp_path)
    answer = answer_over_file("сколько всего на 2027", combined, {"unit": 1000})

    assert answer is not None
    step_prompt = captured["step_prompt"]

    # A budget_items example must be present verbatim.
    sample_question, sample_sql = _BUDGET_ITEMS_SQL_EXAMPLES[0]
    assert sample_question in step_prompt
    assert sample_sql in step_prompt

    # None of the synthetic-warehouse examples (procurements/subtotal/subsidiary)
    # may leak into a file-level prompt.
    for _, sql in _SQL_EXAMPLES:
        assert sql not in step_prompt
    assert "procurements" not in step_prompt


# --- conversation history threaded into every step's prompt ---------------------
def test_history_threaded_into_step_prompt(tmp_path, monkeypatch):
    """A 2-turn history where the second question only makes sense given the
    first ('раздели планы на opex/capex по всем годам' then just 'а за все
    года') -- the prior turn must reach the real next_step() prompt, not be
    silently dropped."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    captured = {}

    class _FakeProvider:
        def generate_sql(self, prompt):
            raise AssertionError("no retry needed in this scenario")

        def generate_json(self, prompt):
            if _SELF_CHECK_MARKER in prompt:
                return _complete()
            if "Уже выполненные шаги" not in prompt:
                captured["step_prompt"] = prompt
                return _query("SELECT SUM(plan_2027) AS total FROM budget_items")
            return _final("Итог найден.")

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

    step_prompt = captured["step_prompt"]
    assert "Предыдущие вопросы и ответы" in step_prompt
    assert "раздели планы на opex/capex по годам" in step_prompt
    assert "9 000 тыс." in step_prompt  # the prior answer text, not just the question
    assert "а за все года" in step_prompt  # the current question is still present


def test_no_history_omits_prior_turns_block(tmp_path, monkeypatch):
    """Without a history argument (the default), the prompt must carry no
    empty 'Предыдущие вопросы' block."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")

    captured = {}

    class _FakeProvider:
        def generate_sql(self, prompt):
            raise AssertionError("no retry needed in this scenario")

        def generate_json(self, prompt):
            if _SELF_CHECK_MARKER in prompt:
                return _complete()
            if "Уже выполненные шаги" not in prompt:
                captured["step_prompt"] = prompt
                return _query("SELECT SUM(plan_2027) AS total FROM budget_items")
            return _final("Итог найден.")

    monkeypatch.setattr(file_sql, "_get_client", lambda: _FakeProvider())

    combined = _subsidiary_b(tmp_path)
    answer = answer_over_file("сколько всего на 2027", combined, {"unit": 1000})

    assert answer is not None
    assert "Предыдущие вопросы" not in captured["step_prompt"]
