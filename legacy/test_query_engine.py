"""Tests for the Russian query engine and the warehouse layer beneath it.

The query engine calls the live Gemini API (paid / daily-quota limited), so the
two tests that hit it are marked `@pytest.mark.api` and can be deselected with
`-m "not api"`. The warehouse-only test runs for free and needs no marker.
"""

import pytest

from legacy.synth import generate_group
from legacy.warehouse import build_warehouse, run_sql
from legacy.query_engine import (
    _SQL_EXAMPLES,
    _format_history,
    _generate_sql,
    _phrase_answer,
    answer,
)


# --- few-shot example selection (regression) -------------------------------------
# _generate_sql() must default to the synthetic `procurements` warehouse examples
# unless a caller (e.g. core.budget_auditor.file_sql) explicitly passes its own
# schema-appropriate set.
def test_generate_sql_defaults_to_synthetic_examples():
    captured = {}

    class _FakeProvider:
        def generate_sql(self, prompt):
            captured["prompt"] = prompt
            return "SELECT 1"

    _generate_sql(_FakeProvider(), "Сколько всего?", "Table: procurements")

    prompt = captured["prompt"]
    # A sample question/SQL pair unique to _SQL_EXAMPLES must be present verbatim.
    sample_question, sample_sql = _SQL_EXAMPLES[0]
    assert sample_question in prompt
    assert sample_sql in prompt
    # And nothing from a foreign schema (e.g. budget_items) leaks in.
    assert "budget_items" not in prompt


def test_generate_sql_uses_explicit_examples_when_given():
    captured = {}

    class _FakeProvider:
        def generate_sql(self, prompt):
            captured["prompt"] = prompt
            return "SELECT 1"

    custom_examples = [("Сколько плана на 2027?", "SELECT SUM(plan_2027) FROM budget_items")]
    _generate_sql(
        _FakeProvider(), "Сколько плана на 2027?", "Table: budget_items",
        examples=custom_examples,
    )

    prompt = captured["prompt"]
    assert "SELECT SUM(plan_2027) FROM budget_items" in prompt
    # The default synthetic-warehouse examples must NOT also appear.
    for _, sql in _SQL_EXAMPLES:
        assert sql not in prompt


# --- number formatting in the phrased answer (regression) -----------------------
# A real answer over a workbook-wide total leaked pandas' scientific notation
# straight into the Russian text ("1.5e+11 тенге"). The model, not our
# code, does the formatting, so the only thing we can pin down is that the
# prompt actually instructs it not to do that.
def test_phrase_answer_prompt_forbids_scientific_notation():
    captured = {}

    class _FakeProvider:
        def generate_json(self, prompt):
            captured["prompt"] = prompt
            return "155 903 400 000 тенге"

        def generate_sql(self, prompt):
            return ""

    result = _phrase_answer(
        _FakeProvider(),
        "Какая общая сумма закупок за все годы?",
        "SELECT SUM(subtotal) FROM procurements",
        "1.5e+11",
        empty=False,
    )

    prompt = captured["prompt"]
    assert "экспоненциальн" in prompt.lower()
    assert "разделител" in prompt.lower()
    assert result == "155 903 400 000 тенге"


# --- conversation history formatting (regression) --------------------------------
def test_format_history_empty_is_blank():
    assert _format_history(None) == ""
    assert _format_history([]) == ""


def test_format_history_includes_question_and_answer():
    block = _format_history([("сколько CAPEX в 2027", "CAPEX на 2027 год: 800 тыс.")])
    assert "Предыдущие вопросы и ответы" in block
    assert "сколько CAPEX в 2027" in block
    assert "CAPEX на 2027 год: 800 тыс." in block


def test_format_history_caps_to_last_three_turns():
    """Older turns are dropped (not summarized) past _MAX_HISTORY_TURNS."""
    history = [(f"вопрос {i}", f"ответ {i}") for i in range(1, 6)]  # 5 turns
    block = _format_history(history)
    for i in (1, 2):
        assert f"вопрос {i}" not in block
    for i in (3, 4, 5):
        assert f"вопрос {i}" in block


def test_warehouse_totals_match_dataframe():
    """The SQL warehouse must return the same per-subsidiary totals as pandas.

    Validates the warehouse layer (schema + run_sql) with no API cost: build the
    warehouse from the synthetic group, then check that a GROUP BY SUM in SQL
    matches a groupby().sum() computed directly on the source DataFrame.
    """
    df, _ = generate_group()
    build_warehouse(df)

    expected = df.groupby("subsidiary")["subtotal"].sum()

    result = run_sql(
        "SELECT subsidiary, SUM(subtotal) as total FROM procurements "
        "GROUP BY subsidiary"
    )
    actual = result.set_index("subsidiary")["total"]

    # Every subsidiary present in the DataFrame must appear in the SQL result.
    assert set(actual.index) == set(expected.index)
    for subsidiary, expected_total in expected.items():
        assert actual[subsidiary] == pytest.approx(expected_total, abs=0.01)


@pytest.mark.api
def test_engine_returns_correct_total():
    """answer() must surface the true total as an actual value in the data.

    We check the returned DataFrame (result["data"]), not the natural-language
    text, so the assertion is about the number the SQL actually computed rather
    than how the model chose to phrase it.
    """
    df, _ = generate_group()
    build_warehouse(df)

    mask = (df["subsidiary"] == "Subsidiary A") & (df["year"] == 2024)
    true_total = round(df.loc[mask, "subtotal"].sum())

    result = answer("Сколько всего потратил Subsidiary A в 2024 году?")

    data = result["data"]
    assert data is not None and not data.empty

    values = data.select_dtypes(include="number").to_numpy().ravel()
    assert any(round(v) == true_total for v in values), (
        f"expected {true_total} in returned data values {list(values)}"
    )


@pytest.mark.api
def test_engine_handles_unknown_gracefully():
    """A query about a nonexistent subsidiary must not crash or invent a number.

    Acceptable outcomes: an error is set, the returned data is None/empty, or the
    data holds no real number (a SUM over zero matching rows comes back as a
    single NULL/NaN row — no figure was fabricated). What is NOT acceptable is a
    concrete number for a company that does not exist in the warehouse.
    """
    df, _ = generate_group()
    build_warehouse(df)

    result = answer("Сколько потратила компания НесуществующаяКорпорация?")

    data = result["data"]
    if result["error"] is not None or data is None or data.empty:
        return

    # Non-empty result is only acceptable if it contains no actual numeric value
    # (i.e. every numeric cell is NULL/NaN — nothing was invented).
    numeric = data.select_dtypes(include="number")
    assert numeric.empty or numeric.isna().all().all(), (
        f"engine returned a concrete number for an unknown subsidiary: {data!r}"
    )
