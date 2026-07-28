"""Tests for the Russian query engine and the warehouse layer beneath it.

The query engine calls the live OpenRouter API (paid / daily-quota limited), so the
two tests that hit it are marked `@pytest.mark.api` and can be deselected with
`-m "not api"`. The warehouse-only test runs for free and needs no marker.
"""

import json

import pytest

from legacy.synth import generate_group
from legacy.warehouse import build_warehouse, run_sql
from legacy.query_engine import (
    _SQL_EXAMPLES,
    _format_history,
    _generate_sql,
    _phrase_answer,
    _strip_decorative_chars,
    answer,
    next_step,
    self_check_answer,
    synthesize_answer,
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


# --- decorative-character stripping (regression) ---------------------------------
# Both the prompt instruction and this post-processing are defense in depth:
# a model that decorates its answer despite being told not to must still come
# back as plain business-register text.
def test_strip_decorative_chars_removes_emoji_and_markdown():
    raw = "📊 Бюджет вырос на **15%** к прошлому году ✅\n- пункт списка\n* второй пункт"
    cleaned = _strip_decorative_chars(raw)
    assert "📊" not in cleaned and "✅" not in cleaned
    assert "**" not in cleaned and "15%" in cleaned
    assert "- пункт" not in cleaned and "пункт списка" in cleaned
    assert "* второй" not in cleaned and "второй пункт" in cleaned


def test_strip_decorative_chars_leaves_clean_text_unchanged():
    clean = "Бюджет вырос на 15% к прошлому году, что требует уточнения."
    assert _strip_decorative_chars(clean) == clean
    # Underscores inside identifiers and a literal multiplication sign must
    # survive -- the stripping regex must not treat every _/* as a marker pair.
    assert _strip_decorative_chars("plan_annual остаётся без изменений") == "plan_annual остаётся без изменений"
    assert _strip_decorative_chars("5 * 3 = 15, не искажать") == "5 * 3 = 15, не искажать"


def test_phrase_answer_strips_decorative_chars_from_response():
    class _FakeProvider:
        def generate_json(self, prompt):
            return "📊 Общая сумма: **155 903 400 000** тенге ✅"

        def generate_sql(self, prompt):
            return ""

    result = _phrase_answer(
        _FakeProvider(),
        "Какая общая сумма закупок за все годы?",
        "SELECT SUM(subtotal) FROM procurements",
        "155903400000",
        empty=False,
    )
    assert result == "Общая сумма: 155 903 400 000 тенге"


def test_phrase_answer_prompt_includes_formatting_instruction():
    captured = {}

    class _FakeProvider:
        def generate_json(self, prompt):
            captured["prompt"] = prompt
            return "155 903 400 000 тенге"

        def generate_sql(self, prompt):
            return ""

    _phrase_answer(
        _FakeProvider(), "Сколько всего?", "SELECT 1", "(пустой результат)", empty=True,
    )
    assert "без эмодзи" in captured["prompt"]
    assert "markdown" in captured["prompt"]


def test_synthesize_answer_strips_decorative_chars_from_response():
    class _FakeProvider:
        def generate_json(self, prompt):
            return "🔥 Итого: **500 000** тенге\n- готово"

        def generate_sql(self, prompt):
            return ""

    result = synthesize_answer(
        _FakeProvider(),
        "Сравни расходы по двум категориям.",
        [("Сколько на A?", "SELECT 1", "300000"), ("Сколько на B?", "SELECT 1", "200000")],
    )
    assert result == "Итого: 500 000 тенге\nготово"


def test_synthesize_answer_prompt_includes_formatting_instruction():
    captured = {}

    class _FakeProvider:
        def generate_json(self, prompt):
            captured["prompt"] = prompt
            return "500 000 тенге"

        def generate_sql(self, prompt):
            return ""

    synthesize_answer(_FakeProvider(), "Вопрос", [("Подвопрос", "SELECT 1", "500000")])
    assert "без эмодзи" in captured["prompt"]


# --- next_step() / self_check_answer(): the bounded step loop's primitives ------
def test_next_step_parses_query_action():
    class _FakeProvider:
        def generate_json(self, prompt):
            return json.dumps({
                "action": "query", "sql": "SELECT SUM(plan_2027) FROM budget_items",
                "reason": "узнать план",
            })

    step = next_step(_FakeProvider(), "Вопрос", "Table: budget_items", scratchpad=[])
    assert step == {
        "action": "query",
        "sql": "SELECT SUM(plan_2027) FROM budget_items",
        "reason": "узнать план",
    }


def test_next_step_final_action_strips_decorative_chars():
    class _FakeProvider:
        def generate_json(self, prompt):
            return json.dumps({"action": "final", "answer": "📊 Итог: **9 800** тенге ✅"})

    step = next_step(_FakeProvider(), "Вопрос", "Table: budget_items", scratchpad=[])
    assert step == {"action": "final", "answer": "Итог: 9 800 тенге"}


def test_next_step_malformed_or_empty_response_returns_error_action():
    class _FakeProvider:
        def generate_json(self, prompt):
            return "not json at all"

    assert next_step(_FakeProvider(), "Вопрос", "Table: t", scratchpad=[]) == {"action": "error"}

    class _EmptyAnswerProvider:
        def generate_json(self, prompt):
            return json.dumps({"action": "final", "answer": "   "})

    assert next_step(_EmptyAnswerProvider(), "Вопрос", "Table: t", scratchpad=[]) == {"action": "error"}

    class _EmptySqlProvider:
        def generate_json(self, prompt):
            return json.dumps({"action": "query", "sql": "", "reason": "..."})

    assert next_step(_EmptySqlProvider(), "Вопрос", "Table: t", scratchpad=[]) == {"action": "error"}


def test_next_step_prompt_requires_a_query_when_scratchpad_empty_not_when_populated():
    captured = {}

    class _FakeProvider:
        def generate_json(self, prompt):
            captured["prompt"] = prompt
            return json.dumps({"action": "final", "answer": "Итог."})

    next_step(_FakeProvider(), "Вопрос", "Table: t", scratchpad=[])
    assert "ОБЯЗАТЕЛЬНО нужно" in captured["prompt"]

    next_step(_FakeProvider(), "Вопрос", "Table: t",
              scratchpad=[{"sql": "SELECT 1", "result": "1", "reason": "проверка"}])
    assert "ОБЯЗАТЕЛЬНО нужно" not in captured["prompt"]
    assert "Уже выполненные шаги" in captured["prompt"]
    assert "проверка" in captured["prompt"]


def test_next_step_prompt_includes_formatting_instruction():
    captured = {}

    class _FakeProvider:
        def generate_json(self, prompt):
            captured["prompt"] = prompt
            return json.dumps({"action": "final", "answer": "Итог."})

    next_step(_FakeProvider(), "Вопрос", "Table: t", scratchpad=[])
    assert "без эмодзи" in captured["prompt"]
    assert "markdown" in captured["prompt"]


def test_self_check_answer_parses_complete_and_missing():
    class _FakeProvider:
        def generate_json(self, prompt):
            return json.dumps({"complete": False, "missing": "не хватает данных за 2025 год"})

    result = self_check_answer(_FakeProvider(), "Вопрос", "Черновой ответ")
    assert result == {"complete": False, "missing": "не хватает данных за 2025 год"}


def test_self_check_answer_malformed_response_defaults_to_complete():
    """A self-check failure must never block the answer -- degrade to
    "ship the draft as-is", not raise or force an extra step."""
    class _FakeProvider:
        def generate_json(self, prompt):
            return "garbage, not json"

    assert self_check_answer(_FakeProvider(), "Вопрос", "Черновой ответ") == {
        "complete": True, "missing": "",
    }


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
