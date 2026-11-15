"""SQL fallback for the file chat.

When the rule-based chat_answer() returns None, load THIS uploaded file's
combined DataFrame into a throwaway SQLite table and route the question through
legacy.query_engine's LLM text-to-SQL (Russian question -> SQL grounded in a
schema description -> SQL runs -> Russian answer phrased from ONLY the returned
numbers). Same pattern query_engine already uses for the synthetic warehouse —
here pointed at a per-file table instead.

Reuse, not duplication: the client, retry/backoff, DailyQuotaExhausted, SQL
generation and answer phrasing all come from legacy.query_engine unchanged;
execution reuses legacy.warehouse.run_sql (its SELECT-only + read-only guard).
Only the table build and the schema description are file-specific.

The table is ephemeral and scoped to one call — a fresh temp DB built from the
current file, dropped in a finally, never the persistent synthetic warehouse.
(A temp file rather than :memory: so warehouse.run_sql's read-only second
connection can reach it.)

answer_over_file() NEVER raises: no key, quota exhaustion, bad SQL, or any other
failure returns None, so app.py falls back to its existing "Не нашёл…" message —
an answer slot never shows an error.

Per SPEC_AUDIT.md architecture rule: pure module, no streamlit imports.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import tempfile

import pandas as pd

from legacy import warehouse
from legacy.llm_providers import DAILY_QUOTA_MESSAGE
from legacy.query_engine import (
    DailyQuotaExhausted,
    _generate_sql_with_error_feedback,
    _get_client,
    next_step,
    self_check_answer,
    synthesize_answer,
)

# Hard cap on how many next_step() calls a single answer_over_file()
# invocation may make in total (main loop + the self-check's optional one
# extra step both draw from this same budget — see _answer_via_step_loop) —
# bounds both LLM call count and temp-DB query count.
_MAX_STEPS = 4

logger = logging.getLogger(__name__)

_TABLE = "budget_items"

# plan_2027 / fact_2025 / forecast_2028 / plan_2027_q1 — the money columns.
_MONEY_RE = re.compile(r"^(plan|fact|forecast)_\d{4}(_q[1-4])?$")

# Descriptive dimensions worth exposing to the SQL layer.
_DIM_COLS = [
    "branch", "cost_type", "statya_code", "statya_name",
    "item_name", "category", "subcategory", "counterparty", "justification",
]

_MONEY_MEANING = {
    "plan": "утверждённый план на",
    "fact": "факт за",
    "forecast": "прогноз на",
}

# Pre-computed anomaly/severity columns from core.budget_auditor.parser's
# flag_rows()/_flag_quarterly(). These ARE exposed to the SQL layer (unlike
# other helper columns) precisely because they are the authoritative already-
# computed answer to "is this row anomalous" questions — re-deriving them from
# raw plan_<year>/plan_<year>_qN columns has been shown to produce a
# materially wrong result (e.g. counting floating-point noise below the 1%
# tolerance as a "discrepancy"). The schema text tells the model to filter on
# these columns, not recompute the underlying condition.
_FLAG_COLS = [
    "quarterly_mismatch_flag", "quarterly_severity", "quarterly_gap_pct",
    "high_change_flag", "delta_pct",
    "missing_justification_flag", "high_price_flag",
]

# quarterly_mismatch_flag/high_change_flag/missing_justification_flag/
# high_price_flag are plain bool Series (never NA) from parser.py — cast
# straight to SQLite INTEGER 0/1. quarterly_gap_pct/delta_pct are float
# percentages (NA where not applicable) — REAL, same as money columns.
# quarterly_severity is a 'critical'/'minor'/NA string — TEXT, same as dims.
_FLAG_BOOL_COLS = {
    "quarterly_mismatch_flag", "high_change_flag",
    "missing_justification_flag", "high_price_flag",
}
_FLAG_NUMERIC_COLS = {"quarterly_gap_pct", "delta_pct"}

_FLAG_DESCRIPTIONS = {
    "quarterly_mismatch_flag": (
        "  - quarterly_mismatch_flag    INTEGER (0/1)   1 значит, что сумма "
        "плановых кварталов (plan_<год>_q1..q4) этой строки расходится с "
        "годовым планом (plan_<год>) больше чем на 1%. Уже посчитано — "
        "фильтруйте по этой колонке, НЕ пересчитывайте сами суммированием "
        "plan_<год>_qN."
    ),
    "quarterly_severity": (
        "  - quarterly_severity         TEXT ('critical'/'minor'/NULL)   "
        "'critical' — расхождение больше 5%; 'minor' — от 1% до 5%; NULL — "
        "расхождения нет или кварталы неполные. Уже посчитано."
    ),
    "quarterly_gap_pct": (
        "  - quarterly_gap_pct          REAL   |годовой план − сумма "
        "кварталов| / годовой план * 100 (в процентах). Уже посчитано — "
        "используйте это значение, не пересчитывайте."
    ),
    "high_change_flag": (
        "  - high_change_flag           INTEGER (0/1)   1 значит, что "
        "изменение плана к предыдущему году (delta_pct) по модулю больше "
        "15%. Уже посчитано."
    ),
    "delta_pct": (
        "  - delta_pct                  REAL   изменение плана к базовому "
        "(предыдущему) году, в процентах: (план − база) / база * 100. Уже "
        "посчитано."
    ),
    "missing_justification_flag": (
        "  - missing_justification_flag INTEGER (0/1)   1 значит, что для "
        "этой строки отсутствует обоснование расхода (justification). Уже "
        "посчитано."
    ),
    "high_price_flag": (
        "  - high_price_flag            INTEGER (0/1)   1 значит, что план "
        "этой строки входит в верхние 10% (90-й перцентиль) по сумме плана "
        "среди строк своего листа. Уже посчитано."
    ),
}

# Few-shot examples grounded in THIS module's real `budget_items` schema
# (branch, cost_type, statya_code, plan_<year>/fact_<year>/forecast_<year>).
# Passed explicitly into every next_step() call instead of falling back to
# query_engine's default _SQL_EXAMPLES, which are written against the
# unrelated synthetic `procurements` table (subtotal/subsidiary/category/year)
# and would otherwise show the model a correct schema description alongside
# contradictory worked examples for a table that doesn't exist here.
_BUDGET_ITEMS_SQL_EXAMPLES = [
    (
        "Сколько всего плана на 2027 год?",
        "SELECT SUM(plan_2027) FROM budget_items",
    ),
    (
        "Сравни план на 2027 год по подразделениям.",
        "SELECT branch, SUM(plan_2027) AS total FROM budget_items "
        "GROUP BY branch",
    ),
    (
        "Раздели план на 2027 год на OPEX и CAPEX.",
        "SELECT cost_type, SUM(plan_2027) AS total FROM budget_items "
        "GROUP BY cost_type",
    ),
    (
        "Сколько потратили на контрагента ТОО Прогресс в 2027 году?",
        "SELECT SUM(fact_2027) FROM budget_items "
        "WHERE counterparty = 'ТОО Прогресс'",
    ),
    (
        "Есть ли расхождения по кварталам?",
        "SELECT branch, cost_type, statya_code, item_name, quarterly_gap_pct, "
        "quarterly_severity FROM budget_items WHERE quarterly_mismatch_flag = 1",
    ),
]


def _split_columns(combined: pd.DataFrame) -> tuple[list[str], list[str]]:
    dims = [c for c in _DIM_COLS if c in combined.columns]
    money = [c for c in combined.columns if _MONEY_RE.match(str(c))]
    return dims, money


def _split_flag_columns(combined: pd.DataFrame) -> list[str]:
    return [c for c in _FLAG_COLS if c in combined.columns]


def _load_temp_db(
    combined: pd.DataFrame, dims: list[str], money: list[str], flags: list[str],
) -> str:
    """Write the selected columns into a fresh temp-file SQLite table; return
    its path. Money columns become REAL, dimensions TEXT, and pre-computed
    flag columns keep the type appropriate to their SQL role (see
    _FLAG_BOOL_COLS/_FLAG_NUMERIC_COLS above; quarterly_severity is TEXT)."""
    frame = combined[dims + money + flags].copy()
    for c in money:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    for c in dims:
        frame[c] = frame[c].astype("string")
    for c in flags:
        if c in _FLAG_BOOL_COLS:
            frame[c] = frame[c].astype(int)
        elif c in _FLAG_NUMERIC_COLS:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
        else:
            frame[c] = frame[c].astype("string")

    fd, path = tempfile.mkstemp(suffix=".db", prefix="filesql_")
    os.close(fd)
    conn = sqlite3.connect(path)
    try:
        frame.to_sql(_TABLE, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()
    return path


def _money_line(col: str) -> str:
    m = re.match(r"^(plan|fact|forecast)_(\d{4})(?:_q([1-4]))?$", col)
    if not m:
        return f"  - {col}    REAL"
    kind, year, quarter = m.group(1), m.group(2), m.group(3)
    meaning = _MONEY_MEANING[kind]
    if quarter:
        return f"  - {col}    REAL   план {quarter} квартал {year} года"
    return f"  - {col}    REAL   {meaning} {year} год"


def _plan_year(combined: pd.DataFrame, meta: dict) -> int | None:
    """Campaign/plan year: combined['plan_year'] (set row-wise by
    aggregate_workbook) first, meta['campaign_year'] as fallback -- same
    idiom already used by narrator_bridge.py for the same uncertainty
    (a file may only ever populate 'Компания' and never 'Год кампании')."""
    year = None
    if "plan_year" in combined.columns and combined["plan_year"].notna().any():
        year = combined["plan_year"].dropna().iloc[0]
    elif meta.get("campaign_year") is not None:
        year = meta.get("campaign_year")
    if year is None or pd.isna(year):
        return None
    return int(year)


def _years_in_money_cols(money: list[str]) -> set[int]:
    years = set()
    for col in money:
        m = re.match(r"^(?:plan|fact|forecast)_(\d{4})(?:_q[1-4])?$", col)
        if m:
            years.add(int(m.group(1)))
    return years


def _structural_notes(combined: pd.DataFrame, meta: dict, dims: list[str], money: list[str]) -> str:
    """Structural facts parser.py already computed and discarded before
    reaching the LLM (Bug D / Bug C schema half): whether an earlier year
    than the campaign year is represented at all, and whether a
    justification/Обоснование column exists for this file. Stated only when
    the gap actually applies -- no earlier year, or no justification column
    -- so a normal file's schema carries neither warning."""
    notes = []

    plan_year = _plan_year(combined, meta)
    if plan_year is not None:
        years = _years_in_money_cols(money)
        earlier = sorted(y for y in years if y < plan_year)
        if not earlier:
            notes.append(
                f"Кампания: {plan_year} год. Более ранних лет (до {plan_year}) в "
                "данных НЕТ — если вопрос про «прошлый год» или сравнение с "
                "предыдущим годом, ответь, что сравнение невозможно, НЕ "
                "подставляй более поздний год вместо более раннего."
            )

    if "justification" not in dims:
        notes.append(
            "В этом файле НЕТ колонки Обоснование — обоснование отсутствует "
            "для 100% статей, это структурный пробел формы, а не результат "
            "фильтрации."
        )

    return "\n".join(notes)


def _meta_block(meta: dict, unit_label: str) -> str:
    """Workbook header metadata (company, campaign year, currency, unit,
    version) as prompt text.

    parser.py reads all of this off the form's header, but none of it used to
    reach the model -- only the table's rows did. So a pure metadata question
    ("какая это компания") had nothing to ground on and came back with a
    confidently invented company name. Values missing from the form are
    spelled out as missing rather than omitted, so the model can't read a
    blank line as an invitation to guess."""
    fields = [
        ("Компания", meta.get("company")),
        ("Год кампании", meta.get("campaign_year")),
        ("Валюта", meta.get("currency")),
        ("Единица измерения", unit_label),
        ("Версия формы", meta.get("version")),
    ]
    lines = []
    for label, value in fields:
        text = str(value).strip() if value is not None else ""
        lines.append(f"  - {label}: {text or 'не указано в форме'}")
    return "\n".join(lines)


def _schema_description(
    combined: pd.DataFrame, meta: dict, dims: list[str], money: list[str],
    flags: list[str],
) -> str:
    """Plain-text schema grounding the model in THIS file's table — dimensions,
    the per-year money columns, the pre-computed anomaly flags, the exact
    distinct branch/cost_type values, and any structural gaps (see
    _structural_notes) — no earlier-than-campaign year, or no justification
    column at all — that parser.py already knows and would otherwise be
    silently discarded before reaching the LLM."""
    unit = meta.get("unit", "")
    unit_label = "тыс. тенге" if str(unit) == "1000" else (str(unit) or "единицах формы")

    def distinct(col: str) -> list[str]:
        if col not in combined.columns:
            return []
        return sorted({str(v) for v in combined[col].dropna().unique()})

    branch_block = "\n".join(f"  - {b}" for b in distinct("branch")) or "  (нет)"
    cost_block = "\n".join(f"  - {c}" for c in distinct("cost_type")) or "  (нет)"
    money_block = "\n".join(_money_line(c) for c in money) or "  (нет числовых колонок)"

    dim_lines = {
        "branch": "  - branch        TEXT   подразделение / филиал (ГО, АФ, КазНИПИ, …)",
        "cost_type": "  - cost_type     TEXT   'OPEX' или 'CAPEX'",
        "statya_code": "  - statya_code   TEXT   код статьи бюджета (например 'IT-OP-01')",
        "statya_name": "  - statya_name   TEXT   наименование статьи",
        "item_name": "  - item_name     TEXT   позиция расхода",
        "category": "  - category      TEXT   вид затрат",
        "subcategory": "  - subcategory   TEXT   подвид затрат",
        "counterparty": "  - counterparty  TEXT   контрагент / поставщик",
        "justification": "  - justification TEXT   обоснование расхода (может быть пустым)",
    }
    dim_block = "\n".join(dim_lines[c] for c in dims if c in dim_lines)
    flag_block = "\n".join(_FLAG_DESCRIPTIONS[c] for c in flags if c in _FLAG_DESCRIPTIONS) or "  (нет)"
    structural_notes = _structural_notes(combined, meta, dims, money)
    meta_block = _meta_block(meta, unit_label)

    return f"""\
Table: {_TABLE}
Плоская таблица одного загруженного файла ИТ-бюджета. Одна строка = одна
статья расхода одного листа/подразделения. Все суммы уже в {unit_label}.

Метаданные файла (из шапки формы, НЕ из таблицы — это и есть ответ на вопросы
«какая это компания», «чья форма», «за какой год»; берите значения дословно
отсюда, не выводите их из данных и не придумывайте):
{meta_block}

Dimension columns (TEXT):
{dim_block}

Money columns (REAL, суммы в {unit_label}):
{money_block}

Precomputed anomaly flags (уже посчитаны пайплайном аудита — фильтруйте по
ним напрямую, НЕ пересчитывайте условие самостоятельно из money-колонок):
{flag_block}

ВАЖНО про структуру:
- В таблице НЕТ единой колонки 'year' и НЕТ колонки 'subtotal'/'amount'. Сумма за
  конкретный год — это отдельная колонка: plan_<год> / fact_<год> / forecast_<год>
  (например, план на 2027 год = SUM(plan_2027)). Кварталы — plan_<год>_q<N>.
- Итоги считаются как SUM(<нужная колонка>). Для разбивки по подразделению или
  типу используйте GROUP BY branch / GROUP BY cost_type.
- Фильтруйте строго по значениям, которые реально есть в данных (см. ниже);
  ничего не переводите и не выдумывайте.

Точные значения branch (используйте написание дословно):
{branch_block}

Точные значения cost_type:
{cost_block}
""" + (f"\n{structural_notes}\n" if structural_notes else "")


def _execute_step_sql(
    client,
    question: str,
    schema: str,
    db_path: str,
    sql: str,
    reason: str,
    history: list[tuple[str, str]] | None,
) -> tuple[str, str]:
    """Execute one step's proposed SQL; on failure, retry once via the same
    self-correction every SQL attempt in this module has always used
    (_generate_sql_with_error_feedback). Never raises: even a failure that
    survives the retry is recorded as an explicit failure note rather than
    aborting the whole answer — the model sees it on the NEXT step (in the
    scratchpad) and can change approach, which is exactly the reactive
    behavior the step loop exists for. Returns (sql_actually_run, result_text).
    """
    retry_question = f"{question}\n\n(Причина этого запроса: {reason})" if reason else question
    try:
        df = warehouse.run_sql(sql, db_path)
        return sql, (df.to_string(index=False) if not df.empty else "(пустой результат)")
    except Exception as exc:
        corrected_sql = _generate_sql_with_error_feedback(
            client, retry_question, schema, sql, str(exc), history=history,
        )
        try:
            df = warehouse.run_sql(corrected_sql, db_path)
        except Exception as retry_exc:
            logger.info(
                "file SQL step retry also failed for question %r — "
                "first attempt: %s | retry: %s",
                question, sql, corrected_sql,
            )
            return corrected_sql, f"(ошибка выполнения запроса: {retry_exc})"
        logger.info(
            "file SQL step self-correction succeeded for question %r — "
            "first attempt: %s | retry: %s",
            question, sql, corrected_sql,
        )
        return corrected_sql, (df.to_string(index=False) if not df.empty else "(пустой результат)")


def _run_steps(
    client,
    question: str,
    schema: str,
    db_path: str,
    history: list[tuple[str, str]] | None,
    scratchpad: list[dict],
    max_calls: int,
) -> tuple[str | None, int]:
    """Run up to max_calls next_step() calls, mutating `scratchpad` in place
    with a {"sql", "result", "reason"} entry for every query executed.
    Returns (final_answer, calls_made) -- final_answer is None if max_calls
    was used up (or the model returned something malformed) without ever
    choosing "final"."""
    calls_made = 0
    while calls_made < max_calls:
        calls_made += 1
        step = next_step(
            client, question, schema, scratchpad, history=history,
            examples=_BUDGET_ITEMS_SQL_EXAMPLES,
        )
        action = step.get("action")
        if action == "final":
            return step["answer"], calls_made
        if action != "query":
            break  # malformed/empty response -- stop, caller forces synthesis
        sql, reason = step["sql"], step.get("reason", "")
        sql_used, result_text = _execute_step_sql(client, question, schema, db_path, sql, reason, history)
        scratchpad.append({"sql": sql_used, "result": result_text, "reason": reason})
    return None, calls_made


def _forced_synthesis(
    client,
    question: str,
    scratchpad: list[dict],
    history: list[tuple[str, str]] | None,
) -> str | None:
    """The step budget ran out (or the model returned something malformed)
    without ever choosing "final" -- force one synthesis call over whatever
    ended up in the scratchpad, rather than loop forever or surface an
    error. None if the scratchpad is empty (nothing to synthesize from)."""
    if not scratchpad:
        return None
    subresults = [(entry.get("reason") or question, entry["sql"], entry["result"]) for entry in scratchpad]
    return synthesize_answer(client, question, subresults, history=history) or None


def _answer_via_step_loop(
    client,
    question: str,
    schema: str,
    db_path: str,
    history: list[tuple[str, str]] | None,
) -> str | None:
    """The bounded step loop: propose-execute-observe up to _MAX_STEPS
    next_step() calls total, forced synthesis if the budget runs out, then a
    single self-check pass that may spend one more of the SAME budget's
    calls to fill a gap the model itself flags. A simple question that's
    fully answered by one query still resolves the same way as before this
    change: one query executed, then the very next call finalizes it.

    Logs how many queries were run and where the final answer came from
    (the model finalizing directly, forced synthesis, or the self-check's
    extra step) -- useful signal for whether _MAX_STEPS is well-tuned once
    real usage starts.
    """
    scratchpad: list[dict] = []
    final_answer, calls_used = _run_steps(client, question, schema, db_path, history, scratchpad, _MAX_STEPS)
    source = "model"

    if final_answer is None:
        final_answer = _forced_synthesis(client, question, scratchpad, history)
        if final_answer is None:
            return None
        calls_used = _MAX_STEPS
        source = "forced_synthesis"

    if calls_used < _MAX_STEPS:
        check = self_check_answer(client, question, final_answer)
        if not check["complete"]:
            follow_up = (
                f"{question}\n\n(По итогам самопроверки в ответе не хватает: "
                f"{check['missing']})" if check["missing"] else question
            )
            extra_answer, extra_calls = _run_steps(
                client, follow_up, schema, db_path, history, scratchpad, max_calls=1,
            )
            calls_used += extra_calls
            if extra_answer is not None:
                final_answer = extra_answer
                source = "self_check_extra_step"
            else:
                synthesized = _forced_synthesis(client, question, scratchpad, history)
                if synthesized is not None:
                    final_answer = synthesized
                    source = "self_check_forced_synthesis"

    logger.info(
        "question %r answered via step loop: source=%s queries_run=%d model_calls=%d",
        question, source, len(scratchpad), calls_used,
    )
    return final_answer


def answer_over_file(
    question: str,
    combined: pd.DataFrame,
    meta: dict,
    history: list[tuple[str, str]] | None = None,
) -> str | None:
    """Answer a question via LLM text-to-SQL over the current file's data.

    A bounded step loop (see _answer_via_step_loop): on each step the model
    sees the original question, the schema, and every {sql, result} pair
    from prior steps in THIS answer, and either proposes one more SQL query
    (with a short reason) or gives a final answer. This lets the model react
    to what an earlier query reveals -- e.g. look up which branch has the
    worst quarterly mismatch, then query that specific branch's suppliers --
    something a fixed up-front decomposition (this module's old classify/
    decompose/synthesize path) could never do, since every sub-question was
    planned before any of them ran. A simple question still resolves with
    exactly one query executed, same as before this change; it just takes
    one additional (cheap) call to confirm nothing more is needed.

    history: the last few (question, answer) text pairs from the chat, in
    order, so a follow-up like "а за все года" is understood against the
    prior turn instead of in isolation -- passed straight through to every
    step's prompt (see legacy.query_engine._format_history). Text only, no
    raw data, so this doesn't touch the existing data-minimization boundary.

    Returns the Russian answer string, or None on ANY failure (missing key,
    empty data, no numeric columns, the step loop never producing an answer)
    -- the caller then shows its existing not-found message. Never raises.

    Daily quota exhaustion is the one failure that does NOT return None: it
    returns DAILY_QUOTA_MESSAGE. "Out of quota until tomorrow" and "I looked
    and couldn't answer that" are different things to tell a user, and
    collapsing both into the caller's generic not-found reply is how a
    quota-out spent months looking like a reasoning failure.
    """
    try:
        if combined is None or combined.empty:
            return None
        dims, money = _split_columns(combined)
        if not money:
            return None  # nothing numeric to query against
        flags = _split_flag_columns(combined)
        client = _get_client()  # raises RuntimeError if no provider is configured
    except Exception:
        return None

    db_path = None
    try:
        db_path = _load_temp_db(combined, dims, money, flags)
        schema = _schema_description(combined, meta or {}, dims, money, flags)
        return _answer_via_step_loop(client, question, schema, db_path, history)
    except DailyQuotaExhausted:
        logger.info("file SQL fallback stopped: daily quota exhausted")
        return DAILY_QUOTA_MESSAGE
    except Exception:
        logger.exception("file SQL fallback failed for question: %s", question)
        return None
    finally:
        if db_path and os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except OSError:
                logger.warning("could not remove temp SQL db %s", db_path)
