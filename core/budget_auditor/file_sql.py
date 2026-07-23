"""SQL fallback for the file chat.

When the rule-based chat_answer() returns None, load THIS uploaded file's
combined DataFrame into a throwaway SQLite table and route the question through
legacy.query_engine's Gemini text-to-SQL (Russian question -> SQL grounded in a
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
from legacy.query_engine import (
    DailyQuotaExhausted,
    _generate_sql,
    _get_client,
    _phrase_answer,
)

logger = logging.getLogger(__name__)

_TABLE = "budget_items"

# plan_2027 / fact_2025 / forecast_2028 / plan_2027_q1 — the money columns.
_MONEY_RE = re.compile(r"^(plan|fact|forecast)_\d{4}(_q[1-4])?$")

# Descriptive dimensions worth exposing to the SQL layer (flags, per-row deltas,
# and helper columns are intentionally left out — the model reasons over raw
# amounts, the same discipline as the synthetic warehouse's schema).
_DIM_COLS = [
    "branch", "cost_type", "statya_code", "statya_name",
    "item_name", "category", "subcategory", "counterparty", "justification",
]

_MONEY_MEANING = {
    "plan": "утверждённый план на",
    "fact": "факт за",
    "forecast": "прогноз на",
}


def _split_columns(combined: pd.DataFrame) -> tuple[list[str], list[str]]:
    dims = [c for c in _DIM_COLS if c in combined.columns]
    money = [c for c in combined.columns if _MONEY_RE.match(str(c))]
    return dims, money


def _load_temp_db(combined: pd.DataFrame, dims: list[str], money: list[str]) -> str:
    """Write the selected columns into a fresh temp-file SQLite table; return
    its path. Money columns become REAL, dimensions TEXT."""
    frame = combined[dims + money].copy()
    for c in money:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    for c in dims:
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


def _schema_description(combined: pd.DataFrame, meta: dict, dims: list[str], money: list[str]) -> str:
    """Plain-text schema grounding the model in THIS file's table — dimensions,
    the per-year money columns, and the exact distinct branch/cost_type values."""
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

    return f"""\
Table: {_TABLE}
Плоская таблица одного загруженного файла ИТ-бюджета компании. Одна строка = одна
статья расхода одного листа/подразделения. Все суммы уже в {unit_label}.

Dimension columns (TEXT):
{dim_block}

Money columns (REAL, суммы в {unit_label}):
{money_block}

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
"""


def answer_over_file(question: str, combined: pd.DataFrame, meta: dict) -> str | None:
    """Answer a question via Gemini text-to-SQL over the current file's data.

    Returns the Russian answer string, or None on ANY failure (missing key,
    quota, empty data, bad SQL, no numeric columns) — the caller then shows its
    existing not-found message. Never raises.
    """
    try:
        if combined is None or combined.empty:
            return None
        dims, money = _split_columns(combined)
        if not money:
            return None  # nothing numeric to query against
        client = _get_client()  # raises RuntimeError if GEMINI_API_KEY is unset
    except Exception:
        return None

    db_path = None
    try:
        db_path = _load_temp_db(combined, dims, money)
        schema = _schema_description(combined, meta or {}, dims, money)
        sql = _generate_sql(client, question, schema)
        df = warehouse.run_sql(sql, db_path)
        result_text = df.to_string(index=False) if not df.empty else "(пустой результат)"
        answer = _phrase_answer(client, question, sql, result_text, empty=df.empty)
        return answer or None
    except DailyQuotaExhausted:
        return None
    except Exception:
        logger.exception("file SQL fallback failed for question: %s", question)
        return None
    finally:
        if db_path and os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except OSError:
                logger.warning("could not remove temp SQL db %s", db_path)
