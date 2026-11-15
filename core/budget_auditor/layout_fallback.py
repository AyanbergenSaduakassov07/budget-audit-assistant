"""LLM-assisted header/column detection for budget_auditor.parser.

Used ONLY when the deterministic pass (parser._detect_layout) can't find a
usable header row, or finds one but recognizes too few of the known
semantic roles to trust (see parser._MIN_MATCHED_ROLES) -- a genuinely
different subsidiary layout, not one of the template variants the
deterministic rules already handle. Never fires for a file that already
parses successfully today.

The model's ONLY job is locating structure: which row is the header, and
which column is which role. It never does math and never decides which rows
are leaf items -- once its mapping is returned and validated, it feeds
straight into parser.parse_sheet(), the exact same deterministic
row-iteration/flagging/aggregation code every other sheet goes through. This
module never re-implements any of that; it is a locator, not a second parser.

Vocabulary reused, not invented: the assignable dimension roles are
core.budget_auditor.file_sql's own _DIM_COLS, minus "branch"/"cost_type" --
those two are sheet-level properties aggregate_workbook() derives from the
sheet NAME (CAPEX vs. OPEX, branch prefix), never a per-row column in this
file family, so asking the model to map them to a column would invite a
mapping that gets silently overwritten downstream anyway. Money columns
follow the exact plan_<year>/fact_<year>/forecast_<year>[_qN] naming
file_sql._MONEY_RE already matches.

Per SPEC.md architecture rule: this module is pure -- no streamlit imports.
"""

from __future__ import annotations

import json
import logging
import re

from openpyxl.utils import column_index_from_string, get_column_letter

from core.budget_auditor.file_sql import _DIM_COLS
from core.budget_narrator import _strip_json_fences
from legacy.query_engine import DailyQuotaExhausted, _generate_content, _get_client

logger = logging.getLogger(__name__)

# First N rows, every column -- enough for the model to see the header band
# and a few data rows beneath it without a full sheet dump.
_SAMPLE_ROWS = 20

# file_sql._DIM_COLS minus branch/cost_type (see module docstring) -- the
# roles the model may assign to a column in THIS mapping.
_ASSIGNABLE_DIM_ROLES = frozenset(_DIM_COLS) - {"branch", "cost_type"}

_MONEY_ROLE_RE = re.compile(r"^(plan|fact|forecast)_(\d{4})(?:_q([1-4]))?$")

# At least one of these must be mapped for a layout to be usable -- a table
# with money columns but no way to identify WHAT each row is isn't worth
# parsing at all.
_REQUIRED_IDENTIFIER_ROLES = ("statya_code", "statya_name", "item_name")


def _extract_sample(ws) -> str:
    """First _SAMPLE_ROWS rows, every column, non-empty values only, as
    compact "{row}{col_letter}={value!r}" lines."""
    lines = []
    max_row = min(ws.max_row or 0, _SAMPLE_ROWS)
    max_col = ws.max_column or 0
    for r in range(1, max_row + 1):
        for c in range(1, max_col + 1):
            v = ws.cell(row=r, column=c).value
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            lines.append(f"{r}{get_column_letter(c)}={v!r}")
    return "\n".join(lines)


def _build_prompt(sample: str) -> str:
    roles = ", ".join(sorted(_ASSIGNABLE_DIM_ROLES))
    return (
        "Ты анализируешь необработанный лист Excel-файла бюджета ИТ-расходов "
        "(корпоративная форма), который не удалось разобрать по стандартному "
        "шаблону — заголовок таблицы отсутствует в ожидаемом месте или "
        "назван иначе, чем в обычных формах. Ниже — непустые ячейки первых "
        "строк листа в формате {номер строки}{буква столбца}={значение}.\n\n"
        f"{sample}\n\n"
        "Определи:\n"
        "1. header_row — номер строки, которая является заголовком таблицы "
        "(содержит названия столбцов; данные начинаются со следующей строки).\n"
        "2. Для каждого столбца этой строки — его роль. Допустимые роли:\n"
        f"   - {roles} (описательные колонки)\n"
        "   - plan_<год>, fact_<год>, forecast_<год> — годовые суммы "
        "(план/факт/прогноз), например plan_2027, fact_2025\n"
        "   - plan_<год>_q<1-4> — квартальная разбивка плана, например "
        "plan_2027_q1\n\n"
        "Столбец, не соответствующий ни одной из этих ролей, не включай в "
        "ответ. Верни ТОЛЬКО валидный JSON без markdown-ограждений (без "
        "```), без пояснений до или после, в точности такой структуры:\n"
        '{"header_row": <int>, "columns": {"<буква столбца>": "<роль>", ...}}'
    )


def _parse_response(raw_text: str) -> dict | None:
    """Parse the model's JSON response; None (never raise) on anything
    malformed."""
    cleaned = _strip_json_fences(raw_text)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _mapping_to_layout(columns) -> tuple[dict, dict]:
    """Convert the LLM's {"A": "role", ...} mapping into the exact (colmap,
    year_cols) shape parser._classify_columns() produces, so parse_sheet()
    consumes either source identically. A column letter or role parser.py/
    file_sql.py doesn't recognize is silently dropped, not fatal -- the
    surrounding _validate_layout() call is what decides pass/fail overall."""
    colmap: dict[str, int] = {}
    year_cols: dict[int, tuple] = {}

    if not isinstance(columns, dict):
        return colmap, year_cols

    for letter, role in columns.items():
        try:
            col_idx = column_index_from_string(str(letter).strip().upper())
        except ValueError:
            continue
        role = str(role).strip()

        m = _MONEY_ROLE_RE.match(role)
        if m:
            kind, year, quarter = m.group(1), int(m.group(2)), m.group(3)
            year_cols[col_idx] = ("quarter", year, int(quarter)) if quarter else (kind, year)
            continue

        if role in _ASSIGNABLE_DIM_ROLES:
            colmap[role] = col_idx

    return colmap, year_cols


def _validate_layout(colmap: dict, year_cols: dict) -> bool:
    """At least one money column AND at least one row-identifying column --
    never trust a mapping short of that, however confidently phrased."""
    return bool(year_cols) and any(role in colmap for role in _REQUIRED_IDENTIFIER_ROLES)


def detect_layout_via_llm(ws) -> tuple[int, dict, dict] | None:
    """Ask the LLM to locate the header row and map columns to parser.py's
    known semantic roles, for a sheet the deterministic pass couldn't
    handle. Returns (header_row, colmap, year_cols) -- parser.parse_sheet()'s
    own layout-override shape -- on a validated mapping, else None.

    Never raises: no provider configured, a quota/network failure, or a
    malformed/invalid response all degrade to None, the same never-raises
    contract as narrator_bridge.narrative_opening and
    file_sql.answer_over_file.
    """
    try:
        sample = _extract_sample(ws)
        if not sample.strip():
            return None

        client = _get_client()
        raw = _generate_content(client, _build_prompt(sample))
        parsed = _parse_response(raw)
        if parsed is None:
            return None

        header_row = parsed.get("header_row")
        if not isinstance(header_row, int) or isinstance(header_row, bool) or header_row < 1:
            return None

        colmap, year_cols = _mapping_to_layout(parsed.get("columns"))
        if not _validate_layout(colmap, year_cols):
            logger.warning(
                "LLM layout fallback produced an unusable mapping "
                "(money_cols=%d identifier_roles=%s); discarding, "
                "falling back to the original deterministic result",
                len(year_cols), [r for r in _REQUIRED_IDENTIFIER_ROLES if r in colmap],
            )
            return None

        logger.info(
            "LLM layout fallback succeeded: header_row=%d roles=%s money_cols=%d",
            header_row, sorted(colmap), len(year_cols),
        )
        return header_row, colmap, year_cols
    except DailyQuotaExhausted:
        logger.info("LLM layout fallback skipped: daily quota exhausted")
        return None
    except Exception:
        logger.exception("LLM layout fallback failed; using original deterministic result")
        return None
