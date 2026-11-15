"""Ingest corporate budget-export workbooks into clean flat DataFrames.

These files come out of a corporate planning tool and have a fundamentally
different shape than the synthetic procurement schema (see core/schema.py):

  - a labelled metadata block in the first ~15 rows (year, version, company,
    currency, and a unit scale such as "1000" meaning values are in thousands);
  - a header that may span two rows (a parent row of column names plus a child
    row of sub-labels for *some* columns);
  - data rows interspersed with fully blank separator rows;
  - one sheet per entity / expense category, where the sheet name is often the
    only entity signal (the "Компания" metadata field is frequently blank).

This module's job stops at producing a clean, flat, per-sheet DataFrame with the
correct unit scaling applied and the sheet's entity attached. Mapping those
columns onto the eventual budget-data schema is a separate, intentionally
out-of-scope next step — nothing here renames columns to the procurement schema.

Per SPEC.md architecture rule: this module is pure — no streamlit imports.
"""

from __future__ import annotations

import re

import openpyxl
import pandas as pd

# Metadata labels searched for in the first rows, mapped to output dict keys.
_METADATA_LABELS = {
    "Год кампании": "year",
    "Версия": "version",
    "Компания": "company",
    "Валюта": "currency",
    "Ед. измерения": "unit_scale",
}

# Keywords that identify the real (top-level) budget header row.
_HEADER_KEYWORDS = (
    "Статья бюджета",
    "План",
    "Прогноз",
    "Виды затрат",
    "Контрагент",
    "Договор",
)

# Substrings (case-insensitive) marking a column as monetary and thus subject to
# unit-scale multiplication.
_MONETARY_KEYWORDS = ("план", "прогноз", "цена")

_METADATA_SCAN_ROWS = 20

# Canonical field -> candidate patterns (matched case-insensitively as regex
# against each flattened column name, patterns tried in order). Subsidiary
# workbooks (Subsidiary A, Subsidiary C, ...) disagree on column layout for the same fields, so
# this is a best-effort detector rather than a fixed positional mapping.
_CANONICAL_MAPPING: dict[str, list[str]] = {
    "fact_prior2":   ["2025 год факт", "факт 2025", "Факт 2025"],
    "plan_prior":    ["2026 год утвержденный план", "план 2026", "утвержденный план 2026", "факт 2026"],
    "plan_annual":   ["План за 2027", "план 2027", "план за.*год"],
    "plan_q1":       ["1 кв"],
    "plan_q2":       ["2 кв"],
    "plan_q3":       ["3 кв"],
    "plan_q4":       ["4 кв"],
    "description":   ["Описание"],
    "justification": ["Обоснование"],
    "forecast_y1":   ["Прогноз 2028", "2028"],
    "forecast_y2":   ["Прогноз 2029", "2029"],
    "forecast_y3":   ["Прогноз 2030", "2030"],
    "forecast_y4":   ["Прогноз 2031", "2031"],
    "counterparty":  ["Контрагент"],
    "article_code":  ["Статья бюджета", "^Статья$"],
    "expense_type":  ["Виды затрат"],
    "contract":      ["Договор"],
}


# --- small cell helpers ------------------------------------------------------
def _cell_text(value) -> str:
    """Trimmed string form of a cell value ('' for None)."""
    return "" if value is None else str(value).strip()


def _is_number(value) -> bool:
    """True for real numeric cells (openpyxl ints/floats), never for bools."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _collapse(text: str) -> str:
    """Strip and collapse internal whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def _to_int(value):
    """Parse an integer from a number or a string like '2027' / '2027 год'."""
    if _is_number(value):
        return int(value)
    match = re.search(r"-?\d+", _cell_text(value))
    return int(match.group()) if match else None


def _to_number(value):
    """Parse a number (int/float) from a value; None if not a recognizable number."""
    if _is_number(value):
        return value
    text = _cell_text(value).replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    num = float(match.group())
    return int(num) if num.is_integer() else num


def _matches_header_keyword(text: str) -> bool:
    low = text.lower()
    return any(kw.lower() in low for kw in _HEADER_KEYWORDS)


def _is_monetary(name: str) -> bool:
    low = name.lower()
    return any(kw in low for kw in _MONETARY_KEYWORDS)


# --- 1. metadata -------------------------------------------------------------
def detect_metadata_block(ws) -> dict:
    """Scan the first rows for labelled metadata fields and read their values.

    Returns a dict with keys year, version, company, currency, unit_scale.
    unit_scale defaults to 1 when "Ед. измерения" is absent or not a recognizable
    number — we never silently assume 1000.
    """
    metadata = {
        "year": None,
        "version": None,
        "company": "",
        "currency": None,
        "unit_scale": 1,
    }

    max_col = ws.max_column or 0
    max_row = min(_METADATA_SCAN_ROWS, ws.max_row or 0)

    for r in range(1, max_row + 1):
        for c in range(1, max_col + 1):
            text = _cell_text(ws.cell(row=r, column=c).value)
            if not text:
                continue
            for label, key in _METADATA_LABELS.items():
                if label.lower() not in text.lower():
                    continue
                raw = _read_label_value(ws, r, c, text)
                if key == "year":
                    metadata["year"] = _to_int(raw)
                elif key == "unit_scale":
                    parsed = _to_number(raw)
                    metadata["unit_scale"] = parsed if parsed is not None else 1
                elif key == "company":
                    metadata["company"] = _cell_text(raw)
                else:
                    value = _cell_text(raw)
                    metadata[key] = value or None

    return metadata


def _read_label_value(ws, row: int, col: int, label_text: str):
    """Read the value for a metadata label: after a ':' in-cell, else to the right."""
    # "Год кампании: 2027" style — value lives in the same cell after the colon.
    if ":" in label_text:
        after = label_text.split(":", 1)[1].strip()
        if after:
            return after
    # Otherwise take the first non-empty cell to the right on the same row.
    max_col = ws.max_column or 0
    for c in range(col + 1, max_col + 1):
        value = ws.cell(row=row, column=c).value
        if _cell_text(value):
            return value
    return None


# --- 2. header detection -----------------------------------------------------
def find_header_row(ws, search_range: int = 30):
    """Locate the real header row and whether a nested sub-header sits below it.

    The header is the scanned row with the most budget-keyword hits (falling back
    to the fullest row if nothing matches). A nested child header is detected when
    the row directly below is a *partial* row of non-numeric, non-keyword labels
    (headers carry text, real data rows carry numbers).

    Returns (header_row, has_nested_header). NOTE: this returns a tuple rather
    than a bare int as the brief's `-> int` hint suggested, because the caller
    (extract_sheet / flatten_headers) needs the nested-header flag this function
    is specified to detect.
    """
    max_col = ws.max_column or 0
    limit = min(search_range, ws.max_row or 0)

    best_row, best_count = 1, -1
    fallback_row, fallback_nonempty = 1, -1

    for r in range(1, limit + 1):
        keyword_hits = 0
        nonempty = 0
        for c in range(1, max_col + 1):
            text = _cell_text(ws.cell(row=r, column=c).value)
            if text:
                nonempty += 1
                if _matches_header_keyword(text):
                    keyword_hits += 1
        if keyword_hits > best_count:
            best_count, best_row = keyword_hits, r
        if nonempty > fallback_nonempty:
            fallback_nonempty, fallback_row = nonempty, r

    header_row = best_row if best_count > 0 else fallback_row
    has_nested = _detect_nested_header(ws, header_row, max_col)
    return header_row, has_nested


def _detect_nested_header(ws, header_row: int, max_col: int) -> bool:
    """True if the row below the header looks like a child/sub-label row."""
    below = header_row + 1
    if below > (ws.max_row or 0):
        return False

    header_nonempty = 0
    below_nonempty = 0
    numeric_below = 0
    sublabel_cells = 0

    for c in range(1, max_col + 1):
        if _cell_text(ws.cell(row=header_row, column=c).value):
            header_nonempty += 1
        below_value = ws.cell(row=below, column=c).value
        below_text = _cell_text(below_value)
        if below_text:
            below_nonempty += 1
            if _is_number(below_value):
                numeric_below += 1
            elif not _matches_header_keyword(below_text):
                sublabel_cells += 1

    # A sub-header: has text sub-labels, carries no numbers (data rows would),
    # and is only partially filled compared with the parent header row.
    return (
        sublabel_cells > 0
        and numeric_below == 0
        and 0 < below_nonempty < header_nonempty
    )


# --- 3. header flattening ----------------------------------------------------
def flatten_headers(ws, header_row: int, has_nested_header: bool) -> list[str]:
    """Build one clean column name per column: 'parent | child' where a child exists."""
    max_col = ws.max_column or 0
    names: list[str] = []
    seen: dict[str, int] = {}

    for c in range(1, max_col + 1):
        parent = _collapse(_cell_text(ws.cell(row=header_row, column=c).value))
        child = ""
        if has_nested_header:
            child = _collapse(_cell_text(ws.cell(row=header_row + 1, column=c).value))
            # A cell that itself matches a top-level keyword is not a sub-label.
            if child and _matches_header_keyword(child):
                child = ""

        if parent and child:
            name = f"{parent} | {child}"
        elif parent:
            name = parent
        elif child:
            name = child
        else:
            name = f"column_{c}"

        # De-duplicate so DataFrame columns stay unique.
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        names.append(name)

    return names


# --- 4. canonical column detection --------------------------------------------
def _find_matching_column(columns: list[str], patterns: list[str]) -> str | None:
    """First existing column matching any pattern, trying patterns in priority order."""
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for col in columns:
            if regex.search(col):
                return col
    return None


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical field aliases for whichever budget columns this sheet has.

    Different subsidiary workbooks put "current year plan", "prior year plan",
    quarterly breakdown, etc. in different columns under different Russian
    labels. This detects whichever of `_CANONICAL_MAPPING`'s fields are present
    and copies them under a stable name (e.g. df["plan_annual"]). Original
    columns are left untouched — this only adds aliases, so it's safe to call
    on any sheet regardless of which fields it has.
    """
    out = df.copy()
    columns = list(df.columns)
    for canonical_key, patterns in _CANONICAL_MAPPING.items():
        match = _find_matching_column(columns, patterns)
        if match is not None:
            out[canonical_key] = df[match]
    return out


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta_abs / delta_pct / delta_flag from the canonical plan columns.

    Requires canonicalize() to have run first. No-op (returns df unchanged) if
    plan_annual wasn't detected — there's nothing to compute a delta against.
    """
    out = df.copy()
    if "plan_annual" not in out.columns:
        return out

    plan_annual = pd.to_numeric(out["plan_annual"], errors="coerce")

    if "plan_prior" not in out.columns:
        out["delta_abs"] = None
        out["delta_pct"] = None
        out["delta_flag"] = "нет данных прошлого года"
        return out

    plan_prior = pd.to_numeric(out["plan_prior"], errors="coerce")

    delta_pct = pd.Series(index=out.index, dtype="float64")
    delta_flag = pd.Series("", index=out.index, dtype=object)

    zero_prior = plan_prior == 0
    has_prior = ~zero_prior

    delta_pct[has_prior] = (
        (plan_annual[has_prior] - plan_prior[has_prior]) / plan_prior[has_prior] * 100
    )

    new_item = zero_prior & (plan_annual > 0)
    delta_pct[new_item] = float("inf")
    delta_flag[new_item] = "новая статья"

    both_zero = zero_prior & (plan_annual == 0)
    delta_pct[both_zero] = 0

    out["delta_abs"] = plan_annual - plan_prior
    out["delta_pct"] = delta_pct
    out["delta_flag"] = delta_flag
    return out


# --- 5. per-sheet extraction -------------------------------------------------
def extract_sheet(ws) -> dict:
    """Turn one worksheet into a clean flat DataFrame with scaling + entity applied."""
    metadata = detect_metadata_block(ws)
    header_row, has_nested = find_header_row(ws)
    columns = flatten_headers(ws, header_row, has_nested)

    ncols = len(columns)
    unit_scale = metadata["unit_scale"]
    monetary_idx = [i for i, name in enumerate(columns) if _is_monetary(name)]
    monetary_names = [columns[i] for i in monetary_idx]

    data_start = header_row + (2 if has_nested else 1)
    max_row = ws.max_row or 0

    rows: list[list] = []
    for r in range(data_start, max_row + 1):
        values = [ws.cell(row=r, column=c).value for c in range(1, ncols + 1)]
        # Skip fully blank separator rows.
        if all(not _cell_text(v) for v in values):
            continue
        # Scale monetary cells — only actual numbers, never text.
        if unit_scale != 1:
            for i in monetary_idx:
                if _is_number(values[i]):
                    values[i] = values[i] * unit_scale
        rows.append(values)

    df = pd.DataFrame(rows, columns=columns)

    # Entity: sheet name when "Компания" is blank, otherwise the company field.
    company = _cell_text(metadata["company"])
    df["entity"] = company if company else ws.title

    # Loud about the transformation so scaling is never silent.
    if unit_scale != 1 and monetary_names:
        print(
            f"[corporate_ingest] sheet '{ws.title}': scaled {len(monetary_names)} "
            f"monetary column(s) by x{unit_scale}: {monetary_names}"
        )

    df = canonicalize(df)
    df = compute_deltas(df)

    plan_annual_source = _find_matching_column(columns, _CANONICAL_MAPPING["plan_annual"])
    budget_year = _to_int(plan_annual_source) if plan_annual_source else None

    return {
        "data": df,
        "metadata": metadata,
        "sheet_name": ws.title,
        "has_prior_year": "plan_prior" in df.columns,
        "budget_year": budget_year,
        "deltas_computed": "delta_flag" in df.columns,
    }


# --- 6. whole-workbook extraction --------------------------------------------
def extract_workbook(file_path: str) -> list[dict]:
    """Extract every sheet of a workbook; no assumptions on sheet count or names."""
    wb = openpyxl.load_workbook(file_path, data_only=True)
    try:
        return [extract_sheet(ws) for ws in wb.worksheets]
    finally:
        wb.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m core.corporate_ingest <path_to_xlsx>")
        sys.exit(1)

    path = sys.argv[1]
    for result in extract_workbook(path):
        df = result["data"]
        print("=" * 70)
        print(f"Sheet:    {result['sheet_name']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Columns:  {list(df.columns)}")
        print(f"Rows:     {len(df)}")
        print("First 3 rows:")
        print(df.head(3).to_string(index=False))
        print()
