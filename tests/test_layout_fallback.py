"""Tests for core.budget_auditor.layout_fallback and its integration into
parser.parse_workbook().

Split in two halves:
  - Unit tests for layout_fallback's own helpers (sample extraction, mapping
    conversion, validation, the never-raises LLM call), all with a fake
    provider -- no live API calls.
  - Integration tests through parse_workbook() itself: a synthetic
    genuinely-different-layout workbook the deterministic pass can't
    handle, confirming the fallback (mocked) produces a correct mapping and
    that flag_rows()/aggregate_workbook() work UNCHANGED on top of it; the
    "fallback also fails" path returns the original clear error rather than
    crashing; and existing recognized-layout fixtures never invoke the
    fallback at all.
"""

import json

import openpyxl

from core.budget_auditor.layout_fallback import (
    _extract_sample,
    _mapping_to_layout,
    _validate_layout,
    detect_layout_via_llm,
)
from core.budget_auditor.parser import aggregate_workbook, flag_rows, parse_workbook
from tests.test_budget_auditor import _full_form_workbook, _quarterly_workbook, _write_meta


# --- fixture: a layout the deterministic parser genuinely can't handle -----------------
def _alternate_layout_workbook(path):
    """No "Статья бюджета" anywhere (column C says "Код статьи" instead) --
    _find_header_row's literal-signature match fails completely, and every
    other header is renamed/reordered relative to every other fixture in
    this test suite. Real data, just a layout the template rules don't
    recognize at all."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws, campaign_year=2027)

    hdr = 8
    headers = {
        3: "Код статьи", 4: "Наименование статьи",
        5: "Категория расхода", 6: "Подкатегория расхода",
        7: "Поставщик", 8: "Обоснование расхода",
        9: "План 2027, тыс.тг", 10: "Факт 2025, тыс.тг",
    }
    for c, t in headers.items():
        ws.cell(row=hdr, column=c, value=t)

    # Code-declaration row (code + name only), THEN a separate leaf row
    # carrying category/subcategory/counterparty/justification/money --
    # the same two-row convention every other fixture in this suite uses
    # (see _full_form_workbook). This is a row-STRUCTURE convention the
    # shared parse_sheet loop assumes regardless of column layout, distinct
    # from the column-layout question this fallback actually addresses.
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-OP-01")
    ws.cell(row=r, column=4, value="Связь")
    r += 1
    ws.cell(row=r, column=5, value="Услуги связи")
    ws.cell(row=r, column=6, value="Сотовая связь")
    ws.cell(row=r, column=7, value="Телеком Плюс")
    ws.cell(row=r, column=8, value="Рост тарифов")
    ws.cell(row=r, column=9, value=2000)
    ws.cell(row=r, column=10, value=1000)

    r += 1
    ws.cell(row=r, column=3, value="IT-OP-02")
    ws.cell(row=r, column=4, value="Лицензии")
    r += 1
    ws.cell(row=r, column=5, value="ПО")
    ws.cell(row=r, column=6, value="Антивирус")
    ws.cell(row=r, column=7, value="Kaspersky")
    ws.cell(row=r, column=8, value="Продление лицензии")
    ws.cell(row=r, column=9, value=500)
    ws.cell(row=r, column=10, value=480)

    wb.save(path)
    return path


_VALID_MAPPING_RESPONSE = json.dumps({
    "header_row": 8,
    "columns": {
        "C": "statya_code", "D": "statya_name",
        "E": "category", "F": "subcategory",
        "G": "counterparty", "H": "justification",
        "I": "plan_2027", "J": "fact_2025",
    },
}, ensure_ascii=False)


# --- _extract_sample ---------------------------------------------------------------------
def test_extract_sample_is_compact_and_skips_blanks(tmp_path):
    path = _alternate_layout_workbook(tmp_path / "alt.xlsx")
    ws = openpyxl.load_workbook(path)["OPEX"]

    sample = _extract_sample(ws)

    assert "8C='Код статьи'" in sample
    assert "9C='IT-OP-01'" in sample
    assert "10I=2000" in sample
    # Blank cells (most of the grid) must never appear as e.g. "1A=None".
    assert "None" not in sample


# --- _mapping_to_layout -------------------------------------------------------------------
def test_mapping_to_layout_converts_letters_roles_and_quarters():
    colmap, year_cols = _mapping_to_layout({
        "C": "statya_code", "D": "statya_name", "I": "plan_2027", "K": "plan_2027_q1",
    })
    assert colmap == {"statya_code": 3, "statya_name": 4}
    assert year_cols[9] == ("plan", 2027)
    assert year_cols[11] == ("quarter", 2027, 1)


def test_mapping_to_layout_ignores_unknown_role_and_bad_letter():
    colmap, year_cols = _mapping_to_layout({
        "C": "statya_code", "Z": "not_a_real_role", "!!": "statya_name",
    })
    assert colmap == {"statya_code": 3}
    assert year_cols == {}


def test_mapping_to_layout_excludes_branch_and_cost_type():
    """branch/cost_type are sheet-level (derived from the sheet name in
    aggregate_workbook), never a real per-row column here -- see the
    module docstring. A model that tries to map them anyway is ignored,
    not trusted."""
    colmap, _ = _mapping_to_layout({"A": "branch", "B": "cost_type", "C": "statya_code"})
    assert colmap == {"statya_code": 3}


def test_mapping_to_layout_handles_non_dict_input():
    assert _mapping_to_layout(None) == ({}, {})
    assert _mapping_to_layout("not a dict") == ({}, {})


# --- _validate_layout ----------------------------------------------------------------------
def test_validate_layout_requires_both_money_and_identifier():
    assert _validate_layout({"statya_code": 3}, {9: ("plan", 2027)}) is True
    assert _validate_layout({"statya_code": 3}, {}) is False  # no money column
    assert _validate_layout({"counterparty": 7}, {9: ("plan", 2027)}) is False  # no identifier
    assert _validate_layout({}, {}) is False


# --- detect_layout_via_llm: never-raises + validation ---------------------------------------
def test_detect_layout_via_llm_returns_none_without_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)
    path = _alternate_layout_workbook(tmp_path / "alt.xlsx")
    ws = openpyxl.load_workbook(path)["OPEX"]
    assert detect_layout_via_llm(ws) is None


def test_detect_layout_via_llm_success_path(tmp_path, monkeypatch):
    path = _alternate_layout_workbook(tmp_path / "alt.xlsx")
    ws = openpyxl.load_workbook(path)["OPEX"]

    monkeypatch.setattr("core.budget_auditor.layout_fallback._get_client", lambda: object())
    monkeypatch.setattr(
        "core.budget_auditor.layout_fallback._generate_content",
        lambda client, prompt: _VALID_MAPPING_RESPONSE,
    )

    result = detect_layout_via_llm(ws)
    assert result is not None
    header_row, colmap, year_cols = result
    assert header_row == 8
    assert colmap["statya_code"] == 3
    assert colmap["counterparty"] == 7
    assert year_cols[9] == ("plan", 2027)
    assert year_cols[10] == ("fact", 2025)


def test_detect_layout_via_llm_invalid_mapping_returns_none(tmp_path, monkeypatch):
    """A mapping with a header_row but no money column at all must be
    discarded, not trusted just because the model was confident."""
    path = _alternate_layout_workbook(tmp_path / "alt.xlsx")
    ws = openpyxl.load_workbook(path)["OPEX"]

    monkeypatch.setattr("core.budget_auditor.layout_fallback._get_client", lambda: object())
    monkeypatch.setattr(
        "core.budget_auditor.layout_fallback._generate_content",
        lambda client, prompt: json.dumps({"header_row": 8, "columns": {"C": "statya_code"}}),
    )
    assert detect_layout_via_llm(ws) is None


def test_detect_layout_via_llm_malformed_json_returns_none(tmp_path, monkeypatch):
    path = _alternate_layout_workbook(tmp_path / "alt.xlsx")
    ws = openpyxl.load_workbook(path)["OPEX"]

    monkeypatch.setattr("core.budget_auditor.layout_fallback._get_client", lambda: object())
    monkeypatch.setattr(
        "core.budget_auditor.layout_fallback._generate_content",
        lambda client, prompt: "this is not JSON at all",
    )
    assert detect_layout_via_llm(ws) is None


def test_detect_layout_via_llm_client_error_returns_none_not_raises(tmp_path, monkeypatch):
    path = _alternate_layout_workbook(tmp_path / "alt.xlsx")
    ws = openpyxl.load_workbook(path)["OPEX"]

    def _raise():
        raise RuntimeError("boom")

    monkeypatch.setattr("core.budget_auditor.layout_fallback._get_client", _raise)
    assert detect_layout_via_llm(ws) is None


# --- integration: parse_workbook() end to end -----------------------------------------------
def test_parse_workbook_uses_llm_fallback_for_unrecognized_layout(tmp_path, monkeypatch):
    """The synthetic alternate-layout workbook -> deterministic pass finds no
    header at all -> LLM fallback (mocked) provides a valid mapping ->
    parse_sheet uses it, and flag_rows()/aggregate_workbook() -- the
    existing deterministic functions, unmodified -- work on top of the
    result exactly as they would for any template-recognized file."""
    path = _alternate_layout_workbook(tmp_path / "alt.xlsx")

    monkeypatch.setattr("core.budget_auditor.layout_fallback._get_client", lambda: object())
    monkeypatch.setattr(
        "core.budget_auditor.layout_fallback._generate_content",
        lambda client, prompt: _VALID_MAPPING_RESPONSE,
    )

    meta, sheets = parse_workbook(path)
    df, stats = sheets["OPEX"]["df"], sheets["OPEX"]["stats"]

    assert stats["parsed_via"] == "llm_fallback"
    assert stats["n_leaf_rows"] == 2
    assert stats["has_justification"] is True
    assert df is not None and len(df) == 2
    assert set(df["statya_code"]) == {"IT-OP-01", "IT-OP-02"}
    assert set(df["counterparty"]) == {"Телеком Плюс", "Kaspersky"}
    assert df["plan_2027"].tolist() == [2000, 500]
    assert df["fact_2025"].tolist() == [1000, 480]

    # The existing deterministic functions, completely unmodified, must work
    # on this result exactly as for any template-recognized file.
    flagged_df = flag_rows(df, stats, threshold_pct=15.0)
    assert "delta_pct" in flagged_df.columns
    combined = aggregate_workbook({"OPEX": {"df": flagged_df, "stats": stats}})
    assert combined["cost_type"].unique().tolist() == ["OPEX"]
    assert combined["plan_2027"].sum() == 2500


def test_parse_workbook_fallback_failure_returns_original_error(tmp_path, monkeypatch):
    """When the deterministic pass fails AND the LLM fallback also fails
    (here: no provider configured), the workbook must still return the
    exact same "no_header_found" error as before this feature existed --
    never a crash, never a new/different failure mode."""
    path = _alternate_layout_workbook(tmp_path / "alt.xlsx")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)

    meta, sheets = parse_workbook(path)
    df, stats = sheets["OPEX"]["df"], sheets["OPEX"]["stats"]

    assert df is None
    assert stats == {"sheet": "OPEX", "error": "no_header_found"}


def test_parse_workbook_never_invokes_fallback_for_recognized_layouts(tmp_path, monkeypatch):
    """Every currently-passing file must go through the deterministic path
    completely unchanged -- the fallback must never even be CALLED for a
    layout already handled, let alone change its result. A raised exception
    from the mocked client would just be swallowed by
    detect_layout_via_llm's own never-raises try/except, so this counts
    invocations directly rather than relying on the raise to fail loudly."""
    calls = {"n": 0}

    def _track_get_client():
        calls["n"] += 1
        raise RuntimeError("LLM layout fallback must not be invoked for a recognized layout")

    monkeypatch.setattr("core.budget_auditor.layout_fallback._get_client", _track_get_client)

    for builder in (_full_form_workbook, _quarterly_workbook):
        path = builder(tmp_path / f"{builder.__name__}.xlsx")
        meta, sheets = parse_workbook(path)
        for sn, d in sheets.items():
            if d["df"] is not None and not d["df"].empty:
                assert d["stats"]["parsed_via"] == "deterministic"

    assert calls["n"] == 0
