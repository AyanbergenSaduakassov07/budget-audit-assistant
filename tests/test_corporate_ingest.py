"""Tests for core.corporate_ingest — building messy budget workbooks in memory.

No API cost: everything here is pure openpyxl + pandas. Absolute imports from the
project root (run with `uv run pytest`).
"""

import pytest
from openpyxl import Workbook

from core.corporate_ingest import (
    detect_metadata_block,
    find_header_row,
    flatten_headers,
    canonicalize,
    compute_deltas,
    extract_sheet,
    extract_workbook,
)


def _build_opex(ws) -> None:
    """OPEX-style sheet: blank company, unit scale 1000, two-row nested header.

    Includes a blank separator row between data rows.
    """
    ws.title = "OPEX"
    ws["A1"], ws["B1"] = "Год кампании", 2027
    ws["A2"], ws["B2"] = "Версия", "Основной план"
    ws["A3"], ws["B3"] = "Компания", ""        # blank -> entity from sheet name
    ws["A4"], ws["B4"] = "Валюта", "KZT"
    ws["A5"], ws["B5"] = "Ед. измерения", 1000  # thousands

    # Parent header on row 8, child sub-labels on row 9 (only for План/Прогноз).
    ws["A8"], ws["B8"], ws["C8"], ws["D8"], ws["E8"] = (
        "Статья бюджета", "Виды затрат", "План", "Прогноз", "Контрагент",
    )
    ws["C9"], ws["D9"] = "год", "год"

    # Data rows 10 and 12, with a blank separator row 11.
    ws["A10"], ws["B10"], ws["C10"], ws["D10"], ws["E10"] = (
        "Аренда", "Прочее", 500, 550, "ТОО Ромашка",
    )
    ws["A12"], ws["B12"], ws["C12"], ws["D12"], ws["E12"] = (
        "Связь", "Услуги", 120, 130, "Телеком Плюс",
    )


def _build_sub_a(ws) -> None:
    """Subsidiary A-style sheet: prior-year plan + current-year plan + quarterly + notes.

    Single-row header. Three planted data rows exercise the three compute_deltas
    branches: normal delta, "новая статья" (prior 0, annual > 0), both zero.
    """
    ws.title = "SUB_A"
    ws["A1"], ws["B1"] = "Год кампании", 2027
    ws["A2"], ws["B2"] = "Ед. измерения", 1

    ws["A6"] = "Статья бюджета"
    ws["B6"] = "2025 год факт"
    ws["C6"] = "2026 год утвержденный план"
    ws["D6"] = "План за 2027 год"
    ws["E6"], ws["F6"], ws["G6"], ws["H6"] = "1 кв", "2 кв", "3 кв", "4 кв"
    ws["I6"], ws["J6"] = "Описание", "Обоснование"

    ws["A7"] = "Лицензии ПО"
    ws["B7"], ws["C7"], ws["D7"] = 700, 1000, 1200
    ws["E7"], ws["F7"], ws["G7"], ws["H7"] = 300, 300, 300, 300
    ws["I7"], ws["J7"] = "SaaS-подписки", "Продление контракта"

    ws["A8"] = "Новый проект"
    ws["B8"], ws["C8"], ws["D8"] = 0, 0, 500
    ws["E8"], ws["F8"], ws["G8"], ws["H8"] = 0, 0, 0, 500
    ws["I8"], ws["J8"] = "Миграция ЦОД", "Новая инициатива"

    ws["A9"] = "Закрытая статья"
    ws["B9"], ws["C9"], ws["D9"] = 0, 0, 0
    ws["E9"], ws["F9"], ws["G9"], ws["H9"] = 0, 0, 0, 0
    ws["I9"], ws["J9"] = "—", "—"


def _build_sub_c(ws) -> None:
    """Subsidiary C-style sheet: only a current-year plan column, no prior-year column."""
    ws.title = "Subsidiary C"
    ws["A1"], ws["B1"] = "Год кампании", 2027
    ws["A2"], ws["B2"] = "Ед. измерения", 1

    ws["A6"] = "Статья бюджета"
    ws["B6"] = "План за 2027 год"
    ws["C6"], ws["D6"], ws["E6"], ws["F6"] = "1 кв", "2 кв", "3 кв", "4 кв"

    ws["A7"] = "Хостинг"
    ws["B7"] = 900
    ws["C7"], ws["D7"], ws["E7"], ws["F7"] = 225, 225, 225, 225


def _build_capex(ws) -> None:
    """CAPEX-style sheet: filled company, non-numeric unit, single-row header."""
    ws.title = "CAPEX"
    ws["A1"], ws["B1"] = "Год кампании:", 2027          # colon-style label
    ws["A2"], ws["C2"] = "Компания", "Павлодарский НПЗ"  # value two cells right
    ws["A3"], ws["B3"] = "Ед. измерения", "тыс."         # not a number -> scale 1

    ws["A6"], ws["B6"], ws["C6"] = "Статья бюджета", "План", "Договор"
    ws["A7"], ws["B7"], ws["C7"] = "Оборудование", 900, "Д-100"
    ws["A8"], ws["B8"], ws["C8"] = "Монтаж", 300, "Д-101"


@pytest.fixture
def opex_ws():
    wb = Workbook()
    _build_opex(wb.active)
    return wb.active


@pytest.fixture
def capex_ws():
    wb = Workbook()
    _build_capex(wb.active)
    return wb.active


@pytest.fixture
def sub_a_ws():
    wb = Workbook()
    _build_sub_a(wb.active)
    return wb.active


@pytest.fixture
def sub_c_ws():
    wb = Workbook()
    _build_sub_c(wb.active)
    return wb.active


# --- detect_metadata_block ---------------------------------------------------
def test_metadata_reads_all_fields(opex_ws):
    meta = detect_metadata_block(opex_ws)
    assert meta["year"] == 2027
    assert meta["version"] == "Основной план"
    assert meta["company"] == ""          # blank stays an empty string
    assert meta["currency"] == "KZT"
    assert meta["unit_scale"] == 1000


def test_metadata_colon_and_offset_value(capex_ws):
    meta = detect_metadata_block(capex_ws)
    assert meta["year"] == 2027                      # from "Год кампании:"
    assert meta["company"] == "Павлодарский НПЗ"     # value two columns to the right


def test_unit_scale_defaults_to_one_when_not_a_number(capex_ws):
    # "Ед. измерения" = "тыс." is not a recognizable number -> must NOT assume 1000.
    assert detect_metadata_block(capex_ws)["unit_scale"] == 1


# --- find_header_row ---------------------------------------------------------
def test_find_header_row_with_nested(opex_ws):
    header_row, has_nested = find_header_row(opex_ws)
    assert header_row == 8
    assert has_nested is True


def test_find_header_row_single(capex_ws):
    header_row, has_nested = find_header_row(capex_ws)
    assert header_row == 6
    assert has_nested is False


# --- flatten_headers ---------------------------------------------------------
def test_flatten_headers_combines_nested(opex_ws):
    header_row, has_nested = find_header_row(opex_ws)
    cols = flatten_headers(opex_ws, header_row, has_nested)
    assert cols == [
        "Статья бюджета",
        "Виды затрат",
        "План | год",
        "Прогноз | год",
        "Контрагент",
    ]


def test_flatten_headers_single_row(capex_ws):
    header_row, has_nested = find_header_row(capex_ws)
    cols = flatten_headers(capex_ws, header_row, has_nested)
    assert cols == ["Статья бюджета", "План", "Договор"]


# --- extract_sheet -----------------------------------------------------------
def test_extract_sheet_scales_only_monetary_numbers(opex_ws):
    result = extract_sheet(opex_ws)
    df = result["data"]

    # Two data rows (blank separator dropped).
    assert len(df) == 2
    # Monetary columns scaled by 1000.
    assert df["План | год"].tolist() == [500000, 120000]
    assert df["Прогноз | год"].tolist() == [550000, 130000]
    # Non-monetary text left untouched.
    assert df["Контрагент"].tolist() == ["ТОО Ромашка", "Телеком Плюс"]


def test_extract_sheet_entity_from_sheet_name_when_company_blank(opex_ws):
    df = extract_sheet(opex_ws)["data"]
    assert (df["entity"] == "OPEX").all()


def test_extract_sheet_entity_from_company_when_present(capex_ws):
    result = extract_sheet(capex_ws)
    df = result["data"]
    assert (df["entity"] == "Павлодарский НПЗ").all()
    # unit_scale 1 -> no scaling applied.
    assert df["План"].tolist() == [900, 300]


def test_extract_sheet_return_shape(opex_ws):
    result = extract_sheet(opex_ws)
    assert set(result) == {
        "data", "metadata", "sheet_name",
        "has_prior_year", "budget_year", "deltas_computed",
    }
    assert result["sheet_name"] == "OPEX"
    assert result["metadata"]["unit_scale"] == 1000
    # OPEX/CAPEX fixtures carry none of the Subsidiary A/Subsidiary C canonical plan columns.
    assert result["has_prior_year"] is False
    assert result["budget_year"] is None
    assert result["deltas_computed"] is False


# --- extract_workbook --------------------------------------------------------
def test_extract_workbook_all_sheets(tmp_path):
    wb = Workbook()
    _build_opex(wb.active)
    _build_capex(wb.create_sheet("CAPEX"))
    path = tmp_path / "budget.xlsx"
    wb.save(path)

    results = extract_workbook(str(path))
    assert [r["sheet_name"] for r in results] == ["OPEX", "CAPEX"]
    assert results[0]["metadata"]["unit_scale"] == 1000
    assert results[1]["metadata"]["company"] == "Павлодарский НПЗ"


# --- canonicalize --------------------------------------------------------------
def test_canonicalize_detects_sub_a_columns(sub_a_ws):
    df = extract_sheet(sub_a_ws)["data"]
    for key in ("plan_prior", "plan_annual", "fact_prior2", "description", "justification"):
        assert key in df.columns
    assert df["plan_prior"].tolist() == [1000, 0, 0]
    assert df["plan_annual"].tolist() == [1200, 500, 0]
    assert df["description"].tolist() == ["SaaS-подписки", "Миграция ЦОД", "—"]


def test_canonicalize_leaves_original_columns_intact(sub_a_ws):
    df = extract_sheet(sub_a_ws)["data"]
    # Original Russian column names must still be present and unchanged.
    assert df["2026 год утвержденный план"].tolist() == [1000, 0, 0]
    assert df["План за 2027 год"].tolist() == [1200, 500, 0]


def test_canonicalize_noop_when_no_known_columns(opex_ws):
    df = extract_sheet(opex_ws)["data"]
    assert "plan_annual" not in df.columns
    assert "plan_prior" not in df.columns


# --- compute_deltas -------------------------------------------------------------
def test_compute_deltas_normal_case(sub_a_ws):
    df = extract_sheet(sub_a_ws)["data"]
    assert df["delta_abs"].tolist()[0] == 200
    assert df["delta_pct"].tolist()[0] == 20.0
    assert df["delta_flag"].tolist()[0] == ""


def test_compute_deltas_new_item_flagged(sub_a_ws):
    df = extract_sheet(sub_a_ws)["data"]
    row = df.iloc[1]
    assert row["delta_abs"] == 500
    assert row["delta_pct"] == float("inf")
    assert row["delta_flag"] == "новая статья"


def test_compute_deltas_both_zero(sub_a_ws):
    df = extract_sheet(sub_a_ws)["data"]
    row = df.iloc[2]
    assert row["delta_abs"] == 0
    assert row["delta_pct"] == 0
    assert row["delta_flag"] == ""


def test_compute_deltas_no_prior_year_column(sub_c_ws):
    df = extract_sheet(sub_c_ws)["data"]
    assert df["delta_abs"].iloc[0] is None
    assert df["delta_pct"].iloc[0] is None
    assert df["delta_flag"].iloc[0] == "нет данных прошлого года"


def test_compute_deltas_returns_unchanged_df_without_plan_annual(opex_ws):
    df = extract_sheet(opex_ws)["data"]
    result = compute_deltas(df)
    assert "delta_abs" not in result.columns
    assert "delta_flag" not in result.columns


# --- extract_sheet: has_prior_year / budget_year / deltas_computed -------------
def test_extract_sheet_sub_a_flags(sub_a_ws):
    result = extract_sheet(sub_a_ws)
    assert result["has_prior_year"] is True
    assert result["budget_year"] == 2027
    assert result["deltas_computed"] is True


def test_extract_sheet_sub_c_flags(sub_c_ws):
    result = extract_sheet(sub_c_ws)
    assert result["has_prior_year"] is False
    assert result["budget_year"] == 2027
    assert result["deltas_computed"] is True
