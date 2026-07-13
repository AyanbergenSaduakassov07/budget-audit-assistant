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
    assert set(result) == {"data", "metadata", "sheet_name"}
    assert result["sheet_name"] == "OPEX"
    assert result["metadata"]["unit_scale"] == 1000


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
