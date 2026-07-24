"""Regression tests for four issues found via direct cell-level inspection of
a real ACM file (a sample budget form):

  1. budget codes that don't match the "AA-AA-00" format must be flagged
     (anomalous_code_flag), not silently forward-filled as if normal.
  2. a footer/grand-total row must be excluded even when the "Итог"/
     "Результат" marker lands in a column other than the code column.
  3. quarters reported нарастающим итогом (cumulative) must not be flagged
     as a critical mismatch when summed against the annual plan.
  4. a counterparty cell naming more than one legal entity (compound cell)
     must not be attributed to a single supplier's concentration risk.

Each fixture reproduces the exact pattern, not just a generic variant.
"""

import openpyxl
import pandas as pd
import pytest

from core.budget_auditor.parser import parse_workbook
from core.budget_auditor.analysis import workbook_summary_text
from core.budget_auditor.narrator_bridge import build_audit_result
from core.budget_audit import concentration_risk, quarterly_reconciliation

from tests.test_budget_auditor import _write_meta, _write_headers, _run_pipeline


# --- 1. anomalous budget code -----------------------------------------------------
def _anomalous_code_workbook(path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws, campaign_year=2026)
    hdr = 9
    _write_headers(ws, hdr, {9: "Контрагент", 11: "План за 2026 год"})
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-CA-01"); ws.cell(row=r, column=4, value="Связь")
    r += 1
    ws.cell(row=r, column=7, value="Услуги связи"); ws.cell(row=r, column=8, value="Интернет")
    ws.cell(row=r, column=9, value="Телеком Плюс"); ws.cell(row=r, column=11, value=1000)
    r += 1  # non-conforming code: missing the leading zero on the sequence number
    ws.cell(row=r, column=3, value="IT-CA-1"); ws.cell(row=r, column=4, value="Оборудование")
    r += 1
    ws.cell(row=r, column=7, value="Оборудование"); ws.cell(row=r, column=8, value="Серверы")
    ws.cell(row=r, column=9, value="Dell"); ws.cell(row=r, column=11, value=2000)
    ws.cell(row=r + 1, column=3, value="Итого"); ws.cell(row=r + 1, column=11, value=3000)
    wb.save(path)
    return path


def test_anomalous_code_flag_marks_non_conforming_codes(tmp_path):
    meta, sheets = parse_workbook(_anomalous_code_workbook(tmp_path / "anomalous.xlsx"))
    df, stats = sheets["OPEX"]["df"], sheets["OPEX"]["stats"]

    assert df["anomalous_code_flag"].tolist() == [False, True]
    assert stats["n_anomalous_code_rows"] == 1

    text = workbook_summary_text(meta, sheets)
    assert "не соответствующий ожидаемому формату" in text
    assert "IT-CA-01" in text


# --- 2. footer marker outside the code column -------------------------------------
def _footer_marker_in_category_column_workbook(path):
    """Real file shape: totals land in a row whose CATEGORY column (not the
    code column) reads "Результат", one row above an "Общий результат" label
    row with no numbers at all."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws, campaign_year=2026)
    hdr = 9
    _write_headers(ws, hdr, {9: "Контрагент", 11: "План за 2026 год"})
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-CA-01"); ws.cell(row=r, column=4, value="Связь")
    r += 1
    ws.cell(row=r, column=7, value="Услуги связи"); ws.cell(row=r, column=8, value="Интернет")
    ws.cell(row=r, column=9, value="Телеком Плюс"); ws.cell(row=r, column=11, value=5000)
    r += 1
    ws.cell(row=r, column=7, value="Услуги связи"); ws.cell(row=r, column=8, value="Сотовая связь")
    ws.cell(row=r, column=9, value="Телеком Плюс"); ws.cell(row=r, column=11, value=2000)
    r += 1  # totals row: "Результат" in the category column, code column blank
    ws.cell(row=r, column=7, value="Результат"); ws.cell(row=r, column=11, value=7000)
    r += 1  # label row one below, no numbers
    ws.cell(row=r, column=3, value="Общий результат")
    wb.save(path)
    return path


def test_footer_row_excluded_even_when_marker_outside_code_column(tmp_path):
    path = _footer_marker_in_category_column_workbook(tmp_path / "footer.xlsx")
    _, sheets = parse_workbook(path)
    df = sheets["OPEX"]["df"]

    # Before the fix: the "Результат" row (category col, not code col) slipped
    # through as a leaf line item -- 5000 + 2000 + 7000 (phantom) = 14000.
    assert len(df) == 2
    assert pd.to_numeric(df["plan_2026"], errors="coerce").sum() == 7000


# --- 3. cumulative (нарастающим итогом) quarters -----------------------------------
def _cumulative_quarters_workbook(path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws)  # campaign year 2026
    hdr = 9
    _write_headers(ws, hdr, {
        9: "Контрагент", 10: "Обоснование", 11: "План за 2026 год",
        12: "1 кв 2026", 13: "2 кв 2026", 14: "3 кв 2026", 15: "4 кв 2026",
    })
    rows = [
        # cumulative: Q4 ~= annual, monotonically non-decreasing -- discrete
        # sum (24000+50000+76000+100000=250000) vs annual 100000 would be a
        # 150% "critical" false positive under a naive sum-and-compare.
        ("Нарастающим", 100_000, 24_000, 50_000, 76_000, 100_000),
        # genuinely discrete and 6% off -- must still be flagged critical.
        ("Критично дискретно", 100_000, 23_500, 23_500, 23_500, 23_500),
    ]
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-CA-01")
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
    ws.cell(row=r + 1, column=11, value=200_000)
    wb.save(path)
    return path


@pytest.fixture
def cumulative_combined(tmp_path):
    _, _, _, combined = _run_pipeline(_cumulative_quarters_workbook(tmp_path / "cumulative.xlsx"))
    return combined


def test_cumulative_quarters_not_flagged_critical(cumulative_combined):
    row = cumulative_combined[cumulative_combined["item_name"] == "Нарастающим"].iloc[0]
    assert bool(row["quarterly_cumulative_detected"]) is True
    assert bool(row["quarterly_mismatch_flag"]) is False
    assert row["quarterly_sum"] == 100_000


def test_discrete_quarters_still_flagged_critical(cumulative_combined):
    row = cumulative_combined[cumulative_combined["item_name"] == "Критично дискретно"].iloc[0]
    assert bool(row["quarterly_cumulative_detected"]) is False
    assert bool(row["quarterly_mismatch_flag"]) is True
    assert row["quarterly_severity"] == "critical"


def test_budget_audit_quarterly_reconciliation_detects_cumulative():
    """Same detection mirrored on core.budget_audit's fixed-column schema."""
    df = pd.DataFrame({
        "plan_annual": [100_000, 100_000],
        "plan_q1": [24_000, 23_500],
        "plan_q2": [50_000, 23_500],
        "plan_q3": [76_000, 23_500],
        "plan_q4": [100_000, 23_500],
        "article_code": ["CUM-01", "DISC-01"],
    })
    result = quarterly_reconciliation(df)
    assert result["article_code"].tolist() == ["DISC-01"]
    assert result.iloc[0]["severity"] == "critical"


# --- 4. compound counterparty cell -------------------------------------------------
def _compound_counterparty_workbook(path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws, campaign_year=2026)
    hdr = 9
    _write_headers(ws, hdr, {9: "Контрагент", 11: "План за 2026 год"})
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-CA-01"); ws.cell(row=r, column=4, value="Услуги")
    r += 1
    ws.cell(row=r, column=7, value="Услуги"); ws.cell(row=r, column=8, value="Связь")
    ws.cell(row=r, column=9, value="ТОО Альфа")
    ws.cell(row=r, column=11, value=30_000)
    r += 1  # compound cell: two legal entities typed into one counterparty field
    ws.cell(row=r, column=7, value="Услуги"); ws.cell(row=r, column=8, value="Оборудование")
    ws.cell(row=r, column=9, value="ТОО Бета, АО Гамма")
    ws.cell(row=r, column=11, value=90_000)
    ws.cell(row=r + 1, column=3, value="Итого"); ws.cell(row=r + 1, column=11, value=120_000)
    wb.save(path)
    return path


def test_compound_counterparty_excluded_from_concentration(tmp_path):
    path = _compound_counterparty_workbook(tmp_path / "compound.xlsx")
    meta, _, _, combined = _run_pipeline(path)
    result = build_audit_result(meta, combined)
    concentration = result["findings"]["concentration"]

    # Without the fix: the compound cell (75% share) would win top_supplier
    # and flip concentration_flag True for a "supplier" that doesn't exist.
    assert concentration["top_supplier"] == "ТОО Альфа"
    assert concentration["top_supplier_pct"] == pytest.approx(25.0)
    assert concentration["concentration_flag"] is False

    breakdown = concentration["supplier_breakdown"]
    compound_row = breakdown[breakdown["counterparty"] == "несколько контрагентов в одной строке"]
    assert len(compound_row) == 1
    assert compound_row.iloc[0]["total_amount"] == 90_000


def test_budget_audit_concentration_risk_excludes_compound_counterparty():
    """Same exclusion mirrored on core.budget_audit's implementation."""
    df = pd.DataFrame({
        "counterparty": ["ТОО Альфа", "ТОО Бета, АО Гамма"],
        "plan_annual": [30_000, 90_000],
        "contract": ["Д-1", "Д-2"],
    })
    result = concentration_risk(df)
    assert result["top_supplier"] == "ТОО Альфа"
    assert result["concentration_flag"] is False
    assert "несколько контрагентов в одной строке" in result["supplier_breakdown"]["counterparty"].tolist()
