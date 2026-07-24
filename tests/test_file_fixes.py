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
  5. the footer-detection fix for #2 was itself a regression (searching
     every column false-positived on free-text prose; a real per-section
     subtotal repeats after EVERY budget code, not just once) -- fixed to
     code-column-only sheet end + a non-terminating per-row subtotal skip.
  6. a non-code value in the code column (a real file's preparer name,
     "НЕКОДОВОЕ_ЗНАЧЕНИЕ") on rows that also carry real leaf data must not be treated as
     a fresh code declaration; multi-row item groups (category on the first
     row only) must forward-fill within the open section.

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
    """Real file shape: a per-section subtotal row whose CATEGORY column
    (not the code column) reads bare "Результат", one row above the true
    "Общий результат" sheet-ending label in the code column."""
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


def test_section_subtotal_row_excluded_without_terminating_sheet(tmp_path):
    path = _footer_marker_in_category_column_workbook(tmp_path / "footer.xlsx")
    _, sheets = parse_workbook(path)
    df = sheets["OPEX"]["df"]

    # A bare "Результат" in the category column is a per-section subtotal --
    # excluded (not a phantom leaf: 5000 + 2000 + 7000 would be 14000), but
    # it must not terminate the sheet either; only the code-column "Общий
    # результат" label one row below does that.
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


# --- 5. footer-detection regression: free text vs. section subtotal vs. true end ---
# The all-columns footer scan added to fix issue 2 above was itself a regression,
# confirmed against 4 real ACM files:
#   (a) the "итог|результат" search matched free-text justification/description
#       prose ("результатов инвентаризации оборудования") and truncated the
#       sheet after 1-2 rows;
#   (b) real forms repeat a bare "Результат" category-column subtotal after
#       EVERY budget-code section (not just once at the sheet's true end) --
#       treating the first occurrence as a sheet-ending break discarded every
#       section after the first;
#   (c) the one true one-time ending label is "Общий результат" specifically
#       in the Статья бюджета (code) column.
# Fix: the sheet-terminating check is code-column-only again; a bare
# "результат"/"итого" in category/subcategory is a non-terminating per-row
# skip; the footer pattern is never applied to free-text columns.
def _justification_false_positive_workbook(path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws, campaign_year=2026)
    hdr = 9
    _write_headers(ws, hdr, {9: "Контрагент", 10: "Обоснование", 11: "План за 2026 год"})
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-OP-01"); ws.cell(row=r, column=4, value="Оборудование")
    r += 1
    ws.cell(row=r, column=7, value="Оборудование"); ws.cell(row=r, column=8, value="Мониторы")
    ws.cell(row=r, column=9, value="Dell")
    ws.cell(row=r, column=10, value=(
        "Потребность определена на основании заявок подразделений, "
        "результатов инвентаризации оборудования и производственной необходимости."
    ))
    ws.cell(row=r, column=11, value=4000)
    r += 1  # a second leaf row that must NOT be truncated away by the false positive
    ws.cell(row=r, column=7, value="Оборудование"); ws.cell(row=r, column=8, value="Клавиатуры")
    ws.cell(row=r, column=9, value="Dell"); ws.cell(row=r, column=11, value=1000)
    ws.cell(row=r + 1, column=3, value="Общий результат"); ws.cell(row=r + 1, column=11, value=5000)
    wb.save(path)
    return path


def test_justification_substring_does_not_truncate_sheet(tmp_path):
    path = _justification_false_positive_workbook(tmp_path / "justification.xlsx")
    _, sheets = parse_workbook(path)
    df = sheets["OPEX"]["df"]

    assert len(df) == 2  # both leaf rows survive, not just the first
    assert pd.to_numeric(df["plan_2026"], errors="coerce").sum() == 5000


def _multi_section_subtotal_workbook(path):
    """Three budget-code sections, each ending in a bare "Результат"
    category-column subtotal -- the real sample.xlsx pattern (IT-OP-01,
    IT-OP-02, IT-OP-03, ... each followed by its own "Результат" row)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws, campaign_year=2026)
    hdr = 9
    _write_headers(ws, hdr, {9: "Контрагент", 11: "План за 2026 год"})
    r = hdr + 1
    for i, (code, name, item, amount) in enumerate([
        ("IT-OP-01", "Связь", "Интернет", 1000),
        ("IT-OP-02", "Лицензии", "Антивирус", 2000),
        ("IT-OP-03", "Обслуживание", "Поддержка", 3000),
    ], start=1):
        ws.cell(row=r, column=3, value=code); ws.cell(row=r, column=4, value=name)
        r += 1
        ws.cell(row=r, column=7, value=name); ws.cell(row=r, column=8, value=item)
        ws.cell(row=r, column=9, value="Вендор"); ws.cell(row=r, column=11, value=amount)
        r += 1  # per-section subtotal: must be skipped, must NOT end the sheet
        ws.cell(row=r, column=7, value="Результат"); ws.cell(row=r, column=11, value=amount)
        r += 1
    ws.cell(row=r, column=3, value="Общий результат"); ws.cell(row=r, column=11, value=6000)
    wb.save(path)
    return path


def test_multiple_section_subtotals_do_not_truncate_later_sections(tmp_path):
    path = _multi_section_subtotal_workbook(tmp_path / "multi_section.xlsx")
    _, sheets = parse_workbook(path)
    df = sheets["OPEX"]["df"]

    # Before the fix: the FIRST "Результат" row (after IT-OP-01) ended the
    # sheet, discarding IT-OP-02 and IT-OP-03 entirely.
    assert set(df["statya_code"]) == {"IT-OP-01", "IT-OP-02", "IT-OP-03"}
    assert len(df) == 3
    assert pd.to_numeric(df["plan_2026"], errors="coerce").sum() == 6000


def test_genuine_sheet_end_label_still_stops_parsing(tmp_path):
    """A real "Общий результат" in the code column must still terminate the
    sheet -- nothing after it (even a stray numeric row) is ingested."""
    path = _multi_section_subtotal_workbook(tmp_path / "sheet_end.xlsx")
    wb = openpyxl.load_workbook(path)
    ws = wb["OPEX"]
    end_row = next(
        r for r in range(1, ws.max_row + 1)
        if ws.cell(row=r, column=3).value == "Общий результат"
    )
    # A stray row below the true end label, shaped like a leaf row.
    ws.cell(row=end_row + 1, column=7, value="Прочее")
    ws.cell(row=end_row + 1, column=8, value="Не должно попасть")
    ws.cell(row=end_row + 1, column=11, value=99_999)
    wb.save(path)

    _, sheets = parse_workbook(path)
    df = sheets["OPEX"]["df"]
    assert "Не должно попасть" not in df["item_name"].tolist()
    assert pd.to_numeric(df["plan_2026"], errors="coerce").sum() == 6000


# --- 6. non-code value in the code column (real file: "НЕКОДОВОЕ_ЗНАЧЕНИЕ") ------------------
# A real file put the preparer's NAME in the Статья бюджета column on rows that
# also carried real category+item+value data, and on multi-row item groups where
# only the first row of the group carried category at all:
#   (a) code="НЕКОДОВОЕ_ЗНАЧЕНИЕ" + category + value on the SAME row -- must be captured as a
#       leaf under the section that's actually open (the last real code), not
#       treated as a fresh (bogus) declaration that discards the row's own data.
#   (b) code="НЕКОДОВОЕ_ЗНАЧЕНИЕ" + category but NO value on that row (a legitimately unfilled
#       item) -- must not corrupt cur_code for the real leaf rows that follow.
#   (c) a multi-row item group: category on the first row only, continuation rows
#       carry just their own value with category/subcategory blank -- must still
#       be recognized as leaf rows via forward-fill within the open section.
def _non_code_value_in_code_column_workbook(path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "CAPEX"
    _write_meta(ws, campaign_year=2027)
    hdr = 9
    _write_headers(ws, hdr, {9: "Контрагент", 11: "План за 2027 год"})
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-CA-01"); ws.cell(row=r, column=4, value="Оборудование")
    r += 1  # (a) non-code value + its own category + its own value
    ws.cell(row=r, column=3, value="НЕКОДОВОЕ_ЗНАЧЕНИЕ")
    ws.cell(row=r, column=7, value="Источники бесперебойного питания")
    ws.cell(row=r, column=9, value="ИБП 2000ВА"); ws.cell(row=r, column=11, value=2151.5)
    r += 1  # (b) non-code value + category but NO value this row -- must not corrupt cur_code
    ws.cell(row=r, column=3, value="НЕКОДОВОЕ_ЗНАЧЕНИЕ")
    ws.cell(row=r, column=7, value="Оснащение гермозон")
    r += 1  # (c) multi-row group: category on first row only
    ws.cell(row=r, column=3, value="НЕКОДОВОЕ_ЗНАЧЕНИЕ")
    ws.cell(row=r, column=7, value="Компьютерная техника")
    ws.cell(row=r, column=9, value="Монитор"); ws.cell(row=r, column=11, value=3847.9)
    r += 1  # continuation rows: blank category, own item + value
    ws.cell(row=r, column=9, value="Системный блок"); ws.cell(row=r, column=11, value=32817.7)
    r += 1
    ws.cell(row=r, column=9, value="Ноутбук"); ws.cell(row=r, column=11, value=8307)
    ws.cell(row=r + 1, column=3, value="Общий результат"); ws.cell(row=r + 1, column=11, value=47124.1)
    wb.save(path)
    return path


def test_non_code_value_with_own_data_captured_under_open_section(tmp_path):
    path = _non_code_value_in_code_column_workbook(tmp_path / "non_code.xlsx")
    _, sheets = parse_workbook(path)
    df = sheets["CAPEX"]["df"]

    # All leaf rows belong to IT-CA-01 -- "НЕКОДОВОЕ_ЗНАЧЕНИЕ" must never appear as a code.
    assert set(df["statya_code"]) == {"IT-CA-01"}
    assert "НЕКОДОВОЕ_ЗНАЧЕНИЕ" not in df["statya_code"].tolist()


def test_non_code_value_without_data_does_not_corrupt_later_sections(tmp_path):
    path = _non_code_value_in_code_column_workbook(tmp_path / "non_code2.xlsx")
    _, sheets = parse_workbook(path)
    df = sheets["CAPEX"]["df"]

    # The "НЕКОДОВОЕ_ЗНАЧЕНИЕ" + category-but-no-value row must not become cur_code, or
    # every leaf row after it (the multi-row group) would misattribute to it.
    assert pd.to_numeric(df["plan_2027"], errors="coerce").sum() == pytest.approx(
        2151.5 + 3847.9 + 32817.7 + 8307
    )


def test_multi_row_item_group_forward_fills_category(tmp_path):
    path = _non_code_value_in_code_column_workbook(tmp_path / "non_code3.xlsx")
    _, sheets = parse_workbook(path)
    df = sheets["CAPEX"]["df"]

    # 4 real leaf rows total: ИБП + Монитор + Системный блок + Ноутбук.
    # "Оснащение гермозон" (category, no value) correctly excluded.
    assert len(df) == 4
    # The continuation row (own value 32817.7, no category of its own) must
    # still be captured, with category forward-filled from the group's first
    # row -- item_name has no col-9 fallback, so it also reads the
    # forward-filled category, same as the group's first row.
    continuation = df[df["plan_2027"] == 32817.7]
    assert len(continuation) == 1
    assert continuation.iloc[0]["category"] == "Компьютерная техника"
    assert continuation.iloc[0]["item_name"] == "Компьютерная техника"


def test_malformed_code_declaration_without_data_still_flags_anomalous(tmp_path):
    """A malformed-but-genuine code declaration (own row has no category/value
    of its own) must still be treated as a declaration -- distinguishing this
    from the "НЕКОДОВОЕ_ЗНАЧЕНИЕ" case is what makes the fix precise, not a blanket
    "ignore any non-code value" rule."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws, campaign_year=2026)
    hdr = 9
    _write_headers(ws, hdr, {9: "Контрагент", 11: "План за 2026 год"})
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-CA-1"); ws.cell(row=r, column=4, value="Оборудование")
    r += 1
    ws.cell(row=r, column=7, value="Оборудование"); ws.cell(row=r, column=8, value="Серверы")
    ws.cell(row=r, column=9, value="Dell"); ws.cell(row=r, column=11, value=2000)
    ws.cell(row=r + 1, column=3, value="Общий результат"); ws.cell(row=r + 1, column=11, value=2000)
    path = tmp_path / "malformed.xlsx"
    wb.save(path)

    _, sheets = parse_workbook(path)
    df = sheets["OPEX"]["df"]
    assert df["statya_code"].tolist() == ["IT-CA-1"]
    assert df["anomalous_code_flag"].tolist() == [True]
