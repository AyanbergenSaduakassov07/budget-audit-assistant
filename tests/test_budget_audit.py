"""Tests for core.budget_audit — synthetic openpyxl fixture, no API cost.

Six planted rows (A-F) exercise every rule and the run_full_audit() priority
logic in one sheet: growth (АТК x3), a critical quarterly mismatch, a missing
field, and one clean row that must survive untouched.

Row C is deliberately double-planted: it's both a "новая статья" (→ АТК via
year_over_year_analysis) *and* missing its contract (→ would be "уточнение"
via missing_fields on its own). run_full_audit() must resolve that in favor
of АТК — that's the "АТК outranks уточнение" priority rule under test.
"""

import pytest
from openpyxl import Workbook

from core.corporate_ingest import extract_sheet
from core.budget_audit import (
    quarterly_reconciliation,
    year_over_year_analysis,
    concentration_risk,
    missing_fields,
    new_line_items,
    run_full_audit,
)

_HEADERS = [
    "Статья бюджета", "Виды затрат", "Контрагент", "Договор",
    "2026 год утвержденный план", "План за 2027 год",
    "1 кв", "2 кв", "3 кв", "4 кв",
]

# article_code, expense_type, counterparty, contract, plan_prior, plan_annual, q1, q2, q3, q4
_ROWS = [
    # A: normal growth, +82% -> АТК. Quarterly matches exactly (182000).
    ("ИТ-01", "Услуги", "ТОО Мегасервис", "Д-001", 100000, 182000, 45500, 45500, 45500, 45500),
    # B: large growth spike -> АТК. Quarterly matches exactly (87610).
    ("ИТ-02", "Связь", "Телеком Плюс", "Д-002", 10000, 87610, 21902, 21902, 21903, 21903),
    # C: brand-new line item (prior=0), plan_annual > 100k -> АТК. Contract left
    # blank on purpose (see module docstring) to test the АТК/уточнение priority rule.
    ("ИТ-03", "Оборудование", "ТОО Мегасервис", "", 0, 198326, 49581, 49582, 49581, 49582),
    # D: flat year-over-year (0%), but quarterly sum is 6% off annual -> critical mismatch.
    ("ИТ-04", "Аренда", "Сервис Плюс", "Д-003", 100000, 100000, 23500, 23500, 23500, 23500),
    # E: counterparty blank -> missing_fields. Growth is a mild +5%, not flagged on its own.
    ("ИТ-05", "Лицензии", "", "Д-005", 100000, 105000, 26250, 26250, 26250, 26250),
    # F: +15% growth, everything else clean -> must end up "согласовано", untouched.
    ("ИТ-06", "Обслуживание", "ТОО Сателлит", "Д-004", 100000, 115000, 28750, 28750, 28750, 28750),
]

TOTAL_BUDGET = sum(row[5] for row in _ROWS)  # 787,936


def _build_audit_sheet(ws) -> None:
    ws.title = "AUDIT"
    ws["A1"], ws["B1"] = "Год кампании", 2027
    ws["A2"], ws["B2"] = "Ед. измерения", 1

    for col_idx, text in enumerate(_HEADERS, start=1):
        ws.cell(row=6, column=col_idx, value=text)

    for r_idx, row in enumerate(_ROWS, start=7):
        for c_idx, value in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=value)


@pytest.fixture
def sheet_result():
    wb = Workbook()
    _build_audit_sheet(wb.active)
    return extract_sheet(wb.active)


@pytest.fixture
def canonical_df(sheet_result):
    return sheet_result["data"]


# --- RULE 1: quarterly_reconciliation ------------------------------------------
def test_quarterly_reconciliation_flags_only_row_d(canonical_df):
    result = quarterly_reconciliation(canonical_df)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["article_code"] == "ИТ-04"
    assert row["difference"] == 6000
    assert row["difference_pct"] == pytest.approx(6.0)
    assert row["severity"] == "critical"


def test_quarterly_reconciliation_ignores_exact_matches(canonical_df):
    result = quarterly_reconciliation(canonical_df)
    assert "ИТ-01" not in result["article_code"].tolist()
    assert "ИТ-06" not in result["article_code"].tolist()


# --- RULE 2: year_over_year_analysis --------------------------------------------
def test_year_over_year_flags_atk_rows(canonical_df):
    result = year_over_year_analysis(canonical_df)
    atk_codes = set(result.loc[result["decision"] == "АТК", "article_code"])
    assert atk_codes == {"ИТ-01", "ИТ-02", "ИТ-03"}


def test_year_over_year_row_b_delta_pct(canonical_df):
    result = year_over_year_analysis(canonical_df)
    row_b = result[result["article_code"] == "ИТ-02"].iloc[0]
    expected_pct = (87610 - 10000) / 10000 * 100
    assert row_b["delta_pct"] == pytest.approx(expected_pct)
    assert row_b["decision"] == "АТК"


def test_year_over_year_new_item_is_inf_delta(canonical_df):
    result = year_over_year_analysis(canonical_df)
    row_c = result[result["article_code"] == "ИТ-03"].iloc[0]
    assert row_c["delta_pct"] == float("inf")
    assert row_c["delta_flag"] == "новая статья"
    assert row_c["decision"] == "АТК"


def test_year_over_year_excludes_normal_rows(canonical_df):
    result = year_over_year_analysis(canonical_df)
    codes = set(result["article_code"])
    # D (0%), E (+5%), F (+15%) are all below the 20% уточнение threshold.
    assert "ИТ-04" not in codes
    assert "ИТ-05" not in codes
    assert "ИТ-06" not in codes


# --- RULE 3: concentration_risk --------------------------------------------------
def test_concentration_risk_flags_top_supplier(canonical_df):
    result = concentration_risk(canonical_df)
    assert result["concentration_flag"] is True
    assert result["top_supplier"] == "ТОО Мегасервис"
    expected_pct = (182000 + 198326) / TOTAL_BUDGET * 100
    assert result["top_supplier_pct"] == pytest.approx(expected_pct)


def test_concentration_risk_single_source_rows(canonical_df):
    breakdown = concentration_risk(canonical_df)["supplier_breakdown"]
    mega = breakdown[breakdown["counterparty"] == "ТОО Мегасервис"].iloc[0]
    assert mega["single_source_rows"] == 1  # only ИТ-03's contract is blank


def test_concentration_risk_excludes_blank_counterparty_from_breakdown(canonical_df):
    breakdown = concentration_risk(canonical_df)["supplier_breakdown"]
    assert "" not in breakdown["counterparty"].tolist()


def test_concentration_risk_total_single_source(canonical_df):
    result = concentration_risk(canonical_df)
    assert result["total_single_source_amount"] == pytest.approx(198326)
    assert result["total_single_source_pct"] == pytest.approx(198326 / TOTAL_BUDGET * 100)


def test_concentration_risk_non_dominant_supplier_not_flagged(canonical_df):
    breakdown = concentration_risk(canonical_df)["supplier_breakdown"]
    kaztel = breakdown[breakdown["counterparty"] == "Телеком Плюс"].iloc[0]
    assert kaztel["risk_flag"] == False  # noqa: E712 — explicit bool check on pandas cell


# --- RULE 4: missing_fields -------------------------------------------------------
def test_missing_fields_flags_row_e(canonical_df):
    result = missing_fields(canonical_df)
    by_code = result.set_index("article_code")
    assert "ИТ-05" in by_code.index
    assert by_code.loc["ИТ-05", "missing_fields"] == "counterparty"


def test_missing_fields_row_c_flagged_for_contract(canonical_df):
    # ИТ-03's contract is blank too — missing_fields catches it independently
    # of the delta/growth rule. run_full_audit resolves the priority (see below).
    result = missing_fields(canonical_df)
    assert "ИТ-03" in result["article_code"].tolist()
    assert result.set_index("article_code").loc["ИТ-03", "missing_fields"] == "contract"


# --- RULE 5: new_line_items -------------------------------------------------------
def test_new_line_items_only_row_c(canonical_df):
    result = new_line_items(canonical_df)
    assert result["article_code"].tolist() == ["ИТ-03"]
    assert result.iloc[0]["recommendation"] == "Запросить обоснование"


def test_new_line_items_recommendation_thresholds():
    import pandas as pd
    df = pd.DataFrame({
        "delta_flag": ["новая статья"] * 3,
        "plan_annual": [50_000, 250_000, 900_000],
    })
    result = new_line_items(df)
    assert result["recommendation"].tolist() == [
        "Уточнить необходимость",
        "Запросить обоснование",
        "Вынести на АТК — крупная новая статья",
    ]


# --- RULE 6: run_full_audit --------------------------------------------------------
def test_run_full_audit_decisions_per_row(sheet_result):
    audit = run_full_audit(sheet_result)
    decisions = audit["all_rows"].set_index("article_code")["decision"].to_dict()
    assert decisions["ИТ-01"] == "АТК"
    assert decisions["ИТ-02"] == "АТК"
    assert decisions["ИТ-03"] == "АТК"  # priority rule: outranks its own missing_fields hit
    assert decisions["ИТ-04"] == "уточнение"
    assert decisions["ИТ-05"] == "уточнение"
    assert decisions["ИТ-06"] == "согласовано"  # must not be flagged by anything


def test_run_full_audit_summary_counts(sheet_result):
    summary = run_full_audit(sheet_result)["summary"]
    assert summary["atk_count"] == 3
    assert summary["atk_amount"] == pytest.approx(182000 + 87610 + 198326)
    assert summary["clarification_count"] == 2
    assert summary["clarification_amount"] == pytest.approx(100000 + 105000)
    assert summary["approved_count"] == 1
    assert summary["approved_amount"] == pytest.approx(115000)
    assert summary["total_rows"] == 6
    assert summary["total_amount"] == pytest.approx(TOTAL_BUDGET)
    assert summary["has_prior_year_comparison"] is True


def test_run_full_audit_top_level_fields(sheet_result):
    audit = run_full_audit(sheet_result)
    assert audit["entity"] == "AUDIT"
    assert audit["sheet_name"] == "AUDIT"
    assert audit["budget_year"] == 2027
    assert audit["unit_scale"] == 1
    assert audit["has_prior_year"] is True
    assert audit["has_critical"] is True


def test_run_full_audit_findings_shapes(sheet_result):
    findings = run_full_audit(sheet_result)["findings"]
    assert len(findings["quarterly_mismatches"]) == 1
    assert len(findings["growth_flags"]) == 3
    assert len(findings["missing_fields"]) == 2  # ИТ-05 (counterparty) + ИТ-03 (contract)
    assert len(findings["new_items"]) == 1
    assert findings["concentration"]["concentration_flag"] is True
