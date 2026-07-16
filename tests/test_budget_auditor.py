"""Tests for core.budget_auditor (parser + analysis) and app.py display formatters.

Fixtures are synthetic workbooks that reproduce the structural variants seen in
the real subsidiary forms (see parser.py module docstring):
  - Subsidiary A style: факт + обоснование columns present
  - Subsidiary B style: no prior-year columns, no обоснование, item names at category level
  - newer-template style: reordered headers ("ФАКТ за 2025 год"), dual plan-year columns
"""

from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from core.budget_auditor.parser import (
    parse_workbook,
    flag_rows,
    aggregate_workbook,
    capex_opex_table,
    year_trend,
    _best_baseline_col,
)
from core.budget_auditor.analysis import (
    workbook_summary_text,
    year_trend_text,
    chat_answer,
)

ROOT = Path(__file__).resolve().parent.parent


# --- workbook builders ---------------------------------------------------------
def _write_meta(ws, campaign_year=2026, company="ТОО Тест-Дочка"):
    meta = [
        ("Год кампании", campaign_year),
        ("Версия", "план"),
        ("Компания", company),
        ("Валюта", "KZT"),
        ("Ед. измерения", 1000),
    ]
    for i, (label, value) in enumerate(meta, start=2):
        ws.cell(row=i, column=3, value=label)
        ws.cell(row=i, column=4, value=value)


def _write_headers(ws, row, extra):
    """Base hierarchy headers + extra {col_index: header_text}."""
    base = {3: "Статья бюджета", 4: "Наименование", 7: "Виды затрат", 8: "Подвид затрат"}
    for c, t in {**base, **extra}.items():
        ws.cell(row=row, column=c, value=t)


def _full_form_workbook(path):
    """Subsidiary A-style OPEX sheet + Subsidiary B-style ГО CAPEX sheet + empty АФ template."""
    wb = openpyxl.Workbook()

    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws)
    hdr = 9
    _write_headers(ws, hdr, {
        9: "Контрагент", 10: "2025 год факт", 11: "План за 2026 год",
        12: "План 1 кв 2026", 13: "Обоснование",
    })
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-OP-01"); ws.cell(row=r, column=4, value="Связь")
    r += 1  # category scaffolding row, no numbers
    ws.cell(row=r, column=7, value="Услуги связи")
    r += 1  # leaf with justification, +100% YoY
    ws.cell(row=r, column=7, value="Услуги связи"); ws.cell(row=r, column=8, value="Сотовая связь")
    ws.cell(row=r, column=9, value="Телеком Плюс")
    ws.cell(row=r, column=10, value=1000); ws.cell(row=r, column=11, value=2000)
    ws.cell(row=r, column=13, value="Рост тарифов")
    r += 1  # leaf WITHOUT justification, +4% YoY
    ws.cell(row=r, column=7, value="Услуги связи"); ws.cell(row=r, column=8, value="Интернет")
    ws.cell(row=r, column=9, value="Телеком Плюс")
    ws.cell(row=r, column=10, value=5000); ws.cell(row=r, column=11, value=5200)
    r += 1  # grand-total footer must be excluded
    ws.cell(row=r, column=3, value="Общий результат"); ws.cell(row=r, column=11, value=7200)
    ws.cell(row=r + 1, column=11, value=7200)

    ws2 = wb.create_sheet("ГО CAPEX")
    hdr2 = 8
    _write_headers(ws2, hdr2, {9: "Контрагент", 11: "План за 2026 год"})
    r = hdr2 + 1
    ws2.cell(row=r, column=3, value="IT-CAP-01"); ws2.cell(row=r, column=4, value="Оборудование")
    r += 1  # Subsidiary B style: item name at category level, no subcategory
    ws2.cell(row=r, column=7, value="Серверы"); ws2.cell(row=r, column=9, value="Dell")
    ws2.cell(row=r, column=11, value=9000)
    r += 1
    ws2.cell(row=r, column=7, value="СХД"); ws2.cell(row=r, column=9, value="HP")
    ws2.cell(row=r, column=11, value=800)
    ws2.cell(row=r + 1, column=3, value="Итого"); ws2.cell(row=r + 1, column=11, value=9800)

    ws3 = wb.create_sheet("АФ")  # untouched empty template
    ws3.cell(row=8, column=3, value="Статья бюджета")

    wb.save(path)
    return path


def _run_pipeline(path, threshold_pct=15.0):
    meta, sheets = parse_workbook(path)
    usable = {sn: d for sn, d in sheets.items() if d["df"] is not None and not d["df"].empty}
    flagged = {
        sn: {"df": flag_rows(d["df"], d["stats"], threshold_pct=threshold_pct), "stats": d["stats"]}
        for sn, d in usable.items()
    }
    return meta, sheets, flagged, aggregate_workbook(flagged)


@pytest.fixture(scope="module")
def full_form(tmp_path_factory):
    path = tmp_path_factory.mktemp("forms") / "full_form.xlsx"
    return _run_pipeline(_full_form_workbook(path))


@pytest.fixture(scope="module")
def app_helpers():
    """Execute app.py's helper section (everything above the page code) so the
    real production formatters and build_flag_view are testable without a
    Streamlit runtime. Page code is excluded — it calls st.stop()."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    head = src.split("# --- page")[0]
    ns = {}
    exec(compile(head, str(ROOT / "app.py"), "exec"), ns)
    return ns


# --- full-form (Subsidiary A-style) pipeline -------------------------------------------
def test_metadata_and_sheet_split(full_form):
    meta, sheets, flagged, combined = full_form
    assert meta["company"] == "ТОО Тест-Дочка"
    assert meta["campaign_year"] == 2026
    assert set(flagged) == {"OPEX", "ГО CAPEX"}
    assert "подана пустой" in workbook_summary_text(meta, sheets)  # АФ template


def test_footer_rows_not_double_counted(full_form):
    *_, combined = full_form
    assert pd.to_numeric(combined["plan_2026"], errors="coerce").sum() == 17000


def test_years_present_and_flags(full_form):
    _, _, flagged, combined = full_form
    assert flagged["OPEX"]["stats"]["years_present"] == [2025, 2026]
    opex = combined[combined["cost_type"] == "OPEX"]
    capex = combined[combined["cost_type"] == "CAPEX"]
    assert opex["high_change_flag"].sum() == 1  # the +100% row only
    assert opex["missing_justification_flag"].sum() == 1
    assert capex["missing_justification_flag"].all()  # column structurally absent
    assert not capex["high_change_flag"].any()  # no baseline in that sheet
    assert capex["branch"].unique().tolist() == ["ГО"]


def test_capex_opex_table_uses_fact_baseline(full_form):
    *_, combined = full_form
    cvo = capex_opex_table(combined)
    assert set(cvo["Тип"]) == {"CAPEX", "OPEX"}
    opex_row = cvo[cvo["Тип"] == "OPEX"].iloc[0]
    assert opex_row["План 2026"] == 7200
    assert opex_row["Δ %"] == pytest.approx(20.0)  # vs fact_2025 = 6000
    assert cvo["Доля"].sum() == pytest.approx(100.0)


def test_summary_text_names_structural_gaps(full_form):
    meta, _, flagged, _ = full_form
    text = workbook_summary_text(meta, flagged)
    assert "нет данных за прошлый год" in text  # ГО CAPEX sheet
    assert "Обоснование" in text


def test_year_trend_consistent_on_full_form(full_form):
    *_, combined = full_form
    yt = year_trend(combined)
    assert not yt["unit_mismatch_suspected"].any()
    assert "согласованным" in year_trend_text(combined)


def test_chat_answers_span_whole_file(full_form):
    meta, _, _, combined = full_form
    for question, fragment in [
        ("сравни OPEX и CAPEX", "OPEX = 7 200"),
        ("сколько capex в 2026", "9 800"),
        ("какая ветка тратит больше", "ГО"),
        ("какие статьи без обоснования", "без обоснования: 3"),
        ("топ самых дорогих статей", "Серверы"),
        ("сколько потратили на сотовую связь", "Сотовая связь"),
    ]:
        reply = chat_answer(question, combined, meta)
        assert reply and fragment in reply, (question, reply)


# --- dual plan-year columns (campaign year must win, not column order) ----------
def test_dual_plan_columns_pick_campaign_year(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws, campaign_year=2027)
    hdr = 9
    _write_headers(ws, hdr, {
        9: "Контрагент", 10: "План за 2026 год", 11: "План за 2027 год",
    })
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-OP-01"); ws.cell(row=r, column=4, value="Связь")
    r += 1
    ws.cell(row=r, column=7, value="Услуги связи"); ws.cell(row=r, column=8, value="Интернет")
    ws.cell(row=r, column=10, value=4000); ws.cell(row=r, column=11, value=4500)
    ws.cell(row=r + 1, column=3, value="Общий результат"); ws.cell(row=r + 1, column=11, value=4500)
    path = tmp_path / "dual_plan.xlsx"
    wb.save(path)

    _, sheets = parse_workbook(path)
    stats = sheets["OPEX"]["stats"]
    df = sheets["OPEX"]["df"]
    assert stats["plan_year"] == 2027  # NOT 2026, despite that column coming first
    assert df["plan_2027"].tolist() == [4500]
    assert df["plan_2026"].tolist() == [4000]  # leftover column still captured
    assert stats["years_present"] == [2026, 2027]


# --- reordered header wording ("ФАКТ за 2025 год" / "ПЛАН за 2026 год") ---------
def test_reordered_fact_plan_headers(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws, campaign_year=2026)
    hdr = 9
    _write_headers(ws, hdr, {
        9: "Контрагент", 10: "ФАКТ за 2025 год", 11: "ПЛАН за 2026 год",
    })
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-OP-01"); ws.cell(row=r, column=4, value="Связь")
    r += 1
    ws.cell(row=r, column=7, value="Услуги связи"); ws.cell(row=r, column=8, value="Интернет")
    ws.cell(row=r, column=10, value=3000); ws.cell(row=r, column=11, value=3300)
    ws.cell(row=r + 1, column=3, value="Итого"); ws.cell(row=r + 1, column=11, value=3300)
    path = tmp_path / "reordered_headers.xlsx"
    wb.save(path)

    _, sheets = parse_workbook(path)
    df = sheets["OPEX"]["df"]
    stats = sheets["OPEX"]["stats"]
    assert df["fact_2025"].tolist() == [3000]
    assert df["plan_2026"].tolist() == [3300]
    assert _best_baseline_col(df, stats["plan_year"]) == "fact_2025"


# --- Subsidiary B style: no baseline anywhere -> pd.NA deltas must render as "н/д" ----
def test_no_baseline_form_renders_nd_not_typeerror(tmp_path, app_helpers):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws)
    hdr = 9
    _write_headers(ws, hdr, {9: "Контрагент", 11: "План за 2026 год"})
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-OP-01"); ws.cell(row=r, column=4, value="Связь")
    for name, value in [("Интернет", 5200), ("Сотовая связь", 2000)]:
        r += 1
        ws.cell(row=r, column=7, value="Услуги связи"); ws.cell(row=r, column=8, value=name)
        ws.cell(row=r, column=9, value="Телеком Плюс"); ws.cell(row=r, column=11, value=value)
    ws.cell(row=r + 1, column=3, value="Итого"); ws.cell(row=r + 1, column=11, value=7200)
    path = tmp_path / "no_baseline.xlsx"
    wb.save(path)

    meta, sheets, flagged, combined = _run_pipeline(path)
    assert combined["delta_abs"].isna().all()
    assert combined["delta_pct"].isna().all()
    assert combined["missing_justification_flag"].all()

    # Render the exact production Styler: the formatter is called lazily on the
    # raw pd.NA cells here — this is where the float(NAType) TypeError lived.
    build_flag_view = app_helpers["build_flag_view"]
    subset = combined[combined["missing_justification_flag"]]
    disp, fmt = build_flag_view(subset, "plan_2026")
    html = disp.style.format(fmt).to_html()
    assert "н/д" in html
    assert "5 200" in html  # real numbers still space-grouped


def test_formatters_handle_all_na_variants(app_helpers):
    for na in (None, pd.NA, float("nan")):
        assert app_helpers["_money"](na) == "н/д"
        assert app_helpers["_pct"](na) == "н/д"
        assert app_helpers["_share"](na) == "н/д"
    assert app_helpers["_money"](1234567) == "1 234 567"
    assert app_helpers["_pct"](-12.34) == "-12.3%"


# --- units-mismatch sanity check -------------------------------------------------
def test_year_trend_flags_unit_mismatch():
    fake = pd.DataFrame({
        "plan_2026": [100.0],
        "plan_2027": [120.0],
        "fact_2025": [9_000_000.0],  # entered in raw tenge, not тыс.
    })
    yt = year_trend(fake)
    assert yt.loc[yt["Год"] == 2025, "unit_mismatch_suspected"].all()
    assert not yt.loc[yt["Год"] != 2025, "unit_mismatch_suspected"].any()
    assert "единиц измерения" in year_trend_text(fake)
