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
    answer_query,
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


# --- routing fixes: bug B (year shadows other filters), bug C (narrow
# justification phrasing), bug E (hardcoded top-5), bug #4 (compound bailout) --
def test_chat_answer_plain_year_total_still_works(full_form):
    """Non-regression for the Bug B reordering: a simple year-only question
    with no other filter to match on must still get the plain whole-file
    year total, not be swallowed by the token-search branch now checked
    first."""
    meta, _, _, combined = full_form
    reply = chat_answer("сколько всего потратили в 2026 году", combined, meta)
    assert reply is not None
    assert "17 000" in reply
    assert "все подразделения" in reply


def test_chat_answer_token_filter_wins_over_year_total(full_form):
    """Bug B: a question naming BOTH a year AND another filter (a
    counterparty here) must be answered by that filter, not the whole
    file's year total, just because a year was also mentioned. Before the
    fix, this returned the whole-file 17 000 total for 2026, silently
    ignoring "Телеком Плюс"."""
    meta, _, _, combined = full_form
    reply = chat_answer("сколько потратили на Телеком Плюс в 2026 году", combined, meta)
    assert reply is not None
    assert "7 200" in reply       # Сотовая связь (2000) + Интернет (5200)
    assert "17 000" not in reply  # NOT the whole-file 2026 total


def test_chat_answer_broadened_justification_phrasing(full_form):
    """Bug C: "не имеют обоснования" (not just the old narrow "без обоснован"/
    "нет обоснован" substrings) must also match the justification branch."""
    meta, _, _, combined = full_form
    reply = chat_answer("какие статьи не имеют обоснования", combined, meta)
    assert reply is not None
    assert "Всего без обоснования: 3" in reply


def test_chat_answer_top_n_parsed_from_question(full_form):
    """Bug E: "топ-3" must return exactly 3 rows, not the hardcoded 5."""
    meta, _, _, combined = full_form
    reply = chat_answer("топ-3 самых крупных", combined, meta)
    assert reply is not None
    assert reply.startswith("Топ-3 самых крупных статей по всему файлу:")
    assert len(reply.splitlines()) == 4  # header + exactly 3 rows
    assert "СХД" not in reply  # the 4th/lowest-value row, correctly excluded


def test_answer_query_broadened_justification_check(full_form):
    """Bug C, answer_query() side: same broadened phrasing check."""
    meta, _, flagged, _ = full_form
    df, stats = flagged["OPEX"]["df"], flagged["OPEX"]["stats"]
    reply = answer_query("какие статьи не имеют обоснования", df, stats, meta)
    assert reply is not None
    assert reply.startswith("1 статей без обоснования")
    assert "Интернет" in reply


def test_answer_query_top_n_parses_requested_count(full_form):
    """Bug E, answer_query() side: requested count honored, not hardcoded 5."""
    meta, _, flagged, _ = full_form
    df, stats = flagged["ГО CAPEX"]["df"], flagged["ГО CAPEX"]["stats"]
    reply = answer_query("топ-1 самых крупных", df, stats, meta)
    assert reply is not None
    assert reply.startswith("Топ-1 самых крупных статей:")
    assert len(reply.splitlines()) == 2  # header + exactly 1 row


def test_chat_answer_compound_bailout_prevents_partial_branch_comparison():
    """Bug #4: "сравни CAPEX ГО и АФ" names two specific branches alongside a
    cost-type keyword -- that means "compare these two branches" (a real
    multi-condition query only file_sql can do), not the whole-company
    OPEX/CAPEX split the plain comparison branch answers. chat_answer() must
    return None so app.py's file_sql fallback handles it instead of a
    confident, wrong, company-wide ratio."""
    combined = pd.DataFrame({
        "branch": ["ГО", "АФ"],
        "cost_type": ["CAPEX", "CAPEX"],
        "item_name": ["Сервер", "Сервер2"],
        "category": ["Оборудование", "Оборудование"],
        "subcategory": [None, None],
        "counterparty": ["Dell", "HP"],
        "plan_year": [2026, 2026],
        "plan_2026": [1000, 2000],
    })
    meta = {"unit": 1000}
    reply = chat_answer("сравни CAPEX ГО и АФ", combined, meta)
    assert reply is None


# --- multi-year OPEX/CAPEX breakdown ("...по всем годам" / "...за все года") ----
def _multi_year_workbook(path):
    """OPEX + CAPEX sheets sharing campaign year 2027, each carrying three
    distinct years (fact_2025 / plan_2026 / plan_2027) -- enough to exercise
    a real per-year breakdown, not just a two-point comparison."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws, campaign_year=2027)
    hdr = 9
    _write_headers(ws, hdr, {
        9: "Контрагент", 10: "2025 год факт",
        11: "План за 2026 год", 12: "План за 2027 год",
    })
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-OP-01"); ws.cell(row=r, column=4, value="Связь")
    r += 1
    ws.cell(row=r, column=7, value="Услуги связи"); ws.cell(row=r, column=8, value="Интернет")
    ws.cell(row=r, column=9, value="Телеком Плюс")
    ws.cell(row=r, column=10, value=8000); ws.cell(row=r, column=11, value=9000); ws.cell(row=r, column=12, value=10000)
    ws.cell(row=r + 1, column=3, value="Итого"); ws.cell(row=r + 1, column=12, value=10000)

    ws2 = wb.create_sheet("CAPEX")
    hdr2 = 9
    _write_headers(ws2, hdr2, {
        9: "Контрагент", 10: "2025 год факт",
        11: "План за 2026 год", 12: "План за 2027 год",
    })
    r = hdr2 + 1
    ws2.cell(row=r, column=3, value="IT-CA-01"); ws2.cell(row=r, column=4, value="Оборудование")
    r += 1
    ws2.cell(row=r, column=7, value="Серверы"); ws2.cell(row=r, column=9, value="Dell")
    ws2.cell(row=r, column=10, value=4000); ws2.cell(row=r, column=11, value=4500); ws2.cell(row=r, column=12, value=5000)
    ws2.cell(row=r + 1, column=3, value="Итого"); ws2.cell(row=r + 1, column=12, value=5000)

    wb.save(path)
    return path


@pytest.fixture(scope="module")
def multi_year_form(tmp_path_factory):
    path = tmp_path_factory.mktemp("multi_year") / "multi_year.xlsx"
    return _run_pipeline(_multi_year_workbook(path))


def test_chat_answer_multi_year_capex_opex_breakdown(multi_year_form):
    meta, _, _, combined = multi_year_form
    reply = chat_answer("раздели планы на opex/capex по всем годам", combined, meta)
    assert reply is not None
    lines = reply.splitlines()
    assert lines[0] == "Разбивка OPEX/CAPEX по годам:"
    assert len(lines) == 4  # header + 2025/2026/2027, not a single-year answer

    assert "2025 (Факт): OPEX = 8 000" in reply
    assert "CAPEX = 4 000" in reply
    assert "2026 (План): OPEX = 9 000" in reply
    assert "2027 (План): OPEX = 10 000" in reply
    assert "OPEX/CAPEX ~ 2.0:1" in reply
    # First year has no prior year to compare against; later years do.
    assert "Δ" not in lines[1]
    assert "Δ +12.5%" in lines[2]  # 8000 -> 9000 and 4000 -> 4500 both +12.5%


def test_chat_answer_sravni_multi_year_without_explicit_capex_opex_keyword(multi_year_form):
    """'сравни' alone (no literal 'capex'/'opex' in the question) still means
    OPEX-vs-CAPEX in this function -- multi-year phrasing must route it to
    the same per-year table, not fall through to the single-year branch."""
    meta, _, _, combined = multi_year_form
    reply = chat_answer("сравни планы за все года", combined, meta)
    assert reply is not None
    assert reply.startswith("Разбивка OPEX/CAPEX по годам:")
    assert "2025" in reply and "2026" in reply and "2027" in reply


def test_chat_answer_multi_year_explicit_range(multi_year_form):
    meta, _, _, combined = multi_year_form
    reply = chat_answer("сравни capex с 2026 до 2027", combined, meta)
    assert reply is not None
    assert "2025" not in reply  # outside the requested range
    assert "2026" in reply and "2027" in reply


def test_chat_answer_single_year_capex_unaffected_by_multi_year_change(multi_year_form):
    """Existing single-year behavior must be untouched when no multi-year
    phrasing is present."""
    meta, _, _, combined = multi_year_form
    reply = chat_answer("сколько capex в 2027", combined, meta)
    assert reply is not None
    assert "Разбивка" not in reply
    assert "5 000" in reply


# --- chat_answer year-availability branches -------------------------------------
def _no_target_year_workbook(path):
    """Subsidiary B-style single sheet: campaign year 2027, plan_2027 AND fact_2025 —
    but no 2025 PLAN and no 2026 data of any kind, so a question about 2026 has
    no column to hit while 2025 only has факт (no план/прогноз) -- the exact
    shape that used to trigger the plan_/forecast_-only bug on year_col."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws, campaign_year=2027)
    hdr = 9
    _write_headers(ws, hdr, {
        9: "Контрагент", 11: "План за 2027 год", 12: "2025 год факт",
    })
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-OP-01"); ws.cell(row=r, column=4, value="Оборудование")
    r += 1  # Subsidiary B style: item name at category level, no subcategory
    ws.cell(row=r, column=7, value="Серверы"); ws.cell(row=r, column=9, value="Dell")
    ws.cell(row=r, column=11, value=9000); ws.cell(row=r, column=12, value=8500)
    r += 1
    ws.cell(row=r, column=7, value="СХД"); ws.cell(row=r, column=9, value="HP")
    ws.cell(row=r, column=11, value=800); ws.cell(row=r, column=12, value=750)
    ws.cell(row=r + 1, column=3, value="Итого"); ws.cell(row=r + 1, column=11, value=9800)
    wb.save(path)
    return path


def test_chat_answer_precise_no_data_for_missing_year(tmp_path):
    meta, _, _, combined = _run_pipeline(_no_target_year_workbook(tmp_path / "subb.xlsx"))
    reply = chat_answer("план за 2026", combined, meta)
    assert reply is not None
    assert "Данных за 2026 год в этом файле нет" in reply
    assert "2027" in reply  # lists the year that IS present
    assert "не нашёл" not in reply.lower()  # NOT the generic fallback


def test_chat_answer_lists_available_years_for_history_question(tmp_path):
    meta, _, _, combined = _run_pipeline(_no_target_year_workbook(tmp_path / "subb2.xlsx"))
    reply = chat_answer("есть ли бюджет за прошлые года", combined, meta)
    assert reply is not None
    assert "2027" in reply
    assert "план" in reply.lower()  # sourced from year_trend()


# --- year_col priority: plan_ > fact_ > forecast_ (regression) ------------------
# A year with ONLY факт data (no план/прогноз) is very common -- prior-year
# actuals with no committed plan for that year. Before this fix, year_col only
# ever checked plan_/forecast_, so asking about such a year fell through to
# "Данных за {year} год в этом файле нет" while the SAME reply then listed that
# year as available in "Доступные годы" -- a self-contradiction (year_col was
# None but available_years() correctly includes fact_ years).
def test_chat_answer_resolves_fact_only_year(tmp_path):
    meta, _, _, combined = _run_pipeline(_no_target_year_workbook(tmp_path / "subb3.xlsx"))
    reply = chat_answer("сколько потратили в 2025 году", combined, meta)
    assert reply is not None
    assert "Данных за 2025 год в этом файле нет" not in reply  # no more self-contradiction
    assert "Факт на 2025 год" in reply
    assert "9 250" in reply  # 8500 + 750


def test_answer_query_resolves_fact_only_year(tmp_path):
    _, sheets = parse_workbook(_no_target_year_workbook(tmp_path / "subb4.xlsx"))
    df, stats = sheets["OPEX"]["df"], sheets["OPEX"]["stats"]
    meta = {"unit": 1000}
    reply = answer_query("что по 2025 году", df, stats, meta)
    assert reply is not None
    assert reply.startswith("Факт на 2025 год")
    assert "9 250" in reply


def test_year_col_priority_prefers_plan_over_fact(tmp_path):
    """When a year has BOTH план and факт (e.g. 2026 in full_form), plan wins."""
    from core.budget_auditor.analysis import _resolve_year_col

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OPEX"
    _write_meta(ws, campaign_year=2026)
    hdr = 9
    _write_headers(ws, hdr, {9: "Контрагент", 10: "2026 год факт", 11: "План за 2026 год"})
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-OP-01"); ws.cell(row=r, column=4, value="Связь")
    r += 1
    ws.cell(row=r, column=7, value="Услуги связи"); ws.cell(row=r, column=8, value="Интернет")
    ws.cell(row=r, column=10, value=3000); ws.cell(row=r, column=11, value=3300)
    ws.cell(row=r + 1, column=3, value="Итого"); ws.cell(row=r + 1, column=11, value=3300)
    path = tmp_path / "both.xlsx"
    wb.save(path)
    _, sheets = parse_workbook(path)
    df = sheets["OPEX"]["df"]
    assert _resolve_year_col(df, 2026) == "plan_2026"


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


# --- conversation history pairing for the AI SQL fallback (Prompt 3) ------------
def test_recent_history_exchanges_pairs_and_skips_file_drop(app_helpers):
    import streamlit as st

    st.session_state["file_chat_history"] = [
        {"role": "user", "content": "вопрос 1"},
        {"role": "assistant", "content": "ответ 1"},
        {"role": "user", "content": "📎 file.xlsx"},
        {"role": "assistant", "kind": "analysis", "name": "file.xlsx", "bytes": b"", "uid": "x"},
        {"role": "user", "content": "вопрос 2"},
        {"role": "assistant", "content": "ответ 2"},
    ]
    try:
        pairs = app_helpers["_recent_history_exchanges"]()
    finally:
        st.session_state.pop("file_chat_history", None)

    # The file-drop pair must not appear as a (possibly malformed) exchange.
    assert pairs == [("вопрос 1", "ответ 1"), ("вопрос 2", "ответ 2")]


def test_recent_history_exchanges_caps_to_three(app_helpers):
    import streamlit as st

    st.session_state["file_chat_history"] = [
        entry
        for i in range(1, 6)
        for entry in (
            {"role": "user", "content": f"вопрос {i}"},
            {"role": "assistant", "content": f"ответ {i}"},
        )
    ]
    try:
        pairs = app_helpers["_recent_history_exchanges"]()
    finally:
        st.session_state.pop("file_chat_history", None)

    assert pairs == [("вопрос 3", "ответ 3"), ("вопрос 4", "ответ 4"), ("вопрос 5", "ответ 5")]


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


# --- quarterly reconciliation flag ----------------------------------------------
# Mirrors test_budget_audit.py's quarterly_reconciliation tests on the
# budget_auditor schema: dynamic plan_<year>_qN columns, campaign year 2026.
def _quarterly_workbook(path):
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
        # (item, annual, q1, q2, q3, q4)
        ("Точно", 100_000, 25_000, 25_000, 25_000, 25_000),          # exact -> clean
        ("Критично", 100_000, 23_500, 23_500, 23_500, 23_500),       # 6% off -> critical
        ("Незначительно", 100_000, 24_500, 24_500, 24_500, 24_500),  # 2% off -> minor
        ("Без кварталов", 50_000, None, None, None, None),           # incomplete -> skipped
    ]
    r = hdr + 1
    ws.cell(row=r, column=3, value="IT-OP-01")
    ws.cell(row=r, column=4, value="Услуги")
    for item, annual, *quarters in rows:
        r += 1
        ws.cell(row=r, column=7, value="Услуги")
        ws.cell(row=r, column=8, value=item)
        ws.cell(row=r, column=9, value="Вендор")
        ws.cell(row=r, column=10, value="обоснование есть")
        ws.cell(row=r, column=11, value=annual)
        for i, q in enumerate(quarters):
            if q is not None:
                ws.cell(row=r, column=12 + i, value=q)
    ws.cell(row=r + 1, column=3, value="Итого")
    ws.cell(row=r + 1, column=11, value=350_000)
    wb.save(path)
    return path


@pytest.fixture
def quarterly_combined(tmp_path):
    _, _, _, combined = _run_pipeline(_quarterly_workbook(tmp_path / "quarters.xlsx"))
    return combined


def test_quarterly_mismatch_flags_only_offending_rows(quarterly_combined):
    flagged = quarterly_combined[quarterly_combined["quarterly_mismatch_flag"]]
    assert set(flagged["item_name"]) == {"Критично", "Незначительно"}

    crit = flagged[flagged["item_name"] == "Критично"].iloc[0]
    assert crit["quarterly_sum"] == 94_000
    assert crit["quarterly_gap_abs"] == 6_000
    assert crit["quarterly_gap_pct"] == pytest.approx(6.0)
    assert crit["quarterly_severity"] == "critical"

    minor = flagged[flagged["item_name"] == "Незначительно"].iloc[0]
    assert minor["quarterly_gap_pct"] == pytest.approx(2.0)
    assert minor["quarterly_severity"] == "minor"


def test_quarterly_mismatch_ignores_exact_and_incomplete(quarterly_combined):
    exact = quarterly_combined[quarterly_combined["item_name"] == "Точно"].iloc[0]
    assert not exact["quarterly_mismatch_flag"]
    assert exact["quarterly_gap_pct"] == pytest.approx(0.0)

    skipped = quarterly_combined[quarterly_combined["item_name"] == "Без кварталов"].iloc[0]
    assert not skipped["quarterly_mismatch_flag"]
    assert pd.isna(skipped["quarterly_gap_pct"])
    assert pd.isna(skipped["quarterly_severity"])


def test_quarterly_flag_absent_columns_never_flags(full_form):
    # The full-form fixture has no quarterly columns at all: the flag columns
    # must still exist (all False / NA) so the UI never KeyErrors on them.
    _, _, _, combined = full_form
    assert "quarterly_mismatch_flag" in combined.columns
    assert not combined["quarterly_mismatch_flag"].any()
    assert combined["quarterly_gap_pct"].isna().all()


def test_quarterly_view_renders_severity_in_russian(quarterly_combined, app_helpers):
    subset = quarterly_combined[quarterly_combined["quarterly_mismatch_flag"]]
    disp, fmt = app_helpers["build_quarterly_view"](subset, "plan_2026")
    out = app_helpers["_format_eager"](disp, fmt)
    assert "критично" in out["Серьёзность"].tolist()
    assert "незначительно" in out["Серьёзность"].tolist()
    assert "94 000" in out["Σ кварталов"].tolist()
