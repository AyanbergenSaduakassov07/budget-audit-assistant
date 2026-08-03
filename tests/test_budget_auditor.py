"""Tests for core.budget_auditor (parser + analysis) and app.py display formatters.

Fixtures are synthetic workbooks that reproduce the structural variants seen in
the real subsidiary forms (see parser.py module docstring):
  - Subsidiary A style: факт + обоснование columns present
  - Subsidiary B style: no prior-year columns, no обоснование, item names at category level
  - newer-template style: reordered headers ("ФАКТ за 2025 год"), dual plan-year columns
"""

import re
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
    MAX_SHEET_ROWS,
)
from core.budget_auditor.analysis import (
    workbook_summary_text,
    year_trend_text,
    chat_answer,
    answer_query,
    format_money,
    _search_tokens,
    _keyword_search_mask,
)

ROOT = Path(__file__).resolve().parent.parent


# --- markdown-table assertions --------------------------------------------------
# Multi-row deterministic answers are markdown tables, not "\n"-joined lines:
# st.write() collapses single newlines inside a paragraph, so the old format
# rendered as one run-on wall of text in the chat.
def _table_rows(reply):
    """Data rows (header and |---| separator excluded) as lists of cells.
    Splits on unescaped pipes only, then unescapes, so a cell containing a
    literal "|" reads back as one cell."""
    rows = []
    for i, line in enumerate(reply.splitlines()):
        if not line.startswith("|") or i < 2:
            continue
        cells = re.split(r"(?<!\\)\|", line)[1:-1]
        rows.append([c.strip().replace("\\|", "|") for c in cells])
    return rows


def _assert_is_markdown_table(reply, n_cols=None):
    assert reply.startswith("|"), reply
    lines = reply.splitlines()
    assert lines[1].startswith("|---"), reply
    if n_cols is not None:
        assert lines[1] == "|" + "---|" * n_cols, reply
    assert _table_rows(reply), "table has no data rows"


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


# --- oversized-file guard (row-count sanity check before parsing) --------------
def test_parse_workbook_rejects_oversized_sheet(tmp_path, monkeypatch):
    """A sheet whose ws.max_row exceeds MAX_SHEET_ROWS must be rejected with a
    clear Russian message BEFORE any per-sheet parsing runs -- a pathological
    file (e.g. a full transactional export, not a ИТ-бюджет form) must fail fast,
    not after fully parsing. Faking max_row via a class-level monkeypatch
    avoids actually writing 5000+ rows to build the fixture."""
    import core.budget_auditor.parser as parser_module
    from openpyxl.worksheet.worksheet import Worksheet

    path = _full_form_workbook(tmp_path / "oversized.xlsx")

    def _must_not_run(*a, **k):
        raise AssertionError("full parse must not run once the row-count guard fires")

    monkeypatch.setattr(parser_module, "read_metadata", _must_not_run)
    monkeypatch.setattr(parser_module, "parse_sheet", _must_not_run)
    monkeypatch.setattr(Worksheet, "max_row", property(lambda self: MAX_SHEET_ROWS + 1))

    with pytest.raises(ValueError, match="Файл слишком большой для обработки"):
        parse_workbook(path)


def test_parse_workbook_accepts_sheet_at_the_row_limit(tmp_path, monkeypatch):
    """The guard must be a strict '>' -- a sheet exactly at MAX_SHEET_ROWS is
    still accepted, not rejected off-by-one."""
    from openpyxl.worksheet.worksheet import Worksheet

    path = _full_form_workbook(tmp_path / "at_limit.xlsx")
    monkeypatch.setattr(Worksheet, "max_row", property(lambda self: MAX_SHEET_ROWS))

    meta, sheets = parse_workbook(path)
    assert meta["company"] == "ТОО Тест-Дочка"


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
    assert reply.startswith("| Статья | Тип | Ветка | Сумма |")
    assert "|---" in reply
    assert len(_table_rows(reply)) == 3  # exactly 3 data rows, header excluded
    assert "Топ-3 самых крупных статей по всему файлу." in reply
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
    assert reply.startswith("| Статья | Сумма |")  # single sheet: no cost_type/branch cols
    assert "|---" in reply
    assert len(_table_rows(reply)) == 1  # exactly 1 data row
    assert "Топ-1 самых крупных статей." in reply


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


def test_chat_answer_bails_out_on_reformat_only_followup():
    """Live bug: "в виде таблицы" as a follow-up carries no business keyword
    of its own -- chat_answer() is stateless and can't know it means "reformat
    the previous answer". Tokenizing it ("виде", "таблицы") for the generic
    keyword-search fallback produced a coincidental substring match against
    item names containing "телевидения" ("виде" is a substring of "видения"),
    returning an unrelated top-items list. Must bail to None instead, so
    app.answer_question() routes to the history-aware LLM fallback."""
    combined = pd.DataFrame({
        "branch": ["(головная)"],
        "cost_type": ["OPEX"],
        "item_name": ["Услуги кабельного телевидения"],
        "category": ["Связь"],
        "subcategory": [None],
        "counterparty": [None],
        "plan_year": [2027],
        "plan_2027": [15890.0],
    })
    meta = {"unit": 1000}
    assert chat_answer("в виде таблицы", combined, meta) is None
    assert chat_answer("покажи списком", combined, meta) is None
    assert chat_answer("сколько CAPEX в 2027", combined, meta) is not None  # unaffected


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
    assert reply.startswith("| Год | OPEX | Δ OPEX % | CAPEX | Δ CAPEX % | OPEX/CAPEX |")
    assert "|---" in reply
    rows = _table_rows(reply)
    assert len(rows) == 3  # 2025/2026/2027, not a single-year answer

    assert rows[0][:2] == ["2025 (Факт)", "8 000 тыс."]
    assert rows[0][3] == "4 000 тыс."
    assert rows[1][1] == "9 000 тыс."
    assert rows[2][1] == "10 000 тыс."
    assert [r[-1] for r in rows] == ["2.0:1"] * 3
    # First year has no prior year to compare against; later years do.
    assert rows[0][2] == rows[0][4] == "—"
    assert rows[1][2] == rows[1][4] == "+12.5%"  # 8000 -> 9000 and 4000 -> 4500


def test_chat_answer_sravni_multi_year_without_explicit_capex_opex_keyword(multi_year_form):
    """'сравни' alone (no literal 'capex'/'opex' in the question) still means
    OPEX-vs-CAPEX in this function -- multi-year phrasing must route it to
    the same per-year table, not fall through to the single-year branch."""
    meta, _, _, combined = multi_year_form
    reply = chat_answer("сравни планы за все года", combined, meta)
    assert reply is not None
    assert reply.startswith("| Год |")
    assert "Разбивка OPEX/CAPEX по годам." in reply
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


# --- markdown-table rendering of multi-row answers -------------------------------
# Live bug: every one of these built its answer as "\n".join(lines) with a label
# line on top. st.write() renders that through standard markdown, which collapses
# a single "\n" inside a paragraph -- so "сравни бюджет за все года" crammed all
# 7 years into one run-on paragraph and "какая ветка тратит больше" did the same
# for the branch totals. Each now returns a real markdown table (explanatory
# sentence moved to its own block AFTER the table, never glued to the last row).
def test_multi_year_breakdown_renders_as_table(multi_year_form):
    meta, _, _, combined = multi_year_form
    reply = chat_answer("сравни бюджет за все года", combined, meta)
    _assert_is_markdown_table(reply, n_cols=6)
    assert reply.splitlines()[0] == "| Год | OPEX | Δ OPEX % | CAPEX | Δ CAPEX % | OPEX/CAPEX |"
    assert reply.endswith("Разбивка OPEX/CAPEX по годам.")


def test_branch_comparison_renders_as_table(full_form):
    meta, _, _, combined = full_form
    reply = chat_answer("какая ветка тратит больше", combined, meta)
    _assert_is_markdown_table(reply, n_cols=2)
    assert reply.splitlines()[0] == "| Подразделение | Сумма |"
    assert _table_rows(reply)[0][0] == "ГО"  # highest spender still ranked first


def test_missing_justification_by_branch_renders_as_table(full_form):
    meta, _, _, combined = full_form
    reply = chat_answer("какие статьи без обоснования", combined, meta)
    _assert_is_markdown_table(reply, n_cols=2)
    assert reply.splitlines()[0] == "| Подразделение | Статей без обоснования |"
    assert sum(int(r[1]) for r in _table_rows(reply)) == 3
    assert reply.endswith("Всего без обоснования: 3 статей.")


def test_workbook_top_items_render_as_table(full_form):
    meta, _, _, combined = full_form
    reply = chat_answer("топ самых дорогих статей", combined, meta)
    _assert_is_markdown_table(reply, n_cols=4)
    assert reply.splitlines()[0] == "| Статья | Тип | Ветка | Сумма |"
    assert _table_rows(reply)[0][0] == "Серверы"


def test_sheet_top_items_render_as_table(full_form):
    meta, _, flagged, _ = full_form
    df, stats = flagged["ГО CAPEX"]["df"], flagged["ГО CAPEX"]["stats"]
    reply = answer_query("самые крупные статьи", df, stats, meta)
    _assert_is_markdown_table(reply, n_cols=2)
    assert reply.splitlines()[0] == "| Статья | Сумма |"


def test_available_years_listing_renders_as_table(tmp_path):
    meta, _, _, combined = _run_pipeline(_no_target_year_workbook(tmp_path / "subb_tbl.xlsx"))
    reply = chat_answer("есть ли бюджет за прошлые года", combined, meta)
    _assert_is_markdown_table(reply, n_cols=3)
    assert reply.splitlines()[0] == "| Год | Тип | Сумма |"
    assert reply.endswith("Годы, представленные в файле.")


def test_unit_mismatch_warning_renders_as_table():
    fake = pd.DataFrame({
        "plan_2026": [100.0],
        "plan_2027": [120.0],
        "fact_2025": [9_000_000.0],  # entered in raw tenge, not тыс.
    })
    reply = year_trend_text(fake)
    _assert_is_markdown_table(reply, n_cols=3)
    assert reply.splitlines()[0] == "| Год | Тип | Сумма |"
    assert _table_rows(reply) == [["2025", "факт", "9 000 000"]]
    assert "единиц измерения" in reply  # explanation kept, just after the table


def test_table_cells_escape_pipes():
    """A pipe inside an item name would otherwise split the row into extra
    columns and break the whole table's alignment."""
    combined = pd.DataFrame({
        "branch": ["ГО|АФ"],
        "cost_type": ["OPEX"],
        "item_name": ["Связь"],
        "category": ["Связь"],
        "subcategory": [None],
        "counterparty": [None],
        "plan_year": [2026],
        "plan_2026": [1000.0],
    })
    reply = chat_answer("какая ветка тратит больше", combined, {"unit": 1000})
    _assert_is_markdown_table(reply, n_cols=2)
    assert r"ГО\|АФ" in reply           # escaped in the raw markdown
    assert _table_rows(reply)[0][0] == "ГО|АФ"  # ...and still one cell when parsed


# --- keyword search: AND across tokens, short tokens kept ------------------------
# Live bug: "Сколько потратили на ACM Digital?" returned an inflated total against a
# manually-verified true total of the correct total The len(t) > 3 cutoff dropped
# "acm" and kept only "digital", and tokens were OR'd across all four searched
# columns -- so any row merely mentioning "digital" anywhere counted as ACM
# Digital spend. No real ACM Digital workbook exists in this repo's fixtures
# (they're all built programmatically), so the shape is reconstructed here:
# two genuine rows plus three decoys that each satisfy the old matcher.
_ACM_DIGITAL_TRUE_TOTAL = 761_000.0


def _acm_digital_df():
    return pd.DataFrame({
        "branch": ["ГО"] * 5,
        "cost_type": ["OPEX"] * 5,
        "item_name": [
            "Сопровождение ИС",
            "Лицензии",
            "Digital signage для холла",   # decoy: "digital" in item_name
            "Аутсорсинг",                  # decoy: "acm" in counterparty only
            "Digital-маркетинг",           # decoy: "digital" in item_name
        ],
        "category": ["Услуги"] * 5,
        "subcategory": [None] * 5,
        "counterparty": [
            "ACM Digital",
            "ACM Digital",
            "ТОО Ромашка",
            "ACM Systems",
            "АО Медиа",
        ],
        "plan_year": [2026] * 5,
        "plan_2026": [500_000.0, 261_000.0, 3_000.0, 900.0, 763.5],
    })


def test_multiword_name_requires_all_tokens_in_one_column():
    """The over-count case. Only the two real ACM Digital rows may be summed;
    every decoy satisfies "digital" OR "acm" but not both in one column."""
    combined = _acm_digital_df()
    reply = chat_answer("Сколько потратили на ACM Digital?", combined, {"unit": 1000})
    assert reply is not None
    assert "Найдено 2 статей" in reply
    assert "Сопровождение ИС" in reply and "Лицензии" in reply
    for decoy in ("Digital signage", "Аутсорсинг", "Digital-маркетинг"):
        assert decoy not in reply
    # 500 000 + 261 493.5, formatted with a space thousands separator
    assert format_money(_ACM_DIGITAL_TRUE_TOTAL, 1000) in reply


def test_short_acronym_token_is_not_dropped():
    """"acm" is 3 chars -- the old len(t) > 3 rule discarded it, leaving the
    search to mean "anything containing 'digital'"."""
    assert _search_tokens("сколько потратили на acm digital") == ["acm", "digital"]
    assert "it" in _search_tokens("расходы на IT")
    assert "sap" in _search_tokens("сколько стоит SAP")


def test_search_tokens_drop_question_scaffolding_and_years():
    """AND-ing tokens only works if question words are gone: one stray
    "сколько" required in every match would sink the whole query."""
    tokens = _search_tokens("сколько всего потратили на серверы в 2026 году")
    assert tokens == ["серверы"]
    assert _search_tokens("сколько всего потратили в 2026 году") == []


def test_and_matching_does_not_span_two_columns():
    """A row whose category supplies one token and counterparty the other must
    NOT match -- that cross-column OR is what pulled in unrelated rows."""
    combined = pd.DataFrame({
        "branch": ["ГО"],
        "cost_type": ["OPEX"],
        "item_name": ["Прочее"],
        "category": ["ACM услуги"],      # supplies "acm"
        "subcategory": [None],
        "counterparty": ["ТОО Digital"],  # supplies "digital"
        "plan_year": [2026],
        "plan_2026": [5_000.0],
    })
    mask = _keyword_search_mask(combined, ["acm", "digital"])
    assert not mask.any()


def test_inflected_russian_token_is_dropped_not_and_ed(full_form):
    """Russian morphology: "сотовую" is not a substring of the item «Сотовая
    связь», so AND-ing the raw word forms would collapse the query to nothing.
    A token matching no row anywhere isn't a filter -- it's dropped, and the
    tokens that DO hit real rows are AND-ed."""
    meta, _, _, combined = full_form
    assert _search_tokens("сколько потратили на сотовую связь") == ["сотовую", "связь"]
    assert not _keyword_search_mask(combined, ["сотовую"]).any()  # matches nothing

    reply = chat_answer("сколько потратили на сотовую связь", combined, meta)
    assert reply is not None
    assert "Сотовая связь" in reply


def test_useful_token_and_ing_keeps_the_decoys_out():
    combined = _acm_digital_df()
    mask = _keyword_search_mask(combined, ["acm", "digital"])
    assert mask.sum() == 2
    assert combined[mask]["plan_2026"].sum() == _ACM_DIGITAL_TRUE_TOTAL


def test_all_tokens_unmatched_yields_no_rows():
    """Nothing in the data uses any of these words -- the branch must find
    nothing rather than widen back to an any-token match."""
    combined = _acm_digital_df()
    assert not _keyword_search_mask(combined, ["вертолёт", "буровая"]).any()


def test_single_token_search_unchanged(full_form):
    """One token means AND and OR are the same thing -- non-regression."""
    meta, _, _, combined = full_form
    reply = chat_answer("Телеком Плюс", combined, meta)
    assert reply is not None
    assert "7 200" in reply


def test_answer_query_single_sheet_uses_the_same_matcher(full_form):
    """The single-sheet function had the identical bug and shares the fix."""
    meta, _, flagged, _ = full_form
    df, stats = flagged["OPEX"]["df"], flagged["OPEX"]["stats"]
    combined_like = df.assign(counterparty=["ACM Digital", "ТОО Ромашка", "ACM Systems"][: len(df)])
    combined_like = combined_like.assign(
        item_name=["Сопровождение", "Digital signage", "Аутсорсинг"][: len(df)]
    )
    reply = answer_query("сколько потратили на ACM Digital", combined_like, stats, meta)
    assert reply is not None
    assert "Найдено 1 совпадающих статей" in reply
    assert "Digital signage" not in reply


# --- company / file identity -----------------------------------------------------
# Live bug: with no branch for it, "какая это компания" fell through to the LLM
# SQL fallback, which has no company column to query and returned a
# confidently invented, unrelated company name. It's a pure metadata lookup
# with one correct answer sitting in meta["company"] -- answered here so the
# LLM is never reached for it.
def _identity_df():
    return pd.DataFrame({
        "branch": ["ГО"],
        "cost_type": ["OPEX"],
        "item_name": ["Услуги связи"],
        "category": ["Связь"],
        "subcategory": [None],
        "counterparty": ["ТОО Альфа"],
        "plan_year": [2026],
        "plan_2026": [1000.0],
    })


def test_company_identity_answered_from_meta():
    combined = _identity_df()
    meta = {"unit": 1000, "company": "ТОО Тест-Дочка"}
    for question in [
        "какая компания",
        "какая это компания",
        "что за компания",
        "название компании",
        "чья это форма",
        "чей файл",
    ]:
        reply = chat_answer(question, combined, meta)
        assert reply == "Компания: ТОО Тест-Дочка (из шапки загруженной формы).", question


def test_company_identity_falls_back_explicitly_when_meta_has_no_company():
    """parser.py leaves company None when the form's header doesn't name it --
    a real structural gap. Say so; never guess."""
    combined = _identity_df()
    for company in (None, "", "   "):
        reply = chat_answer("какая это компания", combined, {"unit": 1000, "company": company})
        assert reply == "Компания: не определена по данным формы.", company
    # key absent entirely, not just None
    assert chat_answer("чья это форма", combined, {"unit": 1000}) == (
        "Компания: не определена по данным формы."
    )


def test_company_identity_answered_from_real_parsed_meta(full_form):
    meta, _, _, combined = full_form
    reply = chat_answer("какая это компания", combined, meta)
    assert reply == f"Компания: {meta['company']} (из шапки загруженной формы)."


def test_company_identity_never_reaches_the_token_search(full_form):
    """The token-search branch would match item names containing "компани" and
    answer with a spend total instead. The identity branch runs first."""
    meta, _, _, combined = full_form
    reply = chat_answer("какая это компания", combined, meta)
    assert "Найдено" not in reply
    assert "Суммарно" not in reply


def test_spending_question_mentioning_company_is_not_hijacked(full_form):
    """"компания" as a bare word in a spend question must NOT trigger the
    identity branch -- the triggers are phrases, not the noun alone."""
    meta, _, _, combined = full_form
    reply = chat_answer("сколько всего потратила компания в 2026 году", combined, meta)
    assert reply is not None
    assert not reply.startswith("Компания:")
    assert "17 000" in reply


# --- supplier concentration ------------------------------------------------------
# Live bug: the top-N branch triggered on a bare "риск" substring, which also
# matches "риск концентрации" -- so every concentration question came back as
# the Топ-5-most-expensive-items table, identically every time, and there was
# no concentration answer path in chat_answer() at all.
def _concentration_df(counterparties, amounts):
    n = len(counterparties)
    return pd.DataFrame({
        "branch": ["ГО"] * n,
        "cost_type": ["OPEX"] * n,
        "item_name": [f"Статья {i}" for i in range(n)],
        "category": ["Услуги"] * n,
        "subcategory": [None] * n,
        "counterparty": counterparties,
        "plan_year": [2026] * n,
        "plan_2026": amounts,
    })


def test_concentration_question_no_longer_hijacked_by_top_n():
    """"риск концентрации" must not fall into the top-N branch on the bare
    "риск" substring."""
    combined = _concentration_df(["ТОО Альфа", "ТОО Бета"], [30_000.0, 10_000.0])
    reply = chat_answer("Есть ли риск концентрации у одного поставщика?", combined, {"unit": 1000})
    assert reply is not None
    assert "Топ-" not in reply                       # NOT the top-N branch's answer
    assert "| Контрагент | Сумма | Доля % |" in reply
    assert "Статья 0" not in reply                   # item names belong to top-N, not here


def test_concentration_answer_names_top_supplier_and_share():
    combined = _concentration_df(["ТОО Альфа", "ТОО Бета"], [75_000.0, 25_000.0])
    reply = chat_answer("риск концентрации поставщиков", combined, {"unit": 1000})
    _assert_is_markdown_table(reply, n_cols=3)
    assert _table_rows(reply) == [
        ["ТОО Альфа", "75 000 тыс.", "75.0%"],
        ["ТОО Бета", "25 000 тыс.", "25.0%"],
    ]
    assert "Крупнейший контрагент -- ТОО Альфа: 75.0% плана на 2026 год" in reply
    assert "Это превышает порог концентрации в 40%" in reply


def test_concentration_answer_below_threshold_is_not_flagged():
    """Also covers the "самый крупный поставщик" phrasing, which carries no
    "риск"/"концентрац" wording of its own."""
    combined = _concentration_df(
        ["ТОО Альфа", "ТОО Бета", "ТОО Дельта"], [35_000.0, 35_000.0, 30_000.0]
    )
    reply = chat_answer("самый крупный поставщик", combined, {"unit": 1000})
    assert "Крупнейший контрагент -- ТОО Альфа: 35.0% плана на 2026 год" in reply
    assert "Это не превышает порог концентрации в 40%" in reply


def test_concentration_answer_excludes_compound_counterparty_from_top_supplier():
    """Mirrors _concentration()'s own coverage: a cell naming two legal
    entities ("ТОО Бета, АО Гамма") holds the largest share here, but must
    never be NAMED as the single top supplier -- attributing its combined
    amount to one company would invent a risk finding. It still shows up in
    the breakdown for visibility."""
    combined = _concentration_df(
        ["ТОО Альфа", "ТОО Бета, АО Гамма", "ТОО Дельта"],
        [30_000.0, 90_000.0, 10_000.0],
    )
    reply = chat_answer("зависимость от одного поставщика", combined, {"unit": 1000})
    _assert_is_markdown_table(reply, n_cols=3)

    rows = _table_rows(reply)
    assert rows[0] == ["несколько контрагентов в одной строке", "90 000 тыс.", "69.2%"]
    # 30 000 of 130 000 -- the compound row's 69.2% is NOT attributed to it.
    assert "Крупнейший контрагент -- ТОО Альфа: 23.1% плана на 2026 год" in reply
    assert "Это не превышает порог концентрации в 40%" in reply
    assert "не приписаны ни одному поставщику" in reply


def test_concentration_answer_without_counterparty_data_says_so():
    combined = _concentration_df([None, None], [30_000.0, 10_000.0])
    reply = chat_answer("риск концентрации", combined, {"unit": 1000})
    assert "Контрагенты в этом файле не заполнены" in reply
    assert "|" not in reply  # no fabricated table


def test_concentration_answer_all_compound_rows_refuses_to_name_a_supplier():
    combined = _concentration_df(["ТОО Альфа, АО Бета"], [40_000.0])
    reply = chat_answer("риск концентрации", combined, {"unit": 1000})
    _assert_is_markdown_table(reply, n_cols=3)
    assert "Крупнейший контрагент" not in reply
    assert "Ни одну строку нельзя отнести к одному поставщику" in reply


def test_concentration_answer_without_plan_column_says_so():
    combined = _concentration_df(["ТОО Альфа"], [40_000.0]).drop(columns=["plan_2026"])
    reply = chat_answer("риск концентрации", combined, {"unit": 1000})
    assert "нет колонки плана за год кампании" in reply


def test_plain_risk_question_still_reaches_top_n(full_form):
    """Non-regression for the narrowed trigger: "риск" with no supplier/
    concentration wording still means "rank the priciest items"."""
    meta, _, _, combined = full_form
    reply = chat_answer("какие статьи в зоне риска", combined, meta)
    assert reply is not None
    assert reply.splitlines()[0] == "| Статья | Тип | Ветка | Сумма |"
    assert "самых крупных статей по всему файлу." in reply


def test_concentration_threshold_shared_with_narrator_bridge():
    """The chat answer quotes the threshold back to the user; it must be the
    same constant _concentration() flags on, not a second hardcoded 40."""
    from core.budget_auditor import analysis, narrator_bridge

    assert analysis._CONCENTRATION_THRESHOLD_PCT is narrator_bridge._CONCENTRATION_THRESHOLD_PCT


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
