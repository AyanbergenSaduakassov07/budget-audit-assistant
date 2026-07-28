"""Tests for core.conclusion_pptx_generator — the .pptx companion to
conclusion_generator's .docx, built from the same build_audit_result() shape.

Slide order under test: Title -> Executive Summary -> Резюме (KPI) ->
Проблемные и рискованные статьи (risk overview) -> CAPEX vs OPEX ->
Динамика по годам -> Концентрация поставщиков (conditional) -> top-risk
item tables (АТК / уточнение, each conditional, each its own slide).
"""

import pandas as pd
import pytest
from pptx import Presentation

from core.budget_auditor.narrator_bridge import build_audit_result
from core.conclusion_generator import _atk_rows, _clarification_rows
from core.conclusion_pptx_generator import generate_pptx_conclusion
from tests.test_budget_auditor import _quarterly_workbook, _run_pipeline


def _n_slides(prs: Presentation) -> int:
    return len(prs.slides._sldIdLst)


def _slide_titles(prs: Presentation) -> list[str]:
    """The ordered list of each slide's title textbox text (always the first
    shape on every content slide -- see _add_title's call order)."""
    return [next(iter(slide.shapes)).text_frame.text for slide in prs.slides]


def _slide_by_title(prs: Presentation, title: str):
    for slide in prs.slides:
        shapes = list(slide.shapes)
        if shapes and shapes[0].has_text_frame and shapes[0].text_frame.text == title:
            return slide
    return None


def _all_texts(slide) -> list[str]:
    return [s.text_frame.text for s in slide.shapes if s.has_text_frame]


@pytest.fixture(scope="module")
def quarterly_result(tmp_path_factory):
    path = tmp_path_factory.mktemp("pptx") / "quarters.xlsx"
    meta, _, _, combined = _run_pipeline(_quarterly_workbook(path))
    return meta, combined


# --- overall deck / slide order --------------------------------------------------------
def test_generate_pptx_conclusion_produces_valid_deck(quarterly_result, tmp_path):
    """Normal audit_result, no narrative attached (build_audit_result() alone
    never sets one -- only narrate_audit() does): title, Резюме, risk
    overview, CAPEX/OPEX, year trend, concentration (100% single supplier
    here), and both top-risk tables (concentration escalates one row to
    АТК; the critical quarterly mismatch lands in уточнение) -- no
    executive-summary slide, in that order."""
    meta, combined = quarterly_result
    result = build_audit_result(meta, combined)

    out = generate_pptx_conclusion(result, str(tmp_path / "заключение.pptx"))
    data = (tmp_path / "заключение.pptx").read_bytes()
    assert out.endswith(".pptx")
    assert data[:2] == b"PK" and len(data) > 5_000  # a real zip/pptx, not a stub

    prs = Presentation(out)
    assert _n_slides(prs) == 8
    assert _slide_titles(prs)[1:] == [
        "Резюме",
        "Проблемные и рискованные статьи",
        "CAPEX vs OPEX",
        "Динамика по годам",
        "Концентрация поставщиков",
        "Выносится на АТК",
        "Требует уточнения",
    ]


def test_generate_pptx_conclusion_default_output_path(quarterly_result, tmp_path, monkeypatch):
    """No output_path -> writes next to cwd using the same
    {entity}_zakluchenie_{year}.pptx convention as the .docx generator."""
    monkeypatch.chdir(tmp_path)
    meta, combined = quarterly_result
    result = build_audit_result(meta, combined)

    out = generate_pptx_conclusion(result)
    assert out.endswith("_zakluchenie_2026.pptx")
    assert Presentation(out) is not None


def test_executive_summary_slide_immediately_after_title(tmp_path):
    with_narrative = {
        "entity": "ТОО Тест",
        "budget_year": 2026,
        "summary": {},
        "narrative": {"executive_summary": "Итоговый вывод по бюджету.", "risk_level": "высокий"},
    }
    out = generate_pptx_conclusion(with_narrative, str(tmp_path / "out.pptx"))
    prs = Presentation(out)
    titles = _slide_titles(prs)
    assert titles == [
        "Заключение по результатам экспертизы",
        "Исполнительное резюме",
        "Резюме",
        "Проблемные и рискованные статьи",
    ]

    exec_slide = _slide_by_title(prs, "Исполнительное резюме")
    texts = _all_texts(exec_slide)
    assert any("Итоговый вывод по бюджету." in t for t in texts)

    title_slide = prs.slides[0]
    assert any("высокий" in t for t in _all_texts(title_slide))


def test_missing_executive_summary_skips_slide_gracefully(tmp_path):
    """No "narrative" key at all (the normal build_audit_result() shape) ->
    no executive-summary slide, never a slide showing the text "None". The
    Резюме and risk-overview slides still always render (they mirror
    app.py's always-on Резюме/render_flags sections)."""
    minimal = {"entity": "ТОО Тест", "budget_year": 2026, "summary": {}}
    out = generate_pptx_conclusion(minimal, str(tmp_path / "out.pptx"))
    prs = Presentation(out)

    assert _slide_by_title(prs, "Исполнительное резюме") is None
    assert _n_slides(prs) == 3  # title, Резюме, risk overview
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                assert "None" not in shape.text_frame.text


def test_unknown_risk_level_treated_as_absent(tmp_path):
    audit_result = {
        "entity": "ТОО Тест",
        "budget_year": 2026,
        "summary": {},
        "narrative": {"executive_summary": "", "risk_level": "неизвестно"},
    }
    out = generate_pptx_conclusion(audit_result, str(tmp_path / "out.pptx"))
    prs = Presentation(out)
    texts = [s.text_frame.text for slide in prs.slides for s in slide.shapes if s.has_text_frame]
    assert not any("неизвестно" in t for t in texts)


def test_no_combined_data_omits_chart_and_concentration_slides_without_raising(tmp_path):
    """No all_rows in this dict shape (e.g. a hand-built audit_result) ->
    CAPEX/OPEX, year-trend, and concentration slides are simply omitted,
    not a crash. Резюме and risk overview still render (all-zero/clean)."""
    audit_result = {
        "entity": "ТОО Без Данных",
        "budget_year": 2026,
        "summary": {"atk_count": 0, "clarification_count": 0, "approved_count": 0, "total_rows": 0},
        "findings": {},
    }
    out = generate_pptx_conclusion(audit_result, str(tmp_path / "out.pptx"))
    prs = Presentation(out)
    assert _n_slides(prs) == 3  # title, Резюме, risk overview
    for slide in prs.slides:
        assert not any(shape.has_chart for shape in slide.shapes if hasattr(shape, "has_chart"))


def test_empty_audit_result_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = generate_pptx_conclusion({})
    prs = Presentation(out)
    assert _n_slides(prs) == 3


# --- risk overview (Проблемные и рискованные статьи) ------------------------------------
def test_risk_overview_slide_counts_match_all_rows_flags(tmp_path):
    all_rows = pd.DataFrame({
        "high_change_flag": [True, False, False],
        "missing_justification_flag": [False, False, False],
        "high_price_flag": [True, True, False],
        "quarterly_mismatch_flag": [True, False, False],
        "quarterly_severity": ["critical", None, None],
        "anomalous_code_flag": [False, False, False],
    })
    audit_result = {"entity": "ТОО Р", "budget_year": 2026, "summary": {}, "all_rows": all_rows}

    # capex_opex_table()/year_trend() aren't meant for a schema this bare;
    # bypass recomputation with empty tables since this test only cares
    # about the risk-overview slide.
    out = generate_pptx_conclusion(
        audit_result, str(tmp_path / "out.pptx"), threshold_pct=20.0,
        capex_opex_df=pd.DataFrame(), year_trend_df=pd.DataFrame(),
    )
    prs = Presentation(out)
    texts = _all_texts(_slide_by_title(prs, "Проблемные и рискованные статьи"))

    assert "1" in texts  # high_change_flag: 1 True
    assert "Резкое изменение год-к-году (порог 20%)" in texts
    assert "✓" in texts  # missing_justification_flag and anomalous_code_flag: both 0
    assert "Все статьи имеют заполненное обоснование." in texts
    assert "2" in texts  # high_price_flag: 2 True
    assert "Топ-10% по стоимости (внутри листа)" in texts
    assert "Несоответствие по кварталам (критичных: 1)" in texts  # 1 mismatch, 1 critical
    assert "Все коды статей соответствуют формату (например IT-CA-01)." in texts


def test_risk_overview_slide_all_clean_shows_checkmarks(tmp_path):
    audit_result = {"entity": "ТОО Чисто", "budget_year": 2026, "summary": {}, "all_rows": pd.DataFrame()}
    out = generate_pptx_conclusion(audit_result, str(tmp_path / "out.pptx"))
    prs = Presentation(out)
    texts = _all_texts(_slide_by_title(prs, "Проблемные и рискованные статьи"))
    assert texts.count("✓") == 5
    assert "Квартальные суммы согласуются с годовым планом (допуск 1%)." in texts


# --- "0 согласовано" honesty caption ----------------------------------------------------
def test_zero_approved_caption_appears_for_structural_gap(tmp_path):
    """approved_count == 0 AND the form never had an «Обоснование» column at
    all -> the caption explains why, right under the executive summary."""
    audit_result = {
        "entity": "ТОО А", "budget_year": 2026,
        "summary": {"approved_count": 0},
        "all_rows": pd.DataFrame(columns=["counterparty"]),  # no "justification" column
        "narrative": {"executive_summary": "Резюме А.", "risk_level": ""},
    }
    out = generate_pptx_conclusion(audit_result, str(tmp_path / "out.pptx"))
    prs = Presentation(out)
    texts = _all_texts(_slide_by_title(prs, "Исполнительное резюме"))
    assert any("структурный пробел формы" in t for t in texts)


def test_zero_approved_caption_absent_when_justification_column_present(tmp_path):
    """approved_count == 0 but the form DOES have an «Обоснование» column --
    a genuine "nothing approved" outcome, not a structural gap -> no caption."""
    audit_result = {
        "entity": "ТОО Б", "budget_year": 2026,
        "summary": {"approved_count": 0},
        "all_rows": pd.DataFrame(columns=["counterparty", "justification"]),
        "narrative": {"executive_summary": "Резюме Б.", "risk_level": ""},
    }
    out = generate_pptx_conclusion(audit_result, str(tmp_path / "out.pptx"))
    prs = Presentation(out)
    texts = _all_texts(_slide_by_title(prs, "Исполнительное резюме"))
    assert not any("структурный пробел формы" in t for t in texts)


def test_zero_approved_caption_absent_when_approved_count_nonzero(tmp_path):
    """No justification column, but approved_count != 0 -> the "0
    согласовано" premise doesn't even hold, so no caption."""
    audit_result = {
        "entity": "ТОО В", "budget_year": 2026,
        "summary": {"approved_count": 3},
        "all_rows": pd.DataFrame(columns=["counterparty"]),
        "narrative": {"executive_summary": "Резюме В.", "risk_level": ""},
    }
    out = generate_pptx_conclusion(audit_result, str(tmp_path / "out.pptx"))
    prs = Presentation(out)
    texts = _all_texts(_slide_by_title(prs, "Исполнительное резюме"))
    assert not any("структурный пробел формы" in t for t in texts)


# --- CAPEX vs OPEX: totals stated plainly ----------------------------------------------
def test_capex_opex_slide_states_totals_plainly(tmp_path):
    capex_opex_df = pd.DataFrame({"Тип": ["CAPEX", "OPEX"], "План 2026": [500000.0, 300000.0]})
    audit_result = {"entity": "ТОО Х", "budget_year": 2026, "summary": {}, "all_rows": pd.DataFrame()}
    out = generate_pptx_conclusion(audit_result, str(tmp_path / "out.pptx"), capex_opex_df=capex_opex_df)
    prs = Presentation(out)
    texts = _all_texts(_slide_by_title(prs, "CAPEX vs OPEX"))
    assert any("CAPEX: 500 000 тыс. тг" in t and "OPEX: 300 000 тыс. тг" in t for t in texts)


# --- year trend: Δ% table, including forecast-to-forecast --------------------------------
def test_year_trend_slide_shows_forecast_to_forecast_delta(tmp_path):
    """Only forecast years present (no plan/fact history before the campaign
    year) -> the table must still show real deltas between forecast years,
    not leave the Δ% column blank just because there's no historical
    baseline."""
    year_trend_df = pd.DataFrame({
        "Год": [2028, 2029],
        "Тип": ["прогноз", "прогноз"],
        "Сумма": [100000.0, 120000.0],
        "unit_mismatch_suspected": [False, False],
    })
    audit_result = {"entity": "ТОО Прогноз", "budget_year": 2029, "summary": {}, "all_rows": pd.DataFrame()}
    out = generate_pptx_conclusion(audit_result, str(tmp_path / "out.pptx"), year_trend_df=year_trend_df)
    prs = Presentation(out)
    slide = _slide_by_title(prs, "Динамика по годам")
    table = next(s.table for s in slide.shapes if s.has_table)
    rows = [[table.cell(r, c).text for c in range(len(table.columns))] for r in range(len(table.rows))]

    assert rows[0] == ["Год", "Тип", "Сумма, тыс. тг", "Δ%"]
    assert rows[1] == ["2028", "прогноз", "100 000", "—"]  # no prior forecast to compare against
    assert rows[2] == ["2029", "прогноз", "120 000", "+20.0%"]  # forecast-to-forecast delta


def test_year_trend_slide_deltas_track_same_kind_only(tmp_path):
    """A план row and a факт row in the same year must not be diffed
    against each other -- Δ% is per kind, across years."""
    year_trend_df = pd.DataFrame({
        "Год": [2025, 2026, 2026],
        "Тип": ["факт", "план", "факт"],
        "Сумма": [90000.0, 110000.0, 100000.0],
        "unit_mismatch_suspected": [False, False, False],
    })
    audit_result = {"entity": "ТОО У", "budget_year": 2026, "summary": {}, "all_rows": pd.DataFrame()}
    out = generate_pptx_conclusion(audit_result, str(tmp_path / "out.pptx"), year_trend_df=year_trend_df)
    prs = Presentation(out)
    slide = _slide_by_title(prs, "Динамика по годам")
    table = next(s.table for s in slide.shapes if s.has_table)
    rows = {
        (table.cell(r, 0).text, table.cell(r, 1).text): table.cell(r, 3).text
        for r in range(1, len(table.rows))
    }
    assert rows[("2025", "факт")] == "—"       # first факт row, nothing before it
    assert rows[("2026", "план")] == "—"       # first план row, nothing before it
    assert rows[("2026", "факт")] == "+11.1%"  # vs. 2025 факт (90000 -> 100000), not vs. 2026 план


# --- supplier concentration --------------------------------------------------------------
def _audit_result_with_concentration(concentration_flag: bool, top_supplier_pct: float,
                                      breakdown: pd.DataFrame | None = None) -> dict:
    return {
        "entity": "ТОО К", "budget_year": 2026, "summary": {}, "all_rows": pd.DataFrame(),
        "findings": {"concentration": {
            "top_supplier": "ТОО Мегасервис",
            "top_supplier_pct": top_supplier_pct,
            "concentration_flag": concentration_flag,
            "supplier_breakdown": breakdown if breakdown is not None else pd.DataFrame(),
        }},
    }


def test_concentration_slide_present_when_flagged(tmp_path):
    out = generate_pptx_conclusion(
        _audit_result_with_concentration(True, 55.0), str(tmp_path / "out.pptx"),
    )
    prs = Presentation(out)
    slide = _slide_by_title(prs, "Концентрация поставщиков")
    assert slide is not None
    texts = _all_texts(slide)
    assert "ТОО Мегасервис" in texts
    assert "55.0% от общего бюджета" in texts


def test_concentration_slide_absent_when_not_flagged_and_low_share(tmp_path):
    out = generate_pptx_conclusion(
        _audit_result_with_concentration(False, 10.0), str(tmp_path / "out.pptx"),
    )
    prs = Presentation(out)
    assert _slide_by_title(prs, "Концентрация поставщиков") is None


def test_concentration_slide_present_when_share_material_even_without_flag(tmp_path):
    """>=25% share is shown even if it didn't cross the >40% flag threshold."""
    out = generate_pptx_conclusion(
        _audit_result_with_concentration(False, 30.0), str(tmp_path / "out.pptx"),
    )
    prs = Presentation(out)
    assert _slide_by_title(prs, "Концентрация поставщиков") is not None


def test_concentration_slide_shows_next_suppliers_table(tmp_path):
    breakdown = pd.DataFrame({
        "counterparty": ["ТОО Мегасервис", "ТОО Б", "ТОО В", "ТОО Г", "ТОО Д", "ТОО Е"],
        "total_amount": [550000.0, 100000.0, 90000.0, 80000.0, 70000.0, 60000.0],
        "share_pct": [55.0, 10.0, 9.0, 8.0, 7.0, 6.0],
    })
    out = generate_pptx_conclusion(
        _audit_result_with_concentration(True, 55.0, breakdown), str(tmp_path / "out.pptx"),
    )
    prs = Presentation(out)
    slide = _slide_by_title(prs, "Концентрация поставщиков")
    table = next(s.table for s in slide.shapes if s.has_table)
    assert len(table.rows) == 6  # header + next 5 suppliers, excluding the top one shown above
    assert table.cell(1, 0).text == "ТОО Б"
    assert table.cell(1, 2).text == "10.0%"


# --- top risk items: both АТК and уточнение tables ---------------------------------------
def test_clarification_slide_appears_when_atk_rows_empty(quarterly_result, tmp_path):
    """The quarterly fixture's real content is 0 growth-flag АТК rows and 1
    critical quarterly mismatch landing in уточнение -- forcing the
    concentration-driven АТК row off (so _atk_rows is truly empty) must
    still leave the уточнение table on its own slide, not disappear the
    whole "top risk" section."""
    meta, combined = quarterly_result
    result = build_audit_result(meta, combined)
    result["findings"]["concentration"]["concentration_flag"] = False
    result["findings"]["concentration"]["top_supplier_pct"] = 0.0
    assert _atk_rows(result) == []
    assert _clarification_rows(result) != []

    out = generate_pptx_conclusion(result, str(tmp_path / "out.pptx"))
    prs = Presentation(out)
    assert _slide_by_title(prs, "Выносится на АТК") is None
    clar_slide = _slide_by_title(prs, "Требует уточнения")
    assert clar_slide is not None
    table = next(s.table for s in clar_slide.shapes if s.has_table)
    assert table.cell(0, 0).text == "Лист"
