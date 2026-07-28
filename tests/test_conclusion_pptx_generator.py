"""Tests for core.conclusion_pptx_generator — the .pptx companion to
conclusion_generator's .docx, built from the same build_audit_result() shape.

Mirrors tests/test_narrator_bridge.py's conclusion-docx coverage
(test_conclusion_docx_renders) but for the pptx path, plus the graceful-
degradation cases specific to this generator (missing narrative, no
combined-data fields for the chart slides).
"""

from pptx import Presentation

from core.budget_auditor.narrator_bridge import build_audit_result
from core.conclusion_pptx_generator import generate_pptx_conclusion
from tests.test_budget_auditor import _quarterly_workbook, _run_pipeline


def _n_slides(prs: Presentation) -> int:
    return len(prs.slides._sldIdLst)


def test_generate_pptx_conclusion_produces_valid_deck(tmp_path):
    """Normal audit_result -> an openable .pptx with one slide per populated
    section: title, summary, CAPEX/OPEX, year trend, top-risk (concentration
    risk escalates this quarterly fixture's single sheet to one АТК row) --
    five slides, no executive_summary since build_audit_result() alone never
    attaches a "narrative" key (only narrate_audit() does)."""
    meta, _, _, combined = _run_pipeline(_quarterly_workbook(tmp_path / "quarters.xlsx"))
    result = build_audit_result(meta, combined)

    out = generate_pptx_conclusion(result, str(tmp_path / "заключение.pptx"))
    data = (tmp_path / "заключение.pptx").read_bytes()
    assert out.endswith(".pptx")
    assert data[:2] == b"PK" and len(data) > 5_000  # a real zip/pptx, not a stub

    prs = Presentation(out)
    assert _n_slides(prs) == 5


def test_generate_pptx_conclusion_default_output_path(tmp_path, monkeypatch):
    """No output_path -> writes next to cwd using the same
    {entity}_zakluchenie_{year}.pptx convention as the .docx generator."""
    monkeypatch.chdir(tmp_path)
    meta, _, _, combined = _run_pipeline(_quarterly_workbook(tmp_path / "quarters.xlsx"))
    result = build_audit_result(meta, combined)

    out = generate_pptx_conclusion(result)
    assert out.endswith("_zakluchenie_2026.pptx")
    assert Presentation(out) is not None


def test_missing_executive_summary_skips_closing_slide_gracefully(tmp_path):
    """No "narrative" key at all (the normal build_audit_result() shape) ->
    slide 6 is simply absent, never a slide showing the text "None"."""
    minimal = {"entity": "ТОО Тест", "budget_year": 2026, "summary": {}}
    out = generate_pptx_conclusion(minimal, str(tmp_path / "out.pptx"))
    prs = Presentation(out)

    # title + summary only: no all_rows -> no chart/top-risk data either.
    assert _n_slides(prs) == 2
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                assert "None" not in shape.text_frame.text


def test_present_executive_summary_adds_closing_slide(tmp_path):
    with_narrative = {
        "entity": "ТОО Тест",
        "budget_year": 2026,
        "summary": {},
        "narrative": {"executive_summary": "Итоговый вывод по бюджету.", "risk_level": "высокий"},
    }
    out = generate_pptx_conclusion(with_narrative, str(tmp_path / "out.pptx"))
    prs = Presentation(out)
    assert _n_slides(prs) == 3  # title + summary + closing

    texts = [s.text_frame.text for slide in prs.slides for s in slide.shapes if s.has_text_frame]
    assert any("Итоговый вывод по бюджету." in t for t in texts)
    assert any("высокий" in t for t in texts)


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


def test_no_combined_data_omits_chart_slides_without_raising(tmp_path):
    """No all_rows in this dict shape (e.g. a hand-built audit_result) ->
    CAPEX/OPEX and year-trend slides are simply omitted, not a crash."""
    audit_result = {
        "entity": "ТОО Без Данных",
        "budget_year": 2026,
        "summary": {"atk_count": 0, "clarification_count": 0, "approved_count": 0, "total_rows": 0},
        "findings": {},
    }
    out = generate_pptx_conclusion(audit_result, str(tmp_path / "out.pptx"))
    prs = Presentation(out)
    assert _n_slides(prs) == 2  # title + summary only
    for slide in prs.slides:
        assert not any(shape.has_chart for shape in slide.shapes if hasattr(shape, "has_chart"))


def test_empty_audit_result_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = generate_pptx_conclusion({})
    assert Presentation(out) is not None
