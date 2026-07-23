"""Tests for core.budget_auditor.narrator_bridge — the adapter between
budget_auditor's combined DataFrame and narrate_audit()'s expected input.

The never-raises guarantee matters here more than anywhere: narrative_opening()
feeds the FIRST text of the assistant message, so every failure mode must
return None (template fallback), never an exception or an error string.
"""

import pandas as pd
import pytest

from core.budget_auditor.narrator_bridge import build_audit_result, narrative_opening
from core.budget_narrator import _build_context
from tests.test_budget_auditor import _full_form_workbook, _quarterly_workbook, _run_pipeline


@pytest.fixture(scope="module")
def quarterly_result(tmp_path_factory):
    path = tmp_path_factory.mktemp("bridge") / "quarters.xlsx"
    meta, _, _, combined = _run_pipeline(_quarterly_workbook(path))
    return meta, combined


@pytest.fixture(scope="module")
def full_form_result(tmp_path_factory):
    path = tmp_path_factory.mktemp("bridge") / "full.xlsx"
    meta, _, _, combined = _run_pipeline(_full_form_workbook(path))
    return meta, combined


# --- adapter shape --------------------------------------------------------------
def test_adapter_buckets_on_quarterly_fixture(quarterly_result):
    meta, combined = quarterly_result
    result = build_audit_result(meta, combined)
    s = result["summary"]

    # No baseline year in the fixture -> no АТК; the two quarterly mismatches
    # land in уточнение; the clean rows are согласовано.
    assert s["atk_count"] == 0
    assert s["clarification_count"] == 2
    assert s["clarification_amount"] == 200_000
    assert s["approved_count"] == 2
    assert s["total_rows"] == 4
    assert s["total_amount"] == 350_000
    assert s["has_prior_year_comparison"] is False

    assert result["entity"] == "ТОО Тест-Дочка"
    assert result["budget_year"] == 2026
    assert len(result["findings"]["quarterly_mismatches"]) == 2
    assert result["findings"]["concentration"]["top_supplier"] == "Вендор"
    assert result["findings"]["concentration"]["top_supplier_pct"] == pytest.approx(100.0)


def test_adapter_buckets_on_full_form(full_form_result):
    meta, combined = full_form_result
    result = build_audit_result(meta, combined)
    s = result["summary"]

    assert s["atk_count"] == int(combined["high_change_flag"].sum())
    assert s["atk_count"] + s["clarification_count"] + s["approved_count"] == s["total_rows"]
    assert s["has_prior_year_comparison"] is True  # Subsidiary A-style sheet has факт baseline

    growth = result["findings"]["growth_flags"]
    assert list(growth.columns) == ["counterparty", "delta_pct"]
    assert len(growth) == s["atk_count"]


def test_adapter_output_passes_narrator_context_minimization(quarterly_result):
    """The adapter dict must flow through narrate_audit's own _build_context
    without raising, and the resulting prompt context must stay aggregate-only."""
    meta, combined = quarterly_result
    context = _build_context(build_audit_result(meta, combined))

    assert context["quarterly_mismatches_count"] == 2
    assert context["concentration_risk"]["top_supplier"] == "Вендор"
    for item in context["top_growth_items"]:
        assert set(item) == {"counterparty", "delta_pct"}  # names + pct, no amounts


def test_adapter_survives_empty_and_missing_columns():
    empty = build_audit_result({}, pd.DataFrame())
    assert empty["summary"]["total_rows"] == 0
    assert empty["findings"]["concentration"]["top_supplier"] == ""
    # And through the narrator's context builder too:
    _build_context(empty)

    no_flags = build_audit_result(None, pd.DataFrame({"item_name": ["x"]}))
    assert no_flags["summary"]["approved_count"] == 1


# --- never-raises guarantee ------------------------------------------------------
def test_narrative_opening_returns_none_without_key(quarterly_result, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    meta, combined = quarterly_result
    assert narrative_opening(meta, combined) is None


def test_narrative_opening_never_raises_on_garbage_input(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert narrative_opening(None, None) is None
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-no-network")
    # With a key set, garbage input must still degrade to None, not raise —
    # build_audit_result tolerates it and the API failure is swallowed.
    import core.budget_narrator as bn
    monkeypatch.setattr(bn, "_get_client", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert narrative_opening(None, None) is None


def test_narrative_opening_success_path(quarterly_result, monkeypatch):
    meta, combined = quarterly_result
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    import core.budget_narrator as bn
    monkeypatch.setattr(bn, "_get_client", lambda: object())
    monkeypatch.setattr(bn, "_generate_content", lambda client, prompt: (
        '{"executive_summary": "Бюджет 350 000 тыс. тг; 2 статьи с расхождением по кварталам.",'
        ' "risk_level": "средний", "atk_items": [], "clarification_items": [],'
        ' "key_risks": [], "note_on_prior_year": ""}'
    ))
    text = narrative_opening(meta, combined)
    assert text.startswith("Бюджет 350 000")
    assert "Уровень риска: средний." in text


def test_narrative_opening_model_error_falls_back(quarterly_result, monkeypatch):
    meta, combined = quarterly_result
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    import core.budget_narrator as bn
    monkeypatch.setattr(bn, "_get_client", lambda: object())
    monkeypatch.setattr(bn, "_generate_content", lambda client, prompt: "NOT JSON AT ALL")
    assert narrative_opening(meta, combined) is None
