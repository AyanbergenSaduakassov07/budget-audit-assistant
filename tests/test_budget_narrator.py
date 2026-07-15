"""Tests for core.budget_narrator — deterministic logic only, no live API cost.

Everything here monkeypatches _get_client/_generate_content rather than
calling Gemini, so it runs in the free `-m "not api"` suite. A live end-to-end
check (actually calling the model) would belong under @pytest.mark.api,
mirroring tests/test_query_engine.py's split — not added here since it wasn't
requested and would spend real quota.
"""

import json

import pandas as pd
import pytest

from core.budget_narrator import (
    _build_context,
    _format_delta_pct,
    _parse_narrative,
    _top_growth_items,
    narrate_audit,
)


def _fake_audit_result() -> dict:
    growth_flags = pd.DataFrame({
        "counterparty": ["A", "B", "C", "D", "E", "F"],
        "plan_annual": [182000, 87610, 198326, 300000, 400000, 500000],
        "delta_pct": [82.0, 776.1, float("inf"), 25.0, 60.0, 30.0],
    })
    quarterly_mismatches = pd.DataFrame({"article_code": ["X1"], "severity": ["critical"]})

    return {
        "entity": "AUDIT",
        "sheet_name": "AUDIT",
        "budget_year": 2027,
        "unit_scale": 1,
        "has_prior_year": True,
        "all_rows": pd.DataFrame(),
        "findings": {
            "quarterly_mismatches": quarterly_mismatches,
            "growth_flags": growth_flags,
            "concentration": {
                "supplier_breakdown": pd.DataFrame(),
                "concentration_flag": True,
                "top_supplier": "ТОО Мегасервис",
                "top_supplier_pct": 48.27,
                "total_single_source_amount": 198326.0,
                "total_single_source_pct": 25.17,
            },
            "missing_fields": pd.DataFrame(),
            "new_items": pd.DataFrame(),
        },
        "summary": {
            "atk_count": 3,
            "atk_amount": 467936.0,
            "clarification_count": 2,
            "clarification_amount": 205000.0,
            "approved_count": 1,
            "approved_amount": 115000.0,
            "total_rows": 6,
            "total_amount": 787936.0,
            "has_prior_year_comparison": True,
        },
        "has_critical": True,
    }


# --- _build_context: whitelisting -----------------------------------------------
def test_build_context_only_includes_whitelisted_fields():
    context = _build_context(_fake_audit_result())
    assert context["entity"] == "AUDIT"
    assert context["budget_year"] == 2027
    assert context["summary"]["atk_amount_tys_tg"] == 467936.0
    assert context["concentration_risk"]["top_supplier"] == "ТОО Мегасервис"
    assert context["quarterly_mismatches_count"] == 1
    assert context["has_prior_year_comparison"] is True

    # No raw per-item DataFrames or amounts must leak into the context.
    assert "all_rows" not in context
    assert "findings" not in context
    for item in context["top_growth_items"]:
        assert set(item) == {"counterparty", "delta_pct"}


def test_build_context_json_serializable():
    # narrate_audit puts this straight into json.dumps() for the prompt —
    # must not contain raw float("inf") or non-serializable objects.
    context = _build_context(_fake_audit_result())
    json.dumps(context, ensure_ascii=False)  # raises if not serializable


# --- _top_growth_items -----------------------------------------------------------
def test_top_growth_items_limits_to_five_sorted_desc():
    growth_flags = _fake_audit_result()["findings"]["growth_flags"]
    items = _top_growth_items(growth_flags)
    assert len(items) == 5
    # inf (row C) sorts first; the untouched 6th-highest (D, 25%) is dropped.
    assert items[0]["counterparty"] == "C"
    assert items[0]["delta_pct"] == "новая статья (нет базы для сравнения)"
    assert "D" not in [i["counterparty"] for i in items]


def test_top_growth_items_empty_dataframe():
    assert _top_growth_items(pd.DataFrame()) == []
    assert _top_growth_items(None) == []


# --- _format_delta_pct -------------------------------------------------------------
def test_format_delta_pct_normal_and_inf():
    assert _format_delta_pct(82.0) == "+82.0%"
    assert _format_delta_pct(float("inf")) == "новая статья (нет базы для сравнения)"
    assert _format_delta_pct(None) == "н/д"


# --- _parse_narrative --------------------------------------------------------------
def test_parse_narrative_valid_json():
    raw = json.dumps({
        "executive_summary": "Резюме.",
        "atk_items": [{"supplier": "A", "reason": "рост", "amount_tys_tg": 182000.0}],
        "clarification_items": [],
        "risk_level": "высокий",
        "key_risks": ["риск 1"],
        "note_on_prior_year": "Сравнение доступно.",
    }, ensure_ascii=False)
    parsed = _parse_narrative(raw)
    assert parsed["executive_summary"] == "Резюме."
    assert parsed["risk_level"] == "высокий"
    assert "error" not in parsed


def test_parse_narrative_strips_code_fences():
    raw = "```json\n" + json.dumps({"executive_summary": "ок"}) + "\n```"
    parsed = _parse_narrative(raw)
    assert parsed["executive_summary"] == "ок"


def test_parse_narrative_malformed_returns_fallback_with_error():
    parsed = _parse_narrative("this is not json at all")
    assert parsed["executive_summary"] == ""
    assert parsed["atk_items"] == []
    assert "error" in parsed


def test_parse_narrative_non_object_json_returns_fallback():
    parsed = _parse_narrative("[1, 2, 3]")
    assert "error" in parsed
    assert parsed["key_risks"] == []


# --- narrate_audit: full flow, API mocked out ---------------------------------------
def test_narrate_audit_merges_narrative_key(monkeypatch):
    canned_response = json.dumps({
        "executive_summary": "Три статьи требуют АТК.",
        "atk_items": [{"supplier": "ТОО Мегасервис", "reason": "рост", "amount_tys_tg": 198326.0}],
        "clarification_items": [],
        "risk_level": "высокий",
        "key_risks": ["концентрация поставщика"],
        "note_on_prior_year": "Сравнение доступно.",
    }, ensure_ascii=False)

    monkeypatch.setattr("core.budget_narrator._get_client", lambda: object())
    monkeypatch.setattr("core.budget_narrator._generate_content", lambda client, prompt: canned_response)

    audit_result = _fake_audit_result()
    result = narrate_audit(audit_result)

    assert result["entity"] == "AUDIT"  # original audit_result fields preserved
    assert result["narrative"]["executive_summary"] == "Три статьи требуют АТК."
    assert result["narrative"]["risk_level"] == "высокий"
    assert "error" not in result["narrative"]


def test_narrate_audit_handles_client_failure_without_crashing(monkeypatch):
    def _raise_client_error():
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    monkeypatch.setattr("core.budget_narrator._get_client", _raise_client_error)

    result = narrate_audit(_fake_audit_result())
    assert result["narrative"]["executive_summary"] == ""
    assert "error" in result["narrative"]
    assert "GEMINI_API_KEY" in result["narrative"]["error"]
