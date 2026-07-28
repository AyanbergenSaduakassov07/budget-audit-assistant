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

    # No baseline year in the fixture -> no АТК. Only the CRITICAL quarterly
    # mismatch escalates to уточнение (minor stays a finding, matching
    # run_full_audit); the other three rows are согласовано.
    assert s["atk_count"] == 0
    assert s["clarification_count"] == 1
    assert s["clarification_amount"] == 100_000
    assert s["approved_count"] == 3
    assert s["approved_amount"] == 250_000
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
    assert {"counterparty", "delta_pct"} <= set(growth.columns)
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
def _clear_provider_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)


def test_narrative_opening_returns_none_without_key(quarterly_result, monkeypatch):
    _clear_provider_env(monkeypatch)
    meta, combined = quarterly_result
    assert narrative_opening(meta, combined) is None


def test_narrative_opening_returns_none_without_any_provider_configured(quarterly_result, monkeypatch):
    """Neither OPENROUTER_API_KEY nor LOCAL_LLM_BASE_URL set -> get_provider()
    raises inside narrative_opening()'s try/except -> None, gracefully."""
    _clear_provider_env(monkeypatch)
    meta, combined = quarterly_result
    assert narrative_opening(meta, combined) is None


def test_narrative_opening_attempts_generation_with_local_llm_base_url(quarterly_result, monkeypatch):
    """No OPENROUTER_API_KEY, only LOCAL_LLM_BASE_URL + LOCAL_LLM_MODEL set ->
    narrative_opening() must still attempt generation (via get_provider()'s
    own precedence), not silently short-circuit to None the way the old
    hardcoded OPENROUTER_API_KEY-only gate did."""
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://llm-host:8000/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "Qwen3-32B")

    meta, combined = quarterly_result
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


def test_narrative_opening_never_raises_on_garbage_input(monkeypatch):
    _clear_provider_env(monkeypatch)
    assert narrative_opening(None, None) is None
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-no-network")
    # With a key set, garbage input must still degrade to None, not raise —
    # build_audit_result tolerates it and the API failure is swallowed.
    import core.budget_narrator as bn
    monkeypatch.setattr(bn, "_get_client", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert narrative_opening(None, None) is None


def test_narrative_opening_success_path(quarterly_result, monkeypatch):
    meta, combined = quarterly_result
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
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
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    import core.budget_narrator as bn
    monkeypatch.setattr(bn, "_get_client", lambda: object())
    monkeypatch.setattr(bn, "_generate_content", lambda client, prompt: "NOT JSON AT ALL")
    assert narrative_opening(meta, combined) is None


# --- conclusion .docx bridge (Prompt 5) -----------------------------------------
def test_item_label_falls_back_to_item_name_before_dash():
    """Two rows sharing the same article_code with no counterparty must NOT
    render as the same indistinguishable "{code} — —" label -- item_name
    (then statya_name) should distinguish them."""
    from core.conclusion_generator import _item_label

    row_a = {"article_code": "IT-CA-01", "counterparty": None, "item_name": "Серверы", "statya_name": "Оборудование"}
    row_b = {"article_code": "IT-CA-01", "counterparty": None, "item_name": "СХД", "statya_name": "Оборудование"}

    label_a = _item_label(row_a)
    label_b = _item_label(row_b)

    assert label_a == "IT-CA-01 — Серверы"
    assert label_b == "IT-CA-01 — СХД"
    assert label_a != label_b


def test_item_label_prefers_counterparty_falls_back_to_statya_name_then_dash():
    from core.conclusion_generator import _item_label

    with_counterparty = {"article_code": "IT-OP-01", "counterparty": "Телеком Плюс", "item_name": "Связь"}
    assert _item_label(with_counterparty) == "IT-OP-01 — Телеком Плюс"

    no_counterparty_or_item_name = {"article_code": "IT-OP-01", "counterparty": "", "item_name": "", "statya_name": "Услуги связи"}
    assert _item_label(no_counterparty_or_item_name) == "IT-OP-01 — Услуги связи"

    all_blank = {"article_code": "IT-OP-01", "counterparty": None, "item_name": None, "statya_name": None}
    assert _item_label(all_blank) == "IT-OP-01 — —"


def test_conclusion_payload_sections(quarterly_result):
    """The adapter dict must satisfy conclusion_generator's payload builder:
    section tables consistent with the summary footers."""
    from core.conclusion_generator import _build_payload

    meta, combined = quarterly_result
    payload = _build_payload(build_audit_result(meta, combined))

    assert "ТОО Тест-Дочка" in payload["intro_text"]
    # АТК: no growth rows; the systemic concentration row is present.
    tabs = [r["tab"] for r in payload["atk_section"]["rows"]]
    assert tabs == ["сквозной вопрос"]
    assert payload["atk_section"]["footer"] == "Итого на АТК: 0 статей"
    # уточнение: exactly the critical quarterly row, footer matches the table.
    clar = payload["clarification_section"]["rows"]
    assert len(clar) == 1
    assert "расхождение по кварталам: 6%" in clar[0]["reason"]
    assert payload["clarification_section"]["footer"] == "Итого требуют уточнения: 1 статей"
    # согласовано: grouped by sheet, 3 rows -> one OPEX group.
    groups = payload["approved_section"]["groups"]
    assert [g["label"] for g in groups] == ["OPEX"]
    assert payload["approved_section"]["footer"] == "Итого согласовано: 3 статей"
    # Резюме: counts line up with the buckets.
    totals = {r["decision"]: r["count"] for r in payload["summary_table"]["rows"]}
    assert totals == {"Выносится на АТК": 0, "Требует уточнения": 1,
                      "Согласовано": 3, "ИТОГО": 4}


def test_conclusion_docx_renders(quarterly_result, tmp_path):
    """Full render through Node; skipped where node isn't installed."""
    import shutil
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    from core.conclusion_generator import generate_conclusion

    meta, combined = quarterly_result
    out = generate_conclusion(build_audit_result(meta, combined),
                              str(tmp_path / "заключение.docx"))
    data = (tmp_path / "заключение.docx").read_bytes()
    assert out.endswith(".docx")
    assert data[:2] == b"PK" and len(data) > 5_000  # a real zip/docx, not a stub
