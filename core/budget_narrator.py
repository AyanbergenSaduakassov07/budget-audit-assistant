"""Turn a run_full_audit() result into a short Russian narrative via Gemini.

Deliberately data-minimized: the prompt sent to the model carries only
aggregate counts/amounts, supplier *names* with delta_pct (never raw тыс. тг
figures per line item), and share percentages — never a raw DataFrame or an
individual cell value from the source workbook. This keeps confidential
per-contract figures out of a third-party API call; the model only ever
reasons over shapes and aggregates, the same way a human reviewer's summary
would cite "significant growth at Supplier X" without republishing the ledger.

Uses the same Gemini client/retry plumbing as legacy.query_engine.

Per SPEC.md architecture rule: this module is pure — no streamlit imports.
"""

from __future__ import annotations

import json
import math

import pandas as pd

from legacy.query_engine import DailyQuotaExhausted, _generate_content, _get_client

_TOP_GROWTH_ITEMS = 5

_RESPONSE_SCHEMA = """{
  "executive_summary": "3-4 sentences in Russian summarizing key findings with amounts in тыс. тг",
  "atk_items": [{"supplier": str, "reason": str, "amount_tys_tg": float}],
  "clarification_items": [{"supplier": str, "reason": str}],
  "risk_level": "высокий / средний / низкий",
  "key_risks": ["2-3 main systemic risks in Russian"],
  "note_on_prior_year": "short sentence on whether YoY comparison was available"
}"""

_FALLBACK_NARRATIVE = {
    "executive_summary": "",
    "atk_items": [],
    "clarification_items": [],
    "risk_level": "неизвестно",
    "key_risks": [],
    "note_on_prior_year": "",
}


def _format_delta_pct(value) -> str:
    """Human-readable delta_pct: percentage string, or a label for edge cases."""
    if value is None:
        return "н/д"
    if isinstance(value, (int, float)) and math.isinf(value):
        return "новая статья (нет базы для сравнения)"
    if pd.isna(value):
        return "н/д"
    return f"{value:+.1f}%"


def _top_growth_items(growth_flags: pd.DataFrame) -> list[dict]:
    """Up to 5 growth-flag rows as {counterparty, delta_pct} — no amounts."""
    if growth_flags is None or growth_flags.empty:
        return []
    if "counterparty" not in growth_flags.columns or "delta_pct" not in growth_flags.columns:
        return []

    ranked = growth_flags.sort_values("delta_pct", ascending=False).head(_TOP_GROWTH_ITEMS)
    return [
        {
            "counterparty": str(row["counterparty"]) if pd.notna(row["counterparty"]) else "н/д",
            "delta_pct": _format_delta_pct(row["delta_pct"]),
        }
        for _, row in ranked.iterrows()
    ]


def _build_context(audit_result: dict) -> dict:
    """The ONLY data allowed into the prompt — aggregates and names, no raw figures."""
    summary = audit_result.get("summary", {}) or {}
    findings = audit_result.get("findings", {}) or {}
    concentration = findings.get("concentration", {}) or {}
    quarterly_mismatches = findings.get("quarterly_mismatches")

    return {
        "entity": audit_result.get("entity"),
        "budget_year": audit_result.get("budget_year"),
        "unit_scale": audit_result.get("unit_scale"),
        "summary": {
            "atk_count": summary.get("atk_count", 0),
            "atk_amount_tys_tg": summary.get("atk_amount", 0),
            "clarification_count": summary.get("clarification_count", 0),
            "clarification_amount_tys_tg": summary.get("clarification_amount", 0),
            "approved_count": summary.get("approved_count", 0),
            "approved_amount_tys_tg": summary.get("approved_amount", 0),
        },
        "top_growth_items": _top_growth_items(findings.get("growth_flags")),
        "concentration_risk": {
            "top_supplier": concentration.get("top_supplier", ""),
            "top_supplier_share_pct": concentration.get("top_supplier_pct", 0.0),
        },
        "quarterly_mismatches_count": 0 if quarterly_mismatches is None else len(quarterly_mismatches),
        "has_prior_year_comparison": bool(summary.get("has_prior_year_comparison", False)),
    }


def _build_prompt(context: dict) -> str:
    context_json = json.dumps(context, ensure_ascii=False, indent=2)
    return (
        "Ты аналитик по закупкам компании. Ниже — сводные данные аудита бюджета "
        "одной дочерней компании (агрегаты и наименования контрагентов, без "
        "сырых построчных сумм). Составь краткое заключение на русском языке "
        "строго на основе этих данных — ничего не выдумывай сверх того, что дано.\n\n"
        f"Данные аудита:\n{context_json}\n\n"
        "Верни ТОЛЬКО валидный JSON без markdown-ограждений (без ```), "
        "без пояснений до или после, в точности такой структуры:\n"
        f"{_RESPONSE_SCHEMA}"
    )


def _strip_json_fences(text: str) -> str:
    """Remove accidental markdown code fences from a JSON response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned.strip()


def _parse_narrative(raw_text: str) -> dict:
    """Parse the model's JSON response; never raise on malformed output."""
    cleaned = _strip_json_fences(raw_text)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        return {
            **_FALLBACK_NARRATIVE,
            "error": f"Не удалось разобрать ответ модели как JSON: {exc}",
        }

    if not isinstance(parsed, dict):
        return {
            **_FALLBACK_NARRATIVE,
            "error": "Ответ модели не является JSON-объектом.",
        }

    return {**_FALLBACK_NARRATIVE, **parsed}


def narrate_audit(audit_result: dict) -> dict:
    """Generate a Russian narrative for a run_full_audit() result via Gemini.

    Returns audit_result merged with a "narrative" key. Never raises: any
    client/API/parse failure is caught and surfaced as narrative["error"]
    with the rest of narrative falling back to empty defaults.
    """
    context = _build_context(audit_result)
    prompt = _build_prompt(context)

    try:
        client = _get_client()
        raw_text = _generate_content(client, prompt)
        narrative = _parse_narrative(raw_text)
    except DailyQuotaExhausted:
        narrative = {
            **_FALLBACK_NARRATIVE,
            "error": "Дневной лимит бесплатного API исчерпан. Попробуйте позже.",
        }
    except Exception as exc:  # never crash the audit pipeline on a narration failure
        narrative = {
            **_FALLBACK_NARRATIVE,
            "error": f"Не удалось сформировать заключение: {exc}",
        }

    return {**audit_result, "narrative": narrative}
