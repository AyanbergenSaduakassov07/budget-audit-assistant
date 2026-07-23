"""Adapter: budget_auditor's parsed workbook -> narrate_audit() input shape.

core.budget_narrator.narrate_audit() expects core.budget_audit.run_full_audit()
output. budget_auditor has no АТК/уточнение decision engine, so its flags map
onto those buckets the same way run_full_audit prioritizes them:

    АТК          <- high_change_flag            (sharp YoY growth)
    уточнение    <- missing_justification_flag or quarterly_mismatch_flag,
                    unless the row is already АТК (АТК outranks)
    согласовано  <- everything else

Data-minimization holds end to end: this module hands narrate_audit bucket
counts and sums, counterparty NAMES with delta_pct, and share percentages;
narrate_audit's own _build_context() then strips everything else before the
prompt is built. No raw line-item figure reaches the API.

narrative_opening() is the app-facing entry point and NEVER raises — the
narrative is the first thing a person sees in the chat, so any failure
(no key, quota, network, malformed model output, unexpected input shape)
degrades silently to None and the caller falls back to the template summary.

Per SPEC_AUDIT.md architecture rule: pure module, no streamlit imports.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def _bool_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Flag column as a clean boolean mask (missing column -> all False)."""
    if name not in df.columns:
        return pd.Series(False, index=df.index)
    return df[name].fillna(False).astype(bool)


def build_audit_result(meta: dict, combined: pd.DataFrame) -> dict:
    """Map (workbook meta, aggregated+flagged DataFrame) into the dict shape
    narrate_audit() reads. Pure and deterministic; no API calls here."""
    meta = meta or {}
    if combined is None:
        combined = pd.DataFrame()

    plan_year = (
        combined["plan_year"].dropna().iloc[0]
        if "plan_year" in combined.columns and combined["plan_year"].notna().any()
        else meta.get("campaign_year")
    )
    # Native int, not numpy int64: narrate_audit json.dumps()-es this dict
    # BEFORE its own try block, and int64 is not JSON serializable.
    if plan_year is not None and not pd.isna(plan_year):
        plan_year = int(plan_year)
    else:
        plan_year = None
    plan_col = f"plan_{plan_year}" if plan_year else None
    plan = (
        pd.to_numeric(combined[plan_col], errors="coerce").fillna(0)
        if plan_col and plan_col in combined.columns
        else pd.Series(0.0, index=combined.index)
    )

    atk_mask = _bool_col(combined, "high_change_flag")
    clar_mask = (
        _bool_col(combined, "missing_justification_flag")
        | _bool_col(combined, "quarterly_mismatch_flag")
    ) & ~atk_mask
    appr_mask = ~atk_mask & ~clar_mask

    growth_cols = [c for c in ("counterparty", "delta_pct") if c in combined.columns]
    growth_flags = (
        combined.loc[atk_mask, growth_cols].copy()
        if len(growth_cols) == 2
        else pd.DataFrame(columns=["counterparty", "delta_pct"])
    )

    concentration = _concentration(combined, plan)
    quarterly_mismatches = combined[_bool_col(combined, "quarterly_mismatch_flag")]
    has_prior = bool(
        combined["delta_pct"].notna().any() if "delta_pct" in combined.columns else False
    )

    return {
        "entity": meta.get("company") or "",
        "budget_year": plan_year,
        "unit_scale": meta.get("unit") or 1,
        "has_prior_year": has_prior,
        "findings": {
            "growth_flags": growth_flags,
            "concentration": concentration,
            "quarterly_mismatches": quarterly_mismatches,
        },
        "summary": {
            "atk_count": int(atk_mask.sum()),
            "atk_amount": float(plan[atk_mask].sum()),
            "clarification_count": int(clar_mask.sum()),
            "clarification_amount": float(plan[clar_mask].sum()),
            "approved_count": int(appr_mask.sum()),
            "approved_amount": float(plan[appr_mask].sum()),
            "total_rows": int(len(combined)),
            "total_amount": float(plan.sum()),
            "has_prior_year_comparison": has_prior,
        },
    }


def _concentration(combined: pd.DataFrame, plan: pd.Series) -> dict:
    """Top-counterparty share of the campaign-year plan — names and percentages
    only, mirroring budget_audit.concentration_risk's output keys."""
    empty = {"top_supplier": "", "top_supplier_pct": 0.0, "concentration_flag": False}
    if "counterparty" not in combined.columns:
        return empty
    total = float(plan.sum())
    if total <= 0:
        return empty
    named = combined["counterparty"].notna() & (combined["counterparty"].astype(str).str.strip() != "")
    if not named.any():
        return empty
    shares = plan[named].groupby(combined.loc[named, "counterparty"]).sum() / total * 100
    top = shares.sort_values(ascending=False)
    return {
        "top_supplier": str(top.index[0]),
        "top_supplier_pct": float(top.iloc[0]),
        "concentration_flag": bool((top > 40).any()),
    }


def narrative_opening(meta: dict, combined: pd.DataFrame) -> str | None:
    """Gemini executive summary + risk level for the opening of the first
    assistant message. Returns None on ANY failure or when GEMINI_API_KEY is
    unset — the caller then uses the template summary. Never raises."""
    try:
        if not os.environ.get("GEMINI_API_KEY"):
            return None
        # Imported lazily so a missing/broken google-genai install degrades to
        # the template summary instead of taking the whole app down at import.
        from core.budget_narrator import narrate_audit

        narrative = narrate_audit(build_audit_result(meta, combined))["narrative"]
        if narrative.get("error"):
            logger.warning("narrative fallback: %s", narrative["error"])
            return None
        text = (narrative.get("executive_summary") or "").strip()
        if not text:
            return None
        risk = (narrative.get("risk_level") or "").strip()
        if risk and risk != "неизвестно":
            text += f"\n\nУровень риска: {risk}."
        return text
    except Exception:
        logger.exception("narrative_opening failed; falling back to template summary")
        return None
