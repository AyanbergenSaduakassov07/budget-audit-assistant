"""Adapter: budget_auditor's parsed workbook -> narrate_audit() input shape.

core.budget_narrator.narrate_audit() expects core.budget_audit.run_full_audit()
output. budget_auditor has no АТК/уточнение decision engine, so its flags map
onto those buckets the same way run_full_audit prioritizes them:

    АТК          <- high_change_flag            (sharp YoY growth)
    уточнение    <- missing_justification_flag or a CRITICAL quarterly
                    mismatch, unless the row is already АТК (АТК outranks;
                    minor quarterly gaps stay findings, not decisions —
                    exactly as run_full_audit escalates only "critical")
    согласовано  <- everything else

The same dict also feeds conclusion_generator.generate_conclusion() (the
.docx Заключение): all_rows carries decision/reason per row, findings frames
carry plan_annual/article_code/severity under their canonical names. The АТК
section may additionally show a "сквозной вопрос" concentration row; footers
count статьи (the summary buckets), matching the generator's own design.

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


def _num_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Numeric column as a Series (missing column -> all-NaN Series, never a
    scalar: pd.to_numeric(df.get(...)) on a missing column returns nan)."""
    if name not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[name], errors="coerce")


def _decorate_rows(combined: pd.DataFrame, plan: pd.Series,
                   atk_mask: pd.Series, clar_mask: pd.Series) -> pd.DataFrame:
    """combined + the row-level columns conclusion_generator reads: canonical
    names (plan_annual, article_code, sheet_name), the final decision, and a
    per-row reason. Reason priority mirrors run_full_audit: growth % first,
    then missing justification, then quarterly gap — first claim wins."""
    rows = combined.copy()
    rows["plan_annual"] = plan
    if "statya_code" in rows.columns:
        rows["article_code"] = rows["statya_code"]
    if "sheet" in rows.columns:
        rows["sheet_name"] = rows["sheet"]

    decision = pd.Series("согласовано", index=rows.index, dtype=object)
    decision[clar_mask] = "уточнение"
    decision[atk_mask] = "АТК"
    rows["decision"] = decision

    reason = pd.Series("", index=rows.index, dtype=object)
    delta = _num_col(rows, "delta_pct")
    for idx in rows.index[atk_mask & delta.notna()]:
        reason[idx] = f"рост {delta[idx]:+.0f}% к прошлому году"
    unclaimed = reason == ""
    reason[unclaimed & _bool_col(rows, "missing_justification_flag")] = "не заполнено: обоснование"
    unclaimed = reason == ""
    gap = _num_col(rows, "quarterly_gap_pct")
    for idx in rows.index[unclaimed & _bool_col(rows, "quarterly_mismatch_flag") & gap.notna()]:
        reason[idx] = f"расхождение по кварталам: {gap[idx]:.0f}%"
    rows["reason"] = reason
    if "quarterly_severity" in rows.columns:
        rows["severity"] = rows["quarterly_severity"]
    return rows


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
    critical_quarterly = (
        combined["quarterly_severity"] == "critical"
        if "quarterly_severity" in combined.columns
        else pd.Series(False, index=combined.index)
    ).fillna(False)
    clar_mask = (
        _bool_col(combined, "missing_justification_flag") | critical_quarterly
    ) & ~atk_mask
    appr_mask = ~atk_mask & ~clar_mask

    all_rows = _decorate_rows(combined, plan, atk_mask, clar_mask)

    growth_cols = [c for c in ("counterparty", "delta_pct", "plan_annual",
                               "article_code", "decision") if c in all_rows.columns]
    growth_flags = (
        all_rows.loc[atk_mask, growth_cols].copy()
        if "counterparty" in growth_cols and "delta_pct" in growth_cols
        else pd.DataFrame(columns=["counterparty", "delta_pct"])
    )

    concentration = _concentration(combined, plan)
    quarterly_mismatches = all_rows[_bool_col(all_rows, "quarterly_mismatch_flag")]
    missing_just = all_rows[_bool_col(all_rows, "missing_justification_flag")]
    has_prior = bool(
        combined["delta_pct"].notna().any() if "delta_pct" in combined.columns else False
    )

    return {
        "entity": meta.get("company") or "",
        "budget_year": plan_year,
        "unit_scale": meta.get("unit") or 1,
        "sheet_name": "все листы",
        "has_prior_year": has_prior,
        "all_rows": all_rows,
        "findings": {
            "growth_flags": growth_flags,
            "concentration": concentration,
            "quarterly_mismatches": quarterly_mismatches,
            "missing_fields": missing_just,
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
    empty = {"top_supplier": "", "top_supplier_pct": 0.0, "concentration_flag": False,
             "supplier_breakdown": pd.DataFrame(columns=["counterparty", "total_amount", "share_pct"])}
    if "counterparty" not in combined.columns:
        return empty
    total = float(plan.sum())
    if total <= 0:
        return empty
    named = combined["counterparty"].notna() & (combined["counterparty"].astype(str).str.strip() != "")
    if not named.any():
        return empty
    amounts = plan[named].groupby(combined.loc[named, "counterparty"]).sum()
    top = (amounts / total * 100).sort_values(ascending=False)
    breakdown = pd.DataFrame({
        "counterparty": top.index,
        "total_amount": amounts.reindex(top.index).values,
        "share_pct": top.values,
    })
    return {
        "top_supplier": str(top.index[0]),
        "top_supplier_pct": float(top.iloc[0]),
        "concentration_flag": bool((top > 40).any()),
        "supplier_breakdown": breakdown,
    }


def narrative_opening(meta: dict, combined: pd.DataFrame) -> str | None:
    """LLM executive summary + risk level for the opening of the first
    assistant message. Returns None on ANY failure or when no provider key
    (OPENROUTER_API_KEY or GEMINI_API_KEY) is set — the caller then uses the
    template summary. Never raises."""
    try:
        if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            return None
        # Imported lazily so a missing/broken provider SDK install degrades to
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
