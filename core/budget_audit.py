"""Rule-based audit checks replicating a manual procurement review.

These rules mirror what a expert checks by hand when writing a Заключение
on a subsidiary's IT budget submission: quarterly arithmetic, year-over-year
growth thresholds, supplier concentration, missing fields, and brand-new line
items. Everything here is a pure function over pandas DataFrames — no LLM
calls, no I/O, no streamlit imports (per SPEC.md's core/ architecture rule).

Operates on the canonical DataFrame produced by
core.corporate_ingest.canonicalize() + compute_deltas() (already wired
together inside extract_sheet()), which supplies plan_prior, plan_annual,
plan_q1..q4, counterparty, article_code, expense_type, contract, delta_abs,
delta_pct, and delta_flag — whichever of those a given subsidiary's column
layout actually has. Each rule degrades to an empty result (rather than
raising) when its required columns are absent, consistent with
corporate_ingest's "detect whichever columns exist" approach.
"""

from __future__ import annotations

import pandas as pd

from core.corporate_ingest import canonicalize

_QUARTER_COLUMNS = ["plan_q1", "plan_q2", "plan_q3", "plan_q4"]
_MISSING_CHECK_COLUMNS = ["counterparty", "contract", "article_code", "expense_type"]

_NEW_ITEM_FLAG = "новая статья"
_NO_PRIOR_YEAR_FLAG = "нет данных прошлого года"


def _blank_mask(series: pd.Series) -> pd.Series:
    """True where a value is null or an empty/whitespace-only string."""
    return series.isna() | series.astype(str).str.strip().eq("")


# --- RULE 1: quarterly reconciliation -----------------------------------------
def quarterly_reconciliation(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where q1+q2+q3+q4 doesn't add up to the annual plan (>1% off)."""
    out_columns = [
        "counterparty", "article_code", "expense_type",
        "plan_annual", "quarterly_sum", "difference", "difference_pct", "severity",
    ]
    required = _QUARTER_COLUMNS + ["plan_annual"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame(columns=out_columns)

    work = df.copy()
    for c in required:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work[work[required].notna().all(axis=1)]
    if work.empty:
        return pd.DataFrame(columns=out_columns)

    quarterly_sum = work[_QUARTER_COLUMNS].sum(axis=1)
    difference = work["plan_annual"] - quarterly_sum
    difference_pct = pd.Series(0.0, index=work.index)
    positive_annual = work["plan_annual"] > 0
    difference_pct[positive_annual] = (
        difference[positive_annual].abs() / work.loc[positive_annual, "plan_annual"] * 100
    )

    result = pd.DataFrame({
        "counterparty": work.get("counterparty", pd.Series(index=work.index, dtype=object)),
        "article_code": work.get("article_code", pd.Series(index=work.index, dtype=object)),
        "expense_type": work.get("expense_type", pd.Series(index=work.index, dtype=object)),
        "plan_annual": work["plan_annual"],
        "quarterly_sum": quarterly_sum,
        "difference": difference,
        "difference_pct": difference_pct,
    })
    result = result[result["difference_pct"] > 1.0].copy()
    result["severity"] = result["difference_pct"].apply(
        lambda pct: "critical" if pct > 5 else "minor"
    )
    return result


# --- RULE 2: year-over-year analysis -------------------------------------------
def year_over_year_analysis(df: pd.DataFrame, atk_threshold: float = 50.0) -> pd.DataFrame:
    """Classify rows by delta_pct/delta_flag; return the АТК / уточнение ones.

    atk_threshold is the growth-% cutoff above which a row goes to АТК instead
    of уточнение (уточнение stays the fixed (20, atk_threshold] band below it).
    """
    out_columns = [
        "counterparty", "article_code", "expense_type",
        "plan_prior", "plan_annual", "delta_abs", "delta_pct", "delta_flag", "decision",
    ]
    if "delta_pct" not in df.columns or "plan_annual" not in df.columns:
        return pd.DataFrame(columns=out_columns)

    work = df.copy()
    plan_annual = pd.to_numeric(work["plan_annual"], errors="coerce")
    delta_pct = pd.to_numeric(work["delta_pct"], errors="coerce")
    delta_flag = work["delta_flag"].fillna("") if "delta_flag" in work.columns else pd.Series("", index=work.index)

    new_item = delta_flag == _NEW_ITEM_FLAG
    no_comparison = delta_flag == _NO_PRIOR_YEAR_FLAG

    # new_item rows carry delta_pct == inf, which would also satisfy a bare
    # ">atk_threshold" check — exclude them from the generic growth clause so
    # the plan_annual-based new-item thresholds (below) are the ones that decide.
    cond_atk = (~new_item & (delta_pct > atk_threshold)) | (new_item & (plan_annual > 100_000))
    cond_utochnenie_growth = ~new_item & delta_pct.gt(20) & delta_pct.le(atk_threshold)
    cond_utochnenie_new = new_item & (plan_annual <= 100_000)

    decision = pd.Series("норма", index=work.index, dtype=object)
    decision[cond_utochnenie_growth] = "уточнение"
    decision[cond_utochnenie_new] = "уточнение"
    decision[no_comparison] = "нет сравнения"
    decision[cond_atk] = "АТК"  # highest priority, applied last

    work["decision"] = decision
    result = work[work["decision"].isin(["АТК", "уточнение"])].copy()

    for col in ("counterparty", "article_code", "expense_type", "plan_prior"):
        if col not in result.columns:
            result[col] = pd.NA

    return result[out_columns]


# --- RULE 3: supplier concentration risk ---------------------------------------
def concentration_risk(df: pd.DataFrame) -> dict:
    """Per-supplier share of total plan_annual; flag suppliers over 40% share."""
    breakdown_columns = ["counterparty", "total_amount", "share_pct", "single_source_rows", "risk_flag"]
    empty_result = {
        "supplier_breakdown": pd.DataFrame(columns=breakdown_columns),
        "concentration_flag": False,
        "top_supplier": "",
        "top_supplier_pct": 0.0,
        "total_single_source_amount": 0.0,
        "total_single_source_pct": 0.0,
    }
    if "counterparty" not in df.columns or "plan_annual" not in df.columns:
        return empty_result

    work = df.copy()
    work["plan_annual"] = pd.to_numeric(work["plan_annual"], errors="coerce").fillna(0)
    total_budget = float(work["plan_annual"].sum())
    if total_budget <= 0:
        return empty_result

    contract_blank = _blank_mask(work["contract"]) if "contract" in work.columns else pd.Series(False, index=work.index)
    total_single_source_amount = float(work.loc[contract_blank, "plan_annual"].sum())

    named = work[~_blank_mask(work["counterparty"])].copy()
    if named.empty:
        return {
            **empty_result,
            "total_single_source_amount": total_single_source_amount,
            "total_single_source_pct": total_single_source_amount / total_budget * 100,
        }

    named["_no_contract"] = contract_blank.loc[named.index]
    grouped = named.groupby("counterparty", as_index=False).agg(
        total_amount=("plan_annual", "sum"),
        single_source_rows=("_no_contract", "sum"),
    )
    grouped["single_source_rows"] = grouped["single_source_rows"].astype(int)
    grouped["share_pct"] = grouped["total_amount"] / total_budget * 100
    grouped["risk_flag"] = grouped["share_pct"] > 40
    grouped = grouped.sort_values("share_pct", ascending=False).reset_index(drop=True)
    breakdown = grouped[breakdown_columns]

    return {
        "supplier_breakdown": breakdown,
        "concentration_flag": bool(breakdown["risk_flag"].any()),
        "top_supplier": str(breakdown.iloc[0]["counterparty"]),
        "top_supplier_pct": float(breakdown.iloc[0]["share_pct"]),
        "total_single_source_amount": total_single_source_amount,
        "total_single_source_pct": total_single_source_amount / total_budget * 100,
    }


# --- RULE 4: missing required fields --------------------------------------------
def missing_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Rows missing counterparty, contract, article_code, or expense_type."""
    present_cols = [c for c in _MISSING_CHECK_COLUMNS if c in df.columns]
    if not present_cols:
        return pd.DataFrame(columns=list(df.columns) + ["missing_fields"])

    masks = {c: _blank_mask(df[c]) for c in present_cols}
    any_missing = pd.concat(masks.values(), axis=1).any(axis=1)
    result = df[any_missing].copy()
    result["missing_fields"] = [
        ", ".join(c for c in present_cols if masks[c].loc[i]) for i in result.index
    ]
    return result


# --- RULE 5: brand-new line items -----------------------------------------------
def new_line_items(df: pd.DataFrame) -> pd.DataFrame:
    """Rows flagged новая статья by compute_deltas, with a review recommendation."""
    out_columns = list(df.columns) + ["recommendation"]
    if "delta_flag" not in df.columns or "plan_annual" not in df.columns:
        return pd.DataFrame(columns=out_columns)

    result = df[df["delta_flag"] == _NEW_ITEM_FLAG].copy()
    plan_annual = pd.to_numeric(result["plan_annual"], errors="coerce")

    def _recommend(amount: float) -> str:
        if pd.notna(amount) and amount > 500_000:
            return "Вынести на АТК — крупная новая статья"
        if pd.notna(amount) and amount > 100_000:
            return "Запросить обоснование"
        return "Уточнить необходимость"

    result["recommendation"] = plan_annual.apply(_recommend)
    return result


# --- RULE 6: full audit ---------------------------------------------------------
def run_full_audit(sheet_result: dict, atk_threshold: float = 50.0) -> dict:
    """Run all five rules over one extract_sheet() result and classify every row.

    atk_threshold: growth-% cutoff passed through to year_over_year_analysis
    (see that docstring). Defaults to 50, matching the original fixed rule.
    """
    canonical = canonicalize(sheet_result["data"])

    quarterly_mismatches = quarterly_reconciliation(canonical)
    growth_flags = year_over_year_analysis(canonical, atk_threshold=atk_threshold)
    concentration = concentration_risk(canonical)
    missing = missing_fields(canonical)
    new_items = new_line_items(canonical)

    atk_idx: set = set()
    utochnenie_idx: set = set()
    # First rule to claim a row also names why — later rules can still add the
    # row to atk_idx/utochnenie_idx (for correct bucket membership) but won't
    # overwrite an already-assigned reason. Order here is the display priority:
    # new-item size > growth % > missing fields > quarterly > concentration.
    # (new-item rows carry delta_pct == inf — RULE5's amount-tiered
    # recommendation is both more informative and avoids formatting inf as a
    # percentage, so it's claimed before the generic growth-% phrasing below.)
    reasons: dict = {}

    if not growth_flags.empty:
        atk_idx |= set(growth_flags.index[growth_flags["decision"] == "АТК"])
        utochnenie_idx |= set(growth_flags.index[growth_flags["decision"] == "уточнение"])
        for idx, r in growth_flags.iterrows():
            if r["delta_flag"] == _NEW_ITEM_FLAG:
                continue  # claimed by the new_items loop below instead
            pct = r["delta_pct"]
            reasons.setdefault(idx, f"{'рост ' if r['decision'] == 'АТК' else ''}+{pct:.0f}% к прошлому году")

    if not new_items.empty:
        amount = pd.to_numeric(new_items["plan_annual"], errors="coerce")
        large = amount > 500_000
        atk_idx |= set(new_items.index[large])
        utochnenie_idx |= set(new_items.index[~large])
        for idx, r in new_items.iterrows():
            reasons.setdefault(idx, r["recommendation"])

    if not missing.empty:
        utochnenie_idx |= set(missing.index)
        for idx, r in missing.iterrows():
            reasons.setdefault(idx, f"не заполнено: {r['missing_fields']}")

    if not quarterly_mismatches.empty:
        critical = quarterly_mismatches[quarterly_mismatches["severity"] == "critical"]
        utochnenie_idx |= set(critical.index)
        for idx, r in critical.iterrows():
            reasons.setdefault(idx, f"расхождение по кварталам: {r['difference_pct']:.0f}%")

    breakdown = concentration["supplier_breakdown"]
    if not breakdown.empty and "counterparty" in canonical.columns:
        risky_suppliers = set(breakdown.loc[breakdown["risk_flag"], "counterparty"])
        if risky_suppliers:
            top_pct = concentration.get("top_supplier_pct", 0.0)
            risky_idx = canonical.index[canonical["counterparty"].isin(risky_suppliers)]
            atk_idx |= set(risky_idx)
            for idx in risky_idx:
                reasons.setdefault(idx, f"концентрация поставщика: {top_pct:.0f}% от общего бюджета")

    utochnenie_idx -= atk_idx  # АТК outranks уточнение when a row hits both

    all_rows = canonical.copy()
    decision = pd.Series("согласовано", index=all_rows.index, dtype=object)
    decision.loc[list(atk_idx)] = "АТК"
    decision.loc[list(utochnenie_idx)] = "уточнение"
    all_rows["decision"] = decision
    all_rows["reason"] = all_rows.index.map(reasons).fillna("")

    def _amount(mask) -> float:
        sub = all_rows.loc[mask, "plan_annual"] if "plan_annual" in all_rows.columns else pd.Series(dtype=float)
        return float(pd.to_numeric(sub, errors="coerce").fillna(0).sum())

    atk_mask = all_rows["decision"] == "АТК"
    clar_mask = all_rows["decision"] == "уточнение"
    appr_mask = all_rows["decision"] == "согласовано"

    metadata = sheet_result.get("metadata", {}) or {}
    entity = str(canonical["entity"].iloc[0]) if "entity" in canonical.columns and len(canonical) else sheet_result.get("sheet_name", "")

    has_critical = bool(
        not quarterly_mismatches.empty and (quarterly_mismatches["severity"] == "critical").any()
    )

    return {
        "entity": entity,
        "sheet_name": sheet_result.get("sheet_name"),
        "budget_year": sheet_result.get("budget_year"),
        "unit_scale": int(metadata.get("unit_scale", 1)),
        "has_prior_year": bool(sheet_result.get("has_prior_year")),
        "all_rows": all_rows,
        "findings": {
            "quarterly_mismatches": quarterly_mismatches,
            "growth_flags": growth_flags,
            "concentration": concentration,
            "missing_fields": missing,
            "new_items": new_items,
        },
        "summary": {
            "atk_count": int(atk_mask.sum()),
            "atk_amount": _amount(atk_mask),
            "clarification_count": int(clar_mask.sum()),
            "clarification_amount": _amount(clar_mask),
            "approved_count": int(appr_mask.sum()),
            "approved_amount": _amount(appr_mask),
            "total_rows": int(len(all_rows)),
            "total_amount": _amount(pd.Series(True, index=all_rows.index)),
            "has_prior_year_comparison": bool(sheet_result.get("has_prior_year")),
        },
        "has_critical": has_critical,
    }
