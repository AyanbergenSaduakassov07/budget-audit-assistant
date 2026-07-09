"""Row-level and summary-level comparison of two procurement sheets.

`compare` is the single entry point. It normalizes both inputs, joins them on
(item_name, supplier), computes every per-row and summary field defined in
SPEC.md, and returns a single dict of results.

Per SPEC.md architecture rule: this module is pure — no streamlit imports.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

if __package__ in (None, ""):
    # Allow running as a plain script (`python core/compare.py`) in addition to
    # `python -m core.compare`, while keeping absolute imports from the project
    # root. Under `-m`, __package__ == "core", so this is skipped.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schema import normalize

_KEY = ["item_name", "supplier"]

# Column order for the returned matched frame (readability only).
_MATCHED_COLUMNS = [
    "item_name",
    "supplier",
    "category",
    "qty_a",
    "qty_b",
    "unit_price_a",
    "unit_price_b",
    "subtotal_a",
    "subtotal_b",
    "delta_subtotal",
    "pct_delta_price",
    "pct_delta_subtotal",
    "severity_score",
    "is_price_anomaly",
    "arithmetic_error_a",
    "arithmetic_error_b",
    "contribution_pct",
    "cumulative_pct",
]


def _with_arithmetic_error(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the SPEC.md arithmetic_error flag on every row."""
    out = df.copy()
    out["arithmetic_error"] = (out["subtotal"] - out["qty"] * out["unit_price"]).abs() > 0.01
    return out


def _anti_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``left`` whose (item_name, supplier) key is absent from ``right``."""
    left_idx = pd.MultiIndex.from_frame(left[_KEY])
    right_idx = pd.MultiIndex.from_frame(right[_KEY])
    return left[~left_idx.isin(right_idx)].reset_index(drop=True)


def _rollup(matched: pd.DataFrame, by: str) -> pd.DataFrame:
    """Group matched rows by ``by``, summing spend on each side with delta/pct."""
    grouped = (
        matched.groupby(by, as_index=False)[["subtotal_a", "subtotal_b"]]
        .sum()
        .sort_values(by)
        .reset_index(drop=True)
    )
    grouped["delta_subtotal"] = (grouped["subtotal_b"] - grouped["subtotal_a"]).round(2)
    grouped["pct_delta"] = grouped.apply(
        lambda r: round((r["subtotal_b"] - r["subtotal_a"]) / r["subtotal_a"] * 100, 2)
        if r["subtotal_a"] != 0
        else 0.0,
        axis=1,
    )
    grouped["subtotal_a"] = grouped["subtotal_a"].round(2)
    grouped["subtotal_b"] = grouped["subtotal_b"].round(2)
    return grouped


def compare(df_a: pd.DataFrame, df_b: pd.DataFrame, threshold_pct: float = 15.0) -> dict:
    """Compare two procurement sheets. See SPEC.md for every field's definition.

    Returns a dict with keys: matched, only_in_a, only_in_b, summary,
    supplier_rollup, category_rollup.
    """
    # --- normalize + independent arithmetic_error on the full sheets ---------
    na = _with_arithmetic_error(normalize(df_a))
    nb = _with_arithmetic_error(normalize(df_b))

    only_in_a = _anti_join(na, nb)
    only_in_b = _anti_join(nb, na)

    # --- matched: inner join, per-row calculations ---------------------------
    matched = na.merge(nb, on=_KEY, how="inner", suffixes=("_a", "_b"))
    # category is an item property; identical across sheets for a matched key.
    matched["category"] = matched["category_a"]

    matched["delta_subtotal"] = (matched["subtotal_b"] - matched["subtotal_a"]).round(2)
    matched["pct_delta_price"] = (
        (matched["unit_price_b"] - matched["unit_price_a"]) / matched["unit_price_a"] * 100
    ).round(2)
    matched["pct_delta_subtotal"] = (
        (matched["subtotal_b"] - matched["subtotal_a"]) / matched["subtotal_a"] * 100
    ).round(2)
    matched["severity_score"] = (matched["pct_delta_price"].abs() / threshold_pct).round(4)
    matched["is_price_anomaly"] = matched["severity_score"] > 1.0

    # --- Pareto contribution: signed share of total variance, sorted desc ----
    abs_delta_total = matched["delta_subtotal"].abs().sum()
    if abs_delta_total > 0:
        matched["contribution_pct"] = (matched["delta_subtotal"] / abs_delta_total * 100).round(2)
    else:
        matched["contribution_pct"] = 0.0
    matched = (
        matched.sort_values("delta_subtotal", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )
    # cumulative over absolute contribution so it climbs monotonically to ~100%.
    matched["cumulative_pct"] = matched["contribution_pct"].abs().cumsum().round(2)

    matched = matched.reindex(columns=_MATCHED_COLUMNS)

    # --- summary-level calculations ------------------------------------------
    total_spend_a = round(float(na["subtotal"].sum()), 2)   # whole sheet A
    total_spend_b = round(float(nb["subtotal"].sum()), 2)   # whole sheet B
    total_pct_delta = (
        round((total_spend_b - total_spend_a) / total_spend_a * 100, 2)
        if total_spend_a != 0
        else 0.0
    )

    weight_sum = float(matched["subtotal_a"].sum())
    if weight_sum > 0:
        weighted_avg_pct_delta_price = round(
            float((matched["pct_delta_price"] * matched["subtotal_a"]).sum()) / weight_sum, 2
        )
    else:
        weighted_avg_pct_delta_price = 0.0

    n_matched = int(len(matched))
    n_anomalies = int(matched["is_price_anomaly"].sum())
    anomaly_rate = round(n_anomalies / n_matched * 100, 2) if n_matched > 0 else 0.0

    summary = {
        "total_spend_a": total_spend_a,
        "total_spend_b": total_spend_b,
        "total_pct_delta": total_pct_delta,
        "weighted_avg_pct_delta_price": weighted_avg_pct_delta_price,
        "anomaly_rate": anomaly_rate,
        "n_matched": n_matched,
        "n_only_in_a": int(len(only_in_a)),
        "n_only_in_b": int(len(only_in_b)),
        "n_anomalies": n_anomalies,
    }

    supplier_rollup = _rollup(matched, "supplier")
    category_rollup = _rollup(matched, "category")

    return {
        "matched": matched,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "summary": summary,
        "supplier_rollup": supplier_rollup,
        "category_rollup": category_rollup,
    }


if __name__ == "__main__":
    from core.synth import generate

    df_a, df_b, ground_truth = generate()
    result = compare(df_a, df_b)

    print("=== ground_truth ===")
    for key, value in ground_truth.items():
        print(f"  {key}: {value}")

    print("\n=== summary ===")
    for key, value in result["summary"].items():
        print(f"  {key}: {value}")

    print("\n=== matched (all computed columns) ===")
    with pd.option_context("display.width", 240, "display.max_columns", None):
        print(result["matched"].to_string(index=False))

    print("\n=== only_in_a ===")
    print(result["only_in_a"].to_string(index=False))
    print("\n=== only_in_b ===")
    print(result["only_in_b"].to_string(index=False))

    print("\n=== supplier_rollup ===")
    print(result["supplier_rollup"].to_string(index=False))
    print("\n=== category_rollup ===")
    print(result["category_rollup"].to_string(index=False))
