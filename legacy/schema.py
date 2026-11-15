"""Canonical schema normalization for procurement DataFrames.

`normalize` is the single entry point: it enforces the canonical row schema from
the canonical schema, cleaning string fields, coercing numeric types, validating invariants,
and dropping anything outside the schema.

Architecture rule: this module is pure — no streamlit imports.
"""

from __future__ import annotations

import pandas as pd

# Canonical schema column order.
CANONICAL_COLUMNS = [
    "item_name",
    "supplier",
    "category",
    "qty",
    "unit_price",
    "subtotal",
]

_STRING_COLUMNS = ["item_name", "supplier", "category"]
_VALID_CATEGORIES = {"license", "hardware", "other"}


def _clean_str(series: pd.Series) -> pd.Series:
    """Lowercase, strip, and collapse internal whitespace to single spaces."""
    return (
        series.astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of ``df`` conforming to the canonical schema.

    Raises ValueError if a required column is missing or a validated invariant
    is violated (null/non-positive qty or unit_price, null subtotal). The input
    DataFrame is never modified in place.
    """
    # Required-column check first, with a clear message naming the offender.
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col!r}")

    # Work on a copy restricted to canonical columns (drops any extras).
    out = df.loc[:, CANONICAL_COLUMNS].copy()

    # --- string fields -------------------------------------------------------
    for col in _STRING_COLUMNS:
        out[col] = _clean_str(out[col])

    # category: anything outside the allowed set collapses to "other".
    out["category"] = out["category"].where(
        out["category"].isin(_VALID_CATEGORIES), "other"
    )

    # --- qty: int64, must be present and > 0 ---------------------------------
    if out["qty"].isnull().any():
        raise ValueError("qty contains null values")
    qty_numeric = pd.to_numeric(out["qty"], errors="coerce")
    if qty_numeric.isnull().any():
        raise ValueError("qty contains non-numeric values")
    out["qty"] = qty_numeric.astype("int64")
    if (out["qty"] <= 0).any():
        raise ValueError("qty must be > 0 for every row")

    # --- unit_price: float rounded to 2 decimals, present and > 0 ------------
    if out["unit_price"].isnull().any():
        raise ValueError("unit_price contains null values")
    unit_price = pd.to_numeric(out["unit_price"], errors="coerce")
    if unit_price.isnull().any():
        raise ValueError("unit_price contains non-numeric values")
    if (unit_price <= 0).any():
        raise ValueError("unit_price must be > 0 for every row")
    out["unit_price"] = unit_price.round(2).astype(float)

    # --- subtotal: float rounded to 2 decimals, present (no positivity rule) -
    if out["subtotal"].isnull().any():
        raise ValueError("subtotal contains null values")
    subtotal = pd.to_numeric(out["subtotal"], errors="coerce")
    if subtotal.isnull().any():
        raise ValueError("subtotal contains non-numeric values")
    out["subtotal"] = subtotal.round(2).astype(float)

    return out.reset_index(drop=True)
