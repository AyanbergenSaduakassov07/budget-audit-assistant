"""Synthetic procurement data generator for the comparison tool.

Produces two subsidiary DataFrames (df_a, df_b) plus a ground_truth dict that
records exactly which items carry each deliberately planted discrepancy, so the
comparison logic can be tested against known answers.

Per SPEC.md architecture rule: this module is pure — no streamlit imports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["item_name", "supplier", "category", "qty", "unit_price", "subtotal"]

# --- Group-level (multi-subsidiary) generation --------------------------------
# The 10 subsidiaries covered by the group-wide dataset. Hardcoded so the
# same seed always yields the same subsidiaries in the same order.
SUBSIDIARIES = [
    "Subsidiary A",
    "ACM Digital",
    "Subsidiary C",
    "Subsidiary D",
    "Subsidiary E",
    "Subsidiary F",
    "Subsidiary G",
    "Subsidiary H",
    "Subsidiary I",
    "Subsidiary J",
]

# Flat output schema for the group DataFrame.
GROUP_COLUMNS = [
    "subsidiary",
    "year",
    "item_name",
    "supplier",
    "category",
    "qty",
    "unit_price",
    "subtotal",
]

# Group item catalog: the same items and KZT price ranges used by generate()
# (Microsoft 365, Adobe CC, AutoCAD, Kaspersky, Dell/LG/HP monitors) plus the
# extra items required for the group build (Zoom, SAP, Cisco, HP servers).
# (item_name, supplier, category, base_unit_price_kzt)
_GROUP_CATALOG = [
    ("microsoft 365 e3", "softline", "license", 210000.0),
    ("adobe creative cloud", "marvel distribution", "license", 320000.0),
    ("autocad license", "autodesk", "license", 480000.0),
    ("kaspersky antivirus", "kaspersky lab", "license", 95000.0),
    ("zoom workplace license", "zoom video", "license", 120000.0),
    ("sap s/4hana license", "sap se", "license", 650000.0),
    ("dell ultrasharp u2723", "dell technologies", "hardware", 185000.0),
    ("lg 27un880 monitor", "lg electronics", "hardware", 165000.0),
    ("hp e24 monitor", "hp inc", "hardware", 210000.0),
    ("cisco catalyst 9300 switch", "cisco systems", "hardware", 850000.0),
    ("hp proliant dl380 server", "hp inc", "hardware", 1200000.0),
]

_GROUP_YEARS = [2022, 2023, 2024]

# Base price inflation applied per procurement year (2022 = list price baseline).
_YEAR_FACTORS = {2022: 1.0, 2023: 1.08, 2024: 1.16}

# --- Planted group-level anomalies (recorded in ground_truth) -----------------
# 1. Cross-subsidiary price gap: one subsidiary pays >30% over the group median
#    for one item in one year.
_GAP_ITEM = "sap s/4hana license"
_GAP_SUBSIDIARY = "Subsidiary E"
_GAP_YEAR = 2024
_GAP_MULTIPLIER = 1.42

# 2. Supplier overcharge: one supplier charges one subsidiary noticeably more
#    than everyone else for the identical item (all years).
_OVERCHARGE_SUPPLIER = "cisco systems"
_OVERCHARGE_SUBSIDIARY = "Subsidiary I"
_OVERCHARGE_ITEM = "cisco catalyst 9300 switch"
_OVERCHARGE_MULTIPLIER = 1.45

# 3. Year-over-year spike: one subsidiary's spend on one category jumps >40%
#    from 2023 to 2024. Enforced as a post-step so the spike holds regardless of
#    how the random 2023 baseline for that block came out (big-ticket hardware
#    rows make raw category totals very high variance).
_YOY_SUBSIDIARY = "Subsidiary H"
_YOY_CATEGORY = "hardware"
_YOY_TARGET_RATIO = 1.55  # force 2024 spend to at least this multiple of 2023

# 4. Arithmetic error: exactly one row where subtotal != qty * unit_price.
_GROUP_ARITH_SUBSIDIARY = "Subsidiary D"
_GROUP_ARITH_ITEM = "microsoft 365 e3"
_GROUP_ARITH_YEAR = 2023
_GROUP_ARITH_MULTIPLIER = 1.15

# Shared catalog for subsidiary A. Prices are list prices in KZT within the
# realistic 2026 ranges from SPEC.md (licenses ~80k–500k, monitors ~150k–300k).
# (item_name, supplier, category, unit_price)
_CATALOG = [
    ("microsoft 365 e3", "softline", "license", 210000.0),
    ("adobe creative cloud", "marvel distribution", "license", 320000.0),
    ("autocad license", "autodesk", "license", 480000.0),        # -> price anomaly
    ("kaspersky antivirus", "kaspersky lab", "license", 95000.0),  # -> missing in B
    ("dell ultrasharp u2723", "dell technologies", "hardware", 185000.0),  # -> arithmetic error in B
    ("lg 27un880 monitor", "lg electronics", "hardware", 165000.0),
    ("hp e24 monitor", "hp inc", "hardware", 210000.0),
]

# Item that exists only in subsidiary B.
_NEW_ITEM = ("zoom workplace license", "zoom video", "license", 120000.0)

# Roles of the four planted discrepancies (by item_name).
_ANOMALY_ITEM = "autocad license"
_MISSING_ITEM = "kaspersky antivirus"
_NEW_ROW_ITEM = _NEW_ITEM[0]
_ARITHMETIC_ITEM = "dell ultrasharp u2723"

# Qty draw ranges per category (inclusive low, exclusive high for rng.integers).
_QTY_RANGES = {"license": (20, 300), "hardware": (5, 60), "other": (5, 100)}

_ANOMALY_MULTIPLIER = 1.28   # df_b price is 28% higher than df_a for the anomaly
_ARITHMETIC_MULTIPLIER = 1.15  # break subtotal by 15% so |error| >> 0.01


def _round2(x: float) -> float:
    return round(float(x), 2)


def _qty(rng: np.random.Generator, category: str) -> int:
    low, high = _QTY_RANGES.get(category, _QTY_RANGES["other"])
    return int(rng.integers(low, high))


def generate(seed: int = 42):
    """Generate (df_a, df_b, ground_truth) deterministically for a given seed.

    Both DataFrames carry exactly the SPEC.md schema columns. df_b differs from
    df_a by four planted discrepancies plus small realistic price negotiation
    jitter on the remaining shared items (kept small enough not to trip an
    anomaly threshold).
    """
    rng = np.random.default_rng(seed)

    # --- Subsidiary A: full catalog at list prices ---------------------------
    a_rows = []
    a_price = {}   # item_name -> unit_price in A, needed to derive the anomaly
    for item_name, supplier, category, price in _CATALOG:
        qty = _qty(rng, category)
        unit_price = _round2(price)
        a_price[item_name] = unit_price
        a_rows.append(
            {
                "item_name": item_name,
                "supplier": supplier,
                "category": category,
                "qty": qty,
                "unit_price": unit_price,
                "subtotal": _round2(qty * unit_price),
            }
        )
    df_a = pd.DataFrame(a_rows, columns=COLUMNS)

    # --- Subsidiary B: catalog minus the missing item, plus discrepancies ----
    b_rows = []
    for item_name, supplier, category, price in _CATALOG:
        if item_name == _MISSING_ITEM:
            continue  # planted discrepancy #2: absent from B entirely

        qty = _qty(rng, category)

        if item_name == _ANOMALY_ITEM:
            # planted discrepancy #1: 28% above A's unit price
            unit_price = _round2(a_price[item_name] * _ANOMALY_MULTIPLIER)
        else:
            # small negotiated difference vs. A's list price (-3%..+5%)
            jitter = 1.0 + float(rng.uniform(-0.03, 0.05))
            unit_price = _round2(a_price[item_name] * jitter)

        subtotal = _round2(qty * unit_price)
        if item_name == _ARITHMETIC_ITEM:
            # planted discrepancy #4: subtotal deliberately inconsistent
            subtotal = _round2(qty * unit_price * _ARITHMETIC_MULTIPLIER)

        b_rows.append(
            {
                "item_name": item_name,
                "supplier": supplier,
                "category": category,
                "qty": qty,
                "unit_price": unit_price,
                "subtotal": subtotal,
            }
        )

    # planted discrepancy #3: new item present only in B
    new_name, new_supplier, new_category, new_price = _NEW_ITEM
    new_qty = _qty(rng, new_category)
    new_unit_price = _round2(new_price)
    b_rows.append(
        {
            "item_name": new_name,
            "supplier": new_supplier,
            "category": new_category,
            "qty": new_qty,
            "unit_price": new_unit_price,
            "subtotal": _round2(new_qty * new_unit_price),
        }
    )
    df_b = pd.DataFrame(b_rows, columns=COLUMNS)

    ground_truth = {
        "price_anomaly": _ANOMALY_ITEM,
        "missing_row": _MISSING_ITEM,
        "new_row": _NEW_ROW_ITEM,
        "arithmetic_error": _ARITHMETIC_ITEM,
    }

    return df_a, df_b, ground_truth


def generate_group(seed: int = 42):
    """Generate group-wide procurement data for all ACM subsidiaries.

    Returns (df, ground_truth):
      - df: a single flat DataFrame with GROUP_COLUMNS, covering every
        subsidiary in SUBSIDIARIES across every year in _GROUP_YEARS. Each
        (subsidiary, year) block has 20-30 line items drawn from the shared
        group catalog.
      - ground_truth: the four deliberately planted group-level anomalies.

    Fully deterministic: the same seed always produces identical data.
    """
    rng = np.random.default_rng(seed)

    rows = []
    arith_planted = False  # discrepancy #4 must break exactly one row

    for subsidiary in SUBSIDIARIES:
        for year in _GROUP_YEARS:
            year_factor = _YEAR_FACTORS[year]

            # Every catalog item appears at least once (so cross-subsidiary and
            # year-over-year comparisons always have a value to compare), then a
            # random number of extra line items brings the block to 20-30 rows.
            line_items = list(_GROUP_CATALOG)
            n_extra = int(rng.integers(9, 20))  # 11 base + 9..19 extra -> 20..30
            extra_idx = rng.integers(0, len(_GROUP_CATALOG), size=n_extra)
            line_items += [_GROUP_CATALOG[i] for i in extra_idx]

            for item_name, supplier, category, base_price in line_items:
                qty = _qty(rng, category)
                jitter = 1.0 + float(rng.uniform(-0.04, 0.06))
                unit_price = base_price * year_factor * jitter

                # #1 cross-subsidiary price gap
                if (
                    item_name == _GAP_ITEM
                    and subsidiary == _GAP_SUBSIDIARY
                    and year == _GAP_YEAR
                ):
                    unit_price *= _GAP_MULTIPLIER

                # #2 supplier overcharge (all years, identical item)
                if item_name == _OVERCHARGE_ITEM and subsidiary == _OVERCHARGE_SUBSIDIARY:
                    unit_price *= _OVERCHARGE_MULTIPLIER

                unit_price = _round2(unit_price)
                subtotal = _round2(qty * unit_price)

                # #4 arithmetic error: break the subtotal of one single row
                if (
                    not arith_planted
                    and subsidiary == _GROUP_ARITH_SUBSIDIARY
                    and item_name == _GROUP_ARITH_ITEM
                    and year == _GROUP_ARITH_YEAR
                ):
                    subtotal = _round2(qty * unit_price * _GROUP_ARITH_MULTIPLIER)
                    arith_planted = True

                rows.append(
                    {
                        "subsidiary": subsidiary,
                        "year": year,
                        "item_name": item_name,
                        "supplier": supplier,
                        "category": category,
                        "qty": qty,
                        "unit_price": unit_price,
                        "subtotal": subtotal,
                    }
                )

    df = pd.DataFrame(rows, columns=GROUP_COLUMNS)

    # #3 year-over-year spike: guarantee the target subsidiary's 2024 spend in
    # the target category is at least _YOY_TARGET_RATIO times its 2023 spend by
    # uniformly scaling the 2024 rows' unit prices (and subtotals). The target
    # subsidiary differs from every other planted anomaly, so nothing else is
    # touched (in particular the arithmetic-error row is left broken).
    cat_mask = (df["subsidiary"] == _YOY_SUBSIDIARY) & (df["category"] == _YOY_CATEGORY)
    spend_2023 = df.loc[cat_mask & (df["year"] == 2023), "subtotal"].sum()
    mask_2024 = cat_mask & (df["year"] == 2024)
    spend_2024 = df.loc[mask_2024, "subtotal"].sum()
    if spend_2024 > 0:
        scale = max(1.0, (_YOY_TARGET_RATIO * spend_2023) / spend_2024)
        df.loc[mask_2024, "unit_price"] = (df.loc[mask_2024, "unit_price"] * scale).round(2)
        df.loc[mask_2024, "subtotal"] = (
            df.loc[mask_2024, "qty"] * df.loc[mask_2024, "unit_price"]
        ).round(2)

    ground_truth = {
        "cross_subsidiary_price_gap": {
            "subsidiary": _GAP_SUBSIDIARY,
            "item_name": _GAP_ITEM,
            "year": _GAP_YEAR,
        },
        "supplier_overcharge": {
            "supplier": _OVERCHARGE_SUPPLIER,
            "subsidiary": _OVERCHARGE_SUBSIDIARY,
            "item_name": _OVERCHARGE_ITEM,
        },
        "yoy_spike": {
            "subsidiary": _YOY_SUBSIDIARY,
            "category": _YOY_CATEGORY,
        },
        "arithmetic_error": {
            "subsidiary": _GROUP_ARITH_SUBSIDIARY,
            "item_name": _GROUP_ARITH_ITEM,
            "year": _GROUP_ARITH_YEAR,
        },
    }

    return df, ground_truth


if __name__ == "__main__":
    df, ground_truth = generate_group()
    print("=== group DataFrame shape ===")
    print(df.shape)
    print("\n=== sample of 10 rows ===")
    print(df.sample(10, random_state=0).to_string(index=False))
    print("\n=== ground_truth ===")
    for key, value in ground_truth.items():
        print(f"  {key}: {value}")
