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


if __name__ == "__main__":
    df_a, df_b, ground_truth = generate()
    print("=== df_a (subsidiary A) ===")
    print(df_a.to_string(index=False))
    print("\n=== df_b (subsidiary B) ===")
    print(df_b.to_string(index=False))
    print("\n=== ground_truth ===")
    for key, value in ground_truth.items():
        print(f"  {key}: {value}")
