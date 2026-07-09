"""Tests for core.compare, driven by the known ground_truth from synth.generate.

All imports are absolute from the project root (run with `uv run pytest`).
"""

import pytest

from core.synth import generate
from core.compare import compare


@pytest.fixture
def data():
    """Real synthetic sheets plus their planted-discrepancy ground truth."""
    df_a, df_b, ground_truth = generate(seed=42)
    return df_a, df_b, ground_truth


@pytest.fixture
def result(data):
    """compare() output for the real df_a vs. df_b at the default threshold."""
    df_a, df_b, _ = data
    return compare(df_a, df_b)


def test_identical_sheets_no_anomalies():
    df_a, _, _ = generate(seed=42)
    result = compare(df_a.copy(), df_a.copy())
    assert result["matched"]["is_price_anomaly"].sum() == 0
    assert result["summary"]["anomaly_rate"] == 0.0
    assert len(result["only_in_a"]) == 0
    assert len(result["only_in_b"]) == 0


def test_price_anomaly_detected(data, result):
    _, _, ground_truth = data
    matched = result["matched"]
    row = matched[matched["item_name"] == ground_truth["price_anomaly"]]
    assert bool(row["is_price_anomaly"].iloc[0]) is True


def test_missing_row_in_only_in_a(data, result):
    _, _, ground_truth = data
    assert ground_truth["missing_row"] in result["only_in_a"]["item_name"].values


def test_new_row_in_only_in_b(data, result):
    _, _, ground_truth = data
    assert ground_truth["new_row"] in result["only_in_b"]["item_name"].values


def test_arithmetic_error_flagged(data, result):
    _, _, ground_truth = data
    matched = result["matched"]
    row = matched[matched["item_name"] == ground_truth["arithmetic_error"]]
    assert bool(row["arithmetic_error_b"].iloc[0]) is True


def test_weighted_avg_differs_from_simple_mean(result):
    matched = result["matched"]
    simple_mean = matched["pct_delta_price"].mean()
    weighted = result["summary"]["weighted_avg_pct_delta_price"]
    assert abs(weighted - simple_mean) > 1.0


def test_pareto_top2_explain_majority(result):
    assert result["matched"]["cumulative_pct"].iloc[1] >= 70.0


def test_threshold_changes_anomaly_count(data):
    df_a, df_b, _ = data
    strict = compare(df_a, df_b, threshold_pct=5.0)["summary"]["n_anomalies"]
    lenient = compare(df_a, df_b, threshold_pct=50.0)["summary"]["n_anomalies"]
    assert strict > lenient
    assert lenient == 0
