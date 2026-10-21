"""Unit tests cho kiểm định thống kê ở Bước 6 (paired cluster bootstrap theo trận)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xai_football.evaluation.significance import (
    align_predictions,
    cluster_bootstrap_indices,
    compare_models,
    holm_adjust,
    two_sided_bootstrap_p,
)


def _toy_data(n_matches: int = 30, per_match: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_matches), per_match)
    y = (rng.random(len(groups)) < 0.2).astype(int)
    strong = np.clip(0.75 * y + 0.25 * rng.random(len(y)), 0, 1)  # gần như tách hoàn hảo
    random_model = rng.random(len(y))  # đoán ngẫu nhiên
    return y, groups, strong, random_model


def test_cluster_bootstrap_resamples_whole_matches():
    groups = np.array([1, 1, 1, 2, 2, 3])
    samples = cluster_bootstrap_indices(groups, n_boot=50, seed=1)
    assert len(samples) == 50
    for idx in samples:
        picked = groups[idx]
        # Mỗi trận được chọn phải xuất hiện đủ toàn bộ possession của nó (bội số kích thước)
        for g, size in ((1, 3), (2, 2), (3, 1)):
            assert np.sum(picked == g) % size == 0
    # Tất định theo seed
    again = cluster_bootstrap_indices(groups, n_boot=50, seed=1)
    assert all(np.array_equal(a, b) for a, b in zip(samples, again, strict=True))


def test_strong_model_beats_random_model_significantly():
    y, groups, strong, rand = _toy_data()
    ci_df, pairs = compare_models(
        y, groups, {"strong": [strong, strong], "random": [rand]}, n_boot=300, seed=0
    )
    row = pairs[(pairs["metric"] == "auc_pr")].iloc[0]
    assert row["model_a"] == "strong" and row["model_b"] == "random"
    assert row["diff_a_minus_b"] > 0.3
    assert row["ci95_low"] > 0
    assert row["significant_holm_0.05"]
    assert row["verdict"] == "strong tốt hơn"

    brier = pairs[pairs["metric"] == "brier_score"].iloc[0]
    assert brier["diff_a_minus_b"] < 0  # strong có Brier thấp hơn
    assert brier["verdict"] == "strong tốt hơn"

    strong_ci = ci_df[(ci_df["model"] == "strong") & (ci_df["metric"] == "auc_pr")].iloc[0]
    assert strong_ci["ci95_low"] <= strong_ci["value"] <= strong_ci["ci95_high"]


def test_identical_models_are_not_significant():
    y, groups, strong, _ = _toy_data()
    _, pairs = compare_models(y, groups, {"a": [strong], "b": [strong]}, n_boot=200, seed=0)
    assert (pairs["diff_a_minus_b"] == 0).all()
    assert not pairs["significant_holm_0.05"].any()


def test_two_sided_p_and_holm():
    assert two_sided_bootstrap_p(np.full(99, 1.0)) == pytest.approx(2 / 100)
    assert two_sided_bootstrap_p(np.linspace(-1, 1, 101)) == pytest.approx(1.0)
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_align_predictions_reorders_and_validates():
    ref = pd.DataFrame({"match_id": [1, 1, 2], "possession_id": [1, 2, 1], "y_true": [0, 1, 0]})
    shuffled = pd.DataFrame(
        {
            "match_id": [2, 1, 1],
            "possession_id": [1, 2, 1],
            "y_true": [0, 1, 0],
            "p": [0.3, 0.9, 0.1],
        }
    )
    assert align_predictions(ref, shuffled, "p").tolist() == [0.1, 0.9, 0.3]

    wrong_label = shuffled.assign(y_true=[1, 1, 0])
    with pytest.raises(ValueError, match="Nhãn"):
        align_predictions(ref, wrong_label, "p")

    missing = shuffled.iloc[:2]
    with pytest.raises(ValueError, match="không phủ"):
        align_predictions(ref, missing, "p")


def test_compare_models_rejects_length_mismatch():
    y, groups, strong, _ = _toy_data()
    with pytest.raises(ValueError):
        compare_models(y, groups, {"a": [strong[:-1]]}, n_boot=10)


@pytest.mark.parametrize("metric", ["auc_pr", "auc_roc", "brier_score"])
def test_fast_metrics_match_sklearn(metric):
    """Bản numpy dùng trong bootstrap phải cho đúng số như sklearn (kể cả khi điểm trùng nhau)."""
    from xai_football.evaluation.significance import METRIC_FUNCS, SKLEARN_METRIC_FUNCS

    rng = np.random.default_rng(7)
    for trial in range(30):
        n = int(rng.integers(20, 3000))
        y = (rng.random(n) < rng.uniform(0.05, 0.5)).astype(int)
        if y.min() == y.max():
            y[0] = 1 - y[0]
        p = rng.random(n)
        if trial % 2 == 0:
            p = np.round(p, 1)  # nhiều điểm trùng nhau
        assert METRIC_FUNCS[metric](y, p) == pytest.approx(
            SKLEARN_METRIC_FUNCS[metric](y, p), abs=1e-12
        )


def test_holm_keeps_nan_as_nan():
    """p-value không tính được (NaN) không được biến thành 1.0 và không làm lệch hệ số Holm."""
    out = holm_adjust([0.01, float("nan"), 0.03])
    assert np.isnan(out[1])
    assert out[0] == pytest.approx(0.02) and out[2] == pytest.approx(0.03)
