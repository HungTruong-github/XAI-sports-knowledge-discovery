"""Unit tests cho module tính toán Hybrid Centrality."""

import numpy as np
import pandas as pd
import pytest

from xai_football.centrality.calculator import (
    compute_hybrid_centrality,
    compute_match_traditional_centrality,
    optimize_alpha_on_set_a,
    percentile_rank,
)


@pytest.fixture
def mock_events_df():
    # 3 cầu thủ (10, 20, 30) chuyền bóng qua lại trong trận 1
    rows = [
        {"match_id": 1, "event_type": "Pass", "player_id": 10, "player_name": "P10", "pass_recipient_id": 20, "pass_outcome": None},
        {"match_id": 1, "event_type": "Pass", "player_id": 20, "player_name": "P20", "pass_recipient_id": 30, "pass_outcome": None},
        {"match_id": 1, "event_type": "Pass", "player_id": 30, "player_name": "P30", "pass_recipient_id": 20, "pass_outcome": None},
        {"match_id": 1, "event_type": "Pass", "player_id": 20, "player_name": "P20", "pass_recipient_id": 10, "pass_outcome": None},
    ]
    return pd.DataFrame(rows)


def test_compute_match_traditional_centrality(mock_events_df):
    cent = compute_match_traditional_centrality(mock_events_df)
    assert len(cent) == 3
    # P20 nằm giữa P10 và P30 nên betweenness của P20 phải cao nhất
    p20 = cent[cent["player_id"] == 20].iloc[0]
    p10 = cent[cent["player_id"] == 10].iloc[0]
    p30 = cent[cent["player_id"] == 30].iloc[0]
    assert p20["betweenness"] >= p10["betweenness"]
    assert p20["betweenness"] >= p30["betweenness"]


def test_percentile_rank():
    s = pd.Series([10.0, 50.0, 20.0, 100.0])
    ranks = percentile_rank(s)
    assert ranks.min() == 0.0
    assert ranks.max() == 1.0
    # Thứ tự phân vị: 10 -> 0.0, 20 -> 0.333, 50 -> 0.667, 100 -> 1.0
    assert ranks.iloc[0] == 0.0
    assert ranks.iloc[3] == 1.0


def test_compute_hybrid_centrality():
    cent_df = pd.DataFrame({
        "player_id": [1, 2, 3],
        "player_name": ["A", "B", "C"],
        "betweenness": [0.1, 0.5, 0.9],
    })
    xai_df = pd.DataFrame({
        "player_id": [1, 2, 3],
        "mean_importance": [0.8, 0.4, 0.2],
    })

    # alpha = 0.5
    hc = compute_hybrid_centrality(cent_df, xai_df, alpha=0.5)
    assert len(hc) == 3
    assert "hybrid_centrality" in hc.columns
    assert (hc["hybrid_centrality"] >= 0.0).all() and (hc["hybrid_centrality"] <= 1.0).all()


def test_optimize_alpha_on_set_a():
    cent_df = pd.DataFrame({
        "player_id": list(range(10)),
        "player_name": [f"P{i}" for i in range(10)],
        "betweenness": np.linspace(0.1, 1.0, 10),
    })
    xai_df = pd.DataFrame({
        "player_id": list(range(10)),
        "mean_importance": np.linspace(0.2, 0.9, 10),
    })
    ref_df = pd.DataFrame({
        "player_id": list(range(10)),
        "minutes_played": [180.0] * 10,
        "xt_per_90": np.linspace(0.01, 0.25, 10),
    })

    best_alpha, curve_df = optimize_alpha_on_set_a(cent_df, xai_df, ref_df)
    assert 0.0 <= best_alpha <= 1.0
    assert len(curve_df) > 0

