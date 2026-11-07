"""Unit tests cho module tính toán Hybrid Centrality."""

import numpy as np
import pandas as pd
import pytest

from xai_football.centrality.calculator import (
    apply_xai_shrinkage,
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
    assert ranks.iloc[0] == 0.0
    assert ranks.iloc[3] == 1.0


def test_apply_xai_shrinkage():
    # Giả lập bảng điểm XAI cấp possession
    rows = []
    # Cầu thủ 1 có 50 lần tham gia, điểm cao 0.9
    for _ in range(50):
        rows.append({"player_id": 1, "mean_importance": 0.9, "mean_importance_no_last": 0.85})
    # Cầu thủ 2 có 40 lần tham gia, điểm thấp 0.1
    for _ in range(40):
        rows.append({"player_id": 2, "mean_importance": 0.1, "mean_importance_no_last": 0.1})
    # Cầu thủ 3 chỉ có 10 lần tham gia (dưới ngưỡng min_possessions=30)
    for _ in range(10):
        rows.append({"player_id": 3, "mean_importance": 0.95, "mean_importance_no_last": 0.9})

    raw_df = pd.DataFrame(rows)
    shrunk_df = apply_xai_shrinkage(raw_df, shrinkage_m=20.0, min_possessions=30)

    # Cầu thủ 3 phải bị loại vì n < 30
    assert len(shrunk_df) == 2
    assert set(shrunk_df["player_id"]) == {1, 2}

    p1 = shrunk_df[shrunk_df["player_id"] == 1].iloc[0]
    p2 = shrunk_df[shrunk_df["player_id"] == 2].iloc[0]

    # Điểm shrunk của P1 phải co về trung bình toàn cục (nhỏ hơn 0.9)
    assert p1["shrunk_importance"] < 0.9
    # Điểm shrunk của P2 phải co về trung bình toàn cục (lớn hơn 0.1)
    assert p2["shrunk_importance"] > 0.1


def test_compute_hybrid_centrality():
    cent_df = pd.DataFrame({
        "player_id": [1, 2, 3],
        "player_name": ["A", "B", "C"],
        "betweenness": [0.1, 0.5, 0.9],
        "is_goalkeeper": [False, False, True],
    })
    xai_df = pd.DataFrame({
        "player_id": [1, 2, 3],
        "mean_importance": [0.8, 0.4, 0.2],
    })

    # Test exclude_goalkeepers=True
    hc = compute_hybrid_centrality(cent_df, xai_df, alpha=0.5, exclude_goalkeepers=True)
    # Thủ môn (player_id 3) phải bị loại
    assert len(hc) == 2
    assert 3 not in hc["player_id"].values
    assert "hybrid_centrality" in hc.columns
    assert (hc["hybrid_centrality"] >= 0.0).all() and (hc["hybrid_centrality"] <= 1.0).all()


def test_optimize_alpha_on_set_a_with_reliability():
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

    # Giả lập half_a1 và half_a2
    h1 = cent_df.copy()
    h1["mean_importance"] = xai_df["mean_importance"]
    h2 = cent_df.copy()
    h2["mean_importance"] = xai_df["mean_importance"]

    best_alpha, curve_df = optimize_alpha_on_set_a(
        cent_df, xai_df, ref_df,
        half_a1_df=h1, half_a2_df=h2,
        split_half_min_reliability_ratio=0.7,
    )
    assert 0.0 <= best_alpha <= 1.0
    assert len(curve_df) > 0
    assert "split_half_reliability" in curve_df.columns
    assert "meets_reliability_constraint" in curve_df.columns


def test_compute_match_traditional_centrality_multi_team():
    """Kiểm tra mạng chuyền bóng được tách độc lập theo từng đội trong cùng trận đấu."""
    rows = [
        # Đội 100: P10 -> P20 -> P30
        {"match_id": 1, "team_id": 100, "event_type": "Pass", "player_id": 10, "player_name": "P10", "pass_recipient_id": 20, "pass_outcome": None},
        {"match_id": 1, "team_id": 100, "event_type": "Pass", "player_id": 20, "player_name": "P20", "pass_recipient_id": 30, "pass_outcome": None},
        {"match_id": 1, "team_id": 100, "event_type": "Pass", "player_id": 30, "player_name": "P30", "pass_recipient_id": 20, "pass_outcome": None},
        {"match_id": 1, "team_id": 100, "event_type": "Pass", "player_id": 20, "player_name": "P20", "pass_recipient_id": 10, "pass_outcome": None},
        # Đội 200: P40 -> P50 -> P60
        {"match_id": 1, "team_id": 200, "event_type": "Pass", "player_id": 40, "player_name": "P40", "pass_recipient_id": 50, "pass_outcome": None},
        {"match_id": 1, "team_id": 200, "event_type": "Pass", "player_id": 50, "player_name": "P50", "pass_recipient_id": 60, "pass_outcome": None},
        {"match_id": 1, "team_id": 200, "event_type": "Pass", "player_id": 60, "player_name": "P60", "pass_recipient_id": 50, "pass_outcome": None},
        {"match_id": 1, "team_id": 200, "event_type": "Pass", "player_id": 50, "player_name": "P50", "pass_recipient_id": 40, "pass_outcome": None},
    ]
    df = pd.DataFrame(rows)
    cent = compute_match_traditional_centrality(df)
    assert len(cent) == 6
    # P20 và P50 đóng vai trò trung tâm trong đội của họ
    p20 = cent[cent["player_id"] == 20].iloc[0]
    p50 = cent[cent["player_id"] == 50].iloc[0]
    assert p20["betweenness"] > 0.0
    assert p50["betweenness"] > 0.0
    # Mẫu số chuẩn hóa của mạng 3 người: normalized betweenness của cầu thủ trung tâm là 1.0
    assert np.isclose(p20["betweenness"], 1.0)
    assert np.isclose(p50["betweenness"], 1.0)
