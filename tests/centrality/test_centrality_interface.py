"""Unit tests kiểm tra interface và ranh giới phương pháp luận của Hybrid Centrality (Decision 0002)."""

import pandas as pd
import pytest

from xai_football.centrality.interface import (
    compute_provisional_hybrid_centrality,
    validate_hybrid_centrality_config,
)


def test_validate_hybrid_centrality_config_rejects_nulls():
    incomplete_cfg = {
        "traditional": {"aggregate_level": None, "combine_measures": None},
        "xai_importance": {"source_explainer": None, "aggregation": None},
        "combination": {"normalization": None, "alpha": None},
    }
    with pytest.raises(ValueError, match="Decision 0002 Pending"):
        validate_hybrid_centrality_config(incomplete_cfg)


def test_validate_hybrid_centrality_config_valid():
    complete_cfg = {
        "traditional": {"aggregate_level": "match", "combine_measures": "mean"},
        "xai_importance": {"source_explainer": "gnn_explainer", "aggregation": "mean"},
        "combination": {"normalization": "minmax", "alpha": 0.5},
    }
    # Không ném ngoại lệ nếu đã cấu hình đầy đủ
    validate_hybrid_centrality_config(complete_cfg)


def test_compute_provisional_centrality_guard():
    trad_df = pd.DataFrame({"player_id": [1, 2], "degree_in": [5.0, 10.0]})
    xai_df = pd.DataFrame({"player_id": [1, 2], "total_importance": [0.8, 1.2]})

    # Không cho phép chạy nếu allow_provisional=False
    with pytest.raises(RuntimeError, match="PROVISIONAL"):
        compute_provisional_hybrid_centrality(trad_df, xai_df, allow_provisional=False)

    # Chạy thành công khi allow_provisional=True
    res = compute_provisional_hybrid_centrality(trad_df, xai_df, alpha=0.6, allow_provisional=True)
    assert "provisional_hybrid_centrality" in res.columns
    assert bool(res["is_official_result"].iloc[0]) is False
    assert res["decision_0002_status"].iloc[0] == "pending_approval"
