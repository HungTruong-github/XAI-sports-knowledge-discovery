"""Unit tests cho Baseline phi đồ thị (Tầng 3a) và trích xuất đặc trưng passing-path."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from xai_football.modeling.baselines.path_features import (
    AVAILABLE_PATH_FEATURES,
    compute_circular_mean_angle,
    extract_path_features_from_graph,
    graphs_to_tabular_dataset,
)


def _mock_graph(n_nodes=4, n_edges=3, y_val=1) -> Data:
    x = torch.randn(n_nodes, 8)
    # edge_attr: [start_x, start_y, end_x, end_y, length, angle_sin, angle_cos, ... 12 dims]
    ea = torch.zeros(n_edges, 12)
    # start_x=0.2, start_y=0.4, end_x=0.8, end_y=0.6, length=0.3
    ea[:, 0] = 0.2
    ea[:, 1] = 0.4
    ea[:, 2] = 0.8
    ea[:, 3] = 0.6
    ea[:, 4] = 0.3
    # 45 degrees: sin = sqrt(2)/2, cos = sqrt(2)/2
    ea[:, 5] = math.sin(math.pi / 4)
    ea[:, 6] = math.cos(math.pi / 4)
    ea[:, 11] = torch.tensor([0.0, 5.0, 10.0])  # duration = 10.0
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    g = Data(x=x, edge_index=edge_index, edge_attr=ea, y=torch.tensor([y_val]))
    g.match_id = 100
    g.possession_id = 5
    return g


def test_circular_mean_angle():
    """Kiểm tra circular mean góc hoạt động chính xác qua ranh giới -pi / +pi."""
    # Hai góc: 0 và pi/2 -> trung bình là pi/4
    sins = np.array([math.sin(0), math.sin(math.pi / 2)])
    coss = np.array([math.cos(0), math.cos(math.pi / 2)])
    mean_ang = compute_circular_mean_angle(sins, coss)
    assert abs(mean_ang - math.pi / 4) < 1e-5

    # Hai góc đối xứng qua trục x âm: +170 deg và -170 deg -> trung bình là pi (180 deg)
    ang1 = math.radians(170)
    ang2 = math.radians(-170)
    sins = np.array([math.sin(ang1), math.sin(ang2)])
    coss = np.array([math.cos(ang1), math.cos(ang2)])
    mean_ang = compute_circular_mean_angle(sins, coss)
    # atan2 của sin=0, cos < 0 là pi
    assert abs(abs(mean_ang) - math.pi) < 1e-4


def test_extract_path_features_schema():
    """Kiểm tra extract_path_features_from_graph trả về đúng schema."""
    g = _mock_graph()
    feat = extract_path_features_from_graph(g)
    assert set(feat.keys()) == set(AVAILABLE_PATH_FEATURES)
    assert feat["n_passes"] == 3.0
    assert feat["n_players_involved"] == 4.0
    assert abs(feat["sequence_length"] - 0.9) < 1e-5
    assert abs(feat["mean_pass_distance"] - 0.3) < 1e-5
    assert abs(feat["mean_pass_angle"] - math.pi / 4) < 1e-5
    assert feat["duration"] == 10.0
    assert abs(feat["start_x"] - 0.2) < 1e-5
    assert abs(feat["start_y"] - 0.4) < 1e-5
    assert abs(feat["final_x"] - 0.8) < 1e-5
    assert abs(feat["final_y"] - 0.6) < 1e-5


def test_custom_feature_subset_and_order():
    """Kiểm tra trích xuất đặc trưng theo thứ tự khai báo trong config."""
    g = _mock_graph()
    subset = ["duration", "mean_pass_angle", "n_passes"]
    feat = extract_path_features_from_graph(g, feature_names=subset)
    assert list(feat.keys()) == subset

    df, y, mids = graphs_to_tabular_dataset([g], feature_names=subset)
    assert list(df.columns) == subset
    assert len(df) == 1
    assert y[0] == 1
    assert mids[0] == 100


def test_raise_error_on_unknown_feature():
    """Ném ValueError rõ ràng nếu config yêu cầu đặc trưng không có trong schema."""
    g = _mock_graph()
    with pytest.raises(ValueError, match="Đặc trưng không hợp lệ"):
        extract_path_features_from_graph(g, feature_names=["n_passes", "non_existent_feature"])


def test_no_label_leakage_in_features():
    """Đảm bảo các đặc trưng trích xuất hoàn toàn không phụ thuộc vào nhãn y."""
    g1 = _mock_graph(y_val=1)
    g0 = _mock_graph(y_val=0)

    f1 = extract_path_features_from_graph(g1)
    f0 = extract_path_features_from_graph(g0)
    assert f1 == f0
