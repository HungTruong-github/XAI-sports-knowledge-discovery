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


def test_edge_feature_reordering_produces_identical_results():
    """Tráo đổi thứ tự cột trong edge_attr kèm schema mới phải cho kết quả giống hệt."""
    g_standard = _mock_graph()
    standard_features = extract_path_features_from_graph(g_standard)

    # Đảo thứ tự cột: ví dụ đưa length (index 4) lên đầu, start_x (0) về sau
    # Tạo g_reordered với edge_attr có thứ tự cột khác
    from xai_football.graphs.features import DEFAULT_EDGE_FEATURE_NAMES

    reordered_schema = list(DEFAULT_EDGE_FEATURE_NAMES)
    # Hoán đổi cột 0 (start_x) và cột 4 (length)
    reordered_schema[0], reordered_schema[4] = reordered_schema[4], reordered_schema[0]

    ea_reordered = g_standard.edge_attr.clone()
    ea_reordered[:, 0] = g_standard.edge_attr[:, 4]
    ea_reordered[:, 4] = g_standard.edge_attr[:, 0]

    g_reordered = Data(
        x=g_standard.x,
        edge_index=g_standard.edge_index,
        edge_attr=ea_reordered,
        y=g_standard.y,
    )
    g_reordered.match_id = g_standard.match_id
    g_reordered.num_nodes = g_standard.num_nodes

    reordered_features = extract_path_features_from_graph(g_reordered, edge_schema=reordered_schema)

    for k in standard_features:
        assert math.isclose(
            standard_features[k], reordered_features[k], rel_tol=1e-5
        ), f"Lệch ở đặc trưng {k}"


def test_missing_required_edge_column_raises_error():
    """Ném ValueError rõ ràng nếu schema thiếu cột bắt buộc."""
    g = _mock_graph()
    # Tạo schema thiếu 'length'
    from xai_football.graphs.features import DEFAULT_EDGE_FEATURE_NAMES

    incomplete_schema = [c for c in DEFAULT_EDGE_FEATURE_NAMES if c != "length"]
    ea_incomplete = g.edge_attr[:, : len(incomplete_schema)]
    g_incomplete = Data(x=g.x, edge_index=g.edge_index, edge_attr=ea_incomplete, y=g.y)

    with pytest.raises(ValueError, match="Edge schema thiếu các cột bắt buộc"):
        extract_path_features_from_graph(g_incomplete, edge_schema=incomplete_schema)


def test_edge_dim_mismatch_raises_error():
    """Ném ValueError khi số cột của edge_attr không khớp với độ dài schema."""
    g = _mock_graph()
    # edge_attr có 12 cột, nhưng truyền schema 10 cột
    with pytest.raises(ValueError, match="Kích thước edge_attr"):
        extract_path_features_from_graph(g, edge_schema=["start_x"] * 10)


def test_non_finite_edge_features_raises_error():
    """Ném ValueError khi edge_attr chứa NaN hoặc Inf."""
    g = _mock_graph()
    g.edge_attr[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN hoặc Inf"):
        extract_path_features_from_graph(g)
