"""Bước 5 — Trích xuất đặc trưng bảng (Tabular features) từ passing path cho baseline Tầng 3(a).

Tham chiếu: configs/model/baseline_tabular.yaml và docs/Quy_trinh_XAI_Football.md, mục 5.2.
Tiền lệ: "Passing path predicts shooting outcome in football".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

# Schema đặc trưng thống nhất cho baseline phi đồ thị
AVAILABLE_PATH_FEATURES: tuple[str, ...] = (
    "n_passes",
    "n_players_involved",
    "sequence_length",
    "mean_pass_distance",
    "mean_pass_angle",
    "duration",
    "start_x",
    "start_y",
    "final_x",
    "final_y",
)


def compute_circular_mean_angle(sin_vals: np.ndarray, cos_vals: np.ndarray) -> float:
    """Tính trung bình góc có hướng bằng circular mean (atan2(mean(sin), mean(cos))).

    Tránh hiện tượng sai lệch tại ranh giới gián đoạn -pi / +pi của góc lượng giác.
    """
    if len(sin_vals) == 0:
        return 0.0
    mean_sin = float(np.mean(sin_vals))
    mean_cos = float(np.mean(cos_vals))
    if math.hypot(mean_sin, mean_cos) < 1e-7:
        return 0.0
    return float(math.atan2(mean_sin, mean_cos))


def extract_path_features_from_graph(
    graph: Any,
    feature_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Trích xuất vector đặc trưng thống kê từ một đối tượng đồ thị possession (PyG Data).

    Args:
        graph: torch_geometric.data.Data với các thuộc tính x, edge_attr, y.
        feature_names: Danh sách đặc trưng cần lấy theo thứ tự. Nếu None, lấy toàn bộ
            `AVAILABLE_PATH_FEATURES`.

    Returns:
        dict chứa các giá trị đặc trưng dạng số thực.

    Raises:
        ValueError: Nếu `feature_names` chứa đặc trưng không nằm trong schema.
    """
    if feature_names is not None:
        invalid = [f for f in feature_names if f not in AVAILABLE_PATH_FEATURES]
        if invalid:
            raise ValueError(
                f"Đặc trưng không hợp lệ trong cấu hình: {invalid}. "
                f"Danh sách đặc trưng hợp lệ: {list(AVAILABLE_PATH_FEATURES)}"
            )

    target_features = (
        list(feature_names) if feature_names is not None else list(AVAILABLE_PATH_FEATURES)
    )

    n_nodes = int(graph.num_nodes) if hasattr(graph, "num_nodes") else 0
    n_edges = int(graph.num_edges) if hasattr(graph, "num_edges") else 0

    # edge_attr theo quy ước features.py:
    # 0: start_x, 1: start_y, 2: end_x, 3: end_y, 4: length, 5: angle_sin, 6: angle_cos,
    # 7: ground, 8: low, 9: high, 10: seq_order, 11: elapsed_sec
    if n_edges > 0 and hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        ea = graph.edge_attr.detach().cpu().numpy()
        start_x = float(ea[0, 0])
        start_y = float(ea[0, 1])
        final_x = float(ea[-1, 2])
        final_y = float(ea[-1, 3])
        lengths = ea[:, 4]
        seq_len = float(np.sum(lengths))
        mean_dist = float(np.mean(lengths))

        # Circular mean cho góc chuyền bóng
        sin_vals = ea[:, 5] if ea.shape[1] > 5 else np.zeros(n_edges)
        cos_vals = ea[:, 6] if ea.shape[1] > 6 else np.ones(n_edges)
        mean_angle = compute_circular_mean_angle(sin_vals, cos_vals)

        elapsed = ea[:, 11] if ea.shape[1] > 11 else np.zeros(n_edges)
        duration = float(np.max(elapsed))
    else:
        start_x, start_y, final_x, final_y = 0.0, 0.0, 0.0, 0.0
        seq_len, mean_dist, mean_angle, duration = 0.0, 0.0, 0.0, 0.0

    all_feat = {
        "n_passes": float(n_edges),
        "n_players_involved": float(n_nodes),
        "sequence_length": seq_len,
        "mean_pass_distance": mean_dist,
        "mean_pass_angle": mean_angle,
        "duration": duration,
        "start_x": start_x,
        "start_y": start_y,
        "final_x": final_x,
        "final_y": final_y,
    }

    return {k: all_feat[k] for k in target_features}


def graphs_to_tabular_dataset(
    graphs: list[Any],
    feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    """Chuyển đổi danh sách đồ thị PyG thành ma trận đặc trưng bảng X và nhãn y.

    Args:
        graphs: Danh sách PyG Data objects.
        feature_names: Thứ tự các đặc trưng cần trích xuất.

    Returns:
        X_df: DataFrame đặc trưng với các cột đúng theo thứ tự khai báo.
        y: numpy array nhãn nhị phân.
        match_ids: Danh sách match_id tương ứng cho từng hàng.
    """
    rows = []
    labels = []
    match_ids = []

    for g in graphs:
        feat = extract_path_features_from_graph(g, feature_names=feature_names)
        rows.append(feat)
        label = int(g.y.item()) if hasattr(g, "y") and g.y is not None else 0
        labels.append(label)
        match_ids.append(int(getattr(g, "match_id", 0)))

    col_order = list(feature_names) if feature_names is not None else list(AVAILABLE_PATH_FEATURES)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[col_order]
    else:
        df = pd.DataFrame(columns=col_order)

    return df, np.array(labels, dtype=int), match_ids
