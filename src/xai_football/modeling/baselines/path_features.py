"""Bước 5 — Trích xuất đặc trưng bảng (Tabular features) từ passing path cho baseline Tầng 3(a).

Tham chiếu: configs/model/baseline_tabular.yaml và docs/Quy_trinh_XAI_Football.md, mục 5.2.
Tiền lệ: "Passing path predicts shooting outcome in football".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_path_features_from_graph(graph) -> dict[str, float]:
    """Trích xuất vector đặc trưng thống kê từ một đối tượng đồ thị possession (PyG Data).

    Args:
        graph: torch_geometric.data.Data với các thuộc tính x, edge_attr, y.

    Returns:
        dict chứa các đặc trưng: n_passes, n_players, sequence_length, mean_pass_distance,
        duration, start_x, start_y, final_x, final_y.
    """
    n_nodes = int(graph.num_nodes)
    n_edges = int(graph.num_edges)

    # edge_attr theo quy ước features.py:
    # 0: start_x, 1: start_y, 2: end_x, 3: end_y, 4: length, 5: angle_sin, 6: angle_cos,
    # 7: ground, 8: low, 9: high, 10: seq_order, 11: elapsed_sec
    if n_edges > 0 and hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        ea = graph.edge_attr.cpu().numpy()
        start_x = float(ea[0, 0])
        start_y = float(ea[0, 1])
        final_x = float(ea[-1, 2])
        final_y = float(ea[-1, 3])
        lengths = ea[:, 4]
        seq_len = float(np.sum(lengths))
        mean_dist = float(np.mean(lengths))
        elapsed = ea[:, 11] if ea.shape[1] > 11 else np.zeros(n_edges)
        duration = float(np.max(elapsed))
    else:
        start_x, start_y, final_x, final_y = 0.0, 0.0, 0.0, 0.0
        seq_len, mean_dist, duration = 0.0, 0.0, 0.0

    return {
        "n_passes": float(n_edges),
        "n_players_involved": float(n_nodes),
        "sequence_length": seq_len,
        "mean_pass_distance": mean_dist,
        "duration": duration,
        "start_x": start_x,
        "start_y": start_y,
        "final_x": final_x,
        "final_y": final_y,
    }


def graphs_to_tabular_dataset(graphs: list) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    """Chuyển đổi danh sách đồ thị PyG thành ma trận đặc trưng bảng X và nhãn y.

    Returns:
        X_df: DataFrame đặc trưng.
        y: numpy array nhãn nhị phân.
        match_ids: Danh sách match_id tương ứng cho từng hàng.
    """
    rows = []
    labels = []
    match_ids = []

    for g in graphs:
        feat = extract_path_features_from_graph(g)
        rows.append(feat)
        label = int(g.y.item()) if hasattr(g, "y") and g.y is not None else 0
        labels.append(label)
        match_ids.append(int(getattr(g, "match_id", 0)))

    return pd.DataFrame(rows), np.array(labels, dtype=int), match_ids
