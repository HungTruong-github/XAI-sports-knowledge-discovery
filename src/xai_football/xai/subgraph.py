"""Bước 7.2 — Trích xuất Subgraph quan trọng và Ánh xạ điểm XAI về đường chuyền/cầu thủ.

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 7.2.
Chức năng:
    - Trích xuất tập cạnh quan trọng dựa trên top-k hoặc ngưỡng threshold.
    - Ánh xạ trực tiếp từ chỉ mục cạnh trong đồ thị về event_id, passer_id, recipient_id gốc.
    - Tổng hợp điểm importance từ các cạnh liên thuộc về từng cầu thủ (in/out/total).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from torch_geometric.data import Data

from xai_football.xai.explainers import ExplanationResult


def select_important_edges(
    edge_mask: np.ndarray,
    top_k: int | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """Xác định các chỉ mục cạnh quan trọng theo top-k hoặc threshold.

    Args:
        edge_mask: Mảng trọng số cạnh 1D [num_edges] trong [0, 1].
        top_k: Số lượng cạnh có điểm cao nhất cần chọn.
        threshold: Ngưỡng điểm tối thiểu để giữ lại.

    Returns:
        Mảng boolean mask 1D kích thước [num_edges], True nếu cạnh được chọn.
    """
    n_edges = len(edge_mask)
    if n_edges == 0:
        return np.zeros(0, dtype=bool)

    if top_k is not None and top_k > 0:
        k = min(int(top_k), n_edges)
        # Lấy top k chỉ số lớn nhất
        top_indices = np.argsort(edge_mask)[-k:]
        selected = np.zeros(n_edges, dtype=bool)
        selected[top_indices] = True
        return selected

    if threshold is not None:
        th = float(threshold)
        return edge_mask >= th

    # Mặc định: chọn các cạnh có điểm >= trung bình
    mean_val = float(edge_mask.mean()) if n_edges > 0 else 0.0
    return edge_mask >= mean_val


def map_explanation_to_passes(
    graph: Data,
    explanation: ExplanationResult,
    top_k: int | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Ánh xạ mặt nạ giải thích về thông tin đường chuyền thực tế từ dữ liệu gốc."""
    n_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
    if n_edges == 0:
        return pd.DataFrame()

    edge_mask = explanation.edge_mask
    is_important = select_important_edges(edge_mask, top_k=top_k, threshold=threshold)

    # Thứ hạng cạnh (1 = quan trọng nhất)
    ranks = (np.argsort(np.argsort(-edge_mask)) + 1).tolist()

    player_ids = (
        graph.player_ids.detach().cpu().numpy()
        if hasattr(graph, "player_ids")
        else np.arange(graph.num_nodes)
    )
    edge_event_ids = (
        list(graph.edge_event_ids) if hasattr(graph, "edge_event_ids") else [None] * n_edges
    )

    rows = []
    for k in range(n_edges):
        src_idx = int(graph.edge_index[0, k].item())
        dst_idx = int(graph.edge_index[1, k].item())

        src_pid = int(player_ids[src_idx]) if src_idx < len(player_ids) else -1
        dst_pid = int(player_ids[dst_idx]) if dst_idx < len(player_ids) else -1
        eid = edge_event_ids[k]

        rows.append(
            {
                "match_id": explanation.match_id,
                "possession_id": explanation.possession_id,
                "edge_index": k,
                "event_id": eid,
                "passer_id": src_pid,
                "recipient_id": dst_pid,
                "src_node_idx": src_idx,
                "dst_node_idx": dst_idx,
                "importance": round(float(edge_mask[k]), 4),
                "rank": ranks[k],
                "is_important": bool(is_important[k]),
                "explainer": explanation.explainer_name,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(by="importance", ascending=False).reset_index(drop=True)


def aggregate_player_importance(
    graph: Data,
    explanation: ExplanationResult,
) -> pd.DataFrame:
    """Tổng hợp điểm XAI từ các cạnh liên thuộc về từng cầu thủ trong possession.

    Công thức:
        - Outgoing importance: Tổng điểm của các đường chuyền cầu thủ thực hiện.
        - Incoming importance: Tổng điểm của các đường chuyền cầu thủ nhận bóng.
        - Total importance: Outgoing + Incoming.
    """
    n_nodes = graph.num_nodes
    n_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0

    player_ids = (
        graph.player_ids.detach().cpu().numpy()
        if hasattr(graph, "player_ids")
        else np.arange(n_nodes)
    )

    out_scores = np.zeros(n_nodes, dtype=float)
    in_scores = np.zeros(n_nodes, dtype=float)
    out_counts = np.zeros(n_nodes, dtype=int)
    in_counts = np.zeros(n_nodes, dtype=int)

    edge_mask = explanation.edge_mask
    if n_edges > 0 and len(edge_mask) == n_edges:
        for k in range(n_edges):
            u = int(graph.edge_index[0, k].item())
            v = int(graph.edge_index[1, k].item())
            val = float(edge_mask[k])
            if u < n_nodes:
                out_scores[u] += val
                out_counts[u] += 1
            if v < n_nodes:
                in_scores[v] += val
                in_counts[v] += 1

    total_scores = out_scores + in_scores

    rows = []
    for i in range(n_nodes):
        pid = int(player_ids[i])
        rows.append(
            {
                "match_id": explanation.match_id,
                "possession_id": explanation.possession_id,
                "player_id": pid,
                "node_idx": i,
                "outgoing_importance": round(float(out_scores[i]), 4),
                "incoming_importance": round(float(in_scores[i]), 4),
                "total_importance": round(float(total_scores[i]), 4),
                "pass_sent_count": int(out_counts[i]),
                "pass_received_count": int(in_counts[i]),
                "explainer": explanation.explainer_name,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(by="total_importance", ascending=False).reset_index(drop=True)
