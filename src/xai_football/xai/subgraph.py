"""Bước 7.2 — Trích xuất Subgraph quan trọng và Ánh xạ điểm XAI về đường chuyền/cầu thủ.

Tham chiếu: configs/xai.yaml, docs/decisions/0002-cong-thuc-hybrid-centrality.md (Quyết định D12).
Chức năng:
    - Trích xuất tập cạnh quan trọng dựa trên top-k hoặc ngưỡng threshold.
    - Ánh xạ trực tiếp từ chỉ mục cạnh trong đồ thị về event_id, passer_id, recipient_id gốc, kèm cờ `is_last_edge`.
    - Tổng hợp điểm importance từ các cạnh liên thuộc về từng cầu thủ:
      + Tổng điểm importance (in/out/total).
      + Trung bình importance trên mỗi lần tham gia (mean_importance).
      + Biến thể triệt tiêu loại trừ cạnh cuối cùng (no_last_edge) để phục vụ D12.
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
        top_indices = np.argsort(edge_mask)[-k:]
        selected = np.zeros(n_edges, dtype=bool)
        selected[top_indices] = True
        return selected

    if threshold is not None:
        th = float(threshold)
        return edge_mask >= th

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
                "is_last_edge": bool(k == (n_edges - 1)),  # D12: cờ đánh dấu đường chuyền cuối
                "explainer": explanation.explainer_name,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(by="importance", ascending=False).reset_index(drop=True)


def aggregate_player_importance(
    graph: Data,
    explanation: ExplanationResult,
) -> pd.DataFrame:
    """Tổng hợp điểm XAI từ các cạnh liên thuộc về từng cầu thủ theo quyết định D12.

    Bao gồm:
        - Tổng điểm và trung bình mỗi lần tham gia (chống thiên lệch cho cầu thủ chạm bóng nhiều).
        - Biến thể loại bỏ cạnh cuối cùng (last-edge ablation) để kiểm tra tính độc lập.
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

    # Điểm loại cạnh cuối
    out_scores_no_last = np.zeros(n_nodes, dtype=float)
    in_scores_no_last = np.zeros(n_nodes, dtype=float)
    out_counts_no_last = np.zeros(n_nodes, dtype=int)
    in_counts_no_last = np.zeros(n_nodes, dtype=int)

    edge_mask = explanation.edge_mask
    if n_edges > 0 and len(edge_mask) == n_edges:
        for k in range(n_edges):
            u = int(graph.edge_index[0, k].item())
            v = int(graph.edge_index[1, k].item())
            val = float(edge_mask[k])
            is_last = (k == (n_edges - 1))

            if u < n_nodes:
                out_scores[u] += val
                out_counts[u] += 1
                if not is_last:
                    out_scores_no_last[u] += val
                    out_counts_no_last[u] += 1

            if v < n_nodes:
                in_scores[v] += val
                in_counts[v] += 1
                if not is_last:
                    in_scores_no_last[v] += val
                    in_counts_no_last[v] += 1

    total_scores = out_scores + in_scores
    participations = out_counts + in_counts

    total_scores_no_last = out_scores_no_last + in_scores_no_last
    participations_no_last = out_counts_no_last + in_counts_no_last

    rows = []
    for i in range(n_nodes):
        pid = int(player_ids[i])
        n_part = int(participations[i])
        n_part_no_last = int(participations_no_last[i])

        mean_imp = round(float(total_scores[i] / n_part), 4) if n_part > 0 else 0.0
        mean_imp_no_last = (
            round(float(total_scores_no_last[i] / n_part_no_last), 4)
            if n_part_no_last > 0
            else 0.0
        )

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
                "participations": n_part,
                "mean_importance": mean_imp,  # D12: trung bình mỗi lần tham gia
                "total_importance_no_last": round(float(total_scores_no_last[i]), 4),
                "mean_importance_no_last": mean_imp_no_last,  # D12: biến thể loại cạnh cuối
                "explainer": explanation.explainer_name,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(by="total_importance", ascending=False).reset_index(drop=True)
