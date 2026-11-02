r"""Bước 8 — Đánh giá chất lượng giải thích (XAI Evaluation Metrics).

Tham chiếu: configs/xai.yaml và docs/decisions/0006-giao-thuc-xai.md (Quyết định D5, D6, D9, D10).
Bộ độ đo đã thống nhất:
    1. Fidelity+ (Sufficiency / Mức độ sụt giảm xác suất khi LOẠI BỎ subgraph quan trọng):
       Fid+ = P(y=1 | G) - P(y=1 | G \ S)
       Giá trị càng cao -> Subgraph bị loại bỏ thực sự là thành phần quyết định dự đoán của mô hình.

    2. Fidelity- (Necessity / Độ bảo toàn xác suất khi CHỈ GIỮ subgraph quan trọng):
       Fid- = P(y=1 | G) - P(y=1 | G[S])
       Giá trị càng thấp (gần 0) -> Subgraph được giữ lại tự nó đủ để duy trì niềm tin dự đoán.

    3. Sparsity (Độ cô đọng của lời giải thích):
       Sparsity = 1 - (|S| / |E|)
       Miền giá trị [0, 1]. Càng cao -> Lời giải thích càng súc tích, chọn lọc.

    4. Stability / Robustness (Tính bền vững trước nhiễu đầu vào):
       Jaccard similarity của tập cạnh quan trọng top-k khi thêm nhiễu Gaussian nhẹ vào đặc trưng đỉnh:
       Jaccard(S_gốc, S_nhiễu) = |S_gốc ∩ S_nhiễu| / |S_gốc ∪ S_nhiễu|

    5. Insertion / Deletion Curve (Đường cong bổ sung / loại trừ tuần tự).

    6. ROAR (Remove and Retrain):
       - D10: Đặt false theo thiết kế, xai_metrics_by_graph.csv ghi nhận "not_run" kèm lý do.

LƯU Ý PHƯƠNG PHÁP:
    - Mọi can thiệp xóa cạnh (Fidelity) giữ nguyên đặc trưng đỉnh (kể cả touch_count)
      để cô lập tác động của cấu trúc cạnh mà không tạo ra hiệu ứng Out-of-Distribution cho đỉnh.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from xai_football.xai.subgraph import select_important_edges


def resolve_top_k(num_edges: int, top_k: int | None = None, top_k_ratio: float = 0.3) -> int:
    """Xác định k theo quyết định D6: k = max(1, round(top_k_ratio * num_edges))."""
    if top_k is not None and top_k > 0:
        return min(int(top_k), num_edges)
    return max(1, int(round(top_k_ratio * num_edges)))


@torch.no_grad()
def _predict_graph_prob(model: nn.Module, data: Data, device: torch.device) -> float:
    """Dự đoán xác suất nhị phân cho một đồ thị Data đơn lẻ (lớp 1 - có cú sút)."""
    model.eval()
    data = data.to(device)
    num_nodes = data.x.shape[0]
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    edge_attr = getattr(data, "edge_attr", None)
    logit = model(x=data.x, edge_index=data.edge_index, edge_attr=edge_attr, batch=batch)
    prob = float(torch.sigmoid(logit.view(-1))[0].item())
    return prob


def compute_fidelity_plus(
    model: nn.Module,
    graph: Data,
    edge_mask: np.ndarray,
    top_k: int | None = None,
    threshold: float | None = None,
    device: torch.device | str = "cpu",
) -> float:
    r"""Tính Fidelity+ (mức giảm xác suất khi xóa các cạnh quan trọng S).

    Công thức: Fid+ = P(y=1 | G) - P(y=1 | G \ S).
    LƯU Ý: Can thiệp xóa cạnh giữ nguyên đặc trưng đỉnh (kể cả touch_count).
    """
    dev = torch.device(device)
    n_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
    if n_edges == 0:
        return 0.0

    orig_prob = _predict_graph_prob(model, graph, dev)

    effective_k = resolve_top_k(n_edges, top_k) if threshold is None else None
    is_imp = select_important_edges(edge_mask, top_k=effective_k, threshold=threshold)
    keep_indices = np.where(~is_imp)[0]

    if len(keep_indices) == 0:
        new_edge_index = torch.zeros((2, 0), dtype=torch.long, device=dev)
        new_edge_attr = (
            torch.zeros((0, graph.edge_attr.shape[1]), dtype=graph.edge_attr.dtype, device=dev)
            if hasattr(graph, "edge_attr") and graph.edge_attr is not None
            else None
        )
    else:
        keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=dev)
        new_edge_index = graph.edge_index[:, keep_tensor]
        new_edge_attr = (
            graph.edge_attr[keep_tensor]
            if hasattr(graph, "edge_attr") and graph.edge_attr is not None
            else None
        )

    g_mod = copy.copy(graph)
    g_mod.edge_index = new_edge_index
    g_mod.edge_attr = new_edge_attr

    drop_prob = _predict_graph_prob(model, g_mod, dev)
    fid_plus = orig_prob - drop_prob
    return round(float(fid_plus), 4)


def compute_fidelity_minus(
    model: nn.Module,
    graph: Data,
    edge_mask: np.ndarray,
    top_k: int | None = None,
    threshold: float | None = None,
    device: torch.device | str = "cpu",
) -> float:
    """Tính Fidelity- (mức giảm xác suất khi CHỈ GIỮ các cạnh quan trọng S).

    Công thức: Fid- = P(y=1 | G) - P(y=1 | G[S]).
    LƯU Ý: Can thiệp xóa cạnh giữ nguyên đặc trưng đỉnh (kể cả touch_count).
    """
    dev = torch.device(device)
    n_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
    if n_edges == 0:
        return 0.0

    orig_prob = _predict_graph_prob(model, graph, dev)

    effective_k = resolve_top_k(n_edges, top_k) if threshold is None else None
    is_imp = select_important_edges(edge_mask, top_k=effective_k, threshold=threshold)
    keep_indices = np.where(is_imp)[0]

    if len(keep_indices) == 0:
        new_edge_index = torch.zeros((2, 0), dtype=torch.long, device=dev)
        new_edge_attr = (
            torch.zeros((0, graph.edge_attr.shape[1]), dtype=graph.edge_attr.dtype, device=dev)
            if hasattr(graph, "edge_attr") and graph.edge_attr is not None
            else None
        )
    else:
        keep_tensor = torch.tensor(keep_indices, dtype=torch.long, device=dev)
        new_edge_index = graph.edge_index[:, keep_tensor]
        new_edge_attr = (
            graph.edge_attr[keep_tensor]
            if hasattr(graph, "edge_attr") and graph.edge_attr is not None
            else None
        )

    g_mod = copy.copy(graph)
    g_mod.edge_index = new_edge_index
    g_mod.edge_attr = new_edge_attr

    kept_prob = _predict_graph_prob(model, g_mod, dev)
    fid_minus = orig_prob - kept_prob
    return round(float(fid_minus), 4)


def compute_sparsity(
    edge_mask: np.ndarray,
    top_k: int | None = None,
    threshold: float | None = None,
) -> float:
    """Tính Sparsity (tỉ lệ các cạnh bị loại khỏi lời giải thích).

    Công thức: Sparsity = 1 - (|S| / |E|).
    """
    n_edges = len(edge_mask)
    if n_edges == 0:
        return 1.0

    effective_k = resolve_top_k(n_edges, top_k) if threshold is None else None
    is_imp = select_important_edges(edge_mask, top_k=effective_k, threshold=threshold)
    n_selected = int(np.sum(is_imp))
    sparsity = 1.0 - (n_selected / n_edges)
    return round(float(sparsity), 4)


def compute_stability(
    explainer_func: Any,
    graph: Data,
    base_mask: np.ndarray,
    n_repeats: int = 5,
    noise_std: float = 0.01,
    top_k: int | None = None,
    threshold: float | None = None,
    seed: int = 42,
) -> float:
    """Tính độ ổn định (Stability) bằng Jaccard similarity qua các lần thêm nhiễu nhẹ vào node features.

    Jaccard(S1, S2) = |S1 ∩ S2| / |S1 ∪ S2|.
    """
    n_edges = len(base_mask)
    if n_edges == 0:
        return 1.0

    effective_k = resolve_top_k(n_edges, top_k) if threshold is None else None
    base_set = set(np.where(select_important_edges(base_mask, top_k=effective_k, threshold=threshold))[0])
    if not base_set:
        return 1.0

    rng = np.random.default_rng(seed)
    jaccards = []

    for _ in range(n_repeats):
        g_noisy = copy.copy(graph)
        noise = torch.tensor(
            rng.normal(0, noise_std, size=graph.x.shape),
            dtype=graph.x.dtype,
            device=graph.x.device,
        )
        g_noisy.x = graph.x + noise

        # Chạy lại giải thích
        res_noisy = explainer_func(g_noisy)
        noisy_mask = res_noisy.edge_mask if hasattr(res_noisy, "edge_mask") else res_noisy
        noisy_set = set(
            np.where(select_important_edges(noisy_mask, top_k=effective_k, threshold=threshold))[0]
        )

        intersection = len(base_set.intersection(noisy_set))
        union = len(base_set.union(noisy_set))
        jaccard = intersection / union if union > 0 else 1.0
        jaccards.append(jaccard)

    return round(float(np.mean(jaccards)), 4)


def compute_insertion_deletion_curves(
    model: nn.Module,
    graph: Data,
    edge_mask: np.ndarray,
    steps: int = 10,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Tính đường cong và AUC cho Deletion Curve và Insertion Curve."""
    dev = torch.device(device)
    n_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
    if n_edges == 0:
        return {"deletion_auc": 0.0, "insertion_auc": 0.0}

    sorted_edge_indices = np.argsort(-edge_mask).tolist()

    # 1. Deletion curve
    del_probs = []
    for step in range(len(sorted_edge_indices) + 1):
        del_edges = set(sorted_edge_indices[:step])
        keep = [i for i in range(n_edges) if i not in del_edges]

        g_cur = copy.copy(graph)
        if len(keep) == 0:
            g_cur.edge_index = torch.zeros((2, 0), dtype=torch.long, device=dev)
            g_cur.edge_attr = (
                torch.zeros((0, graph.edge_attr.shape[1]), dtype=graph.edge_attr.dtype, device=dev)
                if hasattr(graph, "edge_attr") and graph.edge_attr is not None
                else None
            )
        else:
            k_tensor = torch.tensor(keep, dtype=torch.long, device=dev)
            g_cur.edge_index = graph.edge_index[:, k_tensor]
            g_cur.edge_attr = (
                graph.edge_attr[k_tensor]
                if hasattr(graph, "edge_attr") and graph.edge_attr is not None
                else None
            )
        p = _predict_graph_prob(model, g_cur, dev)
        del_probs.append(p)

    # 2. Insertion curve
    ins_probs = []
    for step in range(len(sorted_edge_indices) + 1):
        keep = sorted_edge_indices[:step]
        g_cur = copy.copy(graph)
        if len(keep) == 0:
            g_cur.edge_index = torch.zeros((2, 0), dtype=torch.long, device=dev)
            g_cur.edge_attr = (
                torch.zeros((0, graph.edge_attr.shape[1]), dtype=graph.edge_attr.dtype, device=dev)
                if hasattr(graph, "edge_attr") and graph.edge_attr is not None
                else None
            )
        else:
            k_tensor = torch.tensor(keep, dtype=torch.long, device=dev)
            g_cur.edge_index = graph.edge_index[:, k_tensor]
            g_cur.edge_attr = (
                graph.edge_attr[k_tensor]
                if hasattr(graph, "edge_attr") and graph.edge_attr is not None
                else None
            )
        p = _predict_graph_prob(model, g_cur, dev)
        ins_probs.append(p)

    def _trapezoid(arr: list[float]) -> float:
        y_arr = np.asarray(arr, dtype=float)
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(y_arr))
        elif hasattr(np, "trapz"):
            return float(getattr(np, "trapz")(y_arr))
        if len(y_arr) < 2:
            return 0.0
        return float(np.sum((y_arr[:-1] + y_arr[1:]) / 2.0))

    del_auc = float(_trapezoid(del_probs) / max(len(del_probs) - 1, 1))
    ins_auc = float(_trapezoid(ins_probs) / max(len(ins_probs) - 1, 1))

    return {
        "deletion_auc": round(del_auc, 4),
        "insertion_auc": round(ins_auc, 4),
    }


def compute_sparsity_fidelity_curve(
    model: nn.Module,
    graph: Data,
    edge_mask: np.ndarray,
    device: torch.device | str = "cpu",
) -> list[dict[str, Any]]:
    """D6: Quét k từ 1 đến E-1 để vẽ đường cong Sparsity–Fidelity."""
    n_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
    if n_edges <= 1:
        return []

    results = []
    for k in range(1, n_edges):
        f_plus = compute_fidelity_plus(model, graph, edge_mask, top_k=k, device=device)
        f_minus = compute_fidelity_minus(model, graph, edge_mask, top_k=k, device=device)
        sp = compute_sparsity(edge_mask, top_k=k)
        results.append(
            {
                "k": k,
                "n_edges": n_edges,
                "sparsity": sp,
                "fidelity_plus": f_plus,
                "fidelity_minus": f_minus,
            }
        )
    return results


def evaluate_explanation_quality(
    model: nn.Module,
    graph: Data,
    edge_mask: np.ndarray,
    top_k: int | None = None,
    threshold: float | None = None,
    device: torch.device | str = "cpu",
    explainer_func: Any = None,
    n_stability_repeats: int = 5,
    noise_std: float = 0.01,
) -> dict[str, Any]:
    """Tổng hợp đầy đủ bộ độ đo đánh giá chất lượng của một lời giải thích XAI."""
    fid_plus = compute_fidelity_plus(
        model, graph, edge_mask, top_k=top_k, threshold=threshold, device=device
    )
    fid_minus = compute_fidelity_minus(
        model, graph, edge_mask, top_k=top_k, threshold=threshold, device=device
    )
    sparsity = compute_sparsity(edge_mask, top_k=top_k, threshold=threshold)
    curves = compute_insertion_deletion_curves(model, graph, edge_mask, device=device)

    stability = (
        compute_stability(
            explainer_func=explainer_func,
            graph=graph,
            base_mask=edge_mask,
            n_repeats=n_stability_repeats,
            noise_std=noise_std,
            top_k=top_k,
            threshold=threshold,
        )
        if explainer_func is not None
        else None
    )

    return {
        "fidelity_plus": fid_plus,
        "fidelity_minus": fid_minus,
        "sparsity": sparsity,
        "stability": stability,
        "deletion_auc": curves["deletion_auc"],
        "insertion_auc": curves["insertion_auc"],
        "roar": "not_run",  # D10: ghi nhận not_run
    }
