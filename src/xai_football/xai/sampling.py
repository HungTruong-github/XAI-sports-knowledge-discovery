"""Bước 7 — Lấy mẫu phân tầng (Stratified Sampling) cho thực nghiệm XAI.

Tham chiếu: configs/xai.yaml, docs/decisions/0006-giao-thuc-xai.md (Quyết định D7).
Nguyên tắc:
    - Lọc chặt chẽ các đồ thị có từ `min_edges` cạnh trở lên (mặc định D2: min_edges = 4).
    - Phân tầng cân bằng theo lớp (dương/âm) và theo quy mô đồ thị (4 cạnh, 5-6 cạnh, >=7 cạnh).
    - Phủ đều các trận đấu thuộc tập kiểm thử (đủ 57 trận test).
    - Tất định hoàn toàn theo `seed`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from xai_football.utils.logging import get_logger

logger = get_logger("sampling")


def get_size_bucket(num_edges: int) -> str:
    """Xác định nhóm quy mô đồ thị theo D7."""
    if num_edges < 4:
        return "<4"
    elif num_edges == 4:
        return "4"
    elif num_edges in (5, 6):
        return "5-6"
    else:
        return "7+"


def _stratified_sample_group(
    graphs: list[Data],
    target_count: int,
    rng: np.random.Generator,
) -> list[Data]:
    """Lấy mẫu phân tầng target_count đồ thị từ danh sách graphs theo (match_id, size_bucket)."""
    if len(graphs) <= target_count:
        return list(graphs)

    # Gom nhóm theo (match_id, size_bucket)
    strata: dict[tuple[int, str], list[Data]] = defaultdict(list)
    for g in graphs:
        mid = int(getattr(g, "match_id", 0))
        n_edges = g.edge_index.size(1) if g.edge_index is not None else 0
        bucket = get_size_bucket(n_edges)
        strata[(mid, bucket)].append(g)

    # Trộn thứ tự trong từng stratum bằng rng
    sorted_strata_keys = sorted(strata.keys())
    for key in sorted_strata_keys:
        perm = rng.permutation(len(strata[key]))
        strata[key] = [strata[key][idx] for idx in perm]

    # Lấy mẫu xoay vòng (round-robin) qua các strata để đảm bảo phủ đều các trận và size buckets
    selected: list[Data] = []
    keys_pool = list(sorted_strata_keys)

    while len(selected) < target_count and keys_pool:
        remaining_keys = []
        for key in keys_pool:
            if strata[key]:
                selected.append(strata[key].pop(0))
                if len(selected) >= target_count:
                    break
                if strata[key]:
                    remaining_keys.append(key)
        keys_pool = remaining_keys

    return selected


def stratified_sample(
    graphs: Sequence[Data],
    n_pos: int,
    n_neg: int,
    seed: int = 42,
    min_edges: int = 4,
) -> list[Data]:
    """Lấy mẫu phân tầng đồ thị cho đánh giá XAI theo quyết định D7.

    Args:
        graphs: Danh sách đồ thị nguồn (thường là tập test).
        n_pos: Số lượng mẫu nhãn dương (target = 1) cần lấy.
        n_neg: Số lượng mẫu nhãn âm (target = 0) cần lấy.
        seed: Seed ngẫu nhiên để tái lập tuyệt đối.
        min_edges: Ngưỡng số cạnh tối thiểu (D2: min_edges = 4).

    Returns:
        Danh sách đồ thị đã lấy mẫu, được sắp xếp tất định theo (match_id, possession_id).
    """
    valid_graphs = [
        g
        for g in graphs
        if g.edge_index is not None and g.edge_index.shape[1] >= min_edges
    ]

    pos_graphs: list[Data] = []
    neg_graphs: list[Data] = []

    for g in valid_graphs:
        lbl = int(getattr(g, "y", torch.tensor([0])).view(-1)[0].item())
        if lbl == 1:
            pos_graphs.append(g)
        else:
            neg_graphs.append(g)

    logger.info(
        "Lấy mẫu XAI từ %d đồ thị hợp lệ (>= %d cạnh): %d dương, %d âm",
        len(valid_graphs),
        min_edges,
        len(pos_graphs),
        len(neg_graphs),
    )

    rng = np.random.default_rng(seed)

    sampled_pos = _stratified_sample_group(pos_graphs, n_pos, rng)
    sampled_neg = _stratified_sample_group(neg_graphs, n_neg, rng)

    combined = sampled_pos + sampled_neg

    # Sắp xếp tất định theo (match_id, possession_id)
    combined.sort(
        key=lambda g: (
            int(getattr(g, "match_id", 0)),
            int(getattr(g, "possession_id", 0)),
        )
    )

    logger.info(
        "Đã chọn thành công %d đồ thị (%d dương, %d âm)",
        len(combined),
        len(sampled_pos),
        len(sampled_neg),
    )
    return combined


def create_sample_manifest_df(sampled_graphs: Sequence[Data]) -> pd.DataFrame:
    """Tạo bảng DataFrame kiểm kê danh sách đồ thị đã lấy mẫu."""
    rows: list[dict[str, Any]] = []
    for idx, g in enumerate(sampled_graphs):
        n_edges = g.edge_index.shape[1] if g.edge_index is not None else 0
        n_nodes = g.x.shape[0] if hasattr(g, "x") and g.x is not None else int(getattr(g, "num_nodes", 0))
        target = int(getattr(g, "y", torch.tensor([0])).view(-1)[0].item())
        mid = int(getattr(g, "match_id", 0))
        pid = int(getattr(g, "possession_id", 0))
        team_id = int(getattr(g, "team_id", 0))
        play_pattern = str(getattr(g, "play_pattern", "Unknown"))

        rows.append(
            {
                "graph_index": idx,
                "match_id": mid,
                "possession_id": pid,
                "team_id": team_id,
                "target": target,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "size_bucket": get_size_bucket(n_edges),
                "play_pattern": play_pattern,
            }
        )
    return pd.DataFrame(rows)


def save_sample_manifest(
    sampled_graphs: Sequence[Data],
    output_path: Path | str,
) -> pd.DataFrame:
    """Lưu danh sách kiểm kê xai_sample_manifest.csv."""
    df = create_sample_manifest_df(sampled_graphs)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    logger.info("Đã lưu manifest lấy mẫu XAI vào %s (%d dòng)", p, len(df))
    return df
