"""Bước 3.3-3.4 — Chuyển một possession thành `torch_geometric.data.Data`.

Tham chiếu: docs/Quy_trinh_XAI_Football.md mục 3.3, 3.4; configs/graph.yaml.

Đầu vào là các bảng interim của Bước 2 (possessions.parquet,
possession_events.parquet), không đọc lại JSON thô.

Đồ thị:
    x           [N, F_node]  đặc trưng đỉnh (cầu thủ tham gia possession)
    edge_index  [2, E]       cạnh có hướng người chuyền -> người nhận,
                             mỗi đường chuyền một cạnh (keep_multi_edge)
    edge_attr   [E, F_edge]  đặc trưng cạnh
    y           [1]          nhãn mức đồ thị
Metadata (không phải đặc trưng, dùng để truy ngược kết quả XAI):
    match_id, possession_id, team_id, play_pattern,
    player_ids [N]  — player_id của từng đỉnh
    edge_event_ids  — event_id StatsBomb của từng cạnh
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch_geometric.data import Data

from .features import compute_edge_features, compute_node_features

SUPPORTED_MULTI_EDGE = ("keep_multi_edge",)


def truncate_at_shot(
    passes: Sequence[dict[str, Any]], first_shot_index: int | None
) -> list[dict[str, Any]]:
    """Giữ đường chuyền có event_index NHỎ HƠN cú sút đầu tiên.

    Possession không có cú sút (first_shot_index None/NaN) giữ nguyên.
    """
    if first_shot_index is None or first_shot_index != first_shot_index:  # NaN
        return list(passes)
    return [p for p in passes if p["event_index"] < first_shot_index]


def node_order(passes: Sequence[dict[str, Any]]) -> list[int]:
    """Thứ tự đỉnh = thứ tự xuất hiện đầu tiên theo dòng thời gian — tất định."""
    seen: dict[int, None] = {}
    for p in passes:
        seen.setdefault(int(p["passer_id"]), None)
        seen.setdefault(int(p["recipient_id"]), None)
    return list(seen)


def build_graph(
    possession: dict[str, Any],
    passes: Sequence[dict[str, Any]],
    roles: dict[int, str | None],
    label: int,
    graph_cfg: dict[str, Any],
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
) -> Data:
    """Dựng đồ thị cho một possession.

    `passes` phải đã xếp theo event_index và đã cắt (nếu có) — hàm này không
    lọc thêm, để mọi quyết định loại bỏ nằm ở một chỗ (dataset.py) và được đếm.
    """
    strategy = (graph_cfg.get("edges") or {}).get("multi_edge_strategy")
    if strategy not in SUPPORTED_MULTI_EDGE:
        raise ValueError(
            f"edges.multi_edge_strategy = {strategy!r} chưa được hỗ trợ. "
            f"Hỗ trợ: {SUPPORTED_MULTI_EDGE}"
        )
    if (graph_cfg.get("optional") or {}).get("add_goal_node"):
        raise NotImplementedError("optional.add_goal_node chưa được cài đặt.")
    if not passes:
        raise ValueError("Không dựng được đồ thị từ possession không có đường chuyền.")

    node_cfg = (graph_cfg.get("nodes") or {}).get("features") or {}
    edge_cfg = (graph_cfg.get("edges") or {}).get("features") or {}

    players = node_order(passes)
    local = {pid: i for i, pid in enumerate(players)}

    x = compute_node_features(players, passes, roles, node_cfg, pitch_length, pitch_width)
    edge_attr = compute_edge_features(passes, edge_cfg, pitch_length, pitch_width)
    edge_index = [
        [local[int(p["passer_id"])] for p in passes],
        [local[int(p["recipient_id"])] for p in passes],
    ]

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor([float(label)], dtype=torch.float32),
        num_nodes=len(players),
        player_ids=torch.tensor(players, dtype=torch.long),
        edge_event_ids=[str(p["event_id"]) for p in passes],
        match_id=int(possession["match_id"]),
        possession_id=int(possession["possession_id"]),
        team_id=int(possession["team_id"]),
        play_pattern=possession.get("play_pattern"),
    )
