"""Bước 3.4 — Dựng toàn bộ tập đồ thị possession và lưu cache.

Tham chiếu: docs/Quy_trinh_XAI_Football.md mục 3.4.

Đầu vào (Bước 2):  data/interim/<run_name>/{possessions, possession_events, events}.parquet
Đầu ra:            data/processed/<run_name>/graphs.pt          danh sách Data
                   data/processed/<run_name>/graph_index.parquet một dòng / đồ thị
                   data/processed/<run_name>/feature_names.json  tên đặc trưng

graph_index.parquet là đầu vào của Bước 4 (chia tập theo TRẬN).

Mọi possession bị loại ở bước này đều được đếm trong GraphBuildReport —
không loại âm thầm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import Data

from ..data.labeling import resolve_labeling
from ..utils.io import load_json, save_json
from ..utils.logging import get_logger
from .build import build_graph, truncate_at_shot
from .features import edge_feature_names, node_feature_names, role_of

logger = get_logger(__name__)

PASS_COLUMNS = [
    "match_id",
    "possession_id",
    "event_id",
    "event_index",
    "minute",
    "second",
    "passer_id",
    "recipient_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "height",
]


@dataclass
class GraphBuildReport:
    """Đếm possession đi qua từng bước của Bước 3."""

    possessions_input: int = 0
    shot_possessions_input: int = 0
    dropped_play_pattern: int = 0
    shot_dropped_play_pattern: int = 0
    dropped_too_short_after_truncation: int = 0
    shot_dropped_too_short_after_truncation: int = 0
    passes_removed_by_truncation: int = 0
    graphs_output: int = 0
    positive_graphs: int = 0
    nodes_unknown_role: int = 0
    nodes_total: int = 0
    edges_total: int = 0

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["positive_rate"] = (
            round(self.positive_graphs / self.graphs_output, 4) if self.graphs_output else 0.0
        )
        return row


def load_player_roles(events_parquet: Path) -> dict[tuple[int, int], str | None]:
    """Vai trò của mỗi cầu thủ trong mỗi trận: (match_id, player_id) -> GK/DF/MF/FW.

    Lấy vị trí xuất hiện NHIỀU NHẤT của cầu thủ trong trận (một cầu thủ có thể
    đổi vị trí giữa trận); hòa thì lấy theo tên vị trí để kết quả tất định.
    """
    ev = pd.read_parquet(events_parquet, columns=["match_id", "player_id", "position_name"])
    ev = ev.dropna(subset=["player_id", "position_name"])
    counts = (
        ev.groupby(["match_id", "player_id", "position_name"]).size().reset_index(name="n")
    )
    counts = counts.sort_values(
        ["match_id", "player_id", "n", "position_name"], ascending=[True, True, False, True]
    )
    top = counts.drop_duplicates(["match_id", "player_id"], keep="first")
    return {
        (int(r.match_id), int(r.player_id)): role_of(r.position_name)
        for r in top.itertuples(index=False)
    }


def _label_column(data_cfg: dict[str, Any]) -> str:
    mode, target, _ = resolve_labeling(data_cfg)
    if mode != "possession_level":
        raise ValueError(
            "Bước 3 hiện chỉ hỗ trợ labeling.mode = 'possession_level' "
            "(một possession = một đồ thị = một nhãn, báo cáo 12/09 mục 1.1)."
        )
    return "has_goal" if target == "goal" else "has_shot"


def build_dataset(
    possessions: pd.DataFrame,
    pass_events: pd.DataFrame,
    roles: dict[tuple[int, int], str | None],
    data_cfg: dict[str, Any],
    graph_cfg: dict[str, Any],
) -> tuple[list[Data], pd.DataFrame, GraphBuildReport]:
    """Dựng đồ thị cho mọi possession hợp lệ.

    Thứ tự lọc: (1) play pattern -> (2) cắt tại cú sút -> (3) còn đủ
    min_passes đường chuyền và >= 2 cầu thủ.
    """
    label_col = _label_column(data_cfg)
    min_passes = (data_cfg.get("preprocessing") or {}).get("min_passes_per_possession", 2)
    pitch = data_cfg.get("pitch") or {}
    pitch_length = pitch.get("length", 120)
    pitch_width = pitch.get("width", 80)
    keep_patterns = graph_cfg.get("play_patterns_keep")
    truncate = bool(graph_cfg.get("truncate_at_first_shot", True))
    if truncate and "first_shot_index" not in possessions.columns:
        # Không có cột này mà vẫn chạy thì sẽ KHÔNG cắt gì cả — lỗi âm thầm.
        raise ValueError(
            "possessions.parquet thiếu cột first_shot_index (dữ liệu Bước 2 cũ). "
            "Hãy chạy lại scripts/03_preprocess.py trước."
        )

    passes_by_key: dict[tuple[int, int], list[dict[str, Any]]] = {}
    ordered = pass_events[PASS_COLUMNS].sort_values(["match_id", "possession_id", "event_index"])
    for rec in ordered.to_dict("records"):
        passes_by_key.setdefault((int(rec["match_id"]), int(rec["possession_id"])), []).append(rec)

    report = GraphBuildReport()
    graphs: list[Data] = []
    index_rows: list[dict[str, Any]] = []

    for poss in possessions.sort_values(["match_id", "possession_id"]).to_dict("records"):
        key = (int(poss["match_id"]), int(poss["possession_id"]))
        is_shot = bool(poss["has_shot"])
        report.possessions_input += 1
        report.shot_possessions_input += is_shot

        if keep_patterns is not None and poss.get("play_pattern") not in keep_patterns:
            report.dropped_play_pattern += 1
            report.shot_dropped_play_pattern += is_shot
            continue

        passes = passes_by_key.get(key, [])
        if truncate:
            kept = truncate_at_shot(passes, poss.get("first_shot_index"))
            report.passes_removed_by_truncation += len(passes) - len(kept)
            passes = kept

        n_players = len({p["passer_id"] for p in passes} | {p["recipient_id"] for p in passes})
        if len(passes) < min_passes or n_players < 2:
            report.dropped_too_short_after_truncation += 1
            report.shot_dropped_too_short_after_truncation += is_shot
            continue

        match_roles = {
            pid: roles.get((key[0], pid))
            for pid in {int(p["passer_id"]) for p in passes}
            | {int(p["recipient_id"]) for p in passes}
        }
        label = int(bool(poss[label_col]))
        graph = build_graph(poss, passes, match_roles, label, graph_cfg, pitch_length, pitch_width)
        graphs.append(graph)

        report.graphs_output += 1
        report.positive_graphs += label
        report.nodes_total += graph.num_nodes
        report.edges_total += graph.num_edges
        report.nodes_unknown_role += sum(1 for r in match_roles.values() if r is None)

        index_rows.append(
            {
                "graph_idx": len(graphs) - 1,
                "match_id": key[0],
                "possession_id": key[1],
                "team_id": int(poss["team_id"]),
                "play_pattern": poss.get("play_pattern"),
                "y": label,
                "n_nodes": graph.num_nodes,
                "n_edges": graph.num_edges,
            }
        )

    return graphs, pd.DataFrame(index_rows), report


def save_graphs(
    graphs: list[Data],
    index_df: pd.DataFrame,
    graph_cfg: dict[str, Any],
    out_dir: Path,
) -> None:
    node_cfg = (graph_cfg.get("nodes") or {}).get("features") or {}
    edge_cfg = (graph_cfg.get("edges") or {}).get("features") or {}
    torch.save(graphs, out_dir / "graphs.pt")
    index_df.to_parquet(out_dir / "graph_index.parquet", index=False, engine="pyarrow")
    save_json(
        {
            "node_features": node_feature_names(node_cfg),
            "edge_features": edge_feature_names(edge_cfg),
        },
        out_dir / "feature_names.json",
    )
    logger.info("Đã ghi %d đồ thị vào %s", len(graphs), out_dir)


def load_graphs(out_dir: Path) -> tuple[list[Data], pd.DataFrame, dict[str, list[str]]]:
    """Nạp lại cache. weights_only=False vì tệp chứa đối tượng Data (do chính
    pipeline sinh ra, không phải tệp từ nguồn ngoài)."""
    graphs = torch.load(out_dir / "graphs.pt", weights_only=False)
    index_df = pd.read_parquet(out_dir / "graph_index.parquet")
    names = load_json(out_dir / "feature_names.json")
    return graphs, index_df, names
