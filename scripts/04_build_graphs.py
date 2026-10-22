"""Bước 3 — Dựng đồ thị chuyền bóng cho từng possession.

Cách dùng:
    python scripts/04_build_graphs.py --config configs/data.yaml
    python scripts/04_build_graphs.py --config configs/data.yaml --graph-config configs/graph.yaml

Yêu cầu: đã chạy scripts/03_preprocess.py với cùng config.

Đầu ra:
    data/processed/<run_name>/graphs.pt, graph_index.parquet, feature_names.json
    results/tables/<run_name>/graph_build_report.csv
    results/tables/<run_name>/graph_checks.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from xai_football.data.validation import (  # noqa: E402
    CheckResult,
    has_critical_failure,
    run_all_checks,
)
from xai_football.graphs.dataset import (  # noqa: E402
    PASS_COLUMNS,
    build_dataset,
    load_player_roles,
    save_graphs,
)
from xai_football.graphs.features import edge_feature_names, node_feature_names  # noqa: E402
from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import (  # noqa: E402
    get_interim_dir,
    get_processed_dir,
    get_tables_dir,
)

logger = get_logger("build_graphs")

# Các cột đặc trưng phải nằm trong [0, 1] (chuẩn hóa theo kích thước sân)
UNIT_RANGE_FEATURES = ("mean_x", "mean_y", "start_x", "start_y", "end_x", "end_y", "length")


def graph_integrity_checks(
    graphs, node_names, edge_names, possessions, pass_events, truncate, report
):
    """Kiểm tra cấu trúc đồ thị — FAIL nghĩa là đồ thị sai, không được dùng."""
    checks: list[CheckResult] = []

    # Đỉnh không xác định được vai trò có role one-hot toàn 0. Thường là do tên
    # vị trí không khớp bảng POSITION_TO_ROLE — phải điều tra, không bỏ qua.
    unknown = report.nodes_unknown_role
    checks.append(
        CheckResult(
            "nodes_with_unknown_role",
            unknown,
            "PASS" if unknown == 0 else "WARNING",
            note=f"{unknown}/{report.nodes_total} đỉnh không rõ vai trò" if unknown else "",
        )
    )

    bad_index = sum(
        1
        for g in graphs
        if g.num_edges and (int(g.edge_index.min()) < 0 or int(g.edge_index.max()) >= g.num_nodes)
    )
    checks.append(
        CheckResult("edge_index_within_nodes", bad_index, "PASS" if bad_index == 0 else "FAIL")
    )

    isolated = sum(1 for g in graphs if len(set(g.edge_index.flatten().tolist())) != g.num_nodes)
    checks.append(CheckResult("no_isolated_nodes", isolated, "PASS" if isolated == 0 else "FAIL"))

    bad_shape = sum(
        1
        for g in graphs
        if g.x.shape[1] != len(node_names)
        or g.edge_attr.shape[1] != len(edge_names)
        or g.edge_attr.shape[0] != g.num_edges
        or len(g.edge_event_ids) != g.num_edges
    )
    checks.append(
        CheckResult("feature_dims_match_names", bad_shape, "PASS" if bad_shape == 0 else "FAIL")
    )

    non_finite = sum(
        1 for g in graphs if not (g.x.isfinite().all() and g.edge_attr.isfinite().all())
    )
    checks.append(CheckResult("features_finite", non_finite, "PASS" if non_finite == 0 else "FAIL"))

    out_of_range = 0
    affected_examples: list[str] = []
    for idx, g in enumerate(graphs):
        graph_has_oor = False
        for name in UNIT_RANGE_FEATURES:
            if name in node_names:
                col = g.x[:, node_names.index(name)]
            elif name in edge_names:
                col = g.edge_attr[:, edge_names.index(name)]
            else:
                continue
            c_min, c_max = float(col.min()), float(col.max())
            if c_min < 0 or c_max > 1:
                graph_has_oor = True
                val_str = f"min={c_min:.4f}" if c_min < 0 else f"max={c_max:.4f}"
                mid = getattr(g, "match_id", "N/A")
                pid = getattr(g, "possession_id", "N/A")
                if len(affected_examples) < 2:
                    affected_examples.append(
                        f"graph #{idx} (match={mid}, poss={pid}, {name} {val_str})"
                    )
        if graph_has_oor:
            out_of_range += 1

    if out_of_range == 0:
        oor_note = ""
    else:
        sample_str = "; ".join(affected_examples)
        oor_note = f"{out_of_range} đồ thị có giá trị ngoài [0, 1]: {sample_str}"

    # Tọa độ đã được kẹp về trong sân khi tính đặc trưng (features.clip_to_pitch),
    # nên giá trị ngoài [0, 1] lúc này chỉ có thể do lỗi code -> FAIL.
    checks.append(
        CheckResult(
            "normalized_features_in_[0,1]",
            out_of_range,
            "PASS" if out_of_range == 0 else "FAIL",
            note=oor_note,
        )
    )
    checks.append(
        CheckResult(
            "coords_clipped_to_pitch",
            report.coords_clipped,
            "PASS",
            note=(
                f"{report.coords_clipped} tọa độ ngoài khung 120x80 đã được kẹp về biên sân "
                "(sai số ghi nhận của StatsBomb, ví dụ phạt góc x=120.9)"
                if report.coords_clipped
                else ""
            ),
        )
    )

    # Đối chiếu ngược với dữ liệu Bước 2: cạnh = đường chuyền thật, đúng người
    # chuyền -> người nhận, và (nếu cắt) không có cạnh nào sau cú sút.
    pass_lookup = pass_events.set_index(["match_id", "event_id"])[
        ["event_index", "passer_id", "recipient_id"]
    ].to_dict("index")
    shot_idx = possessions.set_index(["match_id", "possession_id"])["first_shot_index"].to_dict()
    wrong_edge, after_shot = 0, 0
    for g in graphs:
        pids = g.player_ids.tolist()
        first_shot = shot_idx.get((g.match_id, g.possession_id))
        for k, eid in enumerate(g.edge_event_ids):
            rec = pass_lookup.get((g.match_id, eid))
            src, dst = g.edge_index[0, k].item(), g.edge_index[1, k].item()
            if rec is None or (pids[src], pids[dst]) != (rec["passer_id"], rec["recipient_id"]):
                wrong_edge += 1
            elif (
                truncate
                and first_shot is not None
                and not math.isnan(first_shot)
                and rec["event_index"] >= first_shot
            ):
                after_shot += 1
    checks.append(
        CheckResult("edges_match_source_passes", wrong_edge, "PASS" if wrong_edge == 0 else "FAIL")
    )
    checks.append(
        CheckResult(
            "no_edges_after_first_shot",
            after_shot,
            "PASS" if after_shot == 0 else "FAIL",
            note="" if truncate else "truncate_at_first_shot = false — không áp dụng",
        )
    )
    return checks


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--graph-config", default="configs/graph.yaml")
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    graph_cfg = load_yaml(args.graph_config)
    interim_dir = get_interim_dir(data_cfg)
    tables_dir = get_tables_dir(data_cfg)
    processed_dir = get_processed_dir(data_cfg)

    required = ["possessions.parquet", "possession_events.parquet", "events.parquet"]
    missing = [f for f in required if not (interim_dir / f).exists()]
    if missing:
        logger.error("Thiếu %s trong %s. Hãy chạy 03_preprocess.py trước.", missing, interim_dir)
        return 1

    possessions = pd.read_parquet(interim_dir / "possessions.parquet")
    pass_events = pd.read_parquet(interim_dir / "possession_events.parquet", columns=PASS_COLUMNS)
    roles = load_player_roles(interim_dir / "events.parquet")
    logger.info(
        "Đọc %d possession, %d đường chuyền, vai trò của %d cặp (trận, cầu thủ)",
        len(possessions),
        len(pass_events),
        len(roles),
    )

    graphs, index_df, report = build_dataset(possessions, pass_events, roles, data_cfg, graph_cfg)
    if not graphs:
        logger.error("Không dựng được đồ thị nào.")
        return 1

    save_graphs(graphs, index_df, graph_cfg, processed_dir)

    report_df = pd.DataFrame([report.as_row()])
    report_df.to_csv(tables_dir / "graph_build_report.csv", index=False, encoding="utf-8-sig")
    print("\n=== GRAPH BUILD REPORT ===")
    for k, v in report.as_row().items():
        print(f"  {k}: {v}")
    print(
        f"  median_nodes: {float(np.median(index_df['n_nodes']))}, "
        f"median_edges: {float(np.median(index_df['n_edges']))}"
    )

    node_names = node_feature_names((graph_cfg.get("nodes") or {}).get("features") or {})
    edge_names = edge_feature_names((graph_cfg.get("edges") or {}).get("features") or {})
    checks = graph_integrity_checks(
        graphs,
        node_names,
        edge_names,
        possessions,
        pass_events,
        truncate=bool(graph_cfg.get("truncate_at_first_shot", True)),
        report=report,
    )
    checks_df = run_all_checks(checks)
    checks_df.to_csv(tables_dir / "graph_checks.csv", index=False, encoding="utf-8-sig")
    print("\n=== KIỂM TRA ĐỒ THỊ ===")
    print(checks_df.to_string(index=False))

    print(f"\nĐồ thị: {processed_dir}")
    print(f"Bảng:   {tables_dir}")
    return 1 if has_critical_failure(checks_df) else 0


if __name__ == "__main__":
    raise SystemExit(main())
