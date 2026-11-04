"""Giải thích Out-of-Fold bằng mô hình Cross-Fitting (Phase 3).

Tham chiếu:
- docs/decisions/0006-giao-thuc-xai.md (Quyết định D11)
- data/splits/crossfit_5fold.json
Quy tắc:
- Với mỗi fold k in 1..5:
  Load mô hình models/crossfit/fold_{k}/best_model.pt
  Lọc đồ thị test_matches của fold k: >= 4 cạnh (D2), tối đa 40 đồ thị/trận
  Giải thích bằng explainer được cấu hình
  Trích xuất điểm quan trọng cầu thủ (D12) và đường chuyền
- Lưu tổng hợp:
  results/xai/crossfit/explanations_oof.parquet
  results/xai/crossfit/player_importance_oof.parquet
  results/xai/crossfit/crossfit_explain_manifest.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from xai_football.graphs.dataset import load_graphs
from xai_football.modeling.gnn import build_gnn_model
from xai_football.utils.io import load_yaml
from xai_football.utils.logging import get_logger
from xai_football.utils.paths import get_processed_dir
from xai_football.utils.provenance import compute_file_sha256, get_git_info
from xai_football.xai.explainers import build_explainer
from xai_football.xai.subgraph import aggregate_player_importance, map_explanation_to_passes

logger = get_logger("crossfit_explain")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--xai-config", default="configs/xai.yaml")
    parser.add_argument("--split-file", default="data/splits/crossfit_5fold.json")
    parser.add_argument("--explainer", default="pg_explainer", help="Explainer sử dụng cho out-of-fold (pg_explainer hoặc gnn_explainer)")
    parser.add_argument("--max-graphs-per-match", type=int, default=40)
    parser.add_argument("--min-edges", type=int, default=4)
    parser.add_argument("--model-dir", default="models/crossfit", help="Thư mục chứa checkpoints của các fold")
    parser.add_argument("--target-matches-file", default=None, help="Tệp split JSON chứa các trận đấu đích cần giải thích (ví dụ: Set B)")
    parser.add_argument("--target-split-key", default=None, help="Key trong tệp split JSON chứa danh sách trận (ví dụ: set_b)")
    parser.add_argument("--output-dir", default="results/xai/crossfit")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    split_path = Path(args.split_file)
    if not split_path.exists():
        logger.error("Không tìm thấy file split %s", split_path)
        return 1

    with open(split_path, encoding="utf-8") as f:
        crossfit_meta = json.load(f)

    # Nếu chỉ định target matches riêng (ví dụ Set B giải thích bằng mô hình Set A)
    override_test_mids: set[int] | None = None
    if args.target_matches_file:
        t_path = Path(args.target_matches_file)
        if t_path.exists():
            with open(t_path, encoding="utf-8") as f:
                t_data = json.load(f)
            key = args.target_split_key or "matches"
            if key in t_data and isinstance(t_data[key], dict) and "matches" in t_data[key]:
                override_test_mids = set(t_data[key]["matches"])
            elif key in t_data and isinstance(t_data[key], list):
                override_test_mids = set(t_data[key])
            logger.info("Đã tải %d trận đấu đích từ %s (key=%s)", len(override_test_mids) if override_test_mids else 0, t_path, key)

    data_cfg = load_yaml(args.config)
    xai_cfg = load_yaml(args.xai_config)
    model_cfg = load_yaml(args.model_config) if args.model_config else load_yaml("configs/model/mpnn.yaml")

    processed_dir = get_processed_dir(data_cfg)
    graphs, _, _ = load_graphs(processed_dir)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_g = graphs[0]
    node_in_dim = sample_g.x.shape[1]
    edge_in_dim = sample_g.edge_attr.shape[1] if hasattr(sample_g, "edge_attr") and sample_g.edge_attr is not None else 12

    all_exp_rows: list[dict[str, Any]] = []
    all_pass_rows: list[pd.DataFrame] = []
    all_player_rows: list[pd.DataFrame] = []

    folds = crossfit_meta["folds"]
    fold_manifests = {}

    for fold_name, fold_data in folds.items():
        ckpt_path = Path(args.model_dir) / fold_name / "best_model.pt"
        if not ckpt_path.exists():
            logger.error("Không tìm thấy checkpoint %s. Hãy chạy scripts/06d_crossfit_train.py trước.", ckpt_path)
            continue

        ckpt_sha = compute_file_sha256(ckpt_path)
        logger.info("\n========== GIẢI THÍCH OOF VỚI %s (Checkpoint SHA: %s) ==========", fold_name.upper(), ckpt_sha[:8])

        # Load mô hình fold
        model = build_gnn_model(model_cfg, node_in_dim=node_in_dim, edge_in_dim=edge_in_dim)
        checkpoint = torch.load(ckpt_path, map_location=device)
        state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        explainer = build_explainer(args.explainer, model=model, xai_cfg=xai_cfg, device=device)

        # Lọc đồ thị: nếu có override_test_mids (Set B), dùng mô hình Set A để giải thích Set B
        if override_test_mids is not None:
            test_mids = override_test_mids
        else:
            test_mids = set(fold_data["test_matches"])
        # Gom nhóm theo match để giới hạn max_graphs_per_match
        match_graphs: dict[int, list[Any]] = defaultdict(list)
        for g in graphs:
            mid = int(getattr(g, "match_id", 0))
            if mid in test_mids and g.edge_index is not None and g.edge_index.shape[1] >= args.min_edges:
                if len(match_graphs[mid]) < args.max_graphs_per_match:
                    match_graphs[mid].append(g)

        fold_graphs = [g for sublist in match_graphs.values() for g in sublist]
        logger.info("[%s] Số đồ thị OOF thỏa điều kiện (>=%d cạnh, max %d/trận): %d đồ thị trên %d trận",
                    fold_name, args.min_edges, args.max_graphs_per_match, len(fold_graphs), len(match_graphs))

        # Nếu là PGExplainer, huấn luyện nhanh trên tập train_matches của fold
        from xai_football.xai.explainers import FootballPGExplainer
        if isinstance(explainer, FootballPGExplainer):
            train_mids = set(fold_data["train_matches"])
            train_sub = [g for g in graphs if int(getattr(g, "match_id", 0)) in train_mids and g.edge_index.shape[1] >= args.min_edges][:100]
            explainer.train_on_graphs(train_sub)

        for idx, g in enumerate(fold_graphs):
            exp_res = explainer.explain_graph(g, graph_index=idx, target_class=1)
            row_dict = exp_res.to_dict()
            row_dict["fold"] = fold_name
            all_exp_rows.append(row_dict)

            pass_df = map_explanation_to_passes(g, exp_res)
            if not pass_df.empty:
                pass_df["fold"] = fold_name
                all_pass_rows.append(pass_df)

            player_df = aggregate_player_importance(g, exp_res)
            if not player_df.empty:
                player_df["fold"] = fold_name
                all_player_rows.append(player_df)

            if (idx + 1) % 50 == 0 or idx == len(fold_graphs) - 1:
                logger.info("[%s] Đã giải thích OOF %d/%d đồ thị", fold_name, idx + 1, len(fold_graphs))

        fold_manifests[fold_name] = {
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": ckpt_sha,
            "n_graphs_explained": len(fold_graphs),
            "n_matches": len(match_graphs),
        }

        if override_test_mids is not None:
            logger.info("Hoàn tất giải thích các trận đấu đích bằng mô hình của %s", fold_name)
            break

    # Lưu kết quả tổng hợp Out-of-Fold
    if all_exp_rows:
        exp_df = pd.DataFrame(all_exp_rows)
        exp_df.to_parquet(out_dir / "explanations_oof.parquet", index=False)
        logger.info("Đã lưu %d giải thích OOF vào %s", len(exp_df), out_dir / "explanations_oof.parquet")

    if all_player_rows:
        total_player_df = pd.concat(all_player_rows, ignore_index=True)
        total_player_df.to_parquet(out_dir / "player_importance_oof.parquet", index=False)
        logger.info("Đã lưu điểm cầu thủ OOF vào %s (%d dòng)", out_dir / "player_importance_oof.parquet", len(total_player_df))

    if all_pass_rows:
        total_pass_df = pd.concat(all_pass_rows, ignore_index=True)
        total_pass_df.to_parquet(out_dir / "top_passes_oof.parquet", index=False)

    manifest = {
        "step": "Phase 3: Out-of-Fold Cross-fitting XAI",
        "explainer": args.explainer,
        "n_total_explanations": len(all_exp_rows),
        "folds": fold_manifests,
        "git": get_git_info(),
    }
    with open(out_dir / "crossfit_explain_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Hoàn tất giải thích OOF! Toàn bộ kết quả lưu tại: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

