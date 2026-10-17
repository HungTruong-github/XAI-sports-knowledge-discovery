"""Bước 7 — Chạy giải thích mô hình GNN và Baseline (XAI Pipeline).

Tham chiếu: configs/xai.yaml và docs/Quy_trinh_XAI_Football.md, mục 7.
Cách dùng:
    python scripts/08_explain.py --config configs/data.yaml --explainer gnn_explainer --limit 20
    python scripts/08_explain.py --config configs/data.yaml --explainer all --limit 50

Đầu ra:
    results/xai/<run_name>/<model_name>/<run_id>/xai_run_manifest.json
    results/xai/<run_name>/<model_name>/<run_id>/explanations.parquet
    results/xai/<run_name>/<model_name>/<run_id>/xai_metrics_by_graph.csv
    results/xai/<run_name>/<model_name>/<run_id>/xai_metrics_summary.csv
    results/xai/<run_name>/<model_name>/<run_id>/top_passes_importance.csv
    results/xai/<run_name>/<model_name>/<run_id>/shap_feature_importance.csv (nếu bật SHAP)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from xai_football.data.splits import load_match_splits  # noqa: E402
from xai_football.graphs.dataset import load_graphs  # noqa: E402
from xai_football.modeling.baselines import (  # noqa: E402
    graphs_to_tabular_dataset,
)
from xai_football.modeling.gnn import build_gnn_model  # noqa: E402
from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import get_processed_dir  # noqa: E402
from xai_football.utils.provenance import compute_file_sha256, get_git_info  # noqa: E402
from xai_football.xai.explainers import (  # noqa: E402
    FootballPGExplainer,
    build_explainer,
)
from xai_football.xai.metrics import evaluate_explanation_quality  # noqa: E402
from xai_football.xai.shap_baseline import explain_tabular_baseline_shap  # noqa: E402
from xai_football.xai.subgraph import (  # noqa: E402
    aggregate_player_importance,
    map_explanation_to_passes,
)

logger = get_logger("explain")


def load_best_model_for_xai(
    model_name: str,
    run_name: str,
    model_cfg: dict[str, Any],
    sample_graph: Any,
    device: torch.device,
) -> tuple[torch.nn.Module, Path, str]:
    """Tải checkpoint mô hình đã kiểm chứng tốt nhất để giải thích."""
    models_dir = Path("models") / run_name
    exp_dir = Path("experiments") / run_name / model_name

    ckpt_path = models_dir / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        ckpt_path = exp_dir / "best_model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint mô hình {model_name} tại {ckpt_path}. "
            f"Hãy chạy scripts/06_train.py trước."
        )

    node_in_dim = sample_graph.x.shape[1]
    edge_in_dim = (
        sample_graph.edge_attr.shape[1]
        if hasattr(sample_graph, "edge_attr") and sample_graph.edge_attr is not None
        else 12
    )

    model = build_gnn_model(model_cfg, node_in_dim=node_in_dim, edge_in_dim=edge_in_dim)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    ckpt_sha = compute_file_sha256(ckpt_path)
    logger.info("Đã tải mô hình %s từ %s (SHA-256: %s)", model_name, ckpt_path, ckpt_sha[:12])
    return model, ckpt_path, ckpt_sha


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--model-name", default="mpnn", help="Backbone giải thích (mặc định: mpnn)")
    parser.add_argument("--model-config", default=None, help="Đường dẫn file config mô hình")
    parser.add_argument("--xai-config", default="configs/xai.yaml")
    parser.add_argument("--split-config", default="configs/split.yaml")
    parser.add_argument(
        "--explainer",
        default="gnn_explainer",
        choices=["gnn_explainer", "pg_explainer", "all"],
        help="Thuật toán giải thích",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=30, help="Số lượng đồ thị giải thích mẫu")
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-edges", type=int, default=2, help="Số cạnh tối thiểu của đồ thị XAI")
    parser.add_argument("--output-dir", default=None, help="Ghi đè thư mục kết quả")
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    xai_cfg = load_yaml(args.xai_config)
    split_cfg = load_yaml(args.split_config)

    run_name = data_cfg.get("run_name", "official")
    model_name = args.model_name.lower()

    if args.model_config:
        model_cfg = load_yaml(args.model_config)
    else:
        cfg_file = Path(f"configs/model/{model_name}.yaml")
        if not cfg_file.exists():
            cfg_file = Path("configs/model/mpnn.yaml")
        model_cfg = load_yaml(cfg_file)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Thiết bị sử dụng: %s", device)

    # 1. Tải đồ thị và tập split
    processed_dir = get_processed_dir(data_cfg)
    graphs, index_df, feature_names = load_graphs(processed_dir)

    base_split_dir = Path(split_cfg.get("output_dir", "data/splits"))
    split_dir = base_split_dir if run_name == "official" else base_split_dir / run_name
    splits = load_match_splits(split_dir)
    target_mids = set(splits.get(args.split, []))

    eval_graphs = [
        g
        for g in graphs
        if int(getattr(g, "match_id", 0)) in target_mids and g.edge_index.shape[1] >= args.min_edges
    ]
    logger.info(
        "Tìm thấy %d đồ thị trên tập %s có >= %d cạnh",
        len(eval_graphs),
        args.split,
        args.min_edges,
    )

    if not eval_graphs:
        logger.error("Không có đồ thị nào thỏa mãn điều kiện để giải thích.")
        return 1

    # Giới hạn số đồ thị mẫu nếu có cờ limit
    selected_graphs = eval_graphs[: args.limit]
    logger.info("Chọn %d đồ thị để thực hiện giải thích XAI", len(selected_graphs))

    # 2. Tải mô hình
    sample_g = selected_graphs[0]
    model, ckpt_path, ckpt_sha = load_best_model_for_xai(
        model_name, run_name, model_cfg, sample_g, device
    )

    # 3. Khởi tạo Explainers
    explainers_to_run = (
        ["gnn_explainer", "pg_explainer"] if args.explainer == "all" else [args.explainer]
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("results/xai") / run_name / model_name / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    git_info = get_git_info()
    all_explanations_rows = []
    all_metrics_rows = []
    all_pass_rows = []
    all_player_rows = []

    for expl_name in explainers_to_run:
        logger.info("\n========== BẮT ĐẦU GIẢI THÍCH VỚI %s ==========", expl_name.upper())
        explainer = build_explainer(expl_name, model=model, xai_cfg=xai_cfg, device=device)

        # Nếu là PGExplainer, huấn luyện trước trên tập mẫu
        if isinstance(explainer, FootballPGExplainer):
            train_mids = set(splits.get("train", []))
            train_sub_graphs = [g for g in graphs if int(getattr(g, "match_id", 0)) in train_mids][
                :100
            ]
            explainer.train_on_graphs(train_sub_graphs)

        t_start = time.time()
        for idx, g in enumerate(selected_graphs):
            exp_res = explainer.explain_graph(g, graph_index=idx, target_class=1)
            all_explanations_rows.append(exp_res.to_dict())

            # Ánh xạ về đường chuyền
            pass_df = map_explanation_to_passes(g, exp_res, top_k=3)
            if not pass_df.empty:
                all_pass_rows.append(pass_df)

            # Tổng hợp về cầu thủ
            player_df = aggregate_player_importance(g, exp_res)
            if not player_df.empty:
                all_player_rows.append(player_df)

            # Đánh giá chất lượng giải thích (Fidelity, Sparsity, Insertion/Deletion)
            metrics = evaluate_explanation_quality(
                model=model,
                graph=g,
                edge_mask=exp_res.edge_mask,
                top_k=3,
                device=device,
                explainer_func=(
                    explainer.explain_graph if idx < 5 else None
                ),  # stability đo trên 5 đồ thị đầu
                n_stability_repeats=3,
            )

            metric_row = {
                "match_id": exp_res.match_id,
                "possession_id": exp_res.possession_id,
                "graph_index": idx,
                "explainer": expl_name,
                "pred_prob": exp_res.pred_prob,
                "target": exp_res.target,
                "num_edges": exp_res.num_edges,
                **metrics,
            }
            all_metrics_rows.append(metric_row)

            if (idx + 1) % 10 == 0 or idx == len(selected_graphs) - 1:
                logger.info("Đã giải thích %d/%d đồ thị", idx + 1, len(selected_graphs))

        elapsed = time.time() - t_start
        logger.info("Hoàn tất %s sau %.1f giây", expl_name, elapsed)

    # 4. Lưu kết quả XAI
    exp_df = pd.DataFrame(all_explanations_rows)
    metrics_df = pd.DataFrame(all_metrics_rows)

    exp_df.to_parquet(out_dir / "explanations.parquet", index=False)
    exp_df.to_csv(out_dir / "explanations.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_dir / "xai_metrics_by_graph.csv", index=False, encoding="utf-8-sig")

    if all_pass_rows:
        total_pass_df = pd.concat(all_pass_rows, ignore_index=True)
        total_pass_df.to_csv(
            out_dir / "top_passes_importance.csv", index=False, encoding="utf-8-sig"
        )

    if all_player_rows:
        total_player_df = pd.concat(all_player_rows, ignore_index=True)
        total_player_df.to_csv(
            out_dir / "player_importance_by_possession.csv", index=False, encoding="utf-8-sig"
        )

    # 5. Tổng hợp Metric Summary
    num_cols = ["fidelity_plus", "fidelity_minus", "sparsity", "deletion_auc", "insertion_auc"]
    summary_rows = []
    for expl in explainers_to_run:
        sub = metrics_df[metrics_df["explainer"] == expl]
        row_s: dict[str, Any] = {"explainer": expl, "n_graphs": len(sub)}
        for c in num_cols:
            if c in sub:
                row_s[f"{c}_mean"] = round(float(sub[c].mean()), 4)
                row_s[f"{c}_std"] = round(float(sub[c].std(ddof=1)), 4) if len(sub) > 1 else 0.0
        if "stability" in sub and sub["stability"].dropna().any():
            stab_vals = sub["stability"].dropna()
            row_s["stability_mean"] = round(float(stab_vals.mean()), 4)
            row_s["stability_std"] = (
                round(float(stab_vals.std(ddof=1)), 4) if len(stab_vals) > 1 else 0.0
            )
        summary_rows.append(row_s)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "xai_metrics_summary.csv", index=False, encoding="utf-8-sig")

    # 6. Chạy SHAP trên Tabular Baseline nếu cấu hình cho phép
    shap_cfg = xai_cfg.get("shap_baseline", {})
    if shap_cfg.get("enabled", True):
        try:
            logger.info("Chạy SHAP trên Baseline phi đồ thị (Random Forest)...")
            base_cfg = load_yaml("configs/model/baseline_tabular.yaml")
            req_feat = base_cfg.get("features", None)
            X_all, y_all, m_ids = graphs_to_tabular_dataset(graphs, feature_names=req_feat)

            # Fit nhanh Random Forest mẫu
            from sklearn.ensemble import RandomForestClassifier

            rf_sample = RandomForestClassifier(n_estimators=50, random_state=42)
            rf_sample.fit(X_all[:2000], y_all[:2000])

            shap_res = explain_tabular_baseline_shap(rf_sample, X_all[2000:2200], max_samples=100)
            shap_df = shap_res["feature_importance_df"]
            shap_df.to_csv(
                out_dir / "shap_feature_importance.csv", index=False, encoding="utf-8-sig"
            )
            logger.info("Đã lưu SHAP baseline feature importance vào %s", out_dir)
        except Exception as exc:
            logger.warning("Không thể chạy SHAP baseline: %s", exc)

    # 7. Ghi Manifest
    xai_manifest = {
        "run_id": run_id,
        "run_name": run_name,
        "model_name": model_name,
        "model_checkpoint": str(ckpt_path),
        "model_checkpoint_sha256": ckpt_sha,
        "decision_0001_status": "provisional (possession-level shot classification)",
        "explainers": explainers_to_run,
        "n_graphs_explained": len(selected_graphs),
        "split": args.split,
        "git": git_info,
        "metrics_summary": summary_df.to_dict(orient="records"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "xai_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(xai_manifest, f, indent=2, ensure_ascii=False)

    print("\n=== TỔNG HỢP ĐỘ ĐO CHẤT LƯỢNG GIẢI THÍCH (XAI METRICS SUMMARY) ===")
    print(summary_df.to_string(index=False))
    print(f"\nĐã lưu toàn bộ artifact giải thích vào: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
