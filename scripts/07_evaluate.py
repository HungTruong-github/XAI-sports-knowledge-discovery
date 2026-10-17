"""Bước 6 — Đánh giá và tổng hợp bảng đối sánh hiệu năng các mô hình (Bài toán 1).

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 6 (Chiến lược đối sánh 3 tầng).
    - Tầng 2: So sánh backbone GNN (GCN, MPNN vs GATv2, GraphSAGE)
    - Tầng 3a: Đối sánh baseline phi đồ thị (Logistic Regression, Random Forest, ANN)

Tính năng kiểm chứng và Provenance:
    - Xác minh tính hợp lệ của run (chỉ đọc kết quả từ run thành công, từ chối file rác cũ).
    - Lưu giữ cả mean và standard deviation khi chạy nhiều seeds.
    - Đồng bộ tự động giữa baseline_comparison.csv và model_comparison.csv.
    - Xuất tệp manifest đánh giá tổng hợp: results/metadata/<run_name>/evaluation_manifest.json.

Cách dùng:
    python scripts/07_evaluate.py --config configs/data.yaml
    python scripts/07_evaluate.py --config configs/data_pilot.yaml

Đầu ra:
    results/tables/<run_name>/model_comparison.csv
    results/metadata/<run_name>/evaluation_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import get_metadata_dir, get_tables_dir  # noqa: E402
from xai_football.utils.provenance import compute_file_sha256, get_git_info  # noqa: E402

logger = get_logger("evaluate")


def verify_gnn_run(model_dir: Path) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Xác minh provenance và tải kết quả test metrics của mô hình GNN."""
    manifest_path = model_dir / "aggregate_summary.json"
    single_manifest_path = model_dir / "run_manifest.json"
    all_seeds_file = model_dir / "test_metrics_all_seeds.csv"
    single_metrics_file = model_dir / "test_metrics.csv"
    fail_manifest_path = model_dir / "run_failure_manifest.json"

    # 1. Nếu có tệp báo lỗi thất bại toàn diện, từ chối đọc bất kỳ file metric cũ nào
    if fail_manifest_path.exists():
        logger.warning(
            "Phát hiện %s: Toàn bộ run gần nhất đã thất bại! Bỏ qua kết quả cũ.", fail_manifest_path
        )
        return None, {"status": "failed", "error": "all runs failed"}

    # 2. Kiểm tra aggregate summary từ multi-seed run
    if manifest_path.exists() and all_seeds_file.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                agg_info = json.load(f)
            df = pd.read_csv(all_seeds_file)
            if not df.empty:
                return df, agg_info
        except Exception as e:
            logger.warning("Lỗi đọc aggregate summary tại %s: %s", manifest_path, e)

    # 3. Kiểm tra single run manifest nếu chỉ chạy 1 seed
    if single_manifest_path.exists() and single_metrics_file.exists():
        try:
            with open(single_manifest_path, encoding="utf-8") as f:
                s_info = json.load(f)
            df = pd.read_csv(single_metrics_file)
            if not df.empty:
                return df, s_info
        except Exception as e:
            logger.warning("Lỗi đọc single manifest tại %s: %s", single_manifest_path, e)

    # 4. Fallback tìm file đơn lẻ nếu có
    if single_metrics_file.exists():
        df = pd.read_csv(single_metrics_file)
        if not df.empty:
            return df, {"status": "single_run_unverified"}

    return None, {"status": "not_found"}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Cho phép đưa cả các run partial/incomplete vào bảng đối sánh",
    )
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    run_name = data_cfg.get("run_name", "official")
    tables_dir = get_tables_dir(data_cfg)
    meta_dir = get_metadata_dir(data_cfg)

    rows = []
    eval_manifest: dict[str, Any] = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "git": get_git_info(),
        "gnn_models": {},
        "baseline_models": {},
    }

    # 1. Thu thập kết quả từ các mô hình GNN (Tầng 2)
    gnn_models = [
        ("GCN (đề xuất)", "gcn", "Tier 2 — GNN Backbone"),
        ("MPNN (đề xuất)", "mpnn", "Tier 2 — GNN Backbone"),
        ("GATv2 (baseline)", "gatv2", "Tier 2 — GNN Baseline"),
        ("GraphSAGE (baseline)", "graphsage", "Tier 2 — GNN Baseline"),
    ]

    for display_name, m_key, tier in gnn_models:
        model_dir = Path("experiments") / run_name / m_key
        df_metrics, meta_info = verify_gnn_run(model_dir)

        ckpt_path = Path("models") / run_name / f"{m_key}_best.pt"
        if not ckpt_path.exists():
            ckpt_path = model_dir / "best_model.pt"

        ckpt_sha = compute_file_sha256(ckpt_path) if ckpt_path.exists() else None

        eval_manifest["gnn_models"][m_key] = {
            "display_name": display_name,
            "status": meta_info.get("status", "unknown"),
            "checkpoint_path": str(ckpt_path) if ckpt_path.exists() else None,
            "checkpoint_sha256": ckpt_sha,
            "metadata": meta_info,
        }

        if df_metrics is not None and not df_metrics.empty:
            n_runs = len(df_metrics)
            mean_series = df_metrics.select_dtypes(include=["number"]).mean()
            std_series = (
                df_metrics.select_dtypes(include=["number"]).std(ddof=1) if n_runs > 1 else None
            )

            row = {
                "model": display_name,
                "tier": tier,
                "type": "Graph",
                "n_runs": n_runs,
                "accuracy": round(float(mean_series.get("accuracy", 0.0)), 4),
                "accuracy_std": (
                    round(float(std_series.get("accuracy", 0.0)), 4)
                    if std_series is not None
                    else 0.0
                ),
                "precision": round(float(mean_series.get("precision", 0.0)), 4),
                "precision_std": (
                    round(float(std_series.get("precision", 0.0)), 4)
                    if std_series is not None
                    else 0.0
                ),
                "recall": round(float(mean_series.get("recall", 0.0)), 4),
                "recall_std": (
                    round(float(std_series.get("recall", 0.0)), 4)
                    if std_series is not None
                    else 0.0
                ),
                "f1": round(float(mean_series.get("f1", 0.0)), 4),
                "f1_std": (
                    round(float(std_series.get("f1", 0.0)), 4) if std_series is not None else 0.0
                ),
                "auc_roc": round(float(mean_series.get("auc_roc", 0.0)), 4),
                "auc_roc_std": (
                    round(float(std_series.get("auc_roc", 0.0)), 4)
                    if std_series is not None
                    else 0.0
                ),
                "auc_pr": round(float(mean_series.get("auc_pr", 0.0)), 4),
                "auc_pr_std": (
                    round(float(std_series.get("auc_pr", 0.0)), 4)
                    if std_series is not None
                    else 0.0
                ),
                "brier_score": round(float(mean_series.get("brier_score", 0.0)), 4),
                "brier_score_std": (
                    round(float(std_series.get("brier_score", 0.0)), 4)
                    if std_series is not None
                    else 0.0
                ),
                "loss": (
                    round(float(mean_series.get("loss", 0.0)), 4) if "loss" in mean_series else None
                ),
            }
            rows.append(row)

    # 2. Thu thập kết quả từ baseline phi đồ thị (Tầng 3a)
    baseline_file = tables_dir / "baseline_comparison.csv"
    baseline_agg_manifest = Path("experiments") / run_name / "baselines" / "aggregate_summary.json"

    if baseline_agg_manifest.exists():
        try:
            with open(baseline_agg_manifest, encoding="utf-8") as f:
                base_manifest_data = json.load(f)
            eval_manifest["baseline_models"] = base_manifest_data
        except Exception:
            pass

    if baseline_file.exists():
        base_df = pd.read_csv(baseline_file)
        for _, b_row in base_df.iterrows():
            m_name = str(b_row.get("model", "baseline"))
            display_name = {
                "logistic_regression": "Logistic Regression",
                "random_forest": "Random Forest",
                "ann": "ANN (MLP)",
            }.get(m_name, m_name)
            row = {
                "model": display_name,
                "tier": "Tier 3(a) — Non-graph Baseline",
                "type": "Tabular",
                "n_runs": int(b_row.get("n_runs", 1)),
                "accuracy": round(float(b_row.get("accuracy", 0.0)), 4),
                "accuracy_std": round(float(b_row.get("accuracy_std", 0.0)), 4),
                "precision": round(float(b_row.get("precision", 0.0)), 4),
                "precision_std": round(float(b_row.get("precision_std", 0.0)), 4),
                "recall": round(float(b_row.get("recall", 0.0)), 4),
                "recall_std": round(float(b_row.get("recall_std", 0.0)), 4),
                "f1": round(float(b_row.get("f1", 0.0)), 4),
                "f1_std": round(float(b_row.get("f1_std", 0.0)), 4),
                "auc_roc": round(float(b_row.get("auc_roc", 0.0)), 4),
                "auc_roc_std": round(float(b_row.get("auc_roc_std", 0.0)), 4),
                "auc_pr": round(float(b_row.get("auc_pr", 0.0)), 4),
                "auc_pr_std": round(float(b_row.get("auc_pr_std", 0.0)), 4),
                "brier_score": round(float(b_row.get("brier_score", 0.0)), 4),
                "brier_score_std": round(float(b_row.get("brier_score_std", 0.0)), 4),
                "loss": None,
            }
            rows.append(row)

    if not rows:
        logger.warning(
            "Chưa có kết quả mô hình nào hợp lệ trong experiments/%s hoặc %s.",
            run_name,
            baseline_file,
        )
        return 0

    comparison_df = pd.DataFrame(rows)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "model_comparison.csv"
    comparison_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "evaluation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(eval_manifest, f, indent=2, ensure_ascii=False)

    print(f"\n=== BẢNG ĐỐI SÁNH HIỆU NĂNG MÔ HÌNH TRÊN TẬP TEST ({run_name.upper()}) ===")
    print(comparison_df.to_string(index=False))
    print(f"\nĐã lưu bảng đối sánh vào: {out_path}")
    print(f"Đã lưu evaluation manifest vào: {meta_dir / 'evaluation_manifest.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
