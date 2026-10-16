"""Bước 6 — Đánh giá và tổng hợp bảng đối sánh hiệu năng các mô hình (Bài toán 1).

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 6 (Chiến lược đối sánh 3 tầng).
    - Tầng 2: So sánh backbone GNN (GCN, MPNN vs GATv2, GraphSAGE)
    - Tầng 3a: Đối sánh baseline phi đồ thị (Logistic Regression, Random Forest, ANN)

Cách dùng:
    python scripts/07_evaluate.py --config configs/data.yaml
    python scripts/07_evaluate.py --config configs/data_pilot.yaml

Đầu ra:
    results/tables/<run_name>/model_comparison.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import get_tables_dir  # noqa: E402

logger = get_logger("evaluate")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    run_name = data_cfg.get("run_name", "official")
    tables_dir = get_tables_dir(data_cfg)

    rows = []

    # 1. Thu thập kết quả từ các mô hình GNN (Tầng 2)
    gnn_models = [
        ("GCN (đề xuất)", "gcn", "Tier 2 — GNN Backbone"),
        ("MPNN (đề xuất)", "mpnn", "Tier 2 — GNN Backbone"),
        ("GATv2 (baseline)", "gatv2", "Tier 2 — GNN Baseline"),
        ("GraphSAGE (baseline)", "graphsage", "Tier 2 — GNN Baseline"),
    ]

    for display_name, m_key, tier in gnn_models:
        model_dir = Path("experiments") / run_name / m_key
        all_seeds_file = model_dir / "test_metrics_all_seeds.csv"
        single_metrics_file = model_dir / "test_metrics.csv"

        df_metrics = None
        n_runs = 1

        if all_seeds_file.exists():
            df_all = pd.read_csv(all_seeds_file)
            if not df_all.empty:
                df_metrics = df_all
                n_runs = len(df_all)
        elif single_metrics_file.exists():
            df_s = pd.read_csv(single_metrics_file)
            if not df_s.empty:
                df_metrics = df_s
                n_runs = 1
        else:
            # Tìm trong các thư mục con theo seed
            seed_files = list(model_dir.glob("*/test_metrics.csv"))
            if seed_files:
                dfs = [pd.read_csv(sf) for sf in seed_files if sf.is_file()]
                if dfs:
                    df_metrics = pd.concat(dfs, ignore_index=True)
                    n_runs = len(df_metrics)

        if df_metrics is not None and not df_metrics.empty:
            # Lấy giá trị trung bình nếu có nhiều run, hoặc giá trị đơn nếu 1 run
            mean_series = df_metrics.select_dtypes(include=["number"]).mean()
            row = {
                "model": display_name,
                "tier": tier,
                "type": "Graph",
                "n_runs": n_runs,
                "accuracy": round(float(mean_series.get("accuracy", 0.0)), 4),
                "precision": round(float(mean_series.get("precision", 0.0)), 4),
                "recall": round(float(mean_series.get("recall", 0.0)), 4),
                "f1": round(float(mean_series.get("f1", 0.0)), 4),
                "auc_roc": round(float(mean_series.get("auc_roc", 0.0)), 4),
                "auc_pr": round(float(mean_series.get("auc_pr", 0.0)), 4),
                "brier_score": round(float(mean_series.get("brier_score", 0.0)), 4),
                "loss": (
                    round(float(mean_series.get("loss", 0.0)), 4) if "loss" in mean_series else None
                ),
            }
            rows.append(row)

    # 2. Thu thập kết quả từ baseline phi đồ thị (Tầng 3a)
    baseline_file = tables_dir / "baseline_comparison.csv"
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
                "precision": round(float(b_row.get("precision", 0.0)), 4),
                "recall": round(float(b_row.get("recall", 0.0)), 4),
                "f1": round(float(b_row.get("f1", 0.0)), 4),
                "auc_roc": round(float(b_row.get("auc_roc", 0.0)), 4),
                "auc_pr": round(float(b_row.get("auc_pr", 0.0)), 4),
                "brier_score": round(float(b_row.get("brier_score", 0.0)), 4),
                "loss": None,
            }
            rows.append(row)

    if not rows:
        logger.warning(
            "Chưa có kết quả mô hình nào trong experiments/%s hoặc %s.", run_name, baseline_file
        )
        return 0

    comparison_df = pd.DataFrame(rows)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "model_comparison.csv"
    comparison_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n=== BẢNG ĐỐI SÁNH HIỆU NĂNG MÔ HÌNH TRÊN TẬP TEST ({run_name.upper()}) ===")
    print(comparison_df.to_string(index=False))
    print(f"\nĐã lưu bảng đối sánh vào: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
