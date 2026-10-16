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
        metrics_file = Path("experiments") / run_name / m_key / "test_metrics.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            m_dict = df.iloc[0].to_dict()
            row = {
                "model": display_name,
                "tier": tier,
                "type": "Graph",
                "accuracy": m_dict.get("accuracy"),
                "precision": m_dict.get("precision"),
                "recall": m_dict.get("recall"),
                "f1": m_dict.get("f1"),
                "auc_roc": m_dict.get("auc_roc"),
                "auc_pr": m_dict.get("auc_pr"),
                "brier_score": m_dict.get("brier_score"),
                "loss": m_dict.get("loss"),
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
                "accuracy": b_row.get("accuracy"),
                "precision": b_row.get("precision"),
                "recall": b_row.get("recall"),
                "f1": b_row.get("f1"),
                "auc_roc": b_row.get("auc_roc"),
                "auc_pr": b_row.get("auc_pr"),
                "brier_score": b_row.get("brier_score"),
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
