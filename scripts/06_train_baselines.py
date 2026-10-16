"""Bước 5 — Huấn luyện baseline phi đồ thị (Tầng 3a: Logistic Regression, Random Forest, ANN).

Cách dùng:
    python scripts/06_train_baselines.py --config configs/data.yaml
    python scripts/06_train_baselines.py --config configs/data_pilot.yaml

Đầu ra:
    experiments/<run_name>/baselines/tabular_metrics.csv
    results/tables/<run_name>/baseline_comparison.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from xai_football.data.splits import load_match_splits  # noqa: E402
from xai_football.graphs.dataset import load_graphs  # noqa: E402
from xai_football.modeling.baselines import (  # noqa: E402
    graphs_to_tabular_dataset,
    train_and_eval_tabular_baselines,
)
from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import get_processed_dir, get_tables_dir  # noqa: E402
from xai_football.utils.seed import seed_everything  # noqa: E402

logger = get_logger("train_baselines")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model/baseline_tabular.yaml")
    parser.add_argument("--split-config", default="configs/split.yaml")
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    baseline_cfg = load_yaml(args.model_config)
    split_cfg = load_yaml(args.split_config)

    seed_everything(42)

    run_name = data_cfg.get("run_name", "official")
    processed_dir = get_processed_dir(data_cfg)
    tables_dir = get_tables_dir(data_cfg)

    # 1. Tải đồ thị
    graphs, index_df, feature_names = load_graphs(processed_dir)
    logger.info("Đã tải %d đồ thị", len(graphs))

    # 2. Tải split
    base_split_dir = Path(split_cfg.get("output_dir", "data/splits"))
    split_dir = base_split_dir if run_name == "official" else base_split_dir / run_name
    splits = load_match_splits(split_dir)

    train_mids = set(splits["train"])
    val_mids = set(splits["val"])
    test_mids = set(splits["test"])

    # 3. Chuyển đổi đồ thị sang bảng đặc trưng
    X_all, y_all, match_ids = graphs_to_tabular_dataset(graphs)

    tr_mask = [mid in train_mids for mid in match_ids]
    va_mask = [mid in val_mids for mid in match_ids]
    te_mask = [mid in test_mids for mid in match_ids]

    X_train, y_train = X_all[tr_mask], y_all[tr_mask]
    X_val, y_val = X_all[va_mask], y_all[va_mask]
    X_test, y_test = X_all[te_mask], y_all[te_mask]

    logger.info("Dữ liệu bảng: Train=%d, Val=%d, Test=%d", len(X_train), len(X_val), len(X_test))

    # 4. Huấn luyện 3 mô hình
    results = train_and_eval_tabular_baselines(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        config=baseline_cfg,
    )

    # 5. Lưu kết quả
    summary_rows = []
    for model_name, metrics in results["test_metrics"].items():
        row = {"model": model_name, "type": "non_graph"}
        row.update(metrics)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(tables_dir / "baseline_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n=== KẾT QUẢ ĐỐI SÁNH BASELINE PHI ĐỒ THỊ (TEST SET) ===")
    print(summary_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
