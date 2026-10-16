"""Bước 5 — Huấn luyện baseline phi đồ thị (Tầng 3a: Logistic Regression, Random Forest, ANN).

Cách dùng:
    python scripts/06_train_baselines.py --config configs/data.yaml
    python scripts/06_train_baselines.py --config configs/data_pilot.yaml

Đầu ra:
    experiments/<run_name>/baselines/validation_metrics.csv
    experiments/<run_name>/baselines/test_metrics.csv
    experiments/<run_name>/baselines/run_manifest.json
    results/tables/<run_name>/baseline_comparison.csv (test metrics)
    results/tables/<run_name>/val_baseline_comparison.csv (val metrics)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold-strategy", default="val_optimal", choices=["val_optimal", "fixed_0.5"]
    )
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    baseline_cfg = load_yaml(args.model_config)
    split_cfg = load_yaml(args.split_config)

    seed_everything(args.seed)

    run_name = data_cfg.get("run_name", "official")
    processed_dir = get_processed_dir(data_cfg)
    tables_dir = get_tables_dir(data_cfg)
    exp_dir = Path("experiments") / run_name / "baselines"
    exp_dir.mkdir(parents=True, exist_ok=True)

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

    # 3. Trích xuất đặc trưng bảng theo đúng schema config
    req_features = baseline_cfg.get("features", None)
    logger.info("Trích xuất các đặc trưng: %s", req_features)
    X_all, y_all, match_ids = graphs_to_tabular_dataset(graphs, feature_names=req_features)

    tr_mask = [mid in train_mids for mid in match_ids]
    va_mask = [mid in val_mids for mid in match_ids]
    te_mask = [mid in test_mids for mid in match_ids]

    X_train, y_train = X_all[tr_mask], y_all[tr_mask]
    X_val, y_val = X_all[va_mask], y_all[va_mask]
    X_test, y_test = X_all[te_mask], y_all[te_mask]

    logger.info("Dữ liệu bảng: Train=%d, Val=%d, Test=%d", len(X_train), len(X_val), len(X_test))

    # 4. Huấn luyện 3 mô hình
    t0 = time.time()
    results = train_and_eval_tabular_baselines(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        config=baseline_cfg,
        threshold_strategy=args.threshold_strategy,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    # 5. Lưu kết quả chi tiết
    val_rows, test_rows = [], []
    for model_name, m_val in results["val_metrics"].items():
        row_v = {
            "model": model_name,
            "tier": "Tier 3(a) — Non-graph Baseline",
            "type": "Tabular",
            "split": "validation",
        }
        row_v.update(m_val)
        val_rows.append(row_v)

        m_test = results["test_metrics"].get(model_name, {})
        row_t = {
            "model": model_name,
            "tier": "Tier 3(a) — Non-graph Baseline",
            "type": "Tabular",
            "split": "test",
        }
        row_t.update(m_test)
        test_rows.append(row_t)

    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)

    # Lưu vào experiments/
    val_df.to_csv(exp_dir / "validation_metrics.csv", index=False)
    test_df.to_csv(exp_dir / "test_metrics.csv", index=False)

    # Lưu bảng đối sánh vào results/tables/
    tables_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(tables_dir / "baseline_comparison.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(tables_dir / "val_baseline_comparison.csv", index=False, encoding="utf-8-sig")

    # Lưu prediction scores để vẽ ROC/PR curves
    pred_data = {"y_val": y_val.tolist(), "y_test": y_test.tolist()}
    for m_name in results["models"]:
        pred_data[f"{m_name}_val_prob"] = results["val_probs"][m_name].tolist()
        pred_data[f"{m_name}_test_prob"] = results["test_probs"][m_name].tolist()

    pred_df = pd.DataFrame(
        {
            "y_test": y_test,
            "lr_prob": results["test_probs"]["logistic_regression"],
            "rf_prob": results["test_probs"]["random_forest"],
            "ann_prob": results["test_probs"]["ann"],
        }
    )
    pred_df.to_parquet(exp_dir / "test_predictions.parquet", index=False)

    # Ghi manifest
    manifest = {
        "model_type": "tabular_baselines",
        "models": list(results["models"].keys()),
        "features": list(X_all.columns),
        "seed": args.seed,
        "threshold_strategy": args.threshold_strategy,
        "thresholds": results["thresholds"],
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "training_time_seconds": round(elapsed, 2),
    }
    with open(exp_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== KẾT QUẢ ĐỐI SÁNH BASELINE PHI ĐỒ THỊ (TEST SET) ===")
    print(test_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
