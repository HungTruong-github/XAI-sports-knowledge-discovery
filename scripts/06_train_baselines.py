"""Bước 5 — Huấn luyện baseline phi đồ thị (Tầng 3a: Logistic Regression, Random Forest, ANN).

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 5.2.
Cách dùng:
    python scripts/06_train_baselines.py --config configs/data.yaml
    python scripts/06_train_baselines.py --config configs/data.yaml --seeds 42,123,456,789,1024
    python scripts/06_train_baselines.py --config configs/data_pilot.yaml --seed 42

Đầu ra:
    experiments/<run_name>/baselines/<seed>/validation_metrics.csv
    experiments/<run_name>/baselines/<seed>/test_metrics.csv
    experiments/<run_name>/baselines/<seed>/run_manifest.json
    experiments/<run_name>/baselines/test_metrics_all_seeds.csv
    experiments/<run_name>/baselines/aggregate_summary.json
    results/tables/<run_name>/baseline_comparison.csv (test metrics: mean/std)
    results/tables/<run_name>/val_baseline_comparison.csv (val metrics: mean/std)
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

from xai_football.data.splits import load_match_splits  # noqa: E402
from xai_football.graphs.dataset import load_graphs  # noqa: E402
from xai_football.modeling.baselines import (  # noqa: E402
    graphs_to_tabular_dataset,
    train_and_eval_tabular_baselines,
)
from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import (  # noqa: E402
    get_experiments_dir,
    get_processed_dir,
    get_split_dir,
    get_tables_dir,
    resolve_project_path,
)
from xai_football.utils.provenance import compute_dict_sha256, get_git_info  # noqa: E402
from xai_football.utils.seed import seed_everything  # noqa: E402

logger = get_logger("train_baselines")


def resolve_baseline_seeds(
    train_cfg: dict[str, Any] | None,
    cli_seed: int | None = None,
    cli_seeds: str | None = None,
) -> tuple[list[int], bool]:
    """Xác định danh sách seed cho baseline theo thứ tự ưu tiên."""
    if cli_seed is not None and cli_seeds is None:
        return [int(cli_seed)], True
    if cli_seeds:
        return [int(s.strip()) for s in cli_seeds.split(",") if s.strip()], False
    if train_cfg:
        configured_seeds = train_cfg.get("seeds")
        n_runs = int(train_cfg.get("n_runs", len(configured_seeds) if configured_seeds else 1))
        if configured_seeds and isinstance(configured_seeds, list):
            return [int(s) for s in configured_seeds[:n_runs]], False
    return [42], False


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model/baseline_tabular.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--split-config", default="configs/split.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Chạy seed đơn (chế độ debug/test)")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Danh sách seed cách nhau dấu phẩy (vd: 42,123,456,789,1024)",
    )
    parser.add_argument(
        "--threshold-strategy", default="val_optimal", choices=["val_optimal", "fixed_0.5"]
    )
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    baseline_cfg = load_yaml(args.model_config)
    split_cfg = load_yaml(args.split_config)
    train_cfg_path = resolve_project_path(args.train_config)
    train_cfg = load_yaml(train_cfg_path) if train_cfg_path.exists() else None

    seeds, is_debug_run = resolve_baseline_seeds(
        train_cfg, cli_seed=args.seed, cli_seeds=args.seeds
    )
    logger.info("Huấn luyện Baseline: %d seeds=%s (debug_mode=%s)", len(seeds), seeds, is_debug_run)

    run_name = data_cfg.get("run_name", "official")
    processed_dir = get_processed_dir(data_cfg)
    tables_dir = get_tables_dir(data_cfg)

    exp_dir = get_experiments_dir(run_name, "_debug" if is_debug_run else "") / "baselines"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tải đồ thị
    graphs, index_df, feature_names = load_graphs(processed_dir)
    logger.info("Đã tải %d đồ thị", len(graphs))

    # 2. Tải split
    splits = load_match_splits(get_split_dir(split_cfg, run_name))

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

    # Khóa possession của từng hàng test/val — để Bước 6 ghép đúng mẫu với dự đoán GNN
    def _keys(mask: list[bool]) -> pd.DataFrame:
        sel = [g for g, keep in zip(graphs, mask, strict=True) if keep]
        return pd.DataFrame(
            {
                "match_id": [int(g.match_id) for g in sel],
                "possession_id": [int(g.possession_id) for g in sel],
            }
        )

    test_keys = _keys(te_mask)
    val_keys = _keys(va_mask)

    logger.info("Dữ liệu bảng: Train=%d, Val=%d, Test=%d", len(X_train), len(X_val), len(X_test))

    git_info = get_git_info()
    all_val_records: list[dict[str, Any]] = []
    all_test_records: list[dict[str, Any]] = []

    start_total_time = time.time()

    for seed in seeds:
        seed_everything(seed)
        seed_dir = exp_dir / str(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
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
            seed=seed,
        )
        elapsed = time.time() - t0

        seed_val_rows, seed_test_rows = [], []
        for model_name, m_val in results["val_metrics"].items():
            row_v = {
                "model": model_name,
                "tier": "Tier 3(a) — Non-graph Baseline",
                "type": "Tabular",
                "split": "validation",
                "seed": seed,
            }
            row_v.update(m_val)
            seed_val_rows.append(row_v)
            all_val_records.append(row_v)

            m_test = results["test_metrics"].get(model_name, {})
            row_t = {
                "model": model_name,
                "tier": "Tier 3(a) — Non-graph Baseline",
                "type": "Tabular",
                "split": "test",
                "seed": seed,
            }
            row_t.update(m_test)
            seed_test_rows.append(row_t)
            all_test_records.append(row_t)

        pd.DataFrame(seed_val_rows).to_csv(seed_dir / "validation_metrics.csv", index=False)
        pd.DataFrame(seed_test_rows).to_csv(seed_dir / "test_metrics.csv", index=False)

        manifest_seed = {
            "model_type": "tabular_baselines",
            "models": list(results["models"].keys()),
            "features": list(X_all.columns),
            "seed": seed,
            "threshold_strategy": args.threshold_strategy,
            "thresholds": results["thresholds"],
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "training_time_seconds": round(elapsed, 2),
            "git": git_info,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(seed_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_seed, f, indent=2, ensure_ascii=False)

        # Lưu predictions của MỌI seed (kèm khóa possession) — đầu vào kiểm định ở Bước 6
        for split_name, keys, y_true, probs in (
            ("test", test_keys, y_test, results["test_probs"]),
            ("val", val_keys, y_val, results["val_probs"]),
        ):
            pred_df = keys.copy()
            pred_df["y_true"] = y_true
            pred_df["lr_prob"] = probs["logistic_regression"]
            pred_df["rf_prob"] = probs["random_forest"]
            pred_df["ann_prob"] = probs["ann"]
            pred_df.to_parquet(seed_dir / f"{split_name}_predictions.parquet", index=False)

        # Giữ tệp gốc ở thư mục baselines/ (seed đầu tiên) để notebook cũ vẫn đọc được
        if seed == seeds[0]:
            legacy = test_keys.copy()
            legacy["y_test"] = y_test
            legacy["lr_prob"] = results["test_probs"]["logistic_regression"]
            legacy["rf_prob"] = results["test_probs"]["random_forest"]
            legacy["ann_prob"] = results["test_probs"]["ann"]
            legacy.to_parquet(exp_dir / "test_predictions.parquet", index=False)

    total_elapsed = time.time() - start_total_time

    val_df_all = pd.DataFrame(all_val_records)
    test_df_all = pd.DataFrame(all_test_records)

    val_df_all.to_csv(exp_dir / "val_metrics_all_seeds.csv", index=False)
    test_df_all.to_csv(exp_dir / "test_metrics_all_seeds.csv", index=False)

    # Tính toán mean và std cho từng model baseline
    summary_test_rows = []
    summary_val_rows = []
    models_list = ["logistic_regression", "random_forest", "ann"]

    for m_name in models_list:
        sub_t = test_df_all[test_df_all["model"] == m_name]
        sub_v = val_df_all[val_df_all["model"] == m_name]
        numeric_cols = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc_roc",
            "auc_pr",
            "brier_score",
        ]

        # Test row summary
        row_t: dict[str, Any] = {
            "model": m_name,
            "tier": "Tier 3(a) — Non-graph Baseline",
            "type": "Tabular",
            "split": "test",
            "n_runs": len(sub_t),
        }
        for col in numeric_cols:
            mean_val = float(sub_t[col].mean()) if col in sub_t else 0.0
            std_val = float(sub_t[col].std(ddof=1)) if len(sub_t) > 1 and col in sub_t else 0.0
            row_t[col] = round(mean_val, 4)
            row_t[f"{col}_std"] = round(std_val, 4)
        if "threshold" in sub_t:
            row_t["threshold"] = round(float(sub_t["threshold"].mean()), 4)
        summary_test_rows.append(row_t)

        # Val row summary
        row_v: dict[str, Any] = {
            "model": m_name,
            "tier": "Tier 3(a) — Non-graph Baseline",
            "type": "Tabular",
            "split": "validation",
            "n_runs": len(sub_v),
        }
        for col in numeric_cols:
            mean_val = float(sub_v[col].mean()) if col in sub_v else 0.0
            std_val = float(sub_v[col].std(ddof=1)) if len(sub_v) > 1 and col in sub_v else 0.0
            row_v[col] = round(mean_val, 4)
            row_v[f"{col}_std"] = round(std_val, 4)
        if "threshold" in sub_v:
            row_v["threshold"] = round(float(sub_v["threshold"].mean()), 4)
        summary_val_rows.append(row_v)

    summary_test_df = pd.DataFrame(summary_test_rows)
    summary_val_df = pd.DataFrame(summary_val_rows)

    # Lưu aggregate summary manifest
    aggregate_manifest = {
        "model_type": "tabular_baselines",
        "status": "completed",
        # Đủ seed = ĐÚNG danh sách seed cấu hình trong train.yaml (không chỉ đủ số lượng)
        "is_complete_official": (
            not is_debug_run
            and train_cfg is not None
            and set(seeds) >= set(resolve_baseline_seeds(train_cfg)[0])
            and len(seeds) >= int(train_cfg.get("n_runs", 5))
        ),
        "configured_seeds": seeds,
        "n_runs": len(seeds),
        "features": list(X_all.columns),
        "training_time_total_seconds": round(total_elapsed, 2),
        "git": git_info,
        "config_hash": compute_dict_sha256(baseline_cfg),
        "test_metrics_summary": summary_test_df.to_dict(orient="records"),
        "val_metrics_summary": summary_val_df.to_dict(orient="records"),
    }
    with open(exp_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_manifest, f, indent=2, ensure_ascii=False)

    # Đồng bộ bảng kết quả chính thức nếu không phải debug run
    if not is_debug_run:
        tables_dir.mkdir(parents=True, exist_ok=True)
        summary_test_df.to_csv(
            tables_dir / "baseline_comparison.csv", index=False, encoding="utf-8-sig"
        )
        summary_val_df.to_csv(
            tables_dir / "val_baseline_comparison.csv", index=False, encoding="utf-8-sig"
        )

    print(f"\n=== KẾT QUẢ ĐỐI SÁNH BASELINE PHI ĐỒ THỊ ({len(seeds)} SEEDS - TEST SET) ===")
    print(summary_test_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
