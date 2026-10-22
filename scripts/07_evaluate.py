"""Bước 6 — Đánh giá và tổng hợp bảng đối sánh hiệu năng các mô hình (Bài toán 1).

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 6 (Chiến lược đối sánh 3 tầng).
    - Tầng 2: So sánh backbone GNN (GCN, MPNN vs GATv2, GraphSAGE)
    - Tầng 3a: Đối sánh baseline phi đồ thị (Logistic Regression, Random Forest, ANN)

Nguyên tắc:
    - Chỉ nhận run ĐẦY ĐỦ (đủ số seed trong configs/train.yaml, không seed nào lỗi).
      Thiếu/lỗi -> dừng với mã lỗi 1, KHÔNG ghi bảng chính thức. Muốn xem tạm kết
      quả chưa đủ seed thì dùng --allow-partial (bảng có cột `complete` = False).
    - Bảng chính: mean ± std qua các seed trên tập TEST; ngưỡng phân loại của từng
      seed đã được chọn trên tập VALIDATION (không dùng test để chọn gì cả).
    - Kiểm định ý nghĩa thống kê: paired cluster bootstrap theo TRẬN trên tập test
      (src/xai_football/evaluation/significance.py), hiệu chỉnh Holm.
    - Manifest được SINH TỪ CODE (không viết tay): experiment_manifest.json,
      evaluation_manifest.json.

Cách dùng:
    python scripts/07_evaluate.py --config configs/data.yaml
    python scripts/07_evaluate.py --config configs/data.yaml --n-boot 2000
    python scripts/07_evaluate.py --config configs/data.yaml --allow-partial

Đầu ra:
    results/tables/<run>/model_comparison.csv         test: mean/std qua seed
    results/tables/<run>/val_model_comparison.csv     validation: mean/std qua seed
    results/tables/<run>/model_metric_ci.csv          KTC 95% (bootstrap theo trận)
    results/tables/<run>/significance_tests.csv       so sánh từng cặp mô hình
    results/reproducibility/all_models_{test,val}_metrics_all_seeds.csv
    results/reproducibility/baselines_*                 bản sao kết quả baseline
    results/metadata/<run>/experiment_manifest.json
    results/metadata/<run>/evaluation_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from xai_football.evaluation.significance import align_predictions, compare_models  # noqa: E402
from xai_football.utils.io import load_yaml, save_atomic_csv, save_atomic_json  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import (  # noqa: E402
    CONFIGS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    get_experiments_dir,
    get_metadata_dir,
    get_models_dir,
    get_tables_dir,
    resolve_project_path,
)
from xai_football.utils.provenance import (  # noqa: E402
    compute_dict_sha256,
    compute_file_sha256,
    get_git_info,
)

logger = get_logger("evaluate")

METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "brier_score"]

# (tên hiển thị, khóa thư mục, tầng)
GNN_MODELS = [
    ("GCN (đề xuất)", "gcn", "Tier 2 — GNN Backbone"),
    ("MPNN (đề xuất)", "mpnn", "Tier 2 — GNN Backbone"),
    ("GATv2 (baseline)", "gatv2", "Tier 2 — GNN Baseline"),
    ("GraphSAGE (baseline)", "graphsage", "Tier 2 — GNN Baseline"),
]
# (tên hiển thị, khóa trong CSV baseline, cột xác suất trong *_predictions.parquet)
BASELINE_MODELS = [
    ("Logistic Regression", "logistic_regression", "lr_prob"),
    ("Random Forest", "random_forest", "rf_prob"),
    ("ANN (MLP)", "ann", "ann_prob"),
]
BASELINE_TIER = "Tier 3(a) — Non-graph Baseline"


def _rel(path: Path) -> str:
    """Đường dẫn tương đối tính từ gốc repo, dùng '/' để ghi vào manifest."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize_seeds(df: pd.DataFrame) -> dict[str, Any]:
    """mean và std (ddof=1) của từng độ đo qua các seed; std = 0 nếu chỉ 1 seed."""
    out: dict[str, Any] = {"n_runs": len(df)}
    for col in METRICS:
        vals = df[col].astype(float)
        out[col] = round(float(vals.mean()), 4)
        out[f"{col}_std"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0
    if "threshold" in df:
        out["threshold_mean"] = round(float(df["threshold"].astype(float).mean()), 4)
    return out


def load_gnn_results(
    run_name: str, key: str, expected_runs: int
) -> tuple[dict[str, Any] | None, str]:
    """Đọc kết quả multi-seed của một GNN. Trả (kết quả, lý do nếu không đầy đủ)."""
    model_dir = get_experiments_dir(run_name) / key
    if (model_dir / "run_failure_manifest.json").exists():
        return None, "toàn bộ run gần nhất thất bại (run_failure_manifest.json)"

    agg = _read_json(model_dir / "aggregate_summary.json")
    test_csv = model_dir / "test_metrics_all_seeds.csv"
    val_csv = model_dir / "val_metrics_all_seeds.csv"
    if agg is None or not test_csv.exists() or not val_csv.exists():
        return None, f"chưa có kết quả multi-seed trong {_rel(model_dir)}"

    test_df = pd.read_csv(test_csv)
    val_df = pd.read_csv(val_csv)
    seeds = [int(s) for s in agg.get("successful_seeds", [])]

    reasons = []
    if not agg.get("is_complete_official", False):
        reasons.append("aggregate_summary.is_complete_official = false")
    if agg.get("is_debug_run", False):
        reasons.append("aggregate_summary.is_debug_run = true")
    if agg.get("smoke_test", False):
        reasons.append("aggregate_summary.smoke_test = true")
    if int(agg.get("n_failed_runs", 0)) > 0:
        reasons.append(f"{agg.get('n_failed_runs')} seed lỗi")
    if len(seeds) < expected_runs:
        reasons.append(f"chỉ có {len(seeds)}/{expected_runs} seed")

    preds = {}
    for seed in seeds:
        p = model_dir / str(seed) / "test_predictions.parquet"
        if p.exists():
            preds[seed] = pd.read_parquet(p)

    return {
        "dir": model_dir,
        "aggregate": agg,
        "test_df": test_df,
        "val_df": val_df,
        "seeds": seeds,
        "best_seed": agg.get("best_seed_by_val_auc_pr"),
        "predictions": preds,
    }, "; ".join(reasons)


def load_baseline_results(run_name: str, expected_runs: int) -> tuple[dict[str, Any] | None, str]:
    """Đọc kết quả multi-seed của 3 baseline phi đồ thị."""
    base_dir = get_experiments_dir(run_name) / "baselines"
    agg = _read_json(base_dir / "aggregate_summary.json")
    test_csv = base_dir / "test_metrics_all_seeds.csv"
    val_csv = base_dir / "val_metrics_all_seeds.csv"
    if agg is None or not test_csv.exists() or not val_csv.exists():
        return None, f"chưa có kết quả baseline trong {_rel(base_dir)}"

    seeds = [int(s) for s in agg.get("configured_seeds", [])]
    reasons = []
    if not agg.get("is_complete_official", False):
        reasons.append("aggregate_summary.is_complete_official = false")
    if agg.get("is_debug_run", False):
        reasons.append("aggregate_summary.is_debug_run = true")
    if agg.get("smoke_test", False):
        reasons.append("aggregate_summary.smoke_test = true")
    if len(seeds) < expected_runs:
        reasons.append(f"chỉ có {len(seeds)}/{expected_runs} seed")

    preds = {}
    for seed in seeds:
        p = base_dir / str(seed) / "test_predictions.parquet"
        if p.exists():
            preds[seed] = pd.read_parquet(p)

    return {
        "dir": base_dir,
        "aggregate": agg,
        "test_df": pd.read_csv(test_csv),
        "val_df": pd.read_csv(val_csv),
        "seeds": seeds,
        "predictions": preds,
    }, "; ".join(reasons)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Cho phép đưa run chưa đủ seed vào bảng (cột complete = False)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=10000,
        help="Số vòng bootstrap (10.000: sai số Monte Carlo của p ~ ±0,002)",
    )
    parser.add_argument("--boot-seed", type=int, default=42, help="Seed của bootstrap")
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    train_cfg = load_yaml(args.train_config)
    run_name = data_cfg.get("run_name", "official")
    expected_runs = int(train_cfg.get("n_runs", 5))
    tables_dir = get_tables_dir(data_cfg)
    meta_dir = get_metadata_dir(data_cfg)

    included: dict[str, dict[str, Any]] = {}
    excluded: dict[str, str] = {}
    test_rows, val_rows = [], []
    per_seed_test, per_seed_val = [], []

    # ---------------------------------------------------------------
    # 1. Thu thập kết quả GNN (Tầng 2)
    # ---------------------------------------------------------------
    for display, key, tier in GNN_MODELS:
        res, reason = load_gnn_results(run_name, key, expected_runs)
        if res is None or (reason and not args.allow_partial):
            excluded[display] = reason or "không đọc được kết quả"
            logger.error("LOẠI %s: %s", display, excluded[display])
            continue
        complete = not reason
        test_sum = summarize_seeds(res["test_df"])
        val_sum = summarize_seeds(res["val_df"])
        common = {
            "model": display,
            "tier": tier,
            "type": "Graph",
            "complete": complete,
            "seeds": ",".join(str(s) for s in res["seeds"]),
        }
        test_rows.append(
            {
                **common,
                **test_sum,
                "loss": round(float(res["test_df"]["loss"].mean()), 4),
                "val_auc_pr": val_sum["auc_pr"],
                "val_auc_pr_std": val_sum["auc_pr_std"],
                "best_seed_by_val": res["best_seed"],
            }
        )
        val_rows.append({**common, **val_sum})
        per_seed_test.append(res["test_df"].assign(model=display, tier=tier))
        per_seed_val.append(res["val_df"].assign(model=display, tier=tier))
        included[display] = {
            "key": key,
            "kind": "gnn",
            "complete": complete,
            "partial_reason": reason or None,
            **res,
        }

    # ---------------------------------------------------------------
    # 2. Thu thập kết quả baseline phi đồ thị (Tầng 3a)
    # ---------------------------------------------------------------
    base, base_reason = load_baseline_results(run_name, expected_runs)
    if base is None or (base_reason and not args.allow_partial):
        for display, _, _ in BASELINE_MODELS:
            excluded[display] = base_reason or "không đọc được kết quả baseline"
        logger.error("LOẠI baseline phi đồ thị: %s", base_reason)
    else:
        for display, key, prob_col in BASELINE_MODELS:
            t_df = base["test_df"][base["test_df"]["model"] == key]
            v_df = base["val_df"][base["val_df"]["model"] == key]
            if t_df.empty or v_df.empty:
                excluded[display] = f"không có dòng '{key}' trong kết quả baseline"
                continue
            complete = not base_reason
            val_sum = summarize_seeds(v_df)
            common = {
                "model": display,
                "tier": BASELINE_TIER,
                "type": "Tabular",
                "complete": complete,
                "seeds": ",".join(str(s) for s in base["seeds"]),
            }
            test_rows.append(
                {
                    **common,
                    **summarize_seeds(t_df),
                    "loss": None,
                    "val_auc_pr": val_sum["auc_pr"],
                    "val_auc_pr_std": val_sum["auc_pr_std"],
                    "best_seed_by_val": None,
                }
            )
            val_rows.append({**common, **val_sum})
            per_seed_test.append(t_df.assign(model=display, tier=BASELINE_TIER))
            per_seed_val.append(v_df.assign(model=display, tier=BASELINE_TIER))
            included[display] = {
                "key": key,
                "kind": "baseline",
                "prob_col": prob_col,
                "complete": complete,
                "partial_reason": base_reason or None,
                "seeds": base["seeds"],
                "predictions": base["predictions"],
                "dir": base["dir"],
                "aggregate": base["aggregate"],
            }

    if excluded and not args.allow_partial:
        logger.error(
            "Có %d mô hình thiếu/không đầy đủ: %s. KHÔNG ghi bảng chính thức. "
            "Hãy chạy đủ seed, hoặc dùng --allow-partial để xem tạm.",
            len(excluded),
            excluded,
        )
        return 1
    if not test_rows:
        logger.error("Không có kết quả mô hình nào hợp lệ.")
        return 1

    # ---------------------------------------------------------------
    # 3. Bảng đối sánh (mean ± std qua seed)
    # ---------------------------------------------------------------
    order_cols = (
        ["model", "tier", "type", "n_runs", "complete", "seeds"]
        + [c for m in METRICS for c in (m, f"{m}_std")]
        + ["threshold_mean"]
    )
    comparison_df = pd.DataFrame(test_rows)
    comparison_df = comparison_df[
        order_cols + ["loss", "val_auc_pr", "val_auc_pr_std", "best_seed_by_val"]
    ]
    # Seed là số nguyên; baseline không có seed được chọn -> để trống (không thành 789.0)
    comparison_df["best_seed_by_val"] = comparison_df["best_seed_by_val"].astype("Int64")
    val_comparison_df = pd.DataFrame(val_rows)[order_cols]

    all_complete = not excluded and all(v["complete"] for v in included.values())
    is_partial_run = args.allow_partial or not all_complete

    tables_dir.mkdir(parents=True, exist_ok=True)
    if is_partial_run and run_name == "official":
        out_test = tables_dir / "partial_model_comparison.csv"
        out_val = tables_dir / "val_partial_model_comparison.csv"
        logger.warning(
            "Run chưa đầy đủ hoặc chạy với --allow-partial: "
            "Ghi kết quả vào %s và %s. TUYỆT ĐỐI KHÔNG ghi đè bảng canonical official model_comparison.csv.",
            out_test.name,
            out_val.name,
        )
    else:
        out_test = tables_dir / "model_comparison.csv"
        out_val = tables_dir / "val_model_comparison.csv"

    save_atomic_csv(comparison_df, out_test)
    save_atomic_csv(val_comparison_df, out_val)

    # Bản sao kết quả chính thức chỉ được ghi khi MỌI mô hình đủ seed và KHÔNG PHẢI partial
    repro_dir = RESULTS_DIR / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)
    if is_partial_run:
        logger.warning("Run chưa đầy đủ hoặc allow_partial: KHÔNG ghi đè results/reproducibility/.")
    elif all_complete and run_name == "official":
        save_atomic_csv(
            pd.concat(per_seed_test, ignore_index=True),
            repro_dir / "all_models_test_metrics_all_seeds.csv",
        )
        save_atomic_csv(
            pd.concat(per_seed_val, ignore_index=True),
            repro_dir / "all_models_val_metrics_all_seeds.csv",
        )
        if base is not None:
            save_atomic_csv(base["test_df"], repro_dir / "baselines_test_metrics_all_seeds.csv")
            save_atomic_csv(base["val_df"], repro_dir / "baselines_val_metrics_all_seeds.csv")
            save_atomic_json(base["aggregate"], repro_dir / "baselines_aggregate_summary.json")

    # ---------------------------------------------------------------
    # 4. Kiểm định ý nghĩa thống kê (paired cluster bootstrap theo trận)
    # ---------------------------------------------------------------
    # Xóa bảng kiểm định cũ trước — nếu lần này bị bỏ qua thì không để lại số liệu lỗi thời
    for stale in ("model_metric_ci.csv", "significance_tests.csv"):
        (tables_dir / stale).unlink(missing_ok=True)
    significance_info: dict[str, Any] = {"status": "skipped"}
    model_probs: dict[str, list[np.ndarray]] = {}
    reference: pd.DataFrame | None = None
    missing_preds = []
    for display, info in included.items():
        preds = info["predictions"]
        if len(preds) != len(info["seeds"]):
            missing_preds.append(display)
            continue
        prob_col = info.get("prob_col", "y_prob")
        for seed in info["seeds"]:
            frame = preds[seed]
            if not {"match_id", "possession_id"}.issubset(frame.columns):
                missing_preds.append(display)
                break
            if reference is None:
                reference = frame[["match_id", "possession_id", "y_true"]].reset_index(drop=True)
            model_probs.setdefault(display, []).append(
                align_predictions(reference, frame, prob_col)
            )
        else:
            continue
        model_probs.pop(display, None)

    if missing_preds:
        logger.warning(
            "Thiếu file dự đoán có khóa (match_id, possession_id) cho: %s — bỏ qua kiểm định "
            "cho các mô hình này.",
            missing_preds,
        )

    if reference is not None and len(model_probs) >= 2:
        logger.info(
            "Kiểm định bootstrap: %d mô hình, %d vòng, %d trận test",
            len(model_probs),
            args.n_boot,
            reference["match_id"].nunique(),
        )
        ci_df, pairs_df = compare_models(
            reference["y_true"].to_numpy(int),
            reference["match_id"].to_numpy(),
            model_probs,
            n_boot=args.n_boot,
            seed=args.boot_seed,
        )
        save_atomic_csv(ci_df, tables_dir / "model_metric_ci.csv")
        save_atomic_csv(pairs_df, tables_dir / "significance_tests.csv")
        significance_info = {
            "status": "completed",
            "method": (
                "paired cluster bootstrap theo trận trên tập test; độ đo = trung bình qua seed; "
                "KTC 95% theo phân vị; p-value hai phía; hiệu chỉnh Holm theo từng độ đo"
            ),
            "n_boot": args.n_boot,
            "boot_seed": args.boot_seed,
            "n_test_graphs": int(len(reference)),
            "n_test_matches": int(reference["match_id"].nunique()),
            "models": list(model_probs),
            "models_without_predictions": missing_preds,
        }

    # ---------------------------------------------------------------
    # 5. Manifest sinh từ code
    # ---------------------------------------------------------------
    git_info = get_git_info()
    pipeline_csv = tables_dir / "data_pipeline_summary.csv"
    graph_split_csv = tables_dir / "graph_split_summary.csv"
    pipeline = (
        {
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in pd.read_csv(pipeline_csv).iloc[0].to_dict().items()
        }
        if pipeline_csv.exists()
        else {}
    )
    graph_split = (
        pd.read_csv(graph_split_csv).set_index("split") if graph_split_csv.exists() else None
    )

    def _split_val(split: str, col: str) -> int | None:
        if graph_split is None or split not in graph_split.index:
            return None
        return int(graph_split.loc[split, col])

    models_manifest: dict[str, Any] = {}
    config_mismatch: list[str] = []
    for row in test_rows:
        info = included[row["model"]]
        key = info["key"]
        if info["kind"] == "gnn":
            cfg_path = CONFIGS_DIR / "model" / f"{key}.yaml"
            ckpt = get_models_dir(run_name) / f"{key}_best.pt"
            # Mã băm cấu hình LÚC HUẤN LUYỆN (config_snapshot.yaml của từng seed)
            training_hashes = []
            for seed in info["seeds"]:
                snap_path = info["dir"] / str(seed) / "config_snapshot.yaml"
                snap = load_yaml(snap_path) if snap_path.exists() else {}
                training_hashes.append(snap.get("config_hashes", {}).get("model_config"))
        else:
            cfg_path = CONFIGS_DIR / "model" / "baseline_tabular.yaml"
            ckpt = None
            training_hashes = [info["aggregate"].get("config_hash")]
        current_hash = compute_dict_sha256(load_yaml(cfg_path))
        matches_training = all(h == current_hash for h in training_hashes)
        if not matches_training:
            config_mismatch.append(row["model"])
        models_manifest[key] = {
            "display_name": row["model"],
            "tier": row["tier"],
            "type": row["type"],
            "complete": row["complete"],
            "seeds": info["seeds"],
            "config_path": _rel(cfg_path),
            "config_sha256": compute_file_sha256(cfg_path),
            # sha256 nội dung cấu hình (dict) LÚC HUẤN LUYỆN và có khớp cấu hình hiện tại không
            "training_config_dict_sha256": sorted({str(h) for h in training_hashes}),
            "current_config_matches_training": matches_training,
            "experiment_dir": _rel(info["dir"]),
            "best_seed_by_val_auc_pr": row.get("best_seed_by_val"),
            "checkpoint_path": _rel(ckpt) if ckpt is not None and ckpt.exists() else None,
            "checkpoint_sha256": (
                compute_file_sha256(ckpt) if ckpt is not None and ckpt.exists() else None
            ),
            "test_mean_std": {m: [row[m], row[f"{m}_std"]] for m in METRICS},
            "val_auc_pr_mean_std": [row["val_auc_pr"], row["val_auc_pr_std"]],
        }

    experiment_manifest = {
        "generated_by": "scripts/07_evaluate.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": git_info,
        "dataset": {
            "name": "Premier League 2015/2016 (StatsBomb Open Data)",
            "competitions": data_cfg.get("competitions"),
            "expected_matches": data_cfg.get("expected_matches"),
            "n_matches": pipeline.get("n_matches"),
            "clean_possessions": pipeline.get("n_clean_possessions"),
            "labeling": data_cfg.get("labeling"),
            "labeling_status": "TẠM THỜI — Decision 0001 chưa chốt",
        },
        "splits": {
            "strategy": "match_level_stratified",
            **{
                f"{s}_{c}": _split_val(s, col)
                for s in ("train", "val", "test")
                for c, col in (("matches", "n_matches"), ("graphs", "n_graphs"))
            },
        },
        "train_config_path": _rel(resolve_project_path(args.train_config)),
        "train_config_sha256": compute_file_sha256(resolve_project_path(args.train_config)),
        "models_with_config_changed_after_training": config_mismatch,
        "expected_runs_per_model": expected_runs,
        "models": models_manifest,
        "excluded_models": excluded,
        "significance": significance_info,
    }
    eval_manifest = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "git": git_info,
        "allow_partial": args.allow_partial,
        "included_models": {
            k: {
                "complete": v["complete"],
                "partial_reason": v["partial_reason"],
                "seeds": v["seeds"],
                "aggregate_summary": v["aggregate"],
            }
            for k, v in included.items()
        },
        "excluded_models": excluded,
        "significance": significance_info,
        "notes": [
            "Ngưỡng phân loại (accuracy/precision/recall/F1) chọn trên validation, tối đa F1.",
            "Cột loss của GNN là Weighted BCE (pos_weight từ train) — không so sánh được với "
            "baseline dạng bảng.",
            "GNN huấn luyện với pos_weight nên xác suất bị đẩy lên; Brier score phản ánh cả "
            "điều này (không hiệu chỉnh xác suất sau huấn luyện).",
        ],
    }

    meta_dir.mkdir(parents=True, exist_ok=True)
    save_atomic_json(experiment_manifest, meta_dir / "experiment_manifest.json")
    save_atomic_json(eval_manifest, meta_dir / "evaluation_manifest.json")

    show = ["model", "n_runs", "auc_pr", "auc_pr_std", "auc_roc", "auc_roc_std", "brier_score"]
    print(f"\n=== BẢNG ĐỐI SÁNH TRÊN TẬP TEST ({run_name.upper()}, mean ± std qua seed) ===")
    print(comparison_df[show].to_string(index=False))
    if significance_info["status"] == "completed":
        auc_pr_pairs = pairs_df[pairs_df["metric"] == "auc_pr"]
        print("\n=== KIỂM ĐỊNH AUC-PR (bootstrap theo trận, Holm) ===")
        print(
            auc_pr_pairs[
                ["model_a", "model_b", "diff_a_minus_b", "ci95_low", "ci95_high", "p_value_holm"]
            ].to_string(index=False)
        )
    print(f"\nĐã lưu bảng vào: {tables_dir}")
    print(f"Đã lưu manifest vào: {meta_dir}")
    if config_mismatch:
        logger.warning(
            "Cấu hình hiện tại KHÁC cấu hình lúc huấn luyện của: %s (xem experiment_manifest.json)",
            config_mismatch,
        )
    if significance_info["status"] != "completed" and not args.allow_partial:
        logger.error("Không chạy được kiểm định thống kê — run chính thức chưa hoàn tất.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
