"""Bước 5 — Huấn luyện mô hình GNN phân loại possession (Bài toán 1 – Dự đoán).

Cách dùng:
    python scripts/06_train.py --config configs/data.yaml --model-config configs/model/gcn.yaml
    python scripts/06_train.py --config configs/data.yaml --model-config configs/model/mpnn.yaml
    python scripts/06_train.py --config configs/data_pilot.yaml --epochs 10

Đầu ra:
    experiments/<run_name>/<model_name>/<seed>/best_model.pt
    experiments/<run_name>/<model_name>/<seed>/validation_metrics.csv
    experiments/<run_name>/<model_name>/<seed>/test_metrics.csv
    experiments/<run_name>/<model_name>/<seed>/run_manifest.json
    experiments/<run_name>/<model_name>/aggregate_summary.json (khi chạy đủ seeds)
    models/<run_name>/<model_name>_best.pt (chỉ khi hoàn thành official multi-seed run)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from xai_football.data.splits import load_match_splits  # noqa: E402
from xai_football.graphs.dataset import load_graphs  # noqa: E402
from xai_football.modeling.gnn import build_gnn_model  # noqa: E402
from xai_football.training.losses import build_loss_fn  # noqa: E402
from xai_football.training.optimizers import build_optimizer, build_scheduler  # noqa: E402
from xai_football.training.trainer import GNNTrainer  # noqa: E402
from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import (  # noqa: E402
    get_experiments_dir,
    get_models_dir,
    get_processed_dir,
    get_split_dir,
)
from xai_football.utils.provenance import compute_dict_sha256, get_git_info  # noqa: E402
from xai_football.utils.seed import seed_everything  # noqa: E402

logger = get_logger("train")


def train_single_run(
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    split_cfg: dict[str, Any],
    seed: int,
    train_graphs: list[Any],
    val_graphs: list[Any],
    test_graphs: list[Any],
    device: torch.device,
    exp_dir: Path,
    epochs: int | None = None,
    batch_size: int | None = None,
    smoke_test: bool = False,
    git_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Huấn luyện một run độc lập với seed xác định và lưu đầy đủ artifact."""
    seed_everything(seed)
    model_name = (model_cfg.get("name") or "model").lower()

    if smoke_test:
        logger.info("Chạy chế độ SMOKE TEST cho seed %d (subset 200 graphs, 1 epoch)", seed)
        train_sub = train_graphs[:200]
        val_sub = val_graphs[:50]
        test_sub = test_graphs[:50]
        actual_epochs = 1
        actual_batch_size = 32
    else:
        train_sub = train_graphs
        val_sub = val_graphs
        test_sub = test_graphs
        actual_epochs = epochs or int(train_cfg.get("epochs", 200))
        actual_batch_size = batch_size or int(train_cfg.get("batch_size", 64))

    # 1. Tính pos_weight từ tập train
    train_labels = [int(g.y.item()) for g in train_sub if hasattr(g, "y")]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    # 2. DataLoaders
    train_loader = DataLoader(train_sub, batch_size=actual_batch_size, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=actual_batch_size, shuffle=False)
    test_loader = DataLoader(test_sub, batch_size=actual_batch_size, shuffle=False)

    # 3. Model
    sample_g = train_graphs[0]
    node_in_dim = sample_g.x.shape[1]
    if getattr(sample_g, "edge_attr", None) is None:
        raise ValueError("Đồ thị không có edge_attr — hãy chạy lại scripts/04_build_graphs.py.")
    edge_in_dim = sample_g.edge_attr.shape[1]
    model = build_gnn_model(model_cfg, node_in_dim=node_in_dim, edge_in_dim=edge_in_dim)

    # 4. Optimizer, Loss & Scheduler (dùng factory pattern có kiểm tra hợp lệ)
    opt_cfg = train_cfg.get("optimizer", {})
    optimizer = build_optimizer(model, opt_cfg)

    loss_fn = build_loss_fn(train_cfg.get("loss"), pos_weight=pos_weight, device=device)

    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = build_scheduler(optimizer, sched_cfg, epochs=actual_epochs)

    # 5. Early Stopping
    es_cfg = train_cfg.get("early_stopping", {})
    es_patience = int(es_cfg.get("patience", 20))
    es_monitor = str(es_cfg.get("monitor", "val_auc_pr"))
    es_mode = str(es_cfg.get("mode", "min" if "loss" in es_monitor else "max"))
    sched_monitor = str(sched_cfg.get("monitor", es_monitor))

    exp_dir.mkdir(parents=True, exist_ok=True)

    # Lưu snapshot cấu hình kèm provenance
    snapshot = {
        "seed": seed,
        "model_name": model_name,
        "smoke_test": smoke_test,
        "pos_weight_from_train": round(float(pos_weight), 6),
        "n_train_pos": n_pos,
        "n_train_neg": n_neg,
        "node_in_dim": int(node_in_dim),
        "edge_in_dim": int(edge_in_dim),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_info or {},
        "config_hashes": {
            "model_config": compute_dict_sha256(model_cfg),
            "train_config": compute_dict_sha256(train_cfg),
            "data_config": compute_dict_sha256(data_cfg),
            "split_config": compute_dict_sha256(split_cfg),
        },
        "model_config": model_cfg,
        "train_config": train_cfg,
        "data_config": data_cfg,
        "split_config": split_cfg,
    }
    with open(exp_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False, allow_unicode=True)

    trainer = GNNTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        scheduler_monitor=sched_monitor,
        early_stopping_patience=es_patience,
        early_stopping_monitor=es_monitor,
        early_stopping_mode=es_mode,
        threshold_strategy=str(train_cfg.get("threshold_strategy", "val_optimal")),
        device=device,
        output_dir=exp_dir,
        seed=seed,
        model_name=model_name,
    )

    results = trainer.fit(epochs=actual_epochs)
    return results


def resolve_seeds(
    train_cfg: dict[str, Any],
    cli_seed: int | None = None,
    cli_seeds: str | None = None,
    return_debug: bool = False,
) -> list[int] | tuple[list[int], bool]:
    """Xác định danh sách seed cần chạy.

    Ưu tiên:
    1. Tham số dòng lệnh --seed (debug mode cho 1 seed lẻ).
    2. Tham số dòng lệnh --seeds (danh sách seed cụ thể).
    3. Cấu hình train.yaml -> seeds (danh sách đầy đủ theo n_runs).
    4. Cấu hình train.yaml -> seed.
    5. Mặc định [42].

    Nếu return_debug=True, trả về tuple (seeds, is_debug_run).
    """
    if cli_seed is not None:
        seeds, is_debug = [int(cli_seed)], True
    elif cli_seeds:
        seeds, is_debug = [int(s.strip()) for s in cli_seeds.split(",") if s.strip()], False
    else:
        configured_seeds = train_cfg.get("seeds")
        n_runs = int(train_cfg.get("n_runs", len(configured_seeds) if configured_seeds else 1))
        if configured_seeds and isinstance(configured_seeds, list):
            seeds, is_debug = [int(s) for s in configured_seeds[:n_runs]], False
        elif "seed" in train_cfg:
            seeds, is_debug = [int(train_cfg["seed"])], False
        else:
            seeds, is_debug = [42], False

    if return_debug:
        return seeds, is_debug
    return seeds


def compute_metrics_aggregation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Tính toán thống kê mean và std một cách tất định từ danh sách kết quả các runs."""
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "seed"]
    summary: dict[str, Any] = {
        "n_runs": len(rows),
        "mean": {},
        "std": {},
    }
    for col in numeric_cols:
        summary["mean"][col] = round(float(df[col].mean()), 4)
        summary["std"][col] = round(float(df[col].std(ddof=1)), 4) if len(df) > 1 else 0.0
    return summary


def clean_stale_aggregates(base_exp_dir: Path) -> None:
    """Xóa các file tổng hợp cũ trước khi chạy thực nghiệm mới để tránh lẫn lộn provenance."""
    stale_files = [
        "val_metrics_all_seeds.csv",
        "test_metrics_all_seeds.csv",
        "aggregate_summary.json",
        "run_failure_manifest.json",
    ]
    for fname in stale_files:
        p = base_exp_dir / fname
        if p.exists():
            try:
                p.unlink()
            except Exception as e:
                logger.warning("Không thể xóa file cũ %s: %s", p, e)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model/gcn.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--split-config", default="configs/split.yaml")
    parser.add_argument("--device", default=None, help="cpu hoặc cuda (mặc định: tự động nhận)")
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè số epochs từ train.yaml")
    parser.add_argument("--batch-size", type=int, default=None, help="Ghi đè batch size")
    parser.add_argument("--seed", type=int, default=None, help="Chạy một seed cụ thể (debug/test)")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Danh sách seed cách nhau bởi dấu phẩy (vd: 42,123,456)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Chạy smoke test nhanh (1 epoch, subset nhỏ) vào thư mục cách ly",
    )
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    split_cfg = load_yaml(args.split_config)

    # 1. Xác định danh sách seed cần chạy
    seeds, is_debug_run = resolve_seeds(
        train_cfg, cli_seed=args.seed, cli_seeds=args.seeds, return_debug=True
    )
    logger.info(
        "Cấu hình thực nghiệm: %d runs với danh sách seeds=%s (debug_mode=%s)",
        len(seeds),
        seeds,
        is_debug_run,
    )

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Thiết bị sử dụng: %s", device)

    run_name = data_cfg.get("run_name", "official")
    processed_dir = get_processed_dir(data_cfg)

    # Đọc dữ liệu đồ thị
    graphs_pt_path = processed_dir / "graphs.pt"
    if not graphs_pt_path.exists():
        logger.error(
            "Không tìm thấy đồ thị: %s. Hãy chạy scripts/04_build_graphs.py trước.", graphs_pt_path
        )
        return 1

    graphs, index_df, feature_names = load_graphs(processed_dir)
    logger.info("Đã tải %d đồ thị từ %s", len(graphs), processed_dir)

    # Đọc match splits (đường dẫn tính từ gốc repo)
    split_dir = get_split_dir(split_cfg, run_name)
    if not (split_dir / "train_matches.csv").exists():
        logger.error("Chưa có split trong %s. Hãy chạy scripts/05_make_splits.py trước.", split_dir)
        return 1

    splits = load_match_splits(split_dir)
    train_mids = set(splits["train"])
    val_mids = set(splits["val"])
    test_mids = set(splits["test"])

    train_graphs = [g for g in graphs if int(getattr(g, "match_id", 0)) in train_mids]
    val_graphs = [g for g in graphs if int(getattr(g, "match_id", 0)) in val_mids]
    test_graphs = [g for g in graphs if int(getattr(g, "match_id", 0)) in test_mids]

    logger.info(
        "Phân chia đồ thị: Train=%d, Val=%d, Test=%d (Tổng=%d)",
        len(train_graphs),
        len(val_graphs),
        len(test_graphs),
        len(graphs),
    )

    model_name = (model_cfg.get("name") or "model").lower()

    # Cách ly thư mục smoke test và debug run để không bao giờ ghi đè kết quả official
    if args.smoke_test:
        suffix = "_smoke"
    elif is_debug_run:
        suffix = "_debug"
    else:
        suffix = ""
    base_exp_dir = get_experiments_dir(run_name, suffix) / model_name
    models_dir = get_models_dir(run_name, suffix)

    base_exp_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Nếu chạy experiment chính thức mới, dọn dẹp các tệp aggregate cũ để tránh stale artifacts
    if not is_debug_run and not args.smoke_test:
        clean_stale_aggregates(base_exp_dir)

    git_info = get_git_info()
    all_val_rows: list[dict[str, Any]] = []
    all_test_rows: list[dict[str, Any]] = []
    failed_runs: list[dict[str, Any]] = []

    for seed in seeds:
        logger.info(
            "\n========== BẮT ĐẦU RUN: MODEL=%s, SEED=%d ==========", model_name.upper(), seed
        )
        seed_exp_dir = base_exp_dir / str(seed)
        try:
            res = train_single_run(
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                split_cfg=split_cfg,
                seed=seed,
                train_graphs=train_graphs,
                val_graphs=val_graphs,
                test_graphs=test_graphs,
                device=device,
                exp_dir=seed_exp_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                smoke_test=args.smoke_test,
                git_info=git_info,
            )

            if "val_metrics" in res:
                all_val_rows.append({"seed": seed, **res["val_metrics"]})
            if "test_metrics" in res and res["test_metrics"]:
                all_test_rows.append({"seed": seed, **res["test_metrics"]})

        except Exception as exc:
            logger.error("Run với seed %d thất bại: %s", seed, exc, exc_info=True)
            failed_runs.append({"seed": seed, "error": str(exc)})
            seed_exp_dir.mkdir(parents=True, exist_ok=True)
            with open(seed_exp_dir / "run_error.txt", "w", encoding="utf-8") as f:
                f.write(f"Run failed for seed {seed}:\n{exc}\n")

    # Báo cáo các run thất bại nếu có
    if failed_runs:
        logger.warning(
            "CẢNH BÁO: Có %d/%d run thất bại: %s", len(failed_runs), len(seeds), failed_runs
        )

    # 1. Trường hợp TẤT CẢ các run đều thất bại
    if not all_test_rows:
        failure_manifest = {
            "model_name": model_name,
            "status": "failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_requested_seeds": len(seeds),
            "n_successful_runs": 0,
            "n_failed_runs": len(failed_runs),
            "failed_runs": failed_runs,
            "git": git_info,
        }
        with open(base_exp_dir / "run_failure_manifest.json", "w", encoding="utf-8") as f:
            json.dump(failure_manifest, f, indent=2, ensure_ascii=False)
        logger.error("TẤT CẢ %d run đều thất bại! Đã ghi log failure manifest.", len(seeds))
        return 1

    # 2. Trường hợp có ít nhất 1 run thành công
    val_df = pd.DataFrame(all_val_rows)
    test_df = pd.DataFrame(all_test_rows)

    # Lưu tệp per-seed tổng hợp nếu không phải smoke test
    if not args.smoke_test:
        val_df.to_csv(base_exp_dir / "val_metrics_all_seeds.csv", index=False)
        test_df.to_csv(base_exp_dir / "test_metrics_all_seeds.csv", index=False)

    val_agg = compute_metrics_aggregation(all_val_rows)
    test_agg = compute_metrics_aggregation(all_test_rows)

    # Đủ seed = chứa ĐÚNG danh sách seed cấu hình trong train.yaml (không chỉ đủ số lượng)
    configured = set(resolve_seeds(train_cfg))
    is_complete_official = (
        (len(seeds) >= int(train_cfg.get("n_runs", 5)))
        and configured.issubset(seeds)
        and len(failed_runs) == 0
        and not args.smoke_test
        and not is_debug_run
    )

    # Chọn checkpoint tốt nhất dựa trên tập VALIDATION (không dùng test set để chọn checkpoint!)
    best_seed = None
    best_val_auc_pr = -1.0
    for r in all_val_rows:
        val_pr = float(r.get("auc_pr", -1.0))
        if val_pr > best_val_auc_pr:
            best_val_auc_pr = val_pr
            best_seed = r["seed"]

    status_str = "completed" if len(failed_runs) == 0 else "partial"
    aggregate_manifest = {
        "model_name": model_name,
        "status": status_str,
        "is_complete_official": is_complete_official,
        "is_debug_run": is_debug_run,
        "smoke_test": args.smoke_test,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git_info,
        "configured_seeds": seeds,
        "n_requested_seeds": len(seeds),
        "n_successful_runs": len(all_test_rows),
        "n_failed_runs": len(failed_runs),
        "successful_seeds": [r["seed"] for r in all_test_rows],
        "failed_seeds": [r["seed"] for r in failed_runs],
        # Seed có Val AUC-PR cao nhất — checkpoint của seed này được chép sang models/<run>/
        "best_seed_by_val_auc_pr": best_seed,
        "best_val_auc_pr": round(best_val_auc_pr, 4) if best_seed is not None else None,
        "val_metrics_summary": val_agg,
        "test_metrics_summary": test_agg,
    }
    with open(base_exp_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_manifest, f, indent=2, ensure_ascii=False)

    if best_seed is not None and not args.smoke_test and not is_debug_run:
        best_seed_dir = base_exp_dir / str(best_seed)
        for fname in [
            "best_model.pt",
            "training_history.csv",
            "validation_metrics.csv",
            "test_metrics.csv",
            "run_manifest.json",
            "test_predictions.parquet",
            "val_predictions.parquet",
            "config_snapshot.yaml",
        ]:
            src_file = best_seed_dir / fname
            if src_file.exists():
                shutil.copy(src_file, base_exp_dir / fname)
        if (best_seed_dir / "best_model.pt").exists():
            shutil.copy(best_seed_dir / "best_model.pt", models_dir / f"{model_name}_best.pt")
        logger.info(
            "Đã đồng bộ checkpoint tốt nhất từ seed %d (Val AUC-PR: %.4f) vào %s",
            best_seed,
            best_val_auc_pr,
            models_dir,
        )

    if len(seeds) > 1:
        print("\n=== TỔNG HỢP MULTI-RUN VAL METRICS ===")
        print(val_df.describe().to_string())
        print("\n=== TỔNG HỢP MULTI-RUN TEST METRICS ===")
        print(test_df.describe().to_string())

    print(
        f"\nĐã hoàn thành huấn luyện {model_name.upper()}: {len(all_test_rows)} thành công, "
        f"{len(failed_runs)} thất bại trên tổng số {len(seeds)} seed: {seeds} (Status: {status_str})"
    )
    return 0 if not failed_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
