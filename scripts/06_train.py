"""Bước 5 — Huấn luyện mô hình GNN phân loại possession (Bài toán 1 – Dự đoán).

Cách dùng:
    python scripts/06_train.py --config configs/data.yaml --model-config configs/model/gcn.yaml
    python scripts/06_train.py --config configs/data.yaml --model-config configs/model/mpnn.yaml
    python scripts/06_train.py --config configs/data_pilot.yaml --epochs 10

Đầu ra:
    experiments/<run_name>/<model_name>/best_model.pt
    experiments/<run_name>/<model_name>/training_history.csv
    experiments/<run_name>/<model_name>/test_metrics.csv
    models/<run_name>/<model_name>_best.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from xai_football.data.splits import load_match_splits  # noqa: E402
from xai_football.graphs.dataset import load_graphs  # noqa: E402
from xai_football.modeling.gnn import build_gnn_model  # noqa: E402
from xai_football.training.losses import build_loss_fn  # noqa: E402
from xai_football.training.trainer import GNNTrainer  # noqa: E402
from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import get_processed_dir  # noqa: E402
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
    edge_in_dim = (
        sample_g.edge_attr.shape[1]
        if hasattr(sample_g, "edge_attr") and sample_g.edge_attr is not None
        else 12
    )
    model = build_gnn_model(model_cfg, node_in_dim=node_in_dim, edge_in_dim=edge_in_dim)

    # 4. Optimizer, Loss & Scheduler
    opt_cfg = train_cfg.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 0.001))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0001))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = build_loss_fn(train_cfg.get("loss"), pos_weight=pos_weight, device=device)

    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(sched_cfg.get("factor", 0.5)),
        patience=int(sched_cfg.get("patience", 10)),
    )

    # 5. Early Stopping
    es_cfg = train_cfg.get("early_stopping", {})
    es_patience = int(es_cfg.get("patience", 20))
    es_monitor = str(es_cfg.get("monitor", "val_auc_pr"))

    exp_dir.mkdir(parents=True, exist_ok=True)

    # Lưu snapshot cấu hình
    snapshot = {
        "seed": seed,
        "model_name": model_name,
        "smoke_test": smoke_test,
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
        early_stopping_patience=es_patience,
        early_stopping_monitor=es_monitor,
        early_stopping_mode="max",
        threshold_strategy=str(train_cfg.get("threshold_strategy", "val_optimal")),
        device=device,
        output_dir=exp_dir,
        seed=seed,
        model_name=model_name,
    )

    results = trainer.fit(epochs=actual_epochs)
    return results


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
    parser.add_argument("--seed", type=int, default=None, help="Chạy seed cụ thể")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Danh sách seed cách nhau bởi dấu phẩy (vd: 42,123,456)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Chạy smoke test nhanh (1 epoch, subset nhỏ) để kiểm tra pipeline",
    )
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    split_cfg = load_yaml(args.split_config)

    # Xác định danh sách seed cần chạy
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(train_cfg.get("seed", 42))]

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

    # Đọc match splits
    base_split_dir = Path(split_cfg.get("output_dir", "data/splits"))
    split_dir = base_split_dir if run_name == "official" else base_split_dir / run_name
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
    base_exp_dir = Path("experiments") / run_name / model_name
    models_dir = Path("models") / run_name
    models_dir.mkdir(parents=True, exist_ok=True)

    all_val_rows: list[dict[str, Any]] = []
    all_test_rows: list[dict[str, Any]] = []

    for seed in seeds:
        logger.info(
            "\n========== BẮT ĐẦU RUN: MODEL=%s, SEED=%d ==========", model_name.upper(), seed
        )
        seed_exp_dir = base_exp_dir / str(seed)
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
        )

        # Lưu bản sao về base_exp_dir cho run đầu tiên/chính (seed 42) nếu không phải smoke test
        if seed == seeds[0] and not args.smoke_test:
            for fname in [
                "best_model.pt",
                "training_history.csv",
                "validation_metrics.csv",
                "test_metrics.csv",
                "run_manifest.json",
                "test_predictions.parquet",
                "config_snapshot.yaml",
            ]:
                src_file = seed_exp_dir / fname
                if src_file.exists():
                    shutil.copy(src_file, base_exp_dir / fname)
            if (seed_exp_dir / "best_model.pt").exists():
                shutil.copy(seed_exp_dir / "best_model.pt", models_dir / f"{model_name}_best.pt")

        if "val_metrics" in res:
            all_val_rows.append({"seed": seed, **res["val_metrics"]})
        if "test_metrics" in res and res["test_metrics"]:
            all_test_rows.append({"seed": seed, **res["test_metrics"]})

    # Nếu chạy nhiều seed, tính summary mean/std
    if len(seeds) > 1 and not args.smoke_test:
        val_df = pd.DataFrame(all_val_rows)
        test_df = pd.DataFrame(all_test_rows)
        val_df.to_csv(base_exp_dir / "val_metrics_all_seeds.csv", index=False)
        test_df.to_csv(base_exp_dir / "test_metrics_all_seeds.csv", index=False)

        print("\n=== TỔNG HỢP MULTI-RUN VAL METRICS ===")
        print(val_df.describe().to_string())
        print("\n=== TỔNG HỢP MULTI-RUN TEST METRICS ===")
        print(test_df.describe().to_string())

    print(f"\nĐã hoàn thành huấn luyện cho {len(seeds)} seed: {seeds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
