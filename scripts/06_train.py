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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
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
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    split_cfg = load_yaml(args.split_config)

    # 1. Cố định seed
    seed = int(train_cfg.get("seed", 42))
    seed_everything(seed)

    # 2. Xác định device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Thiết bị sử dụng: %s", device)

    run_name = data_cfg.get("run_name", "official")
    processed_dir = get_processed_dir(data_cfg)

    # 3. Đọc dữ liệu đồ thị
    graphs_pt_path = processed_dir / "graphs.pt"
    if not graphs_pt_path.exists():
        logger.error(
            "Không tìm thấy đồ thị: %s. Hãy chạy scripts/04_build_graphs.py trước.", graphs_pt_path
        )
        return 1

    graphs, index_df, feature_names = load_graphs(processed_dir)
    logger.info("Đã tải %d đồ thị từ %s", len(graphs), processed_dir)

    # 4. Đọc match splits
    base_split_dir = Path(split_cfg.get("output_dir", "data/splits"))
    split_dir = base_split_dir if run_name == "official" else base_split_dir / run_name
    if not (split_dir / "train_matches.csv").exists():
        logger.error("Chưa có split trong %s. Hãy chạy scripts/05_make_splits.py trước.", split_dir)
        return 1

    splits = load_match_splits(split_dir)
    train_mids = set(splits["train"])
    val_mids = set(splits["val"])
    test_mids = set(splits["test"])

    # Phân loại đồ thị vào 3 tập theo match_id
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

    # 5. Tính pos_weight từ tập train
    train_labels = [int(g.y.item()) for g in train_graphs if hasattr(g, "y")]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    logger.info("Tập train: %d dương / %d âm -> pos_weight: %.2f", n_pos, n_neg, pos_weight)

    # 6. DataLoader
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 64))
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # 7. Khởi tạo mô hình
    sample_g = graphs[0]
    node_in_dim = sample_g.x.shape[1]
    edge_in_dim = (
        sample_g.edge_attr.shape[1]
        if hasattr(sample_g, "edge_attr") and sample_g.edge_attr is not None
        else 12
    )

    model = build_gnn_model(model_cfg, node_in_dim=node_in_dim, edge_in_dim=edge_in_dim)
    model_name = (model_cfg.get("name") or "model").lower()
    logger.info(
        "Khởi tạo mô hình %s (node_in_dim=%d, edge_in_dim=%d, params=%d)",
        model_name,
        node_in_dim,
        edge_in_dim,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # 8. Optimizer, Loss & Scheduler
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

    # 9. Trainer & Early Stopping
    es_cfg = train_cfg.get("early_stopping", {})
    es_patience = int(es_cfg.get("patience", 20))
    es_monitor = str(es_cfg.get("monitor", "val_auc_pr"))
    epochs = args.epochs or int(train_cfg.get("epochs", 200))

    exp_dir = Path("experiments") / run_name / model_name
    models_dir = Path("models") / run_name
    models_dir.mkdir(parents=True, exist_ok=True)

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
        device=device,
        output_dir=exp_dir,
    )

    # 10. Huấn luyện
    results = trainer.fit(epochs=epochs)

    # Sao chép best model sang thư mục models/
    best_ckpt_src = exp_dir / "best_model.pt"
    if best_ckpt_src.exists():
        import shutil

        shutil.copy(best_ckpt_src, models_dir / f"{model_name}_best.pt")
        logger.info("Đã lưu trọng số chính thức vào: %s", models_dir / f"{model_name}_best.pt")

    print("\n=== KẾT QUẢ HUẤN LUYỆN ===")
    print(f"Mô hình:       {model_name.upper()}")
    print(f"Best Epoch:    {results['best_epoch']}")
    print(f"Best Val PR:   {results['best_metric']:.4f}")
    if results.get("test_metrics"):
        print("Độ đo Test:")
        for k, v in results["test_metrics"].items():
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
