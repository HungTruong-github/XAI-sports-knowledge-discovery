"""Huấn luyện mô hình theo giao thức Cross-Fitting 5-fold (Phase 3).

Tham chiếu:
- docs/decisions/0006-giao-thuc-xai.md (Quyết định D11)
- data/splits/crossfit_5fold.json
Cách dùng:
    python scripts/06d_crossfit_train.py --fold all --epochs 10
    python scripts/06d_crossfit_train.py --fold 1 --epochs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from xai_football.graphs.dataset import load_graphs
from xai_football.modeling.gnn import build_gnn_model
from xai_football.training.losses import build_loss_fn
from xai_football.training.optimizers import build_optimizer, build_scheduler
from xai_football.training.trainer import GNNTrainer
from xai_football.utils.io import load_yaml
from xai_football.utils.logging import get_logger
from xai_football.utils.paths import get_processed_dir
from xai_football.utils.provenance import compute_file_sha256, get_git_info

logger = get_logger("crossfit_train")


def train_single_fold(
    fold_name: str,
    fold_data: dict[str, Any],
    graphs: list[Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    out_dir: Path,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Huấn luyện mô hình trên tập train của một fold cụ thể."""
    train_mids = set(fold_data["train_matches"])
    test_mids = set(fold_data["test_matches"])

    train_graphs = [g for g in graphs if int(getattr(g, "match_id", 0)) in train_mids]
    # Dùng 15% train_graphs làm validation nội bộ để early stopping
    n_val = max(100, int(len(train_graphs) * 0.15))
    val_graphs = train_graphs[-n_val:]
    actual_train = train_graphs[:-n_val]
    test_graphs = [g for g in graphs if int(getattr(g, "match_id", 0)) in test_mids]

    logger.info(
        "[%s] Phân chia: Train=%d, Val=%d, OOF Test=%d",
        fold_name,
        len(actual_train),
        len(val_graphs),
        len(test_graphs),
    )

    sample_g = graphs[0]
    node_in_dim = sample_g.x.shape[1]
    edge_in_dim = sample_g.edge_attr.shape[1] if hasattr(sample_g, "edge_attr") and sample_g.edge_attr is not None else 12

    model = build_gnn_model(model_cfg, node_in_dim=node_in_dim, edge_in_dim=edge_in_dim)
    model.to(device)

    train_loader = DataLoader(actual_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    fold_ckpt_dir = out_dir / fold_name
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # pos_weight
    train_labels = [int(g.y.item()) for g in actual_train if hasattr(g, "y")]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    opt_cfg = train_cfg.get("optimizer", {})
    optimizer = build_optimizer(model, opt_cfg)
    loss_fn = build_loss_fn(train_cfg.get("loss"), pos_weight=pos_weight, device=device)
    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = build_scheduler(optimizer, sched_cfg, epochs=epochs)
    es_cfg = train_cfg.get("early_stopping", {})

    trainer = GNNTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        scheduler_monitor=str(sched_cfg.get("monitor", "val_auc_pr")),
        early_stopping_patience=int(es_cfg.get("patience", 15)),
        early_stopping_monitor="val_auc_pr",
        early_stopping_mode="max",
        threshold_strategy=str(train_cfg.get("threshold_strategy", "val_optimal")),
        device=device,
        output_dir=fold_ckpt_dir,
        seed=42,
        model_name=f"mpnn_{fold_name}",
    )

    metrics = trainer.fit(epochs=epochs)

    ckpt_path = fold_ckpt_dir / "best_model.pt"
    ckpt_sha = compute_file_sha256(ckpt_path) if ckpt_path.exists() else ""

    return {
        "fold": fold_name,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "val_auc_pr": metrics.get("val_metrics", {}).get("auc_pr", 0.0),
        "test_auc_pr": metrics.get("test_metrics", {}).get("auc_pr", 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default="all", help="Chỉ số fold (1..5) hoặc 'all'")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--split-file", default="data/splits/crossfit_5fold.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    split_path = Path(args.split_file)
    if not split_path.exists():
        logger.error("Không tìm thấy file split %s", split_path)
        return 1

    with open(split_path, encoding="utf-8") as f:
        crossfit_meta = json.load(f)

    data_cfg = load_yaml(args.config)
    train_cfg = load_yaml(args.train_config)
    model_cfg = load_yaml(args.model_config) if args.model_config else load_yaml("configs/model/mpnn.yaml")

    processed_dir = get_processed_dir(data_cfg)
    graphs, _, _ = load_graphs(processed_dir)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path("models/crossfit")
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = crossfit_meta["folds"]
    folds_to_run = list(folds.keys()) if args.fold == "all" else [f"fold_{args.fold}"]

    results = []
    for f_name in folds_to_run:
        if f_name not in folds:
            logger.error("Không tìm thấy fold %s trong cấu hình split", f_name)
            continue
        logger.info("\n========== BẮT ĐẦU HUẤN LUYỆN CROSSFIT %s ==========", f_name.upper())
        res = train_single_fold(
            fold_name=f_name,
            fold_data=folds[f_name],
            graphs=graphs,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            out_dir=out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
        results.append(res)

    manifest = {
        "step": "Phase 3: Cross-fitting 5-fold training",
        "model_name": model_cfg.get("name", "mpnn"),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "results": results,
        "git": get_git_info(),
    }
    with open(out_dir / "crossfit_train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Hoàn tất huấn luyện Cross-fitting! Manifest lưu tại %s", out_dir / "crossfit_train_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
