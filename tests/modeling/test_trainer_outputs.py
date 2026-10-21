"""Unit tests: artifact của GNNTrainer phục vụ Bước 6 (dự đoán có khóa, manifest đầy đủ)."""

from __future__ import annotations

import json

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from xai_football.modeling.gnn import build_gnn_model
from xai_football.training.losses import build_loss_fn
from xai_football.training.optimizers import build_optimizer
from xai_football.training.trainer import GNNTrainer
from xai_football.utils.seed import set_seed


def _graphs(n: int, match_offset: int) -> list[Data]:
    graphs = []
    for i in range(n):
        y = float(i % 3 == 0)
        graphs.append(
            Data(
                x=torch.randn(3, 8),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_attr=torch.randn(2, 12),
                y=torch.tensor([y]),
                match_id=match_offset + i // 4,
                possession_id=i,
            )
        )
    return graphs


def test_trainer_writes_keyed_predictions_and_manifest(tmp_path):
    set_seed(0)
    train, val, test = _graphs(24, 100), _graphs(12, 200), _graphs(12, 300)
    model = build_gnn_model({"name": "mpnn", "hidden_dim": 16, "num_layers": 2}, 8, 12)
    trainer = GNNTrainer(
        model=model,
        optimizer=build_optimizer(model, {"name": "adamw", "lr": 0.01}),
        loss_fn=build_loss_fn({"name": "weighted_bce"}, pos_weight=2.0),
        train_loader=DataLoader(train, batch_size=8, shuffle=True),
        val_loader=DataLoader(val, batch_size=8, shuffle=False),
        test_loader=DataLoader(test, batch_size=8, shuffle=False),
        early_stopping_patience=5,
        output_dir=tmp_path,
        seed=0,
        model_name="mpnn",
    )
    trainer.fit(epochs=2)

    for split, graphs in (("test", test), ("val", val)):
        preds = pd.read_parquet(tmp_path / f"{split}_predictions.parquet")
        assert list(preds.columns) == ["match_id", "possession_id", "y_true", "y_prob"]
        assert preds["match_id"].tolist() == [g.match_id for g in graphs]
        assert preds["possession_id"].tolist() == [g.possession_id for g in graphs]
        assert preds["y_true"].tolist() == [int(g.y.item()) for g in graphs]

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["pos_weight"] == 2.0
    assert manifest["epochs_run"] == 2
    assert manifest["n_train_graphs"] == 24
    assert manifest["n_test_graphs"] == 12
    assert manifest["n_parameters"] > 0
