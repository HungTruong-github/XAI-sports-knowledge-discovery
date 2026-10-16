"""Unit tests cho Bước 5: Kiến trúc mô hình GNN, loss functions và metrics."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from xai_football.evaluation.metrics import compute_binary_metrics
from xai_football.modeling.baselines.path_features import (
    graphs_to_tabular_dataset,
)
from xai_football.modeling.gnn import (
    build_gnn_model,
)
from xai_football.modeling.gnn.heads import BinaryClassifierHead
from xai_football.modeling.gnn.readout import GlobalReadout
from xai_football.training.losses import BinaryFocalLoss, build_loss_fn


@pytest.fixture
def dummy_graph_batch():
    """Tạo 2 đồ thị mẫu đóng gói thành 1 Batch."""
    # Graph 1: 3 nodes, 2 edges
    x1 = torch.randn(3, 8)
    e1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    ea1 = torch.randn(2, 12)
    y1 = torch.tensor([1], dtype=torch.long)
    g1 = Data(x=x1, edge_index=e1, edge_attr=ea1, y=y1, match_id=101)

    # Graph 2: 4 nodes, 3 edges
    x2 = torch.randn(4, 8)
    e2 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    ea2 = torch.randn(3, 12)
    y2 = torch.tensor([0], dtype=torch.long)
    g2 = Data(x=x2, edge_index=e2, edge_attr=ea2, y=y2, match_id=102)

    return Batch.from_data_list([g1, g2]), [g1, g2]


def test_readout_and_head(dummy_graph_batch):
    batch, _ = dummy_graph_batch
    for r_type in ("mean", "max", "add", "attention"):
        readout = GlobalReadout(r_type, in_dim=8)
        emb = readout(batch.x, batch.batch)
        assert emb.shape == (2, 8)

    head = BinaryClassifierHead(in_dim=8, hidden_dim=16)
    logits = head(emb)
    assert logits.shape == (2,)


@pytest.mark.parametrize("model_name", ["gcn", "gatv2", "graphsage", "mpnn"])
def test_gnn_models_forward(dummy_graph_batch, model_name):
    batch, _ = dummy_graph_batch
    cfg = {"name": model_name, "hidden_dim": 32, "num_layers": 2}
    model = build_gnn_model(cfg, node_in_dim=8, edge_in_dim=12)

    logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    assert logits.shape == (2,)

    # Kiểm tra node embeddings trích xuất được (cho XAI)
    node_embs = model.get_node_embeddings(batch.x, batch.edge_index, batch.edge_attr)
    assert node_embs.shape[0] == batch.num_nodes
    assert node_embs.shape[1] == 32


def test_losses():
    logits = torch.tensor([1.5, -2.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0])

    loss_fn = build_loss_fn({"name": "weighted_bce"}, pos_weight=5.0)
    loss = loss_fn(logits, targets)
    assert loss.item() > 0

    focal_fn = BinaryFocalLoss(gamma=2.0, alpha=0.5)
    focal_loss = focal_fn(logits, targets)
    assert focal_loss.item() > 0


def test_metrics():
    y_true = [1, 0, 1, 1, 0]
    y_prob = [0.9, 0.1, 0.8, 0.3, 0.2]
    metrics = compute_binary_metrics(y_true, y_prob)

    assert "auc_pr" in metrics
    assert "auc_roc" in metrics
    assert "f1" in metrics
    assert "brier_score" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_tabular_features(dummy_graph_batch):
    _, graph_list = dummy_graph_batch
    X_df, y_arr, match_ids = graphs_to_tabular_dataset(graph_list)

    assert len(X_df) == 2
    assert len(y_arr) == 2
    assert list(y_arr) == [1, 0]
    assert match_ids == [101, 102]
    assert "n_passes" in X_df.columns
    assert "sequence_length" in X_df.columns
