"""Unit tests cho các độ đo đánh giá chất lượng XAI (Fidelity, Sparsity, Stability, Curves)."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from xai_football.xai.metrics import (
    compute_fidelity_minus,
    compute_fidelity_plus,
    compute_insertion_deletion_curves,
    compute_sparsity,
    compute_stability,
    evaluate_explanation_quality,
)


class MockLinearGNN(nn.Module):
    """Mô hình giả lập đơn giản: logit tỉ lệ với số cạnh trong đồ thị."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        n_edges = float(edge_index.shape[1])
        # Càng nhiều cạnh thì logit càng dương (xác suất càng cao)
        return torch.tensor([n_edges - 2.0], dtype=torch.float, device=x.device)


@pytest.fixture
def mock_graph():
    x = torch.zeros((4, 4))
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def test_compute_fidelity_plus_and_minus(mock_graph):
    model = MockLinearGNN()
    edge_mask = np.array([0.1, 0.9, 0.8, 0.2])

    # Chọn top 2 cạnh (cạnh 1 và 2)
    fid_plus = compute_fidelity_plus(model, mock_graph, edge_mask, top_k=2)
    # Xóa 2 cạnh -> số cạnh giảm từ 4 xuống 2 -> xác suất phải giảm -> Fid+ > 0
    assert fid_plus > 0.0

    fid_minus = compute_fidelity_minus(model, mock_graph, edge_mask, top_k=2)
    # Chỉ giữ 2 cạnh -> so với đồ thị gốc 4 cạnh -> xác suất cũng giảm -> Fid- > 0
    assert fid_minus > 0.0


def test_compute_sparsity():
    mask = np.array([0.1, 0.9, 0.8, 0.2])
    # Top 1 cạnh trong 4 cạnh -> Sparsity = 1 - 1/4 = 0.75
    sp_top1 = compute_sparsity(mask, top_k=1)
    assert sp_top1 == 0.75

    # Top 4 cạnh -> Sparsity = 0.0
    sp_top4 = compute_sparsity(mask, top_k=4)
    assert sp_top4 == 0.0

    # Mask rỗng
    assert compute_sparsity(np.zeros(0)) == 1.0


def test_compute_insertion_deletion_curves(mock_graph):
    model = MockLinearGNN()
    edge_mask = np.array([0.1, 0.9, 0.8, 0.2])

    curves = compute_insertion_deletion_curves(model, mock_graph, edge_mask)
    assert "deletion_auc" in curves
    assert "insertion_auc" in curves
    assert 0.0 <= curves["deletion_auc"] <= 1.0
    assert 0.0 <= curves["insertion_auc"] <= 1.0


def test_compute_stability(mock_graph):
    base_mask = np.array([0.1, 0.9, 0.8, 0.2])

    # Hàm giải thích giả lập: luôn trả về cùng mask
    def constant_explainer(g):
        return base_mask

    stab = compute_stability(
        explainer_func=constant_explainer,
        graph=mock_graph,
        base_mask=base_mask,
        n_repeats=3,
        top_k=2,
    )
    # Vì luôn trả về cùng mask nên Jaccard tương đồng = 1.0
    assert stab == 1.0


def test_evaluate_explanation_quality_summary(mock_graph):
    model = MockLinearGNN()
    edge_mask = np.array([0.1, 0.9, 0.8, 0.2])

    res = evaluate_explanation_quality(model, mock_graph, edge_mask, top_k=2)
    assert "fidelity_plus" in res
    assert "fidelity_minus" in res
    assert "sparsity" in res
    assert "deletion_auc" in res
    assert "insertion_auc" in res
    assert "roar" in res
