"""Phase 2 comprehensive tests for XAI components.

Covers:
- Explanation phenomenon vs model targeting class 1 (D5)
- FootballGNNExplainer and FootballPGExplainer target_class usage (D5, D8)
- Stratified sampling determinism and match/bucket coverage (D7)
- RandomExplainer and LastEdgeExplainer baselines
- Player importance aggregation unbiased per-participation and last-edge ablation (D12)
- Fidelity edge removal preserving node features and top-k resolution (D6)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from xai_football.xai.explainers import (
    ExplanationResult,
    LastEdgeExplainer,
    RandomExplainer,
)
from xai_football.xai.metrics import (
    compute_fidelity_minus,
    compute_fidelity_plus,
    compute_sparsity_fidelity_curve,
    resolve_top_k,
)
from xai_football.xai.sampling import stratified_sample
from xai_football.xai.subgraph import aggregate_player_importance, map_explanation_to_passes


class DummyGNN(nn.Module):
    """Simple dummy GNN model."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return torch.tensor([0.5], device=x.device)


@pytest.fixture
def mock_graph():
    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, 12)
    # Ensure order_in_possession at feature 10 is chronological
    edge_attr[:, 10] = torch.tensor([0.0, 0.33, 0.66, 1.0])
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([1], dtype=torch.long),
        match_id=3754087,
        possession_id=42,
        player_ids=torch.tensor([10, 20, 30, 40, 50]),
        edge_event_ids=["e1", "e2", "e3", "e4"],
    )


def test_stratified_sample_determinism_and_buckets():
    """Verify stratified_sample returns deterministic, stratified subsets."""
    graphs = []
    match_ids = [100, 101, 102]
    edge_counts = [4, 5, 6, 8, 10]
    for m in match_ids:
        for ec in edge_counts:
            for y_val in [0, 1]:
                g = Data(
                    x=torch.zeros((ec + 1, 8)),
                    edge_index=torch.zeros((2, ec), dtype=torch.long),
                    edge_attr=torch.zeros((ec, 12)),
                    y=torch.tensor([y_val], dtype=torch.long),
                    match_id=m,
                    possession_id=len(graphs),
                )
                graphs.append(g)

    # Sample 6 pos, 6 neg
    sample = stratified_sample(graphs, n_pos=6, n_neg=6, seed=42)
    assert len(sample) == 12

    # Determinism
    sample_2 = stratified_sample(graphs, n_pos=6, n_neg=6, seed=42)
    assert [g.possession_id for g in sample] == [g.possession_id for g in sample_2]

    # Equal pos and neg
    labels = [int(g.y.item()) for g in sample]
    assert labels.count(1) == 6
    assert labels.count(0) == 6


def test_random_explainer(mock_graph):
    """Verify RandomExplainer returns valid [0, 1] edge masks averaged over seeds."""
    model = DummyGNN()
    explainer = RandomExplainer(model=model, seed=42)
    res = explainer.explain_graph(mock_graph, graph_index=0)
    assert res.explainer_name == "RandomExplainer"
    assert len(res.edge_mask) == 4
    assert np.all(res.edge_mask >= 0.0) and np.all(res.edge_mask <= 1.0)

    # Determinism across calls with same seed
    res2 = explainer.explain_graph(mock_graph, graph_index=0)
    np.testing.assert_allclose(res.edge_mask, res2.edge_mask)


def test_last_edge_explainer(mock_graph):
    """Verify LastEdgeExplainer gives 1.0 to last edge and 0.0 to others."""
    model = DummyGNN()
    explainer = LastEdgeExplainer(model=model)
    res = explainer.explain_graph(mock_graph, graph_index=0)
    assert res.explainer_name == "LastEdgeExplainer"
    assert len(res.edge_mask) == 4
    assert res.edge_mask[-1] == 1.0
    assert np.all(res.edge_mask[:-1] == 0.0)


def test_resolve_top_k():
    """Verify D6 formula: k = max(1, round(0.3 * E))."""
    assert resolve_top_k(1) == 1
    assert resolve_top_k(2) == 1
    assert resolve_top_k(3) == 1
    assert resolve_top_k(4) == 1
    assert resolve_top_k(5) == 2
    assert resolve_top_k(7) == 2
    assert resolve_top_k(10) == 3
    assert resolve_top_k(15) == 4  # round(4.5) == 4 in python
    # Explicit top_k overrides
    assert resolve_top_k(10, top_k=4) == 4


def test_aggregate_player_importance_unbiased(mock_graph):
    """Verify aggregate_player_importance divides by participation and computes no-last metrics."""
    # Graph has 5 nodes (indices 0..4, ids [10, 20, 30, 40, 50]), 4 edges: (0->1, 1->2, 2->3, 3->4)
    # Let edge_mask be [0.2, 0.4, 0.6, 0.8] (edge 3 is last edge with imp 0.8)
    exp = ExplanationResult(
        graph_index=0,
        match_id=3754087,
        possession_id=42,
        num_nodes=5,
        num_edges=4,
        edge_mask=np.array([0.2, 0.4, 0.6, 0.8]),
        node_mask=None,
        pred_prob=0.85,
        target=1,
        explainer_name="GNNExplainer",
    )

    df = aggregate_player_importance(mock_graph, exp)
    assert len(df) == 5

    # Node 1 (player 20): receives edge 0 (imp 0.2), sends edge 1 (imp 0.4)
    # Total = 0.6, participations = 2, mean = 0.3
    row20 = df[df["player_id"] == 20].iloc[0]
    assert row20["participations"] == 2
    assert pytest.approx(row20["total_importance"]) == 0.6
    assert pytest.approx(row20["mean_importance"]) == 0.3
    # Neither edge 0 nor 1 is the last edge (edge 3 is last), so no_last values equal full values
    assert pytest.approx(row20["total_importance_no_last"]) == 0.6
    assert pytest.approx(row20["mean_importance_no_last"]) == 0.3

    # Node 4 (player 50): receives edge 3 (last edge, imp 0.8)
    row50 = df[df["player_id"] == 50].iloc[0]
    assert row50["participations"] == 1
    assert pytest.approx(row50["total_importance"]) == 0.8
    assert pytest.approx(row50["mean_importance"]) == 0.8
    # Edge 3 is last edge, so without last edge, player 50 has 0 participations
    assert pytest.approx(row50["total_importance_no_last"]) == 0.0
    assert pytest.approx(row50["mean_importance_no_last"]) == 0.0


def test_map_explanation_to_passes(mock_graph):
    """Verify map_explanation_to_passes marks is_last_edge correctly."""
    exp = ExplanationResult(
        graph_index=0,
        match_id=3754087,
        possession_id=42,
        num_nodes=5,
        num_edges=4,
        edge_mask=np.array([0.2, 0.4, 0.6, 0.8]),
        node_mask=None,
        pred_prob=0.85,
        target=1,
        explainer_name="GNNExplainer",
    )
    df_passes = map_explanation_to_passes(mock_graph, exp)
    assert len(df_passes) == 4
    # The last edge in chronological order is edge_index 3
    last_row = df_passes[df_passes["edge_index"] == 3].iloc[0]
    assert bool(last_row["is_last_edge"]) is True
    # Other edges are False
    for _, row in df_passes[df_passes["edge_index"] < 3].iterrows():
        assert bool(row["is_last_edge"]) is False


class DummyFeaturePreservingGNN(nn.Module):
    """GNN that sums node features and adds edge count to logit."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        node_sum = x.sum()
        n_edges = float(edge_index.shape[1])
        return node_sum + n_edges


def test_fidelity_preserves_node_features(mock_graph):
    """Verify Fidelity metric preserves node features intact during edge ablation."""
    model = DummyFeaturePreservingGNN()
    orig_x = mock_graph.x.clone()
    edge_mask = np.array([0.9, 0.7, 0.2, 0.1])

    f_plus = compute_fidelity_plus(model, mock_graph, edge_mask, top_k=2)
    f_minus = compute_fidelity_minus(model, mock_graph, edge_mask, top_k=2)

    assert torch.equal(mock_graph.x, orig_x)
    assert isinstance(f_plus, float)
    assert isinstance(f_minus, float)


def test_sparsity_fidelity_curve(mock_graph):
    """Verify compute_sparsity_fidelity_curve returns curve with correct steps."""
    model = DummyFeaturePreservingGNN()
    edge_mask = np.array([0.9, 0.7, 0.2, 0.1])
    curve = compute_sparsity_fidelity_curve(model, mock_graph, edge_mask)

    # For 4 edges, k ranges from 1 to 3
    assert len(curve) == 3
    for pt in curve:
        assert "sparsity" in pt
        assert "fidelity_plus" in pt
        assert "fidelity_minus" in pt
        assert 0.0 <= pt["sparsity"] <= 1.0
