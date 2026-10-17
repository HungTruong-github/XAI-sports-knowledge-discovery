"""Unit tests cho các module Explainer (GNNExplainer & PGExplainer)."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from xai_football.modeling.gnn.gcn import GCNModel
from xai_football.modeling.gnn.mpnn import MPNNModel
from xai_football.xai.explainers import (
    FootballGNNExplainer,
    FootballPGExplainer,
    build_explainer,
)


@pytest.fixture
def sample_graph():
    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, 12)
    y = torch.tensor([1], dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        match_id=3754087,
        possession_id=10,
        player_ids=torch.tensor([101, 102, 103, 104, 105]),
        edge_event_ids=["e1", "e2", "e3", "e4"],
    )


@pytest.fixture
def empty_edge_graph():
    x = torch.randn(3, 8)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, 12), dtype=torch.float)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([0]),
        match_id=3754087,
        possession_id=99,
        player_ids=torch.tensor([101, 102, 103]),
        edge_event_ids=[],
    )


def test_gnn_explainer_mpnn(sample_graph):
    model = MPNNModel(node_in_dim=8, edge_in_dim=12, hidden_dim=16, num_layers=2)
    explainer = FootballGNNExplainer(model=model, epochs=5)

    res = explainer.explain_graph(sample_graph, graph_index=1, target_class=1)

    assert res.explainer_name == "GNNExplainer"
    assert res.match_id == 3754087
    assert res.possession_id == 10
    assert len(res.edge_mask) == 4
    assert np.all(res.edge_mask >= 0.0) and np.all(res.edge_mask <= 1.0)
    assert not np.isnan(res.edge_mask).any()
    assert 0.0 <= res.pred_prob <= 1.0


def test_gnn_explainer_empty_edge_graph(empty_edge_graph):
    model = GCNModel(node_in_dim=8, hidden_dim=16, num_layers=2)
    explainer = FootballGNNExplainer(model=model, epochs=3)

    # Đồ thị 0 cạnh không được văng Exception
    res = explainer.explain_graph(empty_edge_graph, graph_index=0)
    assert res.num_edges == 0
    assert len(res.edge_mask) == 0
    assert res.match_id == 3754087


def test_pg_explainer_mpnn(sample_graph):
    model = MPNNModel(node_in_dim=8, edge_in_dim=12, hidden_dim=16, num_layers=2)
    explainer = FootballPGExplainer(model=model, epochs=2)

    # Huấn luyện trên mẫu
    explainer.train_on_graphs([sample_graph], max_graphs=1)
    assert explainer.is_trained is True

    # Giải thích
    res = explainer.explain_graph(sample_graph, graph_index=0)
    assert res.explainer_name == "PGExplainer"
    assert len(res.edge_mask) == 4
    assert np.all(res.edge_mask >= 0.0) and np.all(res.edge_mask <= 1.0)


def test_build_explainer_factory():
    model = GCNModel(node_in_dim=8, hidden_dim=16, num_layers=2)
    g_exp = build_explainer(
        "gnn_explainer", model=model, xai_cfg={"explainers": {"gnn_explainer": {"epochs": 5}}}
    )
    assert isinstance(g_exp, FootballGNNExplainer)

    p_exp = build_explainer("pg_explainer", model=model)
    assert isinstance(p_exp, FootballPGExplainer)

    with pytest.raises(ValueError, match="không được hỗ trợ"):
        build_explainer("shap_graph_invalid", model=model)
