"""Unit tests cho module trích xuất Subgraph và ánh xạ về cầu thủ/đường chuyền."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from xai_football.xai.explainers import ExplanationResult
from xai_football.xai.subgraph import (
    aggregate_player_importance,
    map_explanation_to_passes,
    select_important_edges,
)


@pytest.fixture
def dummy_explanation():
    return ExplanationResult(
        match_id=1001,
        possession_id=42,
        graph_index=0,
        target=1,
        pred_prob=0.85,
        edge_mask=np.array([0.1, 0.9, 0.4, 0.7]),
        node_mask=None,
        explainer_name="GNNExplainer",
        num_nodes=4,
        num_edges=4,
    )


@pytest.fixture
def dummy_graph():
    return Data(
        x=torch.zeros((4, 8)),
        edge_index=torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]], dtype=torch.long),
        player_ids=torch.tensor([10, 20, 30, 40]),
        edge_event_ids=["p1", "p2", "p3", "p4"],
        match_id=1001,
        possession_id=42,
    )


def test_select_important_edges():
    mask = np.array([0.1, 0.9, 0.4, 0.7])

    # Top-2 edges
    top2 = select_important_edges(mask, top_k=2)
    assert top2.tolist() == [False, True, False, True]

    # Threshold 0.5
    th5 = select_important_edges(mask, threshold=0.5)
    assert th5.tolist() == [False, True, False, True]

    # Mảng rỗng
    assert len(select_important_edges(np.zeros(0))) == 0


def test_map_explanation_to_passes(dummy_graph, dummy_explanation):
    df = map_explanation_to_passes(dummy_graph, dummy_explanation, top_k=2)

    assert len(df) == 4
    # Cạnh quan trọng nhất là index 1 (importance 0.9)
    top_row = df.iloc[0]
    assert top_row["edge_index"] == 1
    assert top_row["importance"] == 0.9
    assert top_row["event_id"] == "p2"
    assert top_row["passer_id"] == 20
    assert top_row["recipient_id"] == 30
    assert top_row["rank"] == 1
    assert bool(top_row["is_important"]) is True


def test_aggregate_player_importance(dummy_graph, dummy_explanation):
    df_players = aggregate_player_importance(dummy_graph, dummy_explanation)

    assert len(df_players) == 4
    # Kiểm tra tính toán:
    # Edge 0: 0->1 (0.1) -> out[0]+=0.1, in[1]+=0.1
    # Edge 1: 1->2 (0.9) -> out[1]+=0.9, in[2]+=0.9
    # Edge 2: 2->3 (0.4) -> out[2]+=0.4, in[3]+=0.4
    # Edge 3: 0->3 (0.7) -> out[0]+=0.7, in[3]+=0.7
    # Player 0: out=0.8, in=0.0 -> total=0.8
    # Player 1: out=0.9, in=0.1 -> total=1.0
    # Player 2: out=0.4, in=0.9 -> total=1.3
    # Player 3: out=0.0, in=1.1 -> total=1.1

    p2_row = df_players[df_players["player_id"] == 30].iloc[0]
    assert p2_row["outgoing_importance"] == 0.4
    assert p2_row["incoming_importance"] == 0.9
    assert p2_row["total_importance"] == 1.3
