"""Unit tests cho Bước 3 — dựng đồ thị possession."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from xai_football.graphs.build import build_graph, node_order, truncate_at_shot
from xai_football.graphs.dataset import build_dataset, load_player_roles
from xai_football.graphs.features import (
    POSITION_TO_ROLE,
    ROLES,
    compute_edge_features,
    edge_feature_names,
    node_feature_names,
    role_of,
)
from xai_football.utils.io import load_yaml

GRAPH_CFG = load_yaml("configs/graph.yaml")
NODE_NAMES = node_feature_names(GRAPH_CFG["nodes"]["features"])
EDGE_NAMES = edge_feature_names(GRAPH_CFG["edges"]["features"])
DATA_CFG = {
    "labeling": {"mode": "possession_level", "target": "shot"},
    "preprocessing": {"min_passes_per_possession": 2},
    "pitch": {"length": 120, "width": 80},
}


def _pass(event_id, index, passer, recipient, sx, sy, ex, ey, second=0, height="Ground Pass"):
    return {
        "match_id": 1,
        "possession_id": 1,
        "event_id": event_id,
        "event_index": index,
        "minute": 0,
        "second": second,
        "passer_id": passer,
        "recipient_id": recipient,
        "start_x": sx,
        "start_y": sy,
        "end_x": ex,
        "end_y": ey,
        "height": height,
    }


@pytest.fixture
def passes():
    # A(10) -> B(11) -> C(12) -> B(11) -> D(13); B->C và C->B là hai cạnh riêng
    return [
        _pass("e1", 1, 10, 11, 20.0, 40.0, 40.0, 40.0, second=0),
        _pass("e2", 3, 11, 12, 40.0, 40.0, 60.0, 20.0, second=3),
        _pass("e3", 5, 12, 11, 60.0, 20.0, 70.0, 40.0, second=6, height="High Pass"),
        _pass("e4", 7, 11, 13, 70.0, 40.0, 100.0, 40.0, second=9),
    ]


POSSESSION = {
    "match_id": 1,
    "possession_id": 1,
    "team_id": 5,
    "play_pattern": "Regular Play",
}
ROLES_MAP = {10: "DF", 11: "MF", 12: "MF", 13: "FW"}


def test_every_statsbomb_position_has_a_role():
    """Đủ 25 vị trí StatsBomb (id 1-25), mỗi vị trí thuộc đúng một vai trò hợp lệ."""
    assert len(POSITION_TO_ROLE) == 25
    assert set(POSITION_TO_ROLE.values()) == set(ROLES)
    assert role_of("Goalkeeper") == "GK"
    assert role_of("Center Forward") == "FW"  # tên thật trong dữ liệu, không phải "Striker"
    assert role_of("Unknown position") is None
    assert role_of(None) is None


def test_graph_structure(passes):
    """Đỉnh = cầu thủ tham gia, cạnh = từng đường chuyền, đúng hướng người chuyền -> nhận."""
    g = build_graph(POSSESSION, passes, ROLES_MAP, 1, GRAPH_CFG)

    assert g.player_ids.tolist() == [10, 11, 12, 13]  # thứ tự xuất hiện đầu tiên
    assert g.num_nodes == 4
    assert g.num_edges == 4  # keep_multi_edge: không gộp B->C và C->B
    edges = [(g.player_ids[s].item(), g.player_ids[d].item()) for s, d in g.edge_index.t()]
    assert edges == [(10, 11), (11, 12), (12, 11), (11, 13)]
    assert g.edge_event_ids == ["e1", "e2", "e3", "e4"]
    assert g.y.tolist() == [1.0]
    assert (g.match_id, g.possession_id, g.team_id) == (1, 1, 5)


def test_feature_dims_match_names(passes):
    g = build_graph(POSSESSION, passes, ROLES_MAP, 0, GRAPH_CFG)
    assert g.x.shape == (4, len(NODE_NAMES))
    assert g.edge_attr.shape == (4, len(EDGE_NAMES))


def test_player_id_is_not_a_feature(passes):
    """player_id chỉ là metadata, không xuất hiện trong x."""
    g = build_graph(POSSESSION, passes, ROLES_MAP, 0, GRAPH_CFG)
    assert not any("id" in name for name in NODE_NAMES)
    for pid in (10, 11, 12, 13):
        assert not (g.x == pid).any()


def test_node_features_values(passes):
    g = build_graph(POSSESSION, passes, ROLES_MAP, 0, GRAPH_CFG)
    b = g.player_ids.tolist().index(11)
    row = dict(zip(NODE_NAMES, g.x[b].tolist(), strict=True))

    assert row["role_MF"] == 1.0 and row["role_DF"] == 0.0
    # B chạm bóng: nhận e1 (40,40), chuyền e2 (40,40), nhận e3 (70,40), chuyền e4 (70,40)
    assert row["mean_x"] == pytest.approx(55.0 / 120)
    assert row["mean_y"] == pytest.approx(40.0 / 80)
    assert row["n_passes_made"] == 2.0
    assert row["n_passes_received"] == 2.0


def test_unknown_role_is_all_zero(passes):
    roles = dict(ROLES_MAP)
    roles[13] = None
    g = build_graph(POSSESSION, passes, roles, 0, GRAPH_CFG)
    d = g.player_ids.tolist().index(13)
    role_cols = [NODE_NAMES.index(f"role_{r}") for r in ROLES]
    assert g.x[d, role_cols].sum().item() == 0.0


def test_edge_features_values(passes):
    rows = compute_edge_features(passes, GRAPH_CFG["edges"]["features"])
    e2 = dict(zip(EDGE_NAMES, rows[1], strict=True))  # (40,40) -> (60,20)

    assert e2["start_x"] == pytest.approx(40 / 120)
    assert e2["end_y"] == pytest.approx(20 / 80)
    assert e2["length"] == pytest.approx(math.hypot(20, -20) / math.hypot(120, 80))
    assert e2["angle_sin"] == pytest.approx(math.sin(math.atan2(-20, 20)))
    assert e2["angle_cos"] == pytest.approx(math.cos(math.atan2(-20, 20)))
    assert e2["order_in_possession"] == pytest.approx(1 / 3)
    assert e2["elapsed_minutes"] == pytest.approx(3 / 60)

    e3 = dict(zip(EDGE_NAMES, rows[2], strict=True))
    assert (e3["height_ground"], e3["height_low"], e3["height_high"]) == (0.0, 0.0, 1.0)


def test_truncate_at_first_shot(passes):
    """Chỉ giữ đường chuyền TRƯỚC cú sút đầu tiên; không có sút thì giữ nguyên."""
    assert [p["event_id"] for p in truncate_at_shot(passes, 5)] == ["e1", "e2"]
    assert truncate_at_shot(passes, None) == passes
    assert truncate_at_shot(passes, float("nan")) == passes


def test_node_order_is_deterministic(passes):
    assert node_order(passes) == node_order(list(passes)) == [10, 11, 12, 13]


def test_unsupported_options_raise(passes):
    merged = {**GRAPH_CFG, "edges": {**GRAPH_CFG["edges"], "multi_edge_strategy": "aggregate"}}
    with pytest.raises(ValueError):
        build_graph(POSSESSION, passes, ROLES_MAP, 0, merged)

    goal_node = {**GRAPH_CFG, "optional": {"add_goal_node": True}}
    with pytest.raises(NotImplementedError):
        build_graph(POSSESSION, passes, ROLES_MAP, 0, goal_node)


def _possessions_df(rows):
    base = {"match_id": 1, "team_id": 5, "has_goal": False, "play_pattern": "Regular Play"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_build_dataset_filters_and_counts():
    """Đếm đúng mọi possession bị loại: play pattern, quá ngắn sau khi cắt tại cú sút."""
    pass_rows = [
        # Possession 1: sút ở index 5 -> còn 2 đường chuyền -> giữ, y=1
        {**_pass("a1", 1, 10, 11, 20, 40, 40, 40), "possession_id": 1},
        {**_pass("a2", 3, 11, 12, 40, 40, 60, 40), "possession_id": 1},
        {**_pass("a3", 7, 12, 13, 60, 40, 80, 40), "possession_id": 1},
        # Possession 2: sút ở index 12 -> chỉ còn 1 đường chuyền -> loại
        {**_pass("b1", 11, 10, 11, 20, 40, 40, 40), "possession_id": 2},
        {**_pass("b2", 13, 11, 12, 40, 40, 60, 40), "possession_id": 2},
        # Possession 3: phạt góc, không sút
        {**_pass("c1", 21, 10, 11, 20, 40, 40, 40), "possession_id": 3},
        {**_pass("c2", 22, 11, 12, 40, 40, 60, 40), "possession_id": 3},
    ]
    possessions = _possessions_df(
        [
            {"possession_id": 1, "has_shot": True, "first_shot_index": 5},
            {"possession_id": 2, "has_shot": True, "first_shot_index": 12},
            {
                "possession_id": 3,
                "has_shot": False,
                "first_shot_index": None,
                "play_pattern": "From Corner",
            },
        ]
    )
    pass_df = pd.DataFrame(pass_rows)

    graphs, index_df, report = build_dataset(possessions, pass_df, {}, DATA_CFG, GRAPH_CFG)
    assert [g.possession_id for g in graphs] == [1, 3]
    assert report.dropped_too_short_after_truncation == 1
    assert report.shot_dropped_too_short_after_truncation == 1
    assert report.passes_removed_by_truncation == 2  # a3 và b2
    assert graphs[0].edge_event_ids == ["a1", "a2"]
    assert index_df["y"].tolist() == [1, 0]
    assert report.nodes_unknown_role == graphs[0].num_nodes + graphs[1].num_nodes

    open_play = {**GRAPH_CFG, "play_patterns_keep": ["Regular Play"]}
    graphs, _, report = build_dataset(possessions, pass_df, {}, DATA_CFG, open_play)
    assert [g.possession_id for g in graphs] == [1]
    assert report.dropped_play_pattern == 1
    assert report.shot_dropped_play_pattern == 0

    no_trunc = {**GRAPH_CFG, "truncate_at_first_shot": False}
    graphs, _, _ = build_dataset(possessions, pass_df, {}, DATA_CFG, no_trunc)
    assert graphs[0].edge_event_ids == ["a1", "a2", "a3"]


def test_build_dataset_refuses_old_interim_without_first_shot_index():
    """Dữ liệu Bước 2 cũ (thiếu first_shot_index) phải báo lỗi, không được âm thầm không cắt."""
    possessions = _possessions_df([{"possession_id": 1, "has_shot": True}])
    with pytest.raises(ValueError, match="first_shot_index"):
        build_dataset(possessions, pd.DataFrame([_pass("a1", 1, 10, 11, 0, 0, 1, 1)]), {},
                      DATA_CFG, GRAPH_CFG)


def test_build_dataset_requires_possession_level_labeling():
    cfg = {**DATA_CFG, "labeling": {"mode": "pass_level", "target": "shot",
                                    "shot_window_seconds": 10}}
    possessions = _possessions_df([{"possession_id": 1, "has_shot": True, "first_shot_index": 5}])
    with pytest.raises(ValueError, match="possession_level"):
        build_dataset(possessions, pd.DataFrame([_pass("a1", 1, 10, 11, 0, 0, 1, 1)]), {},
                      cfg, GRAPH_CFG)


def test_load_player_roles_uses_most_frequent_position(tmp_path):
    ev = pd.DataFrame(
        {
            "match_id": [1, 1, 1, 1, 2],
            "player_id": [10, 10, 10, 11, 10],
            "position_name": [
                "Center Back",
                "Center Back",
                "Center Forward",
                "Goalkeeper",
                "Center Forward",
            ],
        }
    )
    path = tmp_path / "events.parquet"
    ev.to_parquet(path, index=False)
    roles = load_player_roles(path)
    assert roles[(1, 10)] == "DF"  # 2 lần Center Back > 1 lần Striker
    assert roles[(1, 11)] == "GK"
    assert roles[(2, 10)] == "FW"  # theo từng trận
