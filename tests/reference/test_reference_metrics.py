"""Unit tests cho các module Reference Metrics (Player Stats & Expected Threat)."""

import pandas as pd
import pytest

from xai_football.reference.expected_threat import ExpectedThreatModel
from xai_football.reference.player_stats import compute_player_reference_stats


@pytest.fixture
def sample_events_df():
    """Tạo tập sự kiện giả lập gồm 1 pha phối hợp dẫn tới bàn thắng."""
    rows = [
        # Match 1, Poss 1: Pass p10 -> p20 (chưa sút)
        {
            "match_id": 1,
            "event_id": "e1",
            "event_index": 1,
            "event_type": "Pass",
            "possession_id": 1,
            "team_name": "Team A",
            "player_id": 10,
            "player_name": "Player 10",
            "x": 30.0,
            "y": 40.0,
            "pass_end_x": 60.0,
            "pass_end_y": 40.0,
            "pass_outcome": None,
            "shot_outcome": None,
            "shot_xg": None,
        },
        # Pass p20 -> p30 (key pass dẫn tới cú sút)
        {
            "match_id": 1,
            "event_id": "e2",
            "event_index": 2,
            "event_type": "Pass",
            "possession_id": 1,
            "team_name": "Team A",
            "player_id": 20,
            "player_name": "Player 20",
            "x": 60.0,
            "y": 40.0,
            "pass_end_x": 105.0,
            "pass_end_y": 38.0,
            "pass_outcome": None,
            "shot_outcome": None,
            "shot_xg": None,
        },
        # Shot của p30 (Goal, xG=0.45)
        {
            "match_id": 1,
            "event_id": "e3",
            "event_index": 3,
            "event_type": "Shot",
            "possession_id": 1,
            "team_name": "Team A",
            "player_id": 30,
            "player_name": "Player 30",
            "x": 105.0,
            "y": 38.0,
            "pass_end_x": None,
            "pass_end_y": None,
            "pass_outcome": None,
            "shot_outcome": "Goal",
            "shot_xg": 0.45,
        },
    ]
    return pd.DataFrame(rows)


def test_compute_player_reference_stats(sample_events_df):
    """Kiểm tra trích xuất key pass, assist và xA."""
    stats = compute_player_reference_stats(sample_events_df)
    assert len(stats) == 1

    # Player 20 thực hiện đường chuyền quyết định dẫn đến cú sút thành bàn
    row20 = stats[stats["player_id"] == 20].iloc[0]
    assert row20["key_passes"] == 1
    assert row20["assists"] == 1
    assert pytest.approx(row20["xA"]) == 0.45


def test_expected_threat_grid_and_convergence(sample_events_df):
    """Kiểm tra mô hình Expected Threat (xT) hội tụ và tăng dần về phía khung thành đối phương."""
    xt_model = ExpectedThreatModel(n_cols=16, n_rows=12, max_iter=20)
    xt_model.fit(sample_events_df)

    assert xt_model.is_fitted is True
    # Giá trị gần cầu môn đối phương (x=115, y=40) phải cao hơn sân nhà (x=20, y=40)
    v_attack = xt_model.get_value(115.0, 40.0)
    v_defense = xt_model.get_value(20.0, 40.0)
    assert v_attack >= v_defense


def test_compute_action_and_player_xt(sample_events_df):
    """Kiểm tra tính gia số Delta xT và tổng hợp theo cầu thủ."""
    xt_model = ExpectedThreatModel(n_cols=16, n_rows=12)
    xt_model.fit(sample_events_df)

    player_xt = xt_model.compute_player_xt(sample_events_df)
    assert len(player_xt) > 0
    # Cầu thủ chuyền bóng tịnh tiến lên phía trước phải có total_xt >= 0
    assert (player_xt["total_xt"] >= 0.0).all()

