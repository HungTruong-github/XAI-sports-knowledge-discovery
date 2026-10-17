"""Unit tests kiểm tra tính đúng đắn về thứ tự thời gian khi gán nhãn đường chuyền (Priority 2).

Bảo đảm:
1. Cùng một giây: pass followed by shot (index_pass < index_shot) -> nhãn 1.
2. Cùng một giây: shot followed by pass (index_pass > index_shot) -> nhãn 0 (chống gán nhãn nhầm từ sự kiện quá khứ).
3. Khác giây: trong cửa sổ -> 1, ngoài cửa sổ -> 0.
4. Thiếu index hoặc trùng index khi cùng giây: chính sách bảo thủ không gán nhãn dương.
5. Timestamp bất nhất (index báo shot sau pass nhưng time báo shot trước pass) -> 0.
6. Possession không có cú sút -> toàn bộ 0.
7. Phân biệt target="shot" và target="goal".
8. Kiểm tra khác period (hiệp 1 vs hiệp 2) -> 0.
"""

from __future__ import annotations

from xai_football.data.labeling import is_shot_after_pass, label_passes
from xai_football.data.possession import PassEvent, Possession, ShotInfo


def _make_pass(
    event_id: str = "p1",
    index: int = 10,
    minute: int = 1,
    second: int = 30,
    period: int = 1,
) -> PassEvent:
    return PassEvent(
        event_id=event_id,
        index=index,
        period=period,
        minute=minute,
        second=second,
        timestamp=f"00:{minute:02d}:{second:02d}.000",
        passer_id=10,
        passer_name="Player 10",
        recipient_id=11,
        recipient_name="Player 11",
        start_x=40.0,
        start_y=30.0,
        end_x=60.0,
        end_y=35.0,
        length=20.0,
        angle=0.25,
        height="Ground Pass",
        pass_type="Open Play",
        outcome=None,
        team_id=1,
    )


def _make_shot(
    event_id: str = "s1",
    index: int = 12,
    minute: int = 1,
    second: int = 30,
    period: int = 1,
    outcome: str = "Saved",
) -> ShotInfo:
    return ShotInfo(
        event_id=event_id,
        minute=minute,
        second=second,
        outcome=outcome,
        xg=0.15,
        index=index,
        period=period,
    )


def test_pass_followed_by_shot_same_second():
    """Pass (idx=10, t=90s) followed by shot (idx=12, t=90s) trong cùng 1 giây -> hợp lệ (1)."""
    p = _make_pass(event_id="p1", index=10, minute=1, second=30)
    s = _make_shot(event_id="s1", index=12, minute=1, second=30)
    assert is_shot_after_pass(p, s, window_seconds=10.0) is True


def test_shot_followed_by_pass_same_second():
    """Shot (idx=10, t=90s) followed by pass (idx=12, t=90s) trong cùng 1 giây -> KHÔNG hợp lệ (0)."""
    p = _make_pass(event_id="p1", index=12, minute=1, second=30)
    s = _make_shot(event_id="s1", index=10, minute=1, second=30)
    assert is_shot_after_pass(p, s, window_seconds=10.0) is False


def test_events_in_different_seconds():
    """Pass at t=10s, Shot at t=15s (diff=5s <= 10s) -> 1; Shot at t=25s (diff=15s > 10s) -> 0."""
    p = _make_pass(index=10, minute=0, second=10)
    s_within = _make_shot(index=20, minute=0, second=15)
    s_outside = _make_shot(index=30, minute=0, second=25)

    assert is_shot_after_pass(p, s_within, window_seconds=10.0) is True
    assert is_shot_after_pass(p, s_outside, window_seconds=10.0) is False


def test_missing_or_tied_index_handling():
    """Khi thiếu index:
    - Nếu khác giây (dt > 0): xác định được thứ tự -> True
    - Nếu cùng giây (dt == 0): thiếu index phân định -> False (chính sách bảo thủ chống rò rỉ).
    """
    # Cùng giây nhưng không có index
    p_no_idx = _make_pass(index=0, minute=1, second=30)
    s_no_idx = _make_shot(index=0, minute=1, second=30)
    assert is_shot_after_pass(p_no_idx, s_no_idx, window_seconds=10.0) is False

    # Khác giây không có index nhưng trong window
    p_diff = _make_pass(index=0, minute=1, second=20)
    s_diff = _make_shot(index=0, minute=1, second=25)
    assert is_shot_after_pass(p_diff, s_diff, window_seconds=10.0) is True


def test_inconsistent_timestamps():
    """Index báo shot sau pass (idx 20 > idx 10), nhưng timestamp báo shot trước pass (t=80s < t=90s) -> False."""
    p = _make_pass(index=10, minute=1, second=30)  # t = 90s
    s = _make_shot(index=20, minute=1, second=20)  # t = 80s (dt = -10s)
    assert is_shot_after_pass(p, s, window_seconds=10.0) is False


def test_different_periods():
    """Pass ở hiệp 1 (period 1), Shot ở hiệp 2 (period 2) -> không được liên kết nhãn -> False."""
    p = _make_pass(index=10, minute=45, second=0, period=1)
    s = _make_shot(index=20, minute=45, second=5, period=2)
    assert is_shot_after_pass(p, s, window_seconds=10.0) is False


def test_label_passes_possession_integration():
    """Tích hợp label_passes trên một Possession hoàn chỉnh."""
    possession = Possession(
        match_id=1001,
        competition_id=2,
        season_id=27,
        possession_id=1,
        team_id=1,
        team_name="Team A",
        play_pattern="Regular Play",
        period=1,
    )

    # 3 đường chuyền:
    # p1: idx=10, t=10s
    # p2: idx=20, t=20s
    # p3: idx=30, t=30s (sau shot)
    p1 = _make_pass(event_id="p1", index=10, minute=0, second=10)
    p2 = _make_pass(event_id="p2", index=20, minute=0, second=20)
    p3 = _make_pass(event_id="p3", index=30, minute=0, second=30)
    possession.passes = [p1, p2, p3]

    # Cú sút s1: idx=25, t=22s, outcome='Saved'
    s1 = _make_shot(event_id="s1", index=25, minute=0, second=22, outcome="Saved")
    possession.shots = [s1]

    # With window=10s and target='shot':
    # p1 (t=10s): diff=12s > 10s -> 0
    # p2 (t=20s): diff=2s <= 10s, idx 20 < 25 -> 1
    # p3 (t=30s): occurs after shot (idx 30 > 25, dt < 0) -> 0
    labels = label_passes(possession, target="shot", window_seconds=10.0)
    assert labels == [0, 1, 0]

    # With target='goal': s1 không phải Goal -> tất cả 0
    labels_goal = label_passes(possession, target="goal", window_seconds=10.0)
    assert labels_goal == [0, 0, 0]

    # Thêm bàn thắng s2: idx=26, t=24s, outcome='Goal'
    s2 = _make_shot(event_id="s2", index=26, minute=0, second=24, outcome="Goal")
    possession.shots.append(s2)
    labels_goal_after = label_passes(possession, target="goal", window_seconds=10.0)
    assert labels_goal_after == [0, 1, 0]
