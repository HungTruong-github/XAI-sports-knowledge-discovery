"""Unit tests for data collection, canonicalization, possession extraction, and validation."""

from __future__ import annotations

import copy

import pytest

from xai_football.data.canonicalize import canonicalize_events
from xai_football.data.clean import clean_possessions
from xai_football.data.labeling import label_passes
from xai_football.data.possession import PassEvent, Possession, ShotInfo, extract_possessions
from xai_football.data.validation import (
    check_actual_shots_per_match,
    check_duplicate_possession_keys,
    check_match_count,
    check_missing_coordinate_rate,
    check_missing_recipient_rate,
    check_possession_team_consistency,
    has_critical_failure,
    run_all_checks,
)
from xai_football.utils.io import load_yaml
from xai_football.utils.paths import get_interim_dir, get_tables_dir


@pytest.fixture
def dummy_statsbomb_events():
    """Tạo dummy raw StatsBomb events cho 1 trận đấu."""
    return [
        {
            "id": "evt-1",
            "index": 1,
            "period": 1,
            "timestamp": "00:00:05.100",
            "minute": 0,
            "second": 5,
            "type": {"id": 30, "name": "Pass"},
            "possession": 1,
            "possession_team": {"id": 1, "name": "Arsenal"},
            "play_pattern": {"id": 1, "name": "Regular Play"},
            "team": {"id": 1, "name": "Arsenal"},
            "player": {"id": 101, "name": "Player A"},
            "position": {"id": 1, "name": "Goalkeeper"},
            "location": [10.0, 40.0],
            "duration": 1.2,
            "pass": {
                "recipient": {"id": 102, "name": "Player B"},
                "length": 25.0,
                "angle": 0.5,
                "end_location": [35.0, 40.0],
                "type": {"id": 8, "name": "Simple Pass"},
                "outcome": {"id": 8, "name": "Complete"},
            },
        },
        {
            "id": "evt-2",
            "index": 2,
            "period": 1,
            "timestamp": "00:00:08.000",
            "minute": 0,
            "second": 8,
            "type": {"id": 42, "name": "Ball Receipt*"},
            "possession": 1,
            "possession_team": {"id": 1, "name": "Arsenal"},
            "play_pattern": {"id": 1, "name": "Regular Play"},
            "team": {"id": 1, "name": "Arsenal"},
            "player": {"id": 102, "name": "Player B"},
            "position": {"id": 2, "name": "Right Back"},
            "location": [35.0, 40.0],
        },
        {
            "id": "evt-3",
            "index": 3,
            "period": 1,
            "timestamp": "00:00:10.000",
            "minute": 0,
            "second": 10,
            "type": {"id": 30, "name": "Pass"},
            "possession": 1,
            "possession_team": {"id": 1, "name": "Arsenal"},
            "play_pattern": {"id": 1, "name": "Regular Play"},
            "team": {"id": 1, "name": "Arsenal"},
            "player": {"id": 102, "name": "Player B"},
            "position": {"id": 2, "name": "Right Back"},
            "location": [35.0, 40.0],
            "pass": {
                "recipient": {"id": 103, "name": "Player C"},
                "length": 40.0,
                "angle": 0.2,
                "end_location": [75.0, 40.0],
                "outcome": None,  # Complete pass
            },
        },
        {
            "id": "evt-4",
            "index": 4,
            "period": 1,
            "timestamp": "00:00:15.000",
            "minute": 0,
            "second": 15,
            "type": {"id": 16, "name": "Shot"},
            "possession": 1,
            "possession_team": {"id": 1, "name": "Arsenal"},
            "play_pattern": {"id": 1, "name": "Regular Play"},
            "team": {"id": 1, "name": "Arsenal"},
            "player": {"id": 103, "name": "Player C"},
            "position": {"id": 9, "name": "Center Forward"},
            "location": [105.0, 40.0],
            "shot": {
                "statsbomb_xg": 0.35,
                "end_location": [120.0, 42.0, 1.2],
                "outcome": {"id": 97, "name": "Goal"},
            },
        },
    ]


def test_official_config_parameters():
    """A. Test official config values."""
    config = load_yaml("configs/data.yaml")
    assert config["run_name"] == "official"
    comp = config["competitions"][0]
    assert comp["competition_id"] == 2
    assert comp["season_id"] == 27
    assert config["expected_matches"] == 380


def test_possession_unique_key():
    """B. Test unique (match_id, possession_id) identity and validation."""
    keys_valid = [(1, 1), (1, 2), (2, 1)]
    res_valid = check_duplicate_possession_keys(keys_valid)
    assert res_valid.status == "PASS"

    keys_invalid = [(1, 1), (1, 1)]
    res_invalid = check_duplicate_possession_keys(keys_invalid)
    assert res_invalid.status == "FAIL"


def test_successful_and_incomplete_passes():
    """C & D. Test successful pass inclusion and incomplete pass exclusion."""
    events = [
        {
            "id": "p1",
            "index": 1,
            "type": {"name": "Pass"},
            "possession": 1,
            "possession_team": {"id": 1, "name": "Team A"},
            "team": {"id": 1, "name": "Team A"},
            "player": {"id": 10, "name": "P1"},
            "location": [10.0, 20.0],
            "pass": {
                "recipient": {"id": 11, "name": "P2"},
                "end_location": [30.0, 40.0],
                "outcome": None,  # Successful
            },
        },
        {
            "id": "p2",
            "index": 2,
            "type": {"name": "Pass"},
            "possession": 1,
            "possession_team": {"id": 1, "name": "Team A"},
            "team": {"id": 1, "name": "Team A"},
            "player": {"id": 11, "name": "P2"},
            "location": [30.0, 40.0],
            "pass": {
                "recipient": {"id": 12, "name": "P3"},
                "end_location": [50.0, 60.0],
                "outcome": {"name": "Incomplete"},  # Incomplete
            },
        },
    ]
    possessions = extract_possessions(events, match_id=100)
    assert len(possessions) == 1
    p = possessions[0]
    assert p.n_passes == 1  # Only complete pass is included
    assert p.passes[0].event_id == "p1"


def test_recipient_and_player_id_identity(dummy_statsbomb_events):
    """E & F. Test recipient extraction and player_id identity."""
    possessions = extract_possessions(dummy_statsbomb_events, match_id=9999)
    p = possessions[0]
    assert p.passes[0].passer_id == 101
    assert p.passes[0].passer_name == "Player A"
    assert p.passes[0].recipient_id == 102
    assert p.passes[0].recipient_name == "Player B"
    assert p.players == [101, 102, 103]


def test_multiple_shots_and_multi_shot_labeling():
    """G, H, & I. Test multiple shots and multi-shot pass labeling window."""
    p = Possession(
        match_id=1,
        competition_id=2,
        season_id=27,
        possession_id=1,
        team_id=1,
        team_name="Team A",
        play_pattern="Regular Play",
        period=1,
        shots=[
            ShotInfo(event_id="s1", minute=0, second=15, outcome="Saved", xg=0.1),
            ShotInfo(event_id="s2", minute=0, second=25, outcome="Goal", xg=0.5),
        ],
    )

    p.passes = [
        PassEvent(
            "p1",
            1,
            1,
            0,
            10,
            "00:00:10",
            101,
            "A",
            102,
            "B",
            10.0,
            20.0,
            30.0,
            40.0,
            20.0,
            0.0,
            "Low",
            "Simple",
            None,
        ),
        PassEvent(
            "p2",
            2,
            1,
            0,
            20,
            "00:00:20",
            102,
            "B",
            103,
            "C",
            30.0,
            40.0,
            50.0,
            60.0,
            20.0,
            0.0,
            "Low",
            "Simple",
            None,
        ),
        PassEvent(
            "p3",
            3,
            1,
            0,
            30,
            "00:00:30",
            103,
            "C",
            104,
            "D",
            50.0,
            60.0,
            70.0,
            80.0,
            20.0,
            0.0,
            "Low",
            "Simple",
            None,
        ),
    ]

    assert p.has_shot is True
    assert p.has_goal is True

    # With window = 10s:
    # p1 at t=10s is followed by s1 at t=15s (diff=5s <= 10) -> 1
    # p2 at t=20s is followed by s2 at t=25s (diff=5s <= 10) -> 1
    # p3 at t=30s is after both shots -> 0
    labels_shot = label_passes(p, target="shot", window_seconds=10.0)
    assert labels_shot == [1, 1, 0]

    # With target="goal": only s2 (t=25s) is valid
    # p1 at t=10s is 15s before s2 (> 10s) -> 0
    # p2 at t=20s is 5s before s2 (<= 10s) -> 1
    # p3 at t=30s is after s2 -> 0
    labels_goal = label_passes(p, target="goal", window_seconds=10.0)
    assert labels_goal == [0, 1, 0]


def test_clean_possessions_min_passes(dummy_statsbomb_events):
    """J. Test cleaning min_passes threshold filter."""
    possessions = extract_possessions(dummy_statsbomb_events, match_id=9999)
    config_strict = {
        "preprocessing": {
            "min_passes_per_possession": 5,
            "drop_extra_time": True,
            "drop_missing_coordinates": True,
        }
    }
    cleaned, report = clean_possessions(possessions, config_strict)
    assert len(cleaned) == 0
    assert report.dropped_too_short == 1


def test_missing_coordinate_and_recipient_checks():
    """K & L. Test missing coordinate and missing recipient validation checks."""
    res_coord = check_missing_coordinate_rate(n_missing=2, n_total=100, threshold=0.05)
    assert res_coord.status == "PASS"

    res_recip = check_missing_recipient_rate(n_missing=5, n_passes=100, threshold=0.15)
    assert res_recip.status == "PASS"


def test_raw_json_not_modified(dummy_statsbomb_events):
    """M. Test raw JSON is not mutated during preprocessing."""
    original = copy.deepcopy(dummy_statsbomb_events)
    _ = extract_possessions(dummy_statsbomb_events, match_id=9999)
    assert dummy_statsbomb_events == original


def test_canonical_columns_complete(dummy_statsbomb_events):
    """N. Test canonical DataFrame columns completeness."""
    df = canonicalize_events(dummy_statsbomb_events, match_id=9999)
    expected_cols = [
        "match_id",
        "event_id",
        "event_index",
        "period",
        "timestamp",
        "minute",
        "second",
        "possession_id",
        "possession_team_id",
        "possession_team_name",
        "team_id",
        "team_name",
        "player_id",
        "player_name",
        "position_id",
        "position_name",
        "event_type",
        "play_pattern",
        "x",
        "y",
        "pass_recipient_id",
        "pass_recipient_name",
        "pass_end_x",
        "pass_end_y",
        "pass_length",
        "pass_angle",
        "pass_height",
        "pass_type",
        "pass_outcome",
        "shot_outcome",
        "shot_xg",
    ]
    for col in expected_cols:
        assert col in df.columns


def test_official_and_pilot_paths_no_collision():
    """O. Test official vs pilot paths do not collide."""
    cfg_official = {"run_name": "official"}
    cfg_pilot = {"run_name": "pilot"}

    interim_off = get_interim_dir(cfg_official)
    interim_pil = get_interim_dir(cfg_pilot)
    tables_off = get_tables_dir(cfg_official)
    tables_pil = get_tables_dir(cfg_pilot)

    assert interim_off != interim_pil
    assert tables_off != tables_pil
    assert interim_off.name == "official"
    assert interim_pil.name == "pilot"


def test_validation_fail_on_mismatched_match_count():
    """P. Test validation FAIL when official match count != 380 (strict mode)."""
    res_fail = check_match_count(n_matches=300, expected=380, strict=True)
    assert res_fail.status == "FAIL"

    df_checks = run_all_checks([res_fail])
    assert has_critical_failure(df_checks)


def test_official_strict_match_count_379_fails():
    """Q. Official run: 379/380 must FAIL, not WARNING."""
    res = check_match_count(n_matches=379, expected=380, strict=True)
    assert res.status == "FAIL"


def test_official_strict_match_count_380_passes():
    """R. Official run: exactly 380 must PASS."""
    res = check_match_count(n_matches=380, expected=380, strict=True)
    assert res.status == "PASS"


def test_pilot_match_count_not_forced_to_380():
    """S. Pilot: mismatch is only WARNING, not FAIL."""
    res = check_match_count(n_matches=5, expected=380, strict=False)
    assert res.status == "WARNING"
    assert res.status != "FAIL"


def test_actual_shot_count_differs_from_shot_positive_possessions():
    """T. 1 possession with 2 Shot events: shot-positive = 1, actual shots = 2."""
    p = Possession(
        match_id=1,
        competition_id=2,
        season_id=27,
        possession_id=1,
        team_id=1,
        team_name="Team A",
        play_pattern="Regular Play",
        period=1,
        shots=[
            ShotInfo(event_id="s1", minute=0, second=15, outcome="Saved", xg=0.1),
            ShotInfo(event_id="s2", minute=0, second=18, outcome="Goal", xg=0.5),
        ],
    )
    # Shot-positive possessions = 1
    assert p.has_shot is True
    # Actual shot count = 2 (len(p.shots))
    assert len(p.shots) == 2
    # These are distinct concepts
    shot_positive_count = 1 if p.has_shot else 0
    actual_shot_count = len(p.shots)
    assert shot_positive_count != actual_shot_count

    # Validation check uses actual shots
    res = check_actual_shots_per_match(n_actual_shots=2, n_matches=1)
    assert res.status == "WARNING"  # 2 shots/match < 15.0 lower bound


def test_possession_team_consistency_pass():
    """U. Same possession_team_id within a possession -> PASS."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "match_id": [1, 1, 1],
            "possession_id": [1, 1, 1],
            "possession_team_id": [10, 10, 10],
        }
    )
    res = check_possession_team_consistency(df)
    assert res.status == "PASS"
    assert res.value == 0


def test_possession_team_consistency_mixed():
    """V. Mixed possession_team_id within a possession -> WARNING."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "match_id": [1, 1, 1],
            "possession_id": [1, 1, 1],
            "possession_team_id": [10, 10, 20],
        }
    )
    res = check_possession_team_consistency(df)
    assert res.status == "WARNING"
    assert res.value == 1
