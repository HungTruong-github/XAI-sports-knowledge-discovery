"""Tests for data collection, canonicalization, possession extraction, cleaning, and validation."""

from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd

from xai_football.data.canonicalize import canonicalize_events
from xai_football.data.possession import Possession, PassEvent, ShotInfo, extract_possessions
from xai_football.data.clean import clean_possessions
from xai_football.data.validation import check_match_count, run_all_checks, has_critical_failure


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
            "under_pressure": False,
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


def test_canonicalize_events(dummy_statsbomb_events):
    """Test canonicalization flattens StatsBomb events accurately."""
    df = canonicalize_events(dummy_statsbomb_events, match_id=9999)
    assert len(df) == 4
    assert (df["match_id"] == 9999).all()

    # Pass 1 assertions
    pass1 = df[df["event_id"] == "evt-1"].iloc[0]
    assert pass1["event_type"] == "Pass"
    assert pass1["player_id"] == 101
    assert pass1["pass_recipient_id"] == 102
    assert pass1["x"] == 10.0
    assert pass1["pass_end_x"] == 35.0
    assert pass1["pass_outcome"] == "Complete"

    # Shot assertions
    shot = df[df["event_id"] == "evt-4"].iloc[0]
    assert shot["event_type"] == "Shot"
    assert shot["player_id"] == 103
    assert shot["shot_xg"] == 0.35
    assert shot["shot_outcome"] == "Goal"


def test_extract_possessions(dummy_statsbomb_events):
    """Test possession extraction with player_id identity and has_shot/has_goal semantics."""
    possessions = extract_possessions(dummy_statsbomb_events, match_id=9999)
    assert len(possessions) == 1

    p = possessions[0]
    assert p.match_id == 9999
    assert p.possession_id == 1
    assert p.team_name == "Arsenal"
    assert p.n_passes == 2
    assert p.has_shot is True
    assert p.has_goal is True
    assert len(p.shots) == 1
    assert p.shots[0].outcome == "Goal"
    assert p.shots[0].xg == 0.35

    # Check passes identity by player_id
    assert p.passes[0].passer_id == 101
    assert p.passes[0].recipient_id == 102
    assert p.passes[1].passer_id == 102
    assert p.passes[1].recipient_id == 103

    # Backward compatibility properties
    assert p.ends_with_shot is True
    assert p.ends_with_goal is True


def test_clean_possessions(dummy_statsbomb_events):
    """Test possession cleaning retains valid possessions with min_passes_per_possession=2."""
    possessions = extract_possessions(dummy_statsbomb_events, match_id=9999)
    config = {
        "preprocessing": {
            "min_passes_per_possession": 2,
            "drop_extra_time": True,
            "drop_missing_coordinates": True,
        }
    }
    cleaned, report = clean_possessions(possessions, config)
    assert len(cleaned) == 1
    assert report.n_output == 1
    assert report.dropped_too_short == 0


def test_validation_module():
    """Test validation module produces PASS/WARNING/FAIL records."""
    res1 = check_match_count(n_matches=380, expected=380)
    assert res1.status == "PASS"

    res2 = check_match_count(n_matches=100, expected=380)
    assert res2.status == "FAIL"

    df = run_all_checks([res1])
    assert not has_critical_failure(df)
