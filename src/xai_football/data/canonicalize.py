"""Chuyển đổi raw StatsBomb JSON sang canonical event representation.

Mục tiêu: tạo một lớp trung gian giữa raw JSON và logic xử lý, giảm phụ thuộc
vào cấu trúc nested JSON gốc.

Canonical event là một bản phẳng (flat) với các trường chuẩn — không yêu cầu
mọi trường đều có giá trị cho mọi event.

Output: data/interim/events.parquet
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..utils.io import load_json
from ..utils.logging import get_logger
from ..utils.paths import INTERIM_DIR, RAW_DIR, ensure_dir

logger = get_logger(__name__)


def _name(value: Any) -> str | None:
    """Lấy tên từ trường StatsBomb có thể là dict {'id','name'} hoặc chuỗi."""
    if isinstance(value, dict):
        return value.get("name")
    return value


def _id(value: Any) -> int | None:
    """Lấy id từ trường StatsBomb có thể là dict {'id','name'} hoặc int."""
    if isinstance(value, dict):
        v = value.get("id")
        return int(v) if v is not None else None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _nested(event: dict[str, Any], group: str, key: str) -> Any:
    """Đọc event[group][key], trả None nếu thiếu bất kỳ cấp nào."""
    sub = event.get(group)
    return sub.get(key) if isinstance(sub, dict) else None


def canonicalize_event(event: dict[str, Any], match_id: int) -> dict[str, Any]:
    """Chuyển một StatsBomb raw event thành canonical flat dict."""
    location = event.get("location") or [None, None]
    pass_end = _nested(event, "pass", "end_location") or [None, None]

    return {
        "match_id": match_id,
        "event_id": event.get("id"),
        "event_index": int(event.get("index") or 0),
        "period": int(event.get("period") or 0),
        "timestamp": event.get("timestamp"),
        "minute": int(event.get("minute") or 0),
        "second": int(event.get("second") or 0),
        # Possession
        "possession_id": int(event["possession"]) if event.get("possession") is not None else None,
        "possession_team_id": _id(event.get("possession_team")),
        "possession_team_name": _name(event.get("possession_team")),
        # Team
        "team_id": _id(event.get("team")),
        "team_name": _name(event.get("team")),
        # Player
        "player_id": _id(event.get("player")),
        "player_name": _name(event.get("player")),
        "position_id": _id(event.get("position")),
        "position_name": _name(event.get("position")),
        # Event type
        "event_type": _name(event.get("type")),
        "play_pattern": _name(event.get("play_pattern")),
        # Location
        "x": location[0],
        "y": location[1] if len(location) > 1 else None,
        # Pass fields
        "pass_recipient_id": _id(_nested(event, "pass", "recipient")),
        "pass_recipient_name": _name(_nested(event, "pass", "recipient")),
        "pass_end_x": pass_end[0],
        "pass_end_y": pass_end[1] if len(pass_end) > 1 else None,
        "pass_length": _nested(event, "pass", "length"),
        "pass_angle": _nested(event, "pass", "angle"),
        "pass_height": _name(_nested(event, "pass", "height")),
        "pass_type": _name(_nested(event, "pass", "type")),
        "pass_outcome": _name(_nested(event, "pass", "outcome")),
        # Shot fields
        "shot_outcome": _name(_nested(event, "shot", "outcome")),
        "shot_xg": _nested(event, "shot", "statsbomb_xg"),
    }


def canonicalize_events(events: list[dict[str, Any]], match_id: int) -> pd.DataFrame:
    """Chuyển một danh sách StatsBomb raw events thành canonical DataFrame."""
    return pd.DataFrame([canonicalize_event(e, match_id) for e in events])


def canonicalize_match(
    match_id: int,
    competition_id: int,
    season_id: int,
) -> list[dict[str, Any]]:
    """Đọc raw JSON của một trận và chuyển thành danh sách canonical events."""
    from ..utils.paths import match_json_path

    path = match_json_path(competition_id, season_id, match_id)
    raw = load_json(path)
    if isinstance(raw, dict):
        raw = list(raw.values())

    return [canonicalize_event(e, match_id) for e in raw]


def canonicalize_season(
    competition_id: int,
    season_id: int,
    match_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Chuyển toàn bộ trận của một mùa giải sang canonical DataFrame.

    Parameters
    ----------
    competition_id, season_id : int
        Xác định thư mục raw.
    match_ids : list[int] | None
        Nếu None, quét toàn bộ thư mục raw.
    """
    season_dir = RAW_DIR / str(competition_id) / str(season_id)
    if match_ids is None:
        match_files = sorted(
            p for p in season_dir.glob("*.json")
            if not p.name.endswith(".lineups.json")
        )
        match_ids = [int(p.stem) for p in match_files]

    all_records: list[dict[str, Any]] = []
    for mid in match_ids:
        try:
            all_records.extend(canonicalize_match(mid, competition_id, season_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Trận %s: lỗi canonicalize — %s", mid, exc)

    df = pd.DataFrame(all_records)
    logger.info(
        "Canonicalize comp=%s season=%s: %d trận, %d events",
        competition_id, season_id, len(match_ids), len(df),
    )
    return df


def save_canonical_events(df: pd.DataFrame, path: str | None = None) -> str:
    """Persist canonical events vào parquet.

    Returns
    -------
    str
        Đường dẫn file đã ghi.
    """
    if path is None:
        out = ensure_dir(INTERIM_DIR) / "events.parquet"
    else:
        from pathlib import Path
        out = Path(path)
        ensure_dir(out.parent)

    df.to_parquet(out, index=False, engine="pyarrow")
    logger.info("Đã ghi canonical events: %s (%d rows)", out, len(df))
    return str(out)
