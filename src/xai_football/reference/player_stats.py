"""Module tính toán số phút thi đấu và các chỉ số tấn công cơ bản (xA, Key Passes, Assists).

Dữ liệu nguồn:
- Lineups: data/raw/2/27/*.lineups.json (thời gian ra sân chính xác từng cầu thủ)
- Events: data/interim/official/events.parquet (các sự kiện đường chuyền và cú sút)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xai_football.utils.logging import get_logger

logger = get_logger("player_stats")


def _parse_time_str(t_str: str | None, default_val: float = 90.0) -> float:
    """Chuyển đổi chuỗi thời gian 'MM:SS' sang số phút thực (float)."""
    if t_str is None or pd.isna(t_str):
        return default_val
    try:
        parts = str(t_str).strip().split(":")
        if len(parts) == 2:
            return float(parts[0]) + float(parts[1]) / 60.0
        elif len(parts) == 3:
            return float(parts[0]) * 60.0 + float(parts[1]) + float(parts[2]) / 60.0
        return float(parts[0])
    except Exception:
        return default_val


def compute_player_minutes(
    lineups_dir: Path | str = "data/raw/2/27",
    match_ids: set[int] | None = None,
    return_match_level: bool = False,
) -> pd.DataFrame:
    """Tính tổng số phút thi đấu của từng cầu thủ từ các tệp *.lineups.json.

    Args:
        lineups_dir: Thư mục chứa các tệp lineups json của mùa giải.
        match_ids: Tập match_id cần lọc (nếu None, tính toàn bộ 380 trận).
        return_match_level: Nếu True, trả về chi tiết từng trận [match_id, player_id, minutes, is_goalkeeper].

    Returns:
        DataFrame: [player_id, player_name, team_name, minutes_played, match_count, is_goalkeeper]
    """
    p_dir = Path(lineups_dir)
    lineup_files = list(p_dir.glob("*.lineups.json"))
    if not lineup_files:
        logger.warning("Không tìm thấy tệp lineup nào tại %s", p_dir)
        cols = ["match_id", "player_id", "minutes", "is_goalkeeper"] if return_match_level else ["player_id", "player_name", "team_name", "minutes_played", "match_count", "is_goalkeeper"]
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, Any]] = []

    for fpath in lineup_files:
        try:
            mid = int(fpath.stem.split(".")[0])
        except ValueError:
            continue

        if match_ids is not None and mid not in match_ids:
            continue

        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Lỗi đọc tệp %s: %s", fpath, exc)
            continue

        # Lineup json có định dạng dict: {team_name: [players...]}
        for team_name, players in data.items():
            if not isinstance(players, list):
                continue
            for p in players:
                pid = p.get("player_id")
                pname = p.get("player_name")
                positions = p.get("positions", [])
                if not pid or not positions:
                    continue

                total_p_mins = 0.0
                is_gk = any(pos.get("position") == "Goalkeeper" or pos.get("position_id") == 1 for pos in positions)
                for pos in positions:
                    t_from = _parse_time_str(pos.get("from"), default_val=0.0)
                    t_to = _parse_time_str(pos.get("to"), default_val=90.0)
                    duration = max(0.0, t_to - t_from)
                    total_p_mins += duration

                if total_p_mins > 0.0:
                    rows.append({
                        "match_id": mid,
                        "team_name": team_name,
                        "player_id": int(pid),
                        "player_name": pname,
                        "minutes": total_p_mins,
                        "is_goalkeeper": is_gk,
                    })

    if not rows:
        return pd.DataFrame(columns=["player_id", "player_name", "team_name", "minutes_played", "match_count", "is_goalkeeper"])

    df = pd.DataFrame(rows)
    if return_match_level:
        return df

    # Tổng hợp theo duy nhất player_id để đảm bảo 1 dòng / 1 cầu thủ
    agg = df.groupby("player_id").agg(
        player_name=("player_name", "last"),
        team_name=("team_name", "last"),
        minutes_played=("minutes", "sum"),
        match_count=("match_id", "nunique"),
        is_goalkeeper=("is_goalkeeper", "any"),
    ).reset_index()

    agg["minutes_played"] = np.round(agg["minutes_played"], 1)
    logger.info("Đã tính số phút thi đấu cho %d cầu thủ trên %d trận", len(agg), df["match_id"].nunique())
    return agg


def compute_player_reference_stats(
    events_df: pd.DataFrame,
    minutes_df: pd.DataFrame | None = None,
    match_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Trích xuất Key Passes, Assists và Expected Assists (xA) từ dữ liệu sự kiện.

    Định nghĩa chuẩn:
        - Key Pass (Shot Assist): Đường chuyền thành công cuối cùng dẫn tới cú sút trong cùng một possession.
        - xA (Expected Assists): Giá trị shot_xg của cú sút mà đường chuyền đó kiến tạo.
        - Goal Assist: Key Pass mà cú sút dẫn tới bàn thắng (shot_outcome == 'Goal').

    Args:
        events_df: Bảng toàn bộ sự kiện trận đấu (events.parquet).
        minutes_df: Bảng số phút thi đấu của cầu thủ (từ compute_player_minutes).
        match_ids: Lọc tập trận đấu (vd: Set A hoặc Set B).

    Returns:
        DataFrame: [player_id, player_name, team_name, minutes_played, key_passes, assists, xA, key_passes_per_90, xA_per_90]
    """
    df = events_df
    if match_ids is not None:
        df = df[df["match_id"].isin(match_ids)]

    # Lọc các possession có xuất hiện cú sút
    shot_events = df[df["event_type"] == "Shot"].copy()
    if shot_events.empty:
        return pd.DataFrame(columns=[
            "player_id", "player_name", "team_name", "minutes_played",
            "key_passes", "assists", "xA", "key_passes_per_90", "xA_per_90",
        ])

    kp_rows: list[dict[str, Any]] = []

    # Nhóm theo (match_id, possession_id) để truy vết đường chuyền dẫn tới cú sút
    grouped = df.groupby(["match_id", "possession_id"])
    for (mid, pid), group in grouped:
        shots = group[group["event_type"] == "Shot"]
        if shots.empty:
            continue

        for _, shot in shots.iterrows():
            shot_idx = shot["event_index"]
            shot_xg = float(shot["shot_xg"]) if pd.notna(shot["shot_xg"]) else 0.0
            is_goal = (shot.get("shot_outcome") == "Goal")

            # Tìm đường chuyền thành công ngay trước cú sút
            prior_passes = group[
                (group["event_index"] < shot_idx)
                & (group["event_type"] == "Pass")
                & (group["pass_outcome"].isna())
            ]
            if not prior_passes.empty:
                last_pass = prior_passes.iloc[-1]
                passer_id = last_pass.get("player_id")
                passer_name = last_pass.get("player_name")
                team_name = last_pass.get("team_name")

                if pd.notna(passer_id):
                    kp_rows.append({
                        "player_id": int(passer_id),
                        "player_name": passer_name,
                        "team_name": team_name,
                        "key_passes": 1,
                        "assists": 1 if is_goal else 0,
                        "xA": shot_xg,
                    })

    if not kp_rows:
        kp_df = pd.DataFrame(columns=["player_id", "player_name", "team_name", "key_passes", "assists", "xA"])
    else:
        kp_raw = pd.DataFrame(kp_rows)
        kp_df = kp_raw.groupby("player_id").agg(
            player_name=("player_name", "last"),
            team_name=("team_name", "last"),
            key_passes=("key_passes", "sum"),
            assists=("assists", "sum"),
            xA=("xA", "sum"),
        ).reset_index()

    # Ghép với số phút thi đấu nếu có
    if minutes_df is not None and not minutes_df.empty:
        merged = pd.merge(minutes_df, kp_df, on="player_id", how="left")
        merged["player_name"] = merged["player_name_x"].combine_first(merged["player_name_y"])
        merged["team_name"] = merged["team_name_x"].combine_first(merged["team_name_y"])
        merged = merged.drop(columns=["player_name_x", "player_name_y", "team_name_x", "team_name_y"], errors="ignore")
        merged["key_passes"] = merged["key_passes"].fillna(0).astype(int)
        merged["assists"] = merged["assists"].fillna(0).astype(int)
        merged["xA"] = np.round(merged["xA"].fillna(0.0), 3)

        # Tính per 90 (chỉ tính cho cầu thủ có >= 90 phút thi đấu để tránh mẫu nhỏ)
        mins = merged["minutes_played"]
        merged["key_passes_per_90"] = np.where(mins >= 90.0, np.round((merged["key_passes"] / mins) * 90.0, 3), 0.0)
        merged["assists_per_90"] = np.where(mins >= 90.0, np.round((merged["assists"] / mins) * 90.0, 3), 0.0)
        merged["xA_per_90"] = np.where(mins >= 90.0, np.round((merged["xA"] / mins) * 90.0, 3), 0.0)
        return merged.drop_duplicates(subset=["player_id"]).sort_values(by="xA", ascending=False).reset_index(drop=True)

    return kp_df.drop_duplicates(subset=["player_id"]).sort_values(by="xA", ascending=False).reset_index(drop=True)

