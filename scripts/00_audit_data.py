"""Bước 0 — Kiểm tra dataset trước khi tải.

Cách dùng:
    python scripts/00_audit_data.py --config configs/data.yaml

Đầu ra:
    results/tables/dataset_inventory.csv

Script kiểm tra:
  1. competition_id / season_id tồn tại trong StatsBomb Open Data
  2. Số trận thực tế
  3. So sánh với expected_matches nếu có
  4. Thống kê cơ bản: tên giải, mùa, số đội, khoảng ngày
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from xai_football.data.collect import _statsbombpy, resolve_competitions  # noqa: E402
from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import TABLES_DIR, ensure_dir  # noqa: E402

logger = get_logger("audit")


def audit_competition(
    competition_id: int,
    season_id: int,
    competition_name: str = "",
    expected_matches: int | None = None,
) -> dict:
    """Kiểm tra một cặp competition/season."""
    sb = _statsbombpy()

    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    n_matches = len(matches)

    # Thống kê
    result = {
        "competition_name": competition_name,
        "season_name": "",
        "competition_id": competition_id,
        "season_id": season_id,
        "n_matches": n_matches,
        "expected_matches": expected_matches,
        "match_count_ok": True,
        "n_unique_teams": 0,
        "date_min": "",
        "date_max": "",
    }

    if n_matches > 0:
        # Tên mùa giải
        if "season" in matches.columns:
            seasons = matches["season"].unique()
            if len(seasons) > 0:
                s = seasons[0]
                if isinstance(s, dict):
                    result["season_name"] = s.get("season_name", "")
                else:
                    result["season_name"] = str(s)

        # Đội
        teams = set()
        for col in ["home_team", "away_team"]:
            if col in matches.columns:
                for val in matches[col]:
                    if isinstance(val, dict):
                        teams.add(val.get("home_team_name") or val.get("away_team_name") or "")
                    else:
                        teams.add(str(val))
        teams.discard("")
        result["n_unique_teams"] = len(teams)

        # Ngày
        if "match_date" in matches.columns:
            dates = pd.to_datetime(matches["match_date"], errors="coerce")
            result["date_min"] = str(dates.min().date()) if not dates.isna().all() else ""
            result["date_max"] = str(dates.max().date()) if not dates.isna().all() else ""

    # So sánh với expected
    if expected_matches is not None and n_matches != expected_matches:
        result["match_count_ok"] = False
        logger.warning(
            "⚠ %s: kỳ vọng %d trận nhưng thực tế có %d trận!",
            competition_name, expected_matches, n_matches,
        )
    else:
        logger.info(
            "✓ %s: %d trận (khớp kỳ vọng)", competition_name, n_matches,
        )

    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="data.yaml",
                        help="Tệp cấu hình (mặc định: configs/data.yaml)")
    args = parser.parse_args()

    config = load_yaml(args.config)
    expected = config.get("expected_matches")

    entries = resolve_competitions(config)
    rows = []
    any_mismatch = False

    for entry in entries:
        result = audit_competition(
            competition_id=entry["competition_id"],
            season_id=entry["season_id"],
            competition_name=entry["name"],
            expected_matches=expected,
        )
        rows.append(result)
        if not result["match_count_ok"]:
            any_mismatch = True

    table = pd.DataFrame(rows)
    out_path = ensure_dir(TABLES_DIR) / "dataset_inventory.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n=== DATASET INVENTORY ===")
    print(table.to_string(index=False))
    print(f"\nĐã ghi: {out_path}")

    if any_mismatch:
        logger.error("Có mismatch giữa expected và actual match count!")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
