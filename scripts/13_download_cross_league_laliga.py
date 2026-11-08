"""Tải toàn bộ dữ liệu La Liga 2015/16 (380 trận) cho Cross-League Evaluation.

Sử dụng ThreadPoolExecutor để tải đa luồng có retry, lưu vào:
    data/raw/11/27/{match_id}.json
    data/raw/11/27/{match_id}.lineups.json
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from statsbombpy import sb
from statsbombpy.api_client import NoAuthWarning

from xai_football.utils.io import save_json
from xai_football.utils.logging import get_logger
from xai_football.utils.paths import lineup_json_path, match_json_path

warnings.simplefilter("ignore", NoAuthWarning)
logger = get_logger("download_laliga")

COMPETITION_ID = 11
SEASON_ID = 27


def download_single_match(match_id: int, overwrite: bool = False) -> tuple[int, bool, int]:
    """Tải 1 trận đấu (events + lineups). Trả về (match_id, success, event_count)."""
    events_path = match_json_path(COMPETITION_ID, SEASON_ID, match_id)
    lineup_path = lineup_json_path(COMPETITION_ID, SEASON_ID, match_id)

    events_valid = False
    event_count = 0
    if events_path.exists() and not overwrite:
        try:
            with open(events_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, (list, dict)) and len(data) > 0:
                    events_valid = True
                    event_count = len(data)
        except Exception:
            events_valid = False

    lineup_valid = False
    if lineup_path.exists() and not overwrite:
        try:
            with open(lineup_path, encoding="utf-8") as f:
                ldata = json.load(f)
                if isinstance(ldata, (list, dict)) and len(ldata) > 0:
                    lineup_valid = True
        except Exception:
            lineup_valid = False

    # Tải events nếu chưa có
    if not events_valid:
        for attempt in range(4):
            try:
                ev = sb.events(match_id=match_id, fmt="dict")
                records = list(ev.values()) if isinstance(ev, dict) else ev
                save_json(records, events_path)
                event_count = len(records)
                events_valid = True
                break
            except Exception as e:
                time.sleep(1.0 + attempt * 1.5)
                if attempt == 3:
                    logger.warning("Thất bại tải events trận %s: %s", match_id, e)

    # Tải lineups nếu chưa có
    if not lineup_valid:
        for attempt in range(4):
            try:
                lineups = sb.lineups(match_id=match_id)
                formatted = {team: df.to_dict("records") for team, df in lineups.items()}
                save_json(formatted, lineup_path)
                lineup_valid = True
                break
            except Exception as e:
                time.sleep(1.0 + attempt * 1.5)
                if attempt == 3:
                    logger.warning("Thất bại tải lineups trận %s: %s", match_id, e)

    return match_id, (events_valid and lineup_valid), event_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="Số luồng tải đồng thời")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số trận tải")
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè tệp cũ")
    args = parser.parse_args()

    logger.info("Đang truy vấn danh sách trận đấu La Liga 2015/16 từ StatsBomb...")
    matches_df = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    match_ids = matches_df["match_id"].astype(int).tolist()
    if args.limit:
        match_ids = match_ids[:args.limit]

    total = len(match_ids)
    logger.info("Bắt đầu tải %d trận đấu (workers=%d)...", total, args.workers)

    success_count = 0
    total_events = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_single_match, mid, args.overwrite): mid for mid in match_ids}
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            mid, ok, n_ev = fut.result()
            if ok:
                success_count += 1
                total_events += n_ev
            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / max(1.0, elapsed)
                logger.info(
                    "Tiến độ: %d/%d trận (%.1f%%) | Thành công: %d | Sự kiện: %d | Tốc độ: %.1f trận/s",
                    completed, total, 100.0 * completed / total, success_count, total_events, rate
                )

    logger.info(
        "Hoàn tất tải La Liga 2015/16: %d/%d trận thành công, %d sự kiện trong %.1f giây.",
        success_count, total, total_events, time.time() - start_time
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
