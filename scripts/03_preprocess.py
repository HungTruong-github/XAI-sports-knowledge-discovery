"""Bước 2 — Tiền xử lý dữ liệu (Canonicalize, Extract Possessions, Clean, Validate).

Cách dùng:
    python scripts/03_preprocess.py --config configs/data.yaml
    python scripts/03_preprocess.py --config configs/data_pilot.yaml --limit-matches 5

Pipeline:
  1. Canonicalize raw JSON -> data/interim/<run_name>/events.parquet
  2. Extract & clean possessions -> data/interim/<run_name>/possessions.parquet
                                -> data/interim/<run_name>/possession_events.parquet
  3. Validate unique keys (match_id, possession_id)
  4. Label comparison statistics -> results/tables/<run_name>/label_comparison.csv
  5. Play pattern distribution -> results/tables/<run_name>/play_pattern_distribution.csv
  6. Cleaning report -> results/tables/<run_name>/cleaning_report.csv
  7. Sanity checks -> results/tables/<run_name>/sanity_checks.csv
  8. Data pipeline summary -> results/tables/<run_name>/data_pipeline_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from xai_football.data.canonicalize import canonicalize_match  # noqa: E402
from xai_football.data.clean import (  # noqa: E402
    CleaningReport,
    attack_direction_diagnostic,
    clean_possessions,
)
from xai_football.data.collect import resolve_competitions  # noqa: E402
from xai_football.data.labeling import compare_labeling_schemes  # noqa: E402
from xai_football.data.possession import Possession, load_match_possessions  # noqa: E402
from xai_football.data.validation import (  # noqa: E402
    CheckResult,
    check_actual_shots_per_match,
    check_attack_direction,
    check_duplicate_event_ids,
    check_duplicate_possession_keys,
    check_events_per_match,
    check_graph_size_proxy,
    check_match_count,
    check_missing_coordinate_rate,
    check_missing_recipient_rate,
    check_positive_shot_rate,
    check_possession_team_consistency,
    check_possessions_per_match,
    check_successful_pass_coordinate_missingness,
    check_successful_passes_per_match,
    check_unique_teams,
    has_critical_failure,
    run_all_checks,
)
from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import (  # noqa: E402
    RAW_DIR,
    get_interim_dir,
    get_run_name,
    get_tables_dir,
)

logger = get_logger("preprocess")


def _get_match_files(competition_id: int, season_id: int, limit: int | None = None) -> list[Path]:
    season_dir = RAW_DIR / str(competition_id) / str(season_id)
    if not season_dir.exists():
        return []
    files = sorted(p for p in season_dir.glob("*.json") if not p.name.endswith(".lineups.json"))
    if limit is not None:
        files = files[:limit]
    return files


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--limit-matches", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    run_name = get_run_name(config)
    is_official = run_name == "official"

    entries = resolve_competitions(config)
    expected_matches = config.get("expected_matches")

    out_dir = get_tables_dir(config)
    interim_dir = get_interim_dir(config)

    # ================================================================
    # Phase 1: Canonicalize events (chunked writing to prevent high RAM usage)
    # ================================================================
    events_parquet_path = interim_dir / "events.parquet"
    writer = None
    total_canonical_events = 0
    total_canonical_matches = 0
    failed_canonical_match_ids = []

    # Collected metrics from events DataFrame chunk by chunk
    event_ids_all: list[str] = []
    events_per_match_dict: dict[int, int] = {}
    pass_events_total = 0
    pass_events_missing_recipient = 0
    missing_coords_count = 0
    actual_shot_count = 0  # Actual Shot events (not shot-positive possessions)

    for entry in entries:
        comp_id, season_id = entry["competition_id"], entry["season_id"]
        match_files = _get_match_files(comp_id, season_id, args.limit_matches)

        if not match_files:
            logger.warning("Chưa có dữ liệu thô cho %s", entry["name"])
            continue

        logger.info("Canonicalize %s: %d trận...", entry["name"], len(match_files))

        for p in match_files:
            mid = int(p.stem)
            try:
                records = canonicalize_match(mid, comp_id, season_id)
                df_chunk = pd.DataFrame(records)

                if not df_chunk.empty:
                    # Write to Parquet incrementally
                    table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(events_parquet_path, table.schema)
                    writer.write_table(table)

                    total_canonical_events += len(df_chunk)
                    total_canonical_matches += 1
                    events_per_match_dict[mid] = len(df_chunk)

                    # Collect light metrics
                    if "event_id" in df_chunk.columns:
                        event_ids_all.extend(df_chunk["event_id"].dropna().tolist())

                    if "x" in df_chunk.columns and "y" in df_chunk.columns:
                        missing_coords_count += int(df_chunk[["x", "y"]].isna().any(axis=1).sum())

                    passes_chunk = df_chunk[df_chunk["event_type"] == "Pass"]
                    pass_events_total += len(passes_chunk)
                    if "pass_recipient_id" in passes_chunk.columns:
                        pass_events_missing_recipient += int(
                            passes_chunk["pass_recipient_id"].isna().sum()
                        )

                    actual_shot_count += int((df_chunk["event_type"] == "Shot").sum())

            except Exception as exc:  # noqa: BLE001
                logger.error("Trận %s: lỗi canonicalize — %s", mid, exc)
                failed_canonical_match_ids.append(mid)

    if writer is not None:
        writer.close()

    if total_canonical_matches == 0:
        logger.error(
            "Không có match nào được canonicalize thành công! Hãy chạy 01_download_data.py trước."
        )
        return 1

    if failed_canonical_match_ids:
        n_fail_can = len(failed_canonical_match_ids)
        msg = f"Có {n_fail_can} trận lỗi canonicalize: {failed_canonical_match_ids}"
        if is_official:
            logger.error(msg)
            return 1
        logger.warning(msg)

    logger.info(
        "Đã ghi %s (%d events, %d trận)",
        events_parquet_path,
        total_canonical_events,
        total_canonical_matches,
    )

    # ================================================================
    # Phase 2: Extract & Clean Possessions
    # ================================================================
    all_possessions_raw: list[Possession] = []
    all_possessions_clean: list[Possession] = []
    total_report = CleaningReport()
    match_ids_processed: list[int] = []
    possessions_per_match: list[int] = []

    for entry in entries:
        comp_id, season_id = entry["competition_id"], entry["season_id"]
        match_files = _get_match_files(comp_id, season_id, args.limit_matches)

        for path in match_files:
            mid = int(path.stem)
            try:
                raw = load_match_possessions(comp_id, season_id, mid)
                all_possessions_raw.extend(raw)

                kept, report = clean_possessions(raw, config)
                all_possessions_clean.extend(kept)
                total_report = total_report.merge(report)
                match_ids_processed.append(mid)
                possessions_per_match.append(len(kept))
            except Exception as exc:  # noqa: BLE001
                logger.error("Trận %s: lỗi load/clean possession — %s", mid, exc)

    n_matches = len(match_ids_processed)
    logger.info(
        "Tổng: %d trận -> %d raw possessions -> %d clean possessions",
        n_matches,
        len(all_possessions_raw),
        len(all_possessions_clean),
    )

    # ================================================================
    # Phase 3: Unique Key Validation & Persist Possessions
    # ================================================================
    possession_keys = [(p.match_id, p.possession_id) for p in all_possessions_clean]
    n_unique_keys = len(set(possession_keys))
    if len(possession_keys) != n_unique_keys:
        n_dup = len(possession_keys) - n_unique_keys
        logger.error(
            "LỖI NGHIÊM TRỌNG: Phát hiện %d trùng lặp cặp (match_id, possession_id)!", n_dup
        )
        return 1

    possession_rows = []
    possession_event_rows = []
    for p in all_possessions_clean:
        possession_rows.append(
            {
                "match_id": p.match_id,
                "possession_id": p.possession_id,
                "competition_id": p.competition_id,
                "season_id": p.season_id,
                "team_id": p.team_id,
                "team_name": p.team_name,
                "play_pattern": p.play_pattern,
                "period": p.period,
                "n_passes": p.n_passes,
                "n_players": p.n_players,
                "n_unique_edges": p.n_unique_edges,
                "has_shot": p.has_shot,
                "has_goal": p.has_goal,
                "shot_second": p.shot_second,
                "n_shots": len(p.shots),
            }
        )
        for pass_ev in p.passes:
            possession_event_rows.append(
                {
                    "match_id": p.match_id,
                    "possession_id": p.possession_id,
                    "event_id": pass_ev.event_id,
                    "event_index": pass_ev.index,
                    "period": pass_ev.period,
                    "minute": pass_ev.minute,
                    "second": pass_ev.second,
                    "passer_id": pass_ev.passer_id,
                    "passer_name": pass_ev.passer_name,
                    "recipient_id": pass_ev.recipient_id,
                    "recipient_name": pass_ev.recipient_name,
                    "start_x": pass_ev.start_x,
                    "start_y": pass_ev.start_y,
                    "end_x": pass_ev.end_x,
                    "end_y": pass_ev.end_y,
                    "length": pass_ev.length,
                    "angle": pass_ev.angle,
                    "height": pass_ev.height,
                    "pass_type": pass_ev.pass_type,
                    "outcome": pass_ev.outcome,
                }
            )

    possessions_df = pd.DataFrame(possession_rows)
    possession_events_df = pd.DataFrame(possession_event_rows)

    possessions_df.to_parquet(interim_dir / "possessions.parquet", index=False, engine="pyarrow")
    possession_events_df.to_parquet(
        interim_dir / "possession_events.parquet", index=False, engine="pyarrow"
    )
    logger.info(
        "Đã ghi possessions.parquet (%d rows) và possession_events.parquet (%d rows)",
        len(possessions_df),
        len(possession_events_df),
    )

    # ================================================================
    # Phase 4: Label Comparison Table
    # ================================================================
    label_stats = compare_labeling_schemes(all_possessions_clean)
    labels_df = pd.DataFrame([s.as_row() for s in label_stats])
    labels_df.to_csv(out_dir / "label_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n=== SO SÁNH CÁC CÁCH GÁN NHÃN ===")
    print(labels_df.to_string(index=False))

    # ================================================================
    # Phase 5: Play Pattern Distribution
    # ================================================================
    play_pattern_counts: dict[str, int] = {}
    for p in all_possessions_raw:
        pp = p.play_pattern or "Unknown"
        play_pattern_counts[pp] = play_pattern_counts.get(pp, 0) + 1

    pp_df = pd.DataFrame(
        [
            {
                "play_pattern": k,
                "count": v,
                "pct": round(100 * v / len(all_possessions_raw), 2) if all_possessions_raw else 0,
            }
            for k, v in sorted(play_pattern_counts.items(), key=lambda x: -x[1])
        ]
    )
    pp_df.to_csv(out_dir / "play_pattern_distribution.csv", index=False, encoding="utf-8-sig")

    # ================================================================
    # Phase 6: Cleaning Report
    # ================================================================
    cleaning_df = pd.DataFrame([total_report.as_row()])
    cleaning_df.to_csv(out_dir / "cleaning_report.csv", index=False, encoding="utf-8-sig")

    # ================================================================
    # Phase 7: Sanity Checks (Schema: check, value, status, note)
    # ================================================================
    checks: list[CheckResult] = []

    # 1. Match count (official = strict: any mismatch is FAIL)
    checks.append(check_match_count(n_matches, expected_matches, strict=is_official))

    # 2. Unique teams
    all_teams = {p.team_name for p in all_possessions_clean if p.team_name}
    checks.append(check_unique_teams(len(all_teams)))

    # 3. Events per match
    if events_per_match_dict:
        checks.append(check_events_per_match(list(events_per_match_dict.values())))

    # 4. Possessions per match
    if possessions_per_match:
        checks.append(check_possessions_per_match(possessions_per_match))

    # 5. Successful passes per match
    passes_per_match_list = []
    for mid in set(p.match_id for p in all_possessions_clean):
        n_p = sum(pp.n_passes for pp in all_possessions_clean if pp.match_id == mid)
        passes_per_match_list.append(n_p)
    if passes_per_match_list:
        checks.append(check_successful_passes_per_match(passes_per_match_list))

    # 6. Overall event missing coordinate rate
    if total_canonical_events > 0:
        checks.append(check_missing_coordinate_rate(missing_coords_count, total_canonical_events))

    # 7. Successful pass coordinate missingness (edge quality)
    if not possession_events_df.empty:
        n_succ_passes = len(possession_events_df)
        n_miss_start = int(possession_events_df[["start_x", "start_y"]].isna().any(axis=1).sum())
        n_miss_end = int(possession_events_df[["end_x", "end_y"]].isna().any(axis=1).sum())
        checks.append(
            check_successful_pass_coordinate_missingness(n_miss_start, n_miss_end, n_succ_passes)
        )

    # 8. Overall pass missing recipient rate
    if pass_events_total > 0:
        checks.append(
            check_missing_recipient_rate(pass_events_missing_recipient, pass_events_total)
        )

    # 9. Duplicate possession keys
    checks.append(check_duplicate_possession_keys(possession_keys))

    # 10. Duplicate event IDs
    if event_ids_all:
        checks.append(check_duplicate_event_ids(event_ids_all))

    # 11. Shot-positive possession rate
    n_shot_pos = sum(1 for p in all_possessions_clean if p.has_shot)
    checks.append(check_positive_shot_rate(n_shot_pos, len(all_possessions_clean)))

    # 12. Actual shots per match (Shot events, not shot-positive possessions)
    checks.append(check_actual_shots_per_match(actual_shot_count, n_matches))

    # 13. Attack direction diagnostic
    diagnostic = attack_direction_diagnostic(all_possessions_clean)
    checks.append(check_attack_direction(diagnostic))

    # 14. Possession team consistency (from canonical events)
    events_df = pd.read_parquet(
        events_parquet_path,
        columns=["match_id", "possession_id", "possession_team_id"],
    )
    checks.append(check_possession_team_consistency(events_df))

    # 15. Graph size proxy
    if all_possessions_clean:
        checks.append(
            check_graph_size_proxy(
                [p.n_players for p in all_possessions_clean],
                [p.n_passes for p in all_possessions_clean],
            )
        )

    checks_df = run_all_checks(checks)
    checks_df.to_csv(out_dir / "sanity_checks.csv", index=False, encoding="utf-8-sig")

    print("\n=== KIỂM TRA CHẤT LƯỢNG ===")
    print(checks_df.to_string(index=False))

    # ================================================================
    # Phase 8: Data Pipeline Summary
    # ================================================================
    n_successful_passes = sum(p.n_passes for p in all_possessions_clean)
    n_unique_players = len({pid for p in all_possessions_clean for pid in p.players})
    n_goal_pos = sum(1 for p in all_possessions_clean if p.has_goal)
    pass_counts = [p.n_passes for p in all_possessions_clean]
    player_counts = [p.n_players for p in all_possessions_clean]

    succ_pass_miss_coord = 0.0
    succ_pass_miss_recip = 0.0
    if not possession_events_df.empty:
        n_sp = len(possession_events_df)
        n_miss_coord_sp = int(
            possession_events_df[["start_x", "start_y", "end_x", "end_y"]].isna().any(axis=1).sum()
        )
        succ_pass_miss_coord = round(n_miss_coord_sp / n_sp, 4) if n_sp else 0.0
        n_miss_recip_sp = int(possession_events_df["recipient_id"].isna().sum())
        succ_pass_miss_recip = round(n_miss_recip_sp / n_sp, 4) if n_sp else 0.0

    summary = {
        "run_name": run_name,
        "n_matches": n_matches,
        "n_events": total_canonical_events,
        "n_raw_possessions": len(all_possessions_raw),
        "n_clean_possessions": len(all_possessions_clean),
        "n_successful_passes": n_successful_passes,
        "n_unique_players": n_unique_players,
        "n_unique_teams": len(all_teams),
        "shot_positive_count": n_shot_pos,
        "shot_positive_rate": (
            round(n_shot_pos / len(all_possessions_clean), 4) if all_possessions_clean else 0.0
        ),
        "goal_positive_count": n_goal_pos,
        "goal_positive_rate": (
            round(n_goal_pos / len(all_possessions_clean), 4) if all_possessions_clean else 0.0
        ),
        "actual_shot_count": actual_shot_count,
        "actual_shots_per_match": (round(actual_shot_count / n_matches, 2) if n_matches else 0.0),
        "median_passes_per_possession": float(np.median(pass_counts)) if pass_counts else 0,
        "median_players_per_possession": float(np.median(player_counts)) if player_counts else 0,
        "overall_event_missing_coordinate_rate": (
            round(missing_coords_count / total_canonical_events, 4)
            if total_canonical_events
            else 0.0
        ),
        "successful_pass_missing_coordinate_rate": succ_pass_miss_coord,
        "successful_pass_missing_recipient_rate": succ_pass_miss_recip,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(out_dir / "data_pipeline_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== DATA PIPELINE SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\nĐã ghi kết quả vào: {out_dir}")

    if has_critical_failure(checks_df):
        logger.error("Có FAIL nghiêm trọng trong sanity checks!")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
