"""Bước 2 — Tiền xử lý toàn bộ dữ liệu đã tải.

Cách dùng:
    python scripts/03_preprocess.py --config configs/data.yaml
    python scripts/03_preprocess.py --config configs/data_pilot.yaml --limit-matches 5

Pipeline:
  1. Canonicalize raw JSON → data/interim/events.parquet
  2. Extract possessions → data/interim/possessions.parquet
  3. Extract possession-event relationships → data/interim/possession_events.parquet
  4. Clean possessions
  5. Label comparison statistics
  6. Play pattern distribution
  7. Sanity checks → results/tables/sanity_checks.csv
  8. Data pipeline summary → results/tables/data_pipeline_summary.csv

Không sửa file raw.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from xai_football.data.canonicalize import (  # noqa: E402
    canonicalize_season,
    save_canonical_events,
)
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
    check_attack_direction,
    check_duplicate_event_ids,
    check_events_per_match,
    check_graph_size_proxy,
    check_match_count,
    check_missing_coordinate_rate,
    check_missing_recipient_rate,
    check_positive_shot_rate,
    check_possessions_per_match,
    check_shots_per_match,
    check_successful_passes_per_match,
    check_unique_teams,
    has_critical_failure,
    run_all_checks,
)
from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import (  # noqa: E402
    INTERIM_DIR,
    RAW_DIR,
    TABLES_DIR,
    ensure_dir,
)

logger = get_logger("preprocess")


def _get_match_files(
    competition_id: int, season_id: int, limit: int | None = None,
) -> list[Path]:
    season_dir = RAW_DIR / str(competition_id) / str(season_id)
    if not season_dir.exists():
        return []
    files = sorted(
        p for p in season_dir.glob("*.json")
        if not p.name.endswith(".lineups.json")
    )
    if limit is not None:
        files = files[:limit]
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="data.yaml")
    parser.add_argument("--limit-matches", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    entries = resolve_competitions(config)
    expected_matches = config.get("expected_matches")

    out_dir = ensure_dir(TABLES_DIR)
    interim_dir = ensure_dir(INTERIM_DIR)

    # ================================================================
    # Phase 1: Canonicalize events
    # ================================================================
    all_events_dfs = []
    for entry in entries:
        comp_id, season_id = entry["competition_id"], entry["season_id"]
        match_files = _get_match_files(comp_id, season_id, args.limit_matches)
        match_ids = [int(p.stem) for p in match_files]
        if not match_ids:
            logger.warning("Chưa có dữ liệu cho %s — chạy scripts/01_download_data.py trước",
                           entry["name"])
            continue
        logger.info("Canonicalize %s: %d trận...", entry["name"], len(match_ids))
        df = canonicalize_season(comp_id, season_id, match_ids)
        all_events_dfs.append(df)

    if not all_events_dfs:
        logger.error("Không có dữ liệu nào. Hãy chạy scripts/01_download_data.py trước.")
        return 1

    events_df = pd.concat(all_events_dfs, ignore_index=True)
    save_canonical_events(events_df)

    # ================================================================
    # Phase 2: Extract possessions, clean, persist
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
            raw = load_match_possessions(comp_id, season_id, mid)
            all_possessions_raw.extend(raw)

            kept, report = clean_possessions(raw, config)
            all_possessions_clean.extend(kept)
            total_report = total_report.merge(report)
            match_ids_processed.append(mid)
            possessions_per_match.append(len(kept))

    n_matches = len(match_ids_processed)
    logger.info("Tổng: %d trận → %d raw possessions → %d clean possessions",
                n_matches, len(all_possessions_raw), len(all_possessions_clean))

    # ================================================================
    # Phase 3: Persist possessions
    # ================================================================
    possession_rows = []
    possession_event_rows = []
    for p in all_possessions_clean:
        possession_rows.append({
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
        })
        for pass_ev in p.passes:
            possession_event_rows.append({
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
            })

    possessions_df = pd.DataFrame(possession_rows)
    possession_events_df = pd.DataFrame(possession_event_rows)

    possessions_df.to_parquet(interim_dir / "possessions.parquet", index=False, engine="pyarrow")
    possession_events_df.to_parquet(
        interim_dir / "possession_events.parquet", index=False, engine="pyarrow"
    )
    logger.info("Đã ghi possessions.parquet (%d rows) và possession_events.parquet (%d rows)",
                len(possessions_df), len(possession_events_df))

    # ================================================================
    # Phase 4: Label comparison
    # ================================================================
    label_stats = compare_labeling_schemes(all_possessions_clean)
    labels_df = pd.DataFrame([s.as_row() for s in label_stats])
    labels_df.to_csv(out_dir / "label_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n=== SO SÁNH CÁC CÁCH GÁN NHÃN ===")
    print(labels_df.to_string(index=False))

    # ================================================================
    # Phase 5: Play pattern distribution
    # ================================================================
    play_pattern_counts: dict[str, int] = {}
    for p in all_possessions_raw:  # Use raw (before cleaning) for full distribution
        pp = p.play_pattern or "Unknown"
        play_pattern_counts[pp] = play_pattern_counts.get(pp, 0) + 1

    pp_df = pd.DataFrame([
        {"play_pattern": k, "count": v,
         "pct": round(100 * v / len(all_possessions_raw), 2) if all_possessions_raw else 0}
        for k, v in sorted(play_pattern_counts.items(), key=lambda x: -x[1])
    ])
    pp_df.to_csv(out_dir / "play_pattern_distribution.csv", index=False, encoding="utf-8-sig")

    print("\n=== PHÂN BỐ PLAY PATTERN ===")
    print(pp_df.to_string(index=False))

    # ================================================================
    # Phase 6: Cleaning report
    # ================================================================
    cleaning_df = pd.DataFrame([total_report.as_row()])
    cleaning_df.to_csv(out_dir / "cleaning_report.csv", index=False, encoding="utf-8-sig")

    print("\n=== LÀM SẠCH ===")
    print(cleaning_df.to_string(index=False))

    # ================================================================
    # Phase 7: Sanity checks
    # ================================================================
    checks: list[CheckResult] = []

    # Match count
    checks.append(check_match_count(n_matches, expected_matches))

    # Unique teams
    all_teams = set()
    for p in all_possessions_clean:
        if p.team_name:
            all_teams.add(p.team_name)
    checks.append(check_unique_teams(len(all_teams)))

    # Events per match
    if not events_df.empty:
        epm = events_df.groupby("match_id").size().tolist()
        checks.append(check_events_per_match(epm))

    # Possessions per match
    if possessions_per_match:
        checks.append(check_possessions_per_match(possessions_per_match))

    # Successful passes per match
    passes_per_match_list = []
    for mid in set(p.match_id for p in all_possessions_clean):
        n = sum(pp.n_passes for pp in all_possessions_clean if pp.match_id == mid)
        passes_per_match_list.append(n)
    if passes_per_match_list:
        checks.append(check_successful_passes_per_match(passes_per_match_list))

    # Missing coordinate rate
    if not events_df.empty:
        n_with_loc = events_df[["x", "y"]].notna().all(axis=1).sum()
        n_total_events = len(events_df)
        checks.append(check_missing_coordinate_rate(
            n_total_events - int(n_with_loc), n_total_events,
        ))

    # Missing recipient rate
    if not events_df.empty:
        pass_events = events_df[events_df["event_type"] == "Pass"]
        n_pass = len(pass_events)
        n_miss_recip = int(pass_events["pass_recipient_id"].isna().sum())
        checks.append(check_missing_recipient_rate(n_miss_recip, n_pass))

    # Shot rate
    n_shot_pos = sum(1 for p in all_possessions_clean if p.has_shot)
    checks.append(check_positive_shot_rate(n_shot_pos, len(all_possessions_clean)))
    checks.append(check_shots_per_match(n_shot_pos, n_matches))

    # Duplicate event IDs
    if not events_df.empty and "event_id" in events_df.columns:
        checks.append(check_duplicate_event_ids(events_df["event_id"].tolist()))

    # Attack direction diagnostic
    diagnostic = attack_direction_diagnostic(all_possessions_clean)
    checks.append(check_attack_direction(diagnostic))

    # Possession team consistency
    from xai_football.data.validation import check_possession_team_consistency
    checks.append(check_possession_team_consistency(events_df))

    # Graph size proxy
    if all_possessions_clean:
        checks.append(check_graph_size_proxy(
            [p.n_players for p in all_possessions_clean],
            [p.n_passes for p in all_possessions_clean],
        ))

    checks_df = run_all_checks(checks)
    checks_df.to_csv(out_dir / "sanity_checks.csv", index=False, encoding="utf-8-sig")

    print("\n=== KIỂM TRA CHẤT LƯỢNG ===")
    print(checks_df.to_string(index=False))

    print(f"\n=== CHẨN ĐOÁN HƯỚNG TẤN CÔNG (A5) ===")
    print(diagnostic)

    # ================================================================
    # Phase 8: Pipeline summary
    # ================================================================
    n_successful_passes = sum(p.n_passes for p in all_possessions_clean)
    n_unique_players = len(set(
        pid for p in all_possessions_clean for pid in p.players
    ))
    n_goal_pos = sum(1 for p in all_possessions_clean if p.has_goal)

    pass_counts = [p.n_passes for p in all_possessions_clean]
    player_counts = [p.n_players for p in all_possessions_clean]

    # Missing rates from events
    missing_coord_rate = 0.0
    missing_recip_rate = 0.0
    if not events_df.empty:
        n_with_loc = events_df[["x", "y"]].notna().all(axis=1).sum()
        missing_coord_rate = round(1 - int(n_with_loc) / len(events_df), 4)
        pass_events = events_df[events_df["event_type"] == "Pass"]
        if len(pass_events) > 0:
            missing_recip_rate = round(
                int(pass_events["pass_recipient_id"].isna().sum()) / len(pass_events), 4
            )

    summary = {
        "n_matches": n_matches,
        "n_events": len(events_df),
        "n_raw_possessions": len(all_possessions_raw),
        "n_clean_possessions": len(all_possessions_clean),
        "n_successful_passes": n_successful_passes,
        "n_unique_players": n_unique_players,
        "n_unique_teams": len(all_teams),
        "shot_positive_count": n_shot_pos,
        "shot_positive_rate": round(
            n_shot_pos / len(all_possessions_clean), 4
        ) if all_possessions_clean else 0.0,
        "goal_positive_count": n_goal_pos,
        "goal_positive_rate": round(
            n_goal_pos / len(all_possessions_clean), 4
        ) if all_possessions_clean else 0.0,
        "median_passes_per_possession": float(np.median(pass_counts)) if pass_counts else 0,
        "median_players_per_possession": float(np.median(player_counts)) if player_counts else 0,
        "missing_coordinate_rate": missing_coord_rate,
        "missing_recipient_rate": missing_recip_rate,
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
