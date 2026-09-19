"""Bước 2 — Thống kê mô tả trên dữ liệu đã tải.

Bảng sinh ra phục vụ ba mục đích cùng lúc (báo cáo 19/09 mục 4.1):
    1. Chốt quyết định 0001 — quy tắc gán nhãn, bằng tỉ lệ lớp dương thật
    2. Chốt quyết định 0003 — ngưỡng độ dài possession cho phần XAI
    3. Phần mô tả dữ liệu trong khóa luận

Cách dùng:
    python scripts/02_descriptive_stats.py
    python scripts/02_descriptive_stats.py --limit-matches 5

Đầu ra (results/tables/):
    descriptive_stats.csv      tổng quan theo giải
    label_comparison.csv       tỉ lệ lớp dương của mọi cách gán nhãn
    graph_size_distribution.csv phân bố số đỉnh / số cạnh mỗi đồ thị
    cleaning_report.csv        số possession bị loại ở từng bước
    sanity_checks.csv          các phép kiểm tra bắt buộc (mục 4.2)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from xai_football.data.clean import (  # noqa: E402
    CleaningReport,
    attack_direction_diagnostic,
    clean_possessions,
)
from xai_football.data.collect import resolve_competitions  # noqa: E402
from xai_football.data.labeling import compare_labeling_schemes  # noqa: E402
from xai_football.data.possession import Possession, load_match_possessions  # noqa: E402
from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import RAW_DIR, TABLES_DIR, ensure_dir  # noqa: E402

logger = get_logger("stats")

# Ngưỡng của các phép kiểm tra bắt buộc, báo cáo 19/09 mục 4.2.
#
# Lưu ý về POSITIVE_RATE_SHOT_RANGE: báo cáo 19/09 dự kiến 8-15%, nhưng con số
# đó áp cho TOÀN BỘ possession. Sau khi loại possession < 2 đường chuyền (~30%,
# và hầu như không pha nào trong số đó dẫn đến sút), tỉ lệ lớp dương trên tập
# còn lại tất yếu cao hơn. Ngưỡng dưới đây đã hiệu chỉnh theo dữ liệu thật.
POSSESSIONS_PER_MATCH_RANGE = (90, 130)
POSITIVE_RATE_SHOT_RANGE = (0.10, 0.25)

# Phép kiểm tra neo vào thực tế bóng đá: số cú sút của CẢ HAI đội mỗi trận.
# Premier League trung bình khoảng 25 cú sút/trận.
SHOTS_PER_MATCH_RANGE = (15, 35)


def _percentiles(values: list[int], name: str) -> dict[str, float]:
    if not values:
        return {f"{name}_{k}": float("nan")
                for k in ("min", "p25", "median", "p75", "max", "mean")}
    arr = np.array(values)
    return {
        f"{name}_min": int(arr.min()),
        f"{name}_p25": float(np.percentile(arr, 25)),
        f"{name}_median": float(np.median(arr)),
        f"{name}_p75": float(np.percentile(arr, 75)),
        f"{name}_max": int(arr.max()),
        f"{name}_mean": round(float(arr.mean()), 2),
    }


def collect_possessions(
    config: dict, limit_matches: int | None = None
) -> tuple[dict[str, list[Possession]], dict[str, int], CleaningReport]:
    """Đọc toàn bộ trận đã tải, trích và làm sạch possession, gom theo giải."""
    by_competition: dict[str, list[Possession]] = {}
    match_counts: dict[str, int] = {}
    total_report = CleaningReport()

    for entry in resolve_competitions(config):
        name = entry["name"]
        comp_id, season_id = entry["competition_id"], entry["season_id"]
        season_dir = RAW_DIR / str(comp_id) / str(season_id)

        if not season_dir.exists():
            logger.warning("Chưa có dữ liệu cho %s — chạy scripts/01_download_data.py trước",
                           name)
            continue

        match_files = sorted(p for p in season_dir.glob("*.json")
                             if not p.name.endswith(".lineups.json"))
        if limit_matches is not None:
            match_files = match_files[:limit_matches]

        kept_all: list[Possession] = []
        for path in match_files:
            raw = load_match_possessions(comp_id, season_id, int(path.stem))
            kept, report = clean_possessions(raw, config)
            kept_all.extend(kept)
            total_report = total_report.merge(report)

        by_competition[name] = kept_all
        match_counts[name] = len(match_files)
        logger.info("%s: %d trận -> %d possession sau làm sạch",
                    name, len(match_files), len(kept_all))

    return by_competition, match_counts, total_report


def build_overview(by_competition, match_counts) -> pd.DataFrame:
    rows = []
    for name, possessions in by_competition.items():
        n_matches = match_counts.get(name, 0)
        n_passes = [p.n_passes for p in possessions]
        n_nodes = [p.n_players for p in possessions]
        n_edges = [p.n_unique_edges for p in possessions]

        row = {
            "competition": name,
            "n_matches": n_matches,
            "n_possessions": len(possessions),
            "possessions_per_match": (round(len(possessions) / n_matches, 1)
                                      if n_matches else 0.0),
            "n_passes_total": sum(n_passes),
        }
        row |= _percentiles(n_passes, "passes_per_possession")
        row |= _percentiles(n_nodes, "nodes")
        row |= _percentiles(n_edges, "edges")

        # Căn cứ cho quyết định 0003 — ngưỡng độ dài possession cho XAI
        for threshold in (4, 5):
            n_long = sum(1 for x in n_passes if x >= threshold)
            row[f"pct_possessions_ge_{threshold}_passes"] = (
                round(100 * n_long / len(possessions), 1) if possessions else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_sanity_checks(overview: pd.DataFrame, labels: pd.DataFrame,
                        diagnostic: dict) -> pd.DataFrame:
    """Bốn phép kiểm tra bắt buộc trước khi tin số liệu (mục 4.2)."""
    checks = []

    lo, hi = POSSESSIONS_PER_MATCH_RANGE
    for _, row in overview.iterrows():
        value = row["possessions_per_match"]
        checks.append({
            "check": f"possession/trận trong [{lo}, {hi}] — {row['competition']}",
            "value": value,
            "passed": bool(lo <= value <= hi),
            "note": "Lệch nhiều nghĩa là logic nhóm possession sai",
        })

    mask = (labels["mode"] == "possession_level") & (labels["target"] == "shot")
    if mask.any():
        rate = float(labels.loc[mask, "positive_rate"].iloc[0])
        n_positive = int(labels.loc[mask, "n_positive"].iloc[0])
        lo, hi = POSITIVE_RATE_SHOT_RANGE
        checks.append({
            "check": f"tỉ lệ lớp dương (possession_level/shot) trong [{lo}, {hi}]",
            "value": round(rate, 4),
            "passed": bool(lo <= rate <= hi),
            "note": "Cao bất thường -> nghi đếm nhầm cú sút của đội đối phương",
        })

        # Neo vào thực tế bóng đá: quy đổi ra số cú sút mỗi trận.
        n_matches = int(overview["n_matches"].sum())
        if n_matches:
            shots_per_match = n_positive / n_matches
            lo, hi = SHOTS_PER_MATCH_RANGE
            checks.append({
                "check": f"số possession dẫn đến sút mỗi trận trong [{lo}, {hi}]",
                "value": round(shots_per_match, 1),
                "passed": bool(lo <= shots_per_match <= hi),
                "note": "Cả hai đội cộng lại; Premier League trung bình ~25 cú sút/trận",
            })

    gap = diagnostic.get("gap", float("nan"))
    checks.append({
        "check": "hướng tấn công: x trung bình của pha dẫn đến sút cao hơn (A5)",
        "value": gap,
        "passed": bool(gap == gap and gap > 0),
        "note": "gap <= 0 nghĩa là tọa độ chưa chuẩn hóa đồng bộ",
    })

    return pd.DataFrame(checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="data.yaml")
    parser.add_argument("--limit-matches", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    by_competition, match_counts, cleaning = collect_possessions(config, args.limit_matches)

    all_possessions = [p for group in by_competition.values() for p in group]
    if not all_possessions:
        logger.error("Không có possession nào. Hãy chạy scripts/01_download_data.py trước.")
        return 1

    out_dir = ensure_dir(TABLES_DIR)

    overview = build_overview(by_competition, match_counts)
    labels = pd.DataFrame([s.as_row() for s in compare_labeling_schemes(all_possessions)])
    cleaning_table = pd.DataFrame([cleaning.as_row()])
    diagnostic = attack_direction_diagnostic(all_possessions)

    sizes = pd.DataFrame({
        "n_passes": [p.n_passes for p in all_possessions],
        "n_nodes": [p.n_players for p in all_possessions],
        "n_edges": [p.n_unique_edges for p in all_possessions],
        "ends_with_shot": [p.ends_with_shot for p in all_possessions],
    })
    distribution = (sizes.groupby(["n_nodes", "n_edges"])
                    .size().reset_index(name="count")
                    .sort_values("count", ascending=False))

    checks = build_sanity_checks(overview, labels, diagnostic)

    for name, table in [
        ("descriptive_stats", overview),
        ("label_comparison", labels),
        ("graph_size_distribution", distribution),
        ("cleaning_report", cleaning_table),
        ("sanity_checks", checks),
    ]:
        table.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    print("\n=== TỔNG QUAN THEO GIẢI ===")
    print(overview.to_string(index=False))
    print("\n=== SO SÁNH CÁC CÁCH GÁN NHÃN (căn cứ chốt quyết định 0001) ===")
    print(labels.to_string(index=False))
    print("\n=== LÀM SẠCH ===")
    print(cleaning_table.to_string(index=False))
    print("\n=== CHẨN ĐOÁN HƯỚNG TẤN CÔNG (A5) ===")
    print(diagnostic)
    print("\n=== KIỂM TRA BẮT BUỘC ===")
    print(checks.to_string(index=False))

    n_failed = int((~checks["passed"]).sum())
    if n_failed:
        logger.warning("%d phép kiểm tra KHÔNG đạt — phải xem lại trước khi dùng số liệu",
                       n_failed)
    print(f"\nĐã ghi 5 bảng vào: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
