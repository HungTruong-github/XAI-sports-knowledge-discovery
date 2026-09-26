"""Bước 2 — Thống kê mô tả trên dữ liệu interim đã tiền xử lý.

Trách nhiệm của script này:
  - Đọc data/interim/<run_name>/possessions.parquet
  - Sinh descriptive_stats.csv
  - Sinh graph_size_distribution.csv
  - KHÔNG ghi đè label_comparison.csv, cleaning_report.csv, hay
    sanity_checks.csv (do preprocess đảm nhận).

Cách dùng:
    python scripts/02_descriptive_stats.py --config configs/data.yaml
    python scripts/02_descriptive_stats.py --config configs/data_pilot.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import get_interim_dir, get_tables_dir  # noqa: E402

logger = get_logger("stats")


def _percentiles(values: list[int] | pd.Series, name: str) -> dict[str, float]:
    if len(values) == 0:
        return {f"{name}_{k}": float("nan") for k in ("min", "p25", "median", "p75", "max", "mean")}
    arr = np.array(values)
    return {
        f"{name}_min": int(arr.min()),
        f"{name}_p25": float(np.percentile(arr, 25)),
        f"{name}_median": float(np.median(arr)),
        f"{name}_p75": float(np.percentile(arr, 75)),
        f"{name}_max": int(arr.max()),
        f"{name}_mean": round(float(arr.mean()), 2),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    interim_dir = get_interim_dir(config)
    tables_dir = get_tables_dir(config)

    possessions_parquet = interim_dir / "possessions.parquet"
    if not possessions_parquet.exists():
        logger.error(
            "Không tìm thấy %s. Hãy chạy scripts/03_preprocess.py trước.",
            possessions_parquet,
        )
        return 1

    logger.info("Đọc interim dataset từ: %s", possessions_parquet)
    possessions_df = pd.read_parquet(possessions_parquet)

    if possessions_df.empty:
        logger.error("Interim dataset rỗng.")
        return 1

    # 1. Thống kê tổng quan (descriptive_stats.csv)
    n_matches = possessions_df["match_id"].nunique()
    n_possessions = len(possessions_df)
    n_passes = possessions_df["n_passes"]
    n_nodes = possessions_df["n_players"]
    n_edges = possessions_df["n_unique_edges"]

    overview_row = {
        "run_name": config.get("run_name", "official"),
        "n_matches": n_matches,
        "n_possessions": n_possessions,
        "possessions_per_match": round(n_possessions / n_matches, 1) if n_matches else 0.0,
        "n_passes_total": int(n_passes.sum()),
    }
    overview_row |= _percentiles(n_passes, "passes_per_possession")
    overview_row |= _percentiles(n_nodes, "nodes")
    overview_row |= _percentiles(n_edges, "edges")

    for threshold in (4, 5):
        n_long = int((n_passes >= threshold).sum())
        overview_row[f"pct_possessions_ge_{threshold}_passes"] = (
            round(100 * n_long / n_possessions, 1) if n_possessions else 0.0
        )

    overview_df = pd.DataFrame([overview_row])
    overview_path = tables_dir / "descriptive_stats.csv"
    overview_df.to_csv(overview_path, index=False, encoding="utf-8-sig")

    # 2. Phân bố kích thước đồ thị (graph_size_distribution.csv)
    distribution = (
        possessions_df.groupby(["n_players", "n_unique_edges"])
        .size()
        .reset_index(name="count")
        .rename(columns={"n_players": "n_nodes", "n_unique_edges": "n_edges"})
        .sort_values("count", ascending=False)
    )
    dist_path = tables_dir / "graph_size_distribution.csv"
    distribution.to_csv(dist_path, index=False, encoding="utf-8-sig")

    print("\n=== THỐNG KÊ MÔ TẢ DATASET ===")
    print(overview_df.to_string(index=False))

    print("\n=== PHÂN BỐ KÍCH THƯỚC ĐỒ THỊ (TOP 10) ===")
    print(distribution.head(10).to_string(index=False))

    print(f"\nĐã ghi: {overview_path}")
    print(f"Đã ghi: {dist_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
