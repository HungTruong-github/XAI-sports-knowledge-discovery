"""Bước 1 — Tải dữ liệu thô từ StatsBomb Open Data.

Cách dùng:
    python scripts/01_download_data.py                        # tải theo configs/data.yaml
    python scripts/01_download_data.py --limit-matches 5       # chạy thử 5 trận
    python scripts/01_download_data.py --list-competitions     # xem kho có gì

Đầu ra:
    data/raw/{competition_id}/{season_id}/{match_id}.json
    results/tables/collection_report.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from xai_football.data.collect import download_all, list_competitions  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402

logger = get_logger("download")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="data.yaml", help="Tệp cấu hình (mặc định: configs/data.yaml)"
    )
    parser.add_argument(
        "--limit-matches",
        type=int,
        default=None,
        help="Chỉ tải N trận đầu mỗi giải — dùng để chạy thử",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Tải lại cả những trận đã có sẵn trong data/raw"
    )
    parser.add_argument(
        "--list-competitions",
        action="store_true",
        help="Chỉ in danh sách giải-mùa có trong kho rồi thoát",
    )
    args = parser.parse_args()

    if args.list_competitions:
        df = list_competitions()
        cols = ["competition_id", "season_id", "competition_name", "season_name"]
        print(df[cols].to_string(index=False))
        print(f"\nTổng: {len(df)} cặp giải-mùa")
        return 0

    reports = download_all(
        config_path=args.config,
        limit_matches=args.limit_matches,
        overwrite=args.overwrite,
    )

    from xai_football.utils.io import load_yaml
    from xai_football.utils.paths import get_tables_dir

    config = load_yaml(args.config)
    table = pd.DataFrame([r.as_row() for r in reports])
    out_dir = get_tables_dir(config)
    out_path = out_dir / "collection_report.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")

    print()
    print(table.to_string(index=False))
    print(f"\nĐã ghi: {out_path}")

    failed = {r.competition_name: r.failed_match_ids for r in reports if r.failed_match_ids}
    if failed:
        logger.warning("Các trận tải lỗi: %s", failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
