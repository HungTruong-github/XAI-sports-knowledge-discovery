"""Bước 4 — Chia tập train/val/test theo TRẬN ĐẤU (chống rò rỉ dữ liệu).

Cách dùng:
    python scripts/05_make_splits.py --config configs/data.yaml
    python scripts/05_make_splits.py --config configs/data_pilot.yaml

Yêu cầu: đã chạy scripts/03_preprocess.py (hoặc 04_build_graphs.py) để có possessions.parquet.

Đầu ra:
    data/splits/train_matches.csv
    data/splits/val_matches.csv
    data/splits/test_matches.csv
    data/splits/match_splits.csv
    data/splits/split_summary.csv
    results/tables/<run_name>/split_summary.csv
    results/tables/<run_name>/split_checks.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from xai_football.data.splits import (  # noqa: E402
    compute_graph_split_summary,
    compute_match_stats,
    label_column_for,
    make_stratified_match_split,
    save_match_splits,
    summarize_splits,
    validate_splits,
)
from xai_football.data.validation import has_critical_failure, run_all_checks  # noqa: E402
from xai_football.utils.io import load_yaml  # noqa: E402
from xai_football.utils.logging import get_logger  # noqa: E402
from xai_football.utils.paths import (  # noqa: E402
    get_interim_dir,
    get_processed_dir,
    get_split_dir,
    get_tables_dir,
)

logger = get_logger("make_splits")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--split-config", default="configs/split.yaml")
    args = parser.parse_args()

    data_cfg = load_yaml(args.config)
    split_cfg = load_yaml(args.split_config)

    interim_dir = get_interim_dir(data_cfg)
    tables_dir = get_tables_dir(data_cfg)
    run_name = data_cfg.get("run_name", "official")

    possessions_path = interim_dir / "possessions.parquet"
    if not possessions_path.exists():
        logger.error(
            "Không tìm thấy %s. Hãy chạy scripts/03_preprocess.py trước.",
            possessions_path,
        )
        return 1

    possessions_df = pd.read_parquet(possessions_path)
    logger.info("Đọc %d possession từ %s", len(possessions_df), possessions_path)

    # 1. Tính thống kê từng trận (phân tầng theo đúng nhãn trong configs/data.yaml)
    label_col = label_column_for(data_cfg)
    match_stats_df = compute_match_stats(possessions_df, label_col=label_col)
    logger.info("Phân tầng theo cột nhãn: %s", label_col)
    n_matches = len(match_stats_df)
    logger.info("Tính thống kê cho %d trận đấu", n_matches)

    # 2. Đọc tỉ lệ và tham số chia từ config
    ratios = split_cfg.get("ratios", {"train": 0.70, "val": 0.15, "test": 0.15})
    train_ratio = float(ratios.get("train", 0.70))
    val_ratio = float(ratios.get("val", 0.15))
    test_ratio = float(ratios.get("test", 0.15))
    seed = int(split_cfg.get("seed", 42))

    # 3. Thực hiện phân chia phân tầng theo trận
    splits = make_stratified_match_split(
        match_stats_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        n_bins=5,
        seed=seed,
    )

    # 4. Kiểm tra toàn vẹn phân chia (chống rò rỉ dữ liệu)
    checks = validate_splits(splits, match_stats_df["match_id"].tolist(), match_stats_df)
    checks_df = run_all_checks(checks)

    # 5. Lưu kết quả ra thư mục splits (đường dẫn tính từ gốc repo)
    out_split_dir = get_split_dir(split_cfg, run_name)
    save_match_splits(splits, match_stats_df, out_split_dir)

    # Lưu bảng tổng kết vào results/tables/<run_name>/
    summary_df = summarize_splits(splits, match_stats_df)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(tables_dir / "split_summary.csv", index=False, encoding="utf-8-sig")
    checks_df.to_csv(tables_dir / "split_checks.csv", index=False, encoding="utf-8-sig")

    print("\n=== SPLIT SUMMARY (POSSESSION-LEVEL) ===")
    print(summary_df.to_string(index=False))

    # 6. Nếu đã có graph_index.parquet, tính và lưu graph-level split summary
    processed_dir = get_processed_dir(data_cfg)
    graph_index_path = processed_dir / "graph_index.parquet"
    if graph_index_path.exists():
        graph_index_df = pd.read_parquet(graph_index_path)
        graph_summary_df = compute_graph_split_summary(splits, graph_index_df, possessions_df)
        graph_summary_df.to_csv(
            tables_dir / "graph_split_summary.csv", index=False, encoding="utf-8-sig"
        )
        graph_summary_df.to_csv(
            out_split_dir / "graph_split_summary.csv", index=False, encoding="utf-8-sig"
        )
        print("\n=== GRAPH-LEVEL SPLIT SUMMARY (POST-GRAPH CONSTRUCTION) ===")
        print(graph_summary_df.to_string(index=False))

    print("\n=== KIỂM TRA TOÀN VẸN PHÂN CHIA (CHỐNG RÒ RỈ) ===")
    print(checks_df.to_string(index=False))

    print(f"\nDanh sách match_id đã lưu: {out_split_dir}")
    print(f"Báo cáo đã xuất:          {tables_dir}")

    return 1 if has_critical_failure(checks_df) else 0


if __name__ == "__main__":
    raise SystemExit(main())
