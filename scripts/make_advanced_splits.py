"""Tạo các phân chia trận đấu nâng cao (Cross-fitting 5-fold và Hybrid Set A/B).

Tham chiếu:
- docs/decisions/0002-cong-thuc-hybrid-centrality.md (Quyết định D15)
- docs/decisions/0006-giao-thuc-xai.md (Quyết định D11)
Đầu ra:
- data/splits/crossfit_5fold.json (5-fold stratified match-level, seed 42)
- data/splits/hybrid_AB_split.json (Set A / Set B stratified match-level, seed 42)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from xai_football.utils.logging import get_logger

logger = get_logger("advanced_splits")


def main() -> int:
    poss_path = Path("data/interim/official/possessions.parquet")
    if not poss_path.exists():
        logger.error("Không tìm thấy tệp %s", poss_path)
        return 1

    poss_df = pd.read_parquet(poss_path)
    # Tính tỷ lệ cú sút của từng trận đấu
    match_stats = poss_df.groupby("match_id")["has_shot"].agg(["count", "mean"]).reset_index()
    match_stats.columns = ["match_id", "n_possessions", "shot_rate"]
    match_stats = match_stats.sort_values(by="match_id").reset_index(drop=True)

    n_matches = len(match_stats)
    logger.info("Tổng số trận đấu: %d", n_matches)

    # Phân thành 5 nhóm quantile theo shot_rate để phân tầng cân bằng
    match_stats["strata"] = pd.qcut(match_stats["shot_rate"], q=5, labels=False)

    out_dir = Path("data/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tạo Cross-fitting 5-fold split
    skf_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    crossfit_dict = {
        "description": "5-fold match-level stratified cross-fitting split for out-of-fold XAI",
        "n_matches": n_matches,
        "seed": 42,
        "folds": {},
    }

    match_ids = match_stats["match_id"].values
    strata = match_stats["strata"].values

    for fold_idx, (train_idx, val_idx) in enumerate(skf_5.split(match_ids, strata)):
        fold_train = [int(m) for m in match_ids[train_idx]]
        fold_val = [int(m) for m in match_ids[val_idx]]
        crossfit_dict["folds"][f"fold_{fold_idx + 1}"] = {
            "train_matches": fold_train,
            "test_matches": fold_val,
            "n_train": len(fold_train),
            "n_test": len(fold_val),
        }

    crossfit_path = out_dir / "crossfit_5fold.json"
    with open(crossfit_path, "w", encoding="utf-8") as f:
        json.dump(crossfit_dict, f, indent=2)
    logger.info("Đã lưu crossfit split vào %s", crossfit_path)

    # 2. Tạo Hybrid Set A / Set B split (Quyết định D15)
    skf_2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    set_a_idx, set_b_idx = next(skf_2.split(match_ids, strata))

    set_a_matches = sorted([int(m) for m in match_ids[set_a_idx]])
    set_b_matches = sorted([int(m) for m in match_ids[set_b_idx]])

    assert len(set_a_matches) == 190, f"Set A phải có 190 trận, thực tế: {len(set_a_matches)}"
    assert len(set_b_matches) == 190, f"Set B phải có 190 trận, thực tế: {len(set_b_matches)}"
    assert set(set_a_matches).isdisjoint(set(set_b_matches)), "Set A và Set B không được giao nhau!"

    hybrid_dict = {
        "description": "Set A (tuning alpha, fitting xT) and Set B (evaluation) split per Decision D15",
        "n_matches": n_matches,
        "seed": 42,
        "set_a": {
            "purpose": "Học tham số alpha của Hybrid Centrality và huấn luyện mô hình xT",
            "n_matches": len(set_a_matches),
            "matches": set_a_matches,
        },
        "set_b": {
            "purpose": "Đánh giá độc lập chất lượng Hybrid Centrality và các độ đo tham chiếu",
            "n_matches": len(set_b_matches),
            "matches": set_b_matches,
        },
    }

    hybrid_path = out_dir / "hybrid_AB_split.json"
    with open(hybrid_path, "w", encoding="utf-8") as f:
        json.dump(hybrid_dict, f, indent=2)
    logger.info("Đã lưu hybrid A/B split vào %s", hybrid_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
