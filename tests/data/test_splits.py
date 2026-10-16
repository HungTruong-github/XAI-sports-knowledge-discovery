"""Unit tests cho Bước 4: Chia tập dữ liệu theo trận (match-level split)."""

import pandas as pd
import pytest

from xai_football.data.splits import (
    compute_match_stats,
    load_match_splits,
    make_stratified_match_split,
    save_match_splits,
    validate_splits,
)


@pytest.fixture
def sample_possessions_df():
    """Tạo DataFrame possession giả lập 10 trận đấu với tỉ lệ nhãn dương đa dạng."""
    rows = []
    for mid in range(1001, 1011):
        n_poss = 50 + (mid % 3) * 10
        # Thay đổi tỉ lệ sút từ 10% đến 25%
        pos_rate = 0.10 + (mid - 1001) * 0.015
        n_pos = int(round(n_poss * pos_rate))
        for i in range(n_poss):
            rows.append(
                {
                    "match_id": mid,
                    "possession_id": i,
                    "has_shot": i < n_pos,
                }
            )
    return pd.DataFrame(rows)


def test_compute_match_stats(sample_possessions_df):
    stats = compute_match_stats(sample_possessions_df)
    assert len(stats) == 10
    assert list(stats.columns) == ["match_id", "n_possessions", "n_positive", "positive_label_rate"]
    assert (stats["positive_label_rate"] >= 0).all() and (stats["positive_label_rate"] <= 1).all()


def test_make_stratified_match_split_disjoint(sample_possessions_df):
    stats = compute_match_stats(sample_possessions_df)
    splits = make_stratified_match_split(
        stats,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    tr = set(splits["train"])
    va = set(splits["val"])
    te = set(splits["test"])

    # Không rò rỉ dữ liệu giữa các tập
    assert len(tr & va) == 0, "Rò rỉ train - val"
    assert len(tr & te) == 0, "Rò rỉ train - test"
    assert len(va & te) == 0, "Rò rỉ val - test"

    # Tổng số trận phải đúng bằng 10
    assert len(tr) + len(va) + len(te) == 10
    assert tr | va | te == set(stats["match_id"])


def test_validate_splits_pass(sample_possessions_df):
    stats = compute_match_stats(sample_possessions_df)
    splits = make_stratified_match_split(
        stats, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    checks = validate_splits(splits, stats["match_id"].tolist(), stats)
    for c in checks:
        assert c.status in ("PASS", "WARNING")


def test_validate_splits_detects_leakage():
    # Tạo splits bị rò rỉ có chủ đích
    leaky_splits = {
        "train": [1, 2, 3],
        "val": [3, 4],  # match_id 3 bị trùng giữa train và val
        "test": [5],
    }
    checks = validate_splits(leaky_splits, [1, 2, 3, 4, 5])
    failed = [c for c in checks if c.status == "FAIL"]
    assert len(failed) > 0
    assert failed[0].check == "no_leakage_train_val"


def test_save_and_load_splits(tmp_path, sample_possessions_df):
    stats = compute_match_stats(sample_possessions_df)
    splits = make_stratified_match_split(stats, seed=42)

    save_match_splits(splits, stats, tmp_path)

    assert (tmp_path / "train_matches.csv").exists()
    assert (tmp_path / "val_matches.csv").exists()
    assert (tmp_path / "test_matches.csv").exists()
    assert (tmp_path / "match_splits.csv").exists()
    assert (tmp_path / "split_summary.csv").exists()

    loaded = load_match_splits(tmp_path)
    assert loaded["train"] == splits["train"]
    assert loaded["val"] == splits["val"]
    assert loaded["test"] == splits["test"]


def test_invalid_ratios(sample_possessions_df):
    stats = compute_match_stats(sample_possessions_df)
    with pytest.raises(ValueError, match="Tổng các tỉ lệ phải bằng 1.0"):
        make_stratified_match_split(stats, train_ratio=0.8, val_ratio=0.2, test_ratio=0.2)
