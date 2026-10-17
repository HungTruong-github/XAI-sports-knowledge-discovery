"""Unit tests kiểm tra các bất biến (invariants) của phân chia tập dữ liệu (Splits).

Bảo đảm 100% không rò rỉ thông tin giữa train, validation và test theo match_id.
"""

from pathlib import Path

import pandas as pd
import pytest

from xai_football.data.splits import load_match_splits


@pytest.fixture
def official_split_dir():
    return Path("data/splits")


def test_split_matches_disjoint(official_split_dir):
    """Bắt buộc 3 tập train, val, test không có bất kỳ match_id nào trùng lặp."""
    splits = load_match_splits(official_split_dir)
    train_m = set(splits["train"])
    val_m = set(splits["val"])
    test_m = set(splits["test"])

    # Tập giao phải rỗng tuyệt đối
    assert len(train_m.intersection(val_m)) == 0, "Rò rỉ giữa train và val!"
    assert len(train_m.intersection(test_m)) == 0, "Rò rỉ giữa train và test!"
    assert len(val_m.intersection(test_m)) == 0, "Rò rỉ giữa val và test!"

    # Tổng số trận phải đúng 380 trận
    total_unique = len(train_m | val_m | test_m)
    assert total_unique == 380, f"Tổng số trận là {total_unique}, kỳ vọng 380!"
    assert len(train_m) == 266, f"Train matches = {len(train_m)}, kỳ vọng 266 (70%)"
    assert len(val_m) == 57, f"Val matches = {len(val_m)}, kỳ vọng 57 (15%)"
    assert len(test_m) == 57, f"Test matches = {len(test_m)}, kỳ vọng 57 (15%)"


def test_graph_index_matches_belong_to_splits(official_split_dir):
    """Mọi match_id trong graph_index.parquet phải thuộc đúng một split đã khai báo."""
    index_path = Path("data/processed/official/graph_index.parquet")
    if not index_path.exists():
        pytest.skip("Chưa có graph_index.parquet trên đĩa")

    df_index = pd.read_parquet(index_path)
    splits = load_match_splits(official_split_dir)
    all_allowed_matches = set(splits["train"]) | set(splits["val"]) | set(splits["test"])

    graph_matches = set(df_index["match_id"].unique())
    leak_matches = graph_matches - all_allowed_matches
    assert (
        len(leak_matches) == 0
    ), f"Phát hiện graph thuộc match_id không nằm trong split: {leak_matches}"
