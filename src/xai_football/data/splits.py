"""Bước 4 — Chia tập train/val/test theo TRẬN ĐẤU (match-level split).

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 4.
Cấu hình:   configs/split.yaml.

BẮT BUỘC chia theo trận, không chia theo possession:
    Các possession trong cùng một trận có tương quan chiến thuật,
    đội hình và phong độ; chia theo possession sẽ gây RÒ RỈ DỮ LIỆU.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from xai_football.data.validation import CheckResult
from xai_football.utils.logging import get_logger

logger = get_logger("splits")


def label_column_for(data_cfg: dict) -> str:
    """Cột nhãn trong possessions.parquet ứng với labeling.target của configs/data.yaml.

    Bảo đảm phân tầng ở Bước 4 dùng đúng nhãn mà mô hình sẽ học (shot hoặc goal).
    """
    target = (data_cfg.get("labeling") or {}).get("target")
    if target not in ("shot", "goal"):
        raise ValueError(f"labeling.target không hợp lệ hoặc chưa điền: {target!r}")
    return "has_goal" if target == "goal" else "has_shot"


def compute_match_stats(possessions_df: pd.DataFrame, label_col: str | None = None) -> pd.DataFrame:
    """Tính thống kê cấp độ trận đấu: số possession, số nhãn dương và tỉ lệ nhãn dương.

    Args:
        possessions_df: DataFrame possession (yêu cầu các cột match_id và cột nhãn).
        label_col: Cột nhãn dùng để phân tầng. None = tự chọn 'has_shot' (hoặc 'label').
            Pipeline chính thức truyền giá trị từ `label_column_for(data_cfg)`.

    Returns:
        DataFrame chứa: match_id, n_possessions, n_positive, positive_label_rate.
    """
    if "match_id" not in possessions_df.columns:
        raise ValueError("DataFrame thiếu cột 'match_id'")

    if label_col is None:
        label_col = "has_shot" if "has_shot" in possessions_df.columns else "label"
    if label_col not in possessions_df.columns:
        raise ValueError(f"DataFrame thiếu cột nhãn '{label_col}'")

    grouped = possessions_df.groupby("match_id", as_index=False).agg(
        n_possessions=(
            "possession_id" if "possession_id" in possessions_df.columns else label_col,
            "count",
        ),
        n_positive=(label_col, lambda s: int(s.sum())),
    )
    grouped["positive_label_rate"] = (grouped["n_positive"] / grouped["n_possessions"]).round(4)

    return grouped.sort_values("match_id").reset_index(drop=True)


def make_stratified_match_split(
    match_stats_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    n_bins: int = 5,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Chia trận đấu thành 3 tập train/val/test phân tầng (stratified) theo tỉ lệ nhãn dương.

    Args:
        match_stats_df: DataFrame chứa match_id và positive_label_rate.
        train_ratio: Tỉ lệ tập train (mặc định 0.70).
        val_ratio: Tỉ lệ tập validation (mặc định 0.15).
        test_ratio: Tỉ lệ tập test (mặc định 0.15).
        n_bins: Số lượng bin phân tầng cho tỉ lệ nhãn dương.
        seed: Random seed cố định để đảm bảo tính tái lập.

    Returns:
        dict gồm {"train": [match_id, ...], "val": [...], "test": [...]}.
    """
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-5):
        raise ValueError(
            f"Tổng các tỉ lệ phải bằng 1.0, nhận được: {train_ratio + val_ratio + test_ratio}"
        )

    n_total = len(match_stats_df)
    if n_total < 3:
        raise ValueError(f"Số lượng trận ({n_total}) quá ít để chia 3 tập.")

    df = match_stats_df.copy()

    # Phân tầng liên tục thành các bins rời rạc (quantile bins)
    # Nếu số trận ít hoặc trùng lặp nhiều, tự động giảm n_bins
    actual_bins = min(n_bins, n_total // 3)
    if actual_bins >= 2:
        try:
            df["bin"] = pd.qcut(
                df["positive_label_rate"],
                q=actual_bins,
                labels=False,
                duplicates="drop",
            )
        except Exception:
            df["bin"] = pd.cut(df["positive_label_rate"], bins=actual_bins, labels=False)
    else:
        df["bin"] = 0

    # Kiểm tra tính khả thi của stratify ở lần chia 1
    can_stratify_1 = df["bin"].nunique() > 1 and df["bin"].value_counts().min() >= 2

    # Lần chia 1: Tách train vs (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["bin"] if can_stratify_1 else None,
        random_state=seed,
        shuffle=True,
    )

    # Kiểm tra tính khả thi của stratify ở lần chia 2
    can_stratify_2 = temp_df["bin"].nunique() > 1 and temp_df["bin"].value_counts().min() >= 2

    # Lần chia 2: Tách val vs test từ tập temp
    val_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_test_ratio,
        stratify=temp_df["bin"] if can_stratify_2 else None,
        random_state=seed,
        shuffle=True,
    )

    train_ids = sorted(train_df["match_id"].astype(int).tolist())
    val_ids = sorted(val_df["match_id"].astype(int).tolist())
    test_ids = sorted(test_df["match_id"].astype(int).tolist())

    # Kiểm tra rò rỉ dữ liệu (không có match_id nào trùng lặp)
    s_train, s_val, s_test = set(train_ids), set(val_ids), set(test_ids)
    assert not (s_train & s_val), "RÒ RỈ: Match ID trùng giữa Train và Val!"
    assert not (s_train & s_test), "RÒ RỈ: Match ID trùng giữa Train và Test!"
    assert not (s_val & s_test), "RÒ RỈ: Match ID trùng giữa Val và Test!"
    assert len(s_train) + len(s_val) + len(s_test) == n_total, "Tổng số trận sau chia không khớp!"

    logger.info(
        "Đã chia %d trận: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        n_total,
        len(train_ids),
        100 * len(train_ids) / n_total,
        len(val_ids),
        100 * len(val_ids) / n_total,
        len(test_ids),
        100 * len(test_ids) / n_total,
    )

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def summarize_splits(
    splits: dict[str, list[int]],
    match_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """Tạo bảng thống kê tổng hợp phân bố dữ liệu trên từng tập train/val/test."""
    df_lookup = match_stats_df.set_index("match_id")
    total_matches = len(match_stats_df)
    total_poss = match_stats_df["n_possessions"].sum()

    rows = []
    for split_name in ("train", "val", "test"):
        mids = splits.get(split_name, [])
        subset = df_lookup.loc[mids] if mids else pd.DataFrame()
        n_matches = len(mids)
        n_poss = int(subset["n_possessions"].sum()) if not subset.empty else 0
        n_pos = int(subset["n_positive"].sum()) if not subset.empty else 0
        pos_rate = round(n_pos / n_poss, 4) if n_poss > 0 else 0.0

        rows.append(
            {
                "split": split_name,
                "n_matches": n_matches,
                "pct_matches": round(100 * n_matches / total_matches, 2) if total_matches else 0.0,
                "n_possessions": n_poss,
                "pct_possessions": round(100 * n_poss / total_poss, 2) if total_poss else 0.0,
                "n_positive": n_pos,
                "positive_label_rate": pos_rate,
            }
        )

    return pd.DataFrame(rows)


def compute_graph_split_summary(
    splits: dict[str, list[int]],
    graph_index_df: pd.DataFrame,
    possessions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tạo bảng thống kê tổng hợp phân bố đồ thị trên từng tập train/val/test sau khi dựng graph.

    Args:
        splits: dict gồm {"train": [match_id, ...], "val": [...], "test": [...]}.
        graph_index_df: DataFrame index của đồ thị (yêu cầu match_id, y).
        possessions_df: DataFrame possession ban đầu (tùy chọn, để tính số possession bị loại).

    Returns:
        DataFrame chứa: split, n_matches, pct_matches, n_graphs, pct_graphs,
        n_positive, n_negative, positive_rate, n_dropped_graphs, match_overlap.
    """
    total_graphs = len(graph_index_df)
    total_matches = sum(len(mids) for mids in splits.values())

    s_train = set(splits.get("train", []))
    s_val = set(splits.get("val", []))
    s_test = set(splits.get("test", []))
    no_overlap_train = len(s_train & s_val) == 0 and len(s_train & s_test) == 0
    no_overlap_val = len(s_val & s_train) == 0 and len(s_val & s_test) == 0
    no_overlap_test = len(s_test & s_train) == 0 and len(s_test & s_val) == 0
    overlap_status = {"train": no_overlap_train, "val": no_overlap_val, "test": no_overlap_test}

    rows = []
    for split_name in ("train", "val", "test"):
        mids = set(splits.get(split_name, []))
        sub_graphs = graph_index_df[graph_index_df["match_id"].isin(mids)]
        n_graphs = len(sub_graphs)
        n_pos = int((sub_graphs["y"] == 1).sum()) if n_graphs > 0 else 0
        n_neg = int((sub_graphs["y"] == 0).sum()) if n_graphs > 0 else 0
        pos_rate = round(n_pos / n_graphs, 4) if n_graphs > 0 else 0.0

        n_dropped = None
        if possessions_df is not None:
            sub_poss = possessions_df[possessions_df["match_id"].isin(mids)]
            n_dropped = len(sub_poss) - n_graphs

        row = {
            "split": split_name,
            "n_matches": len(mids),
            "pct_matches": round(100 * len(mids) / total_matches, 2) if total_matches else 0.0,
            "n_graphs": n_graphs,
            "pct_graphs": round(100 * n_graphs / total_graphs, 2) if total_graphs else 0.0,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_rate": pos_rate,
            "n_dropped_graphs": n_dropped,
            "match_overlap": "NONE" if overlap_status[split_name] else "DETECTED",
        }
        rows.append(row)

    return pd.DataFrame(rows)


def save_match_splits(
    splits: dict[str, list[int]],
    match_stats_df: pd.DataFrame,
    output_dir: Path | str,
) -> None:
    """Ghi danh sách match_id và bảng thống kê ra thư mục data/splits/.

    Args:
        splits: dict chứa match_id cho train, val, test.
        match_stats_df: DataFrame thống kê từng trận.
        output_dir: Thư mục đích (data/splits).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Ghi từng file CSV riêng biệt cho train, val, test
    for split_name in ("train", "val", "test"):
        mids = splits.get(split_name, [])
        df_split = pd.DataFrame({"match_id": mids})
        df_split.to_csv(out_path / f"{split_name}_matches.csv", index=False)

    # 2. Ghi bảng map match_id -> split
    split_map: dict[int, str] = {}
    for s_name, ids in splits.items():
        for mid in ids:
            split_map[mid] = s_name

    mapping_df = match_stats_df.copy()
    mapping_df["split"] = mapping_df["match_id"].map(split_map)
    mapping_df.to_csv(out_path / "match_splits.csv", index=False)

    # 3. Ghi bảng tóm tắt split_summary.csv
    summary_df = summarize_splits(splits, match_stats_df)
    summary_df.to_csv(out_path / "split_summary.csv", index=False)
    logger.info("Đã lưu các file split vào: %s", out_path)


def load_match_splits(splits_dir: Path | str) -> dict[str, list[int]]:
    """Đọc danh sách match_id đã lưu từ thư mục data/splits/.

    Returns:
        dict gồm {"train": [match_id, ...], "val": [...], "test": [...]}.
    """
    p = Path(splits_dir)
    splits = {}
    for split_name in ("train", "val", "test"):
        fpath = p / f"{split_name}_matches.csv"
        if not fpath.exists():
            raise FileNotFoundError(f"Không tìm thấy file split: {fpath}")
        df = pd.read_csv(fpath)
        splits[split_name] = df["match_id"].astype(int).tolist()
    return splits


def validate_splits(
    splits: dict[str, list[int]],
    all_match_ids: list[int] | set[int],
    match_stats_df: pd.DataFrame | None = None,
) -> list[CheckResult]:
    """Kiểm tra toàn vẹn phân chia tập dữ liệu — chống rò rỉ và mất mát dữ liệu."""
    checks: list[CheckResult] = []

    s_train = set(splits.get("train", []))
    s_val = set(splits.get("val", []))
    s_test = set(splits.get("test", []))
    s_all = set(all_match_ids)

    # 1. Rò rỉ giữa train và val
    overlap_tr_val = len(s_train & s_val)
    checks.append(
        CheckResult(
            "no_leakage_train_val",
            overlap_tr_val,
            "PASS" if overlap_tr_val == 0 else "FAIL",
            note=f"{overlap_tr_val} trận trùng giữa Train và Val" if overlap_tr_val else "",
        )
    )

    # 2. Rò rỉ giữa train và test
    overlap_tr_test = len(s_train & s_test)
    checks.append(
        CheckResult(
            "no_leakage_train_test",
            overlap_tr_test,
            "PASS" if overlap_tr_test == 0 else "FAIL",
            note=f"{overlap_tr_test} trận trùng giữa Train và Test" if overlap_tr_test else "",
        )
    )

    # 3. Rò rỉ giữa val và test
    overlap_val_test = len(s_val & s_test)
    checks.append(
        CheckResult(
            "no_leakage_val_test",
            overlap_val_test,
            "PASS" if overlap_val_test == 0 else "FAIL",
            note=f"{overlap_val_test} trận trùng giữa Val và Test" if overlap_val_test else "",
        )
    )

    # 4. Độ bao phủ toàn bộ match_id
    union_splits = s_train | s_val | s_test
    missing_matches = len(s_all - union_splits)
    checks.append(
        CheckResult(
            "splits_cover_all_matches",
            missing_matches,
            "PASS" if missing_matches == 0 else "FAIL",
            note=f"Thiếu {missing_matches} trận chưa được phân tập" if missing_matches else "",
        )
    )

    # 5. Phân tầng: kiểm tra độ lệch tỉ lệ nhãn dương giữa các tập
    if match_stats_df is not None:
        summary = summarize_splits(splits, match_stats_df)
        rates = summary["positive_label_rate"].tolist()
        max_diff = max(rates) - min(rates) if rates else 0.0
        checks.append(
            CheckResult(
                "stratification_rate_balance",
                round(max_diff, 4),
                "PASS" if max_diff <= 0.05 else "WARNING",
                note=f"Độ lệch tỉ lệ nhãn dương lớn nhất giữa các tập: {max_diff:.4f}",
            )
        )

    return checks
