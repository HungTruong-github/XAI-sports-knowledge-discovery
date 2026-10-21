"""Bước 6 — Kiểm định ý nghĩa thống kê khi so sánh các mô hình (Bài toán 1).

Vì sao cần: bảng mean ± std qua 5 seed chỉ phản ánh dao động do khởi tạo ngẫu
nhiên. Nó KHÔNG cho biết chênh lệch giữa hai mô hình có vượt qua dao động do
chọn tập test hay không (tập test chỉ có 57 trận).

Phương pháp — paired cluster bootstrap theo TRẬN:
    1. Mọi mô hình được đánh giá trên cùng một tập đồ thị test (ghép theo
       khóa match_id, possession_id).
    2. Mỗi vòng bootstrap lấy mẫu có hoàn lại 57 trận test (không lấy mẫu từng
       possession, vì các possession cùng trận tương quan với nhau — cùng lý do
       Bước 4 chia tập theo trận).
    3. Trên mẫu đó tính độ đo của TỪNG seed rồi lấy trung bình qua seed, cho
       từng mô hình. Chênh lệch giữa hai mô hình được tính trên CÙNG mẫu
       (paired), nên nhiễu chung do chọn trận bị triệt tiêu.
    4. Khoảng tin cậy 95% = phân vị 2.5% và 97.5% của phân phối chênh lệch.
       p-value hai phía = 2 * min(P(diff <= 0), P(diff >= 0)), có hiệu chỉnh +1.
    5. Nhiều cặp so sánh cùng lúc -> hiệu chỉnh Holm–Bonferroni theo từng độ đo.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _binary_clf_curve(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Số TP, FP tích lũy tại mỗi ngưỡng phân biệt (giảm dần) — như sklearn."""
    order = np.argsort(-p, kind="mergesort")
    y_s, p_s = y[order], p[order]
    idx = np.r_[np.flatnonzero(np.diff(p_s)), len(y_s) - 1]
    tps = np.cumsum(y_s)[idx].astype(float)
    fps = 1.0 + idx - tps
    return tps, fps


def fast_average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """AUC-PR (average precision) — cùng công thức với sklearn.average_precision_score.

    Viết lại bằng numpy vì bootstrap gọi hàm này hàng trăm nghìn lần; bản sklearn tốn
    phần lớn thời gian cho kiểm tra đầu vào. Có test đối chiếu với sklearn.
    """
    tps, fps = _binary_clf_curve(y, p)
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def fast_roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC-ROC — cùng kết quả với sklearn.roc_auc_score (diện tích hình thang)."""
    tps, fps = _binary_clf_curve(y, p)
    tpr = np.r_[0.0, tps / tps[-1]]
    fpr = np.r_[0.0, fps / fps[-1]]
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


# Độ đo không phụ thuộc ngưỡng — phù hợp cho so sánh giữa các mô hình
METRIC_FUNCS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auc_pr": fast_average_precision,
    "auc_roc": fast_roc_auc,
    "brier_score": lambda y, p: float(np.mean((p - y) ** 2)),
}
# Bản sklearn gốc — dùng trong test để đối chiếu
SKLEARN_METRIC_FUNCS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auc_pr": lambda y, p: float(average_precision_score(y, p)),
    "auc_roc": lambda y, p: float(roc_auc_score(y, p)),
    "brier_score": lambda y, p: float(brier_score_loss(y, p)),
}
# Với Brier score, giá trị THẤP hơn là tốt hơn
LOWER_IS_BETTER = {"brier_score"}


def iter_cluster_bootstrap_indices(
    groups: Sequence[int] | np.ndarray, n_boot: int = 10000, seed: int = 42
) -> Iterator[np.ndarray]:
    """Sinh lần lượt `n_boot` mẫu bootstrap theo cụm (cụm = trận đấu).

    Mỗi mẫu là mảng chỉ số hàng: chọn có hoàn lại đúng bằng số trận gốc, rồi lấy
    toàn bộ possession của mỗi trận được chọn. Sinh từng mẫu (generator) để không
    phải giữ cả n_boot mẫu trong RAM (10.000 mẫu x 6.646 dòng ~ 0,5 GB).
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rows_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        yield np.concatenate([rows_by_group[g] for g in picked])


def cluster_bootstrap_indices(
    groups: Sequence[int] | np.ndarray, n_boot: int = 2000, seed: int = 42
) -> list[np.ndarray]:
    """Như `iter_cluster_bootstrap_indices` nhưng trả về list (cùng chuỗi mẫu với cùng seed)."""
    return list(iter_cluster_bootstrap_indices(groups, n_boot=n_boot, seed=seed))


def bootstrap_metric_matrix(
    y_true: np.ndarray,
    seed_probs: Sequence[np.ndarray],
    samples: Sequence[np.ndarray],
    metric: str,
) -> np.ndarray:
    """Độ đo trung bình qua seed của một mô hình trên từng mẫu bootstrap.

    Returns:
        Mảng [n_boot]. Mẫu nào chỉ có một lớp (không tính được AUC) -> NaN.
    """
    func = METRIC_FUNCS[metric]
    y_true = np.asarray(y_true).astype(int)
    out = np.full(len(samples), np.nan)
    for b, idx in enumerate(samples):
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        out[b] = float(np.mean([func(y_b, np.asarray(p)[idx]) for p in seed_probs]))
    return out


def point_metric(y_true: np.ndarray, seed_probs: Sequence[np.ndarray], metric: str) -> float:
    """Độ đo trung bình qua seed trên toàn bộ tập test (không bootstrap)."""
    func = METRIC_FUNCS[metric]
    y_true = np.asarray(y_true).astype(int)
    return float(np.mean([func(y_true, np.asarray(p)) for p in seed_probs]))


def two_sided_bootstrap_p(diffs: np.ndarray) -> float:
    """p-value hai phía cho giả thuyết H0: chênh lệch = 0, từ phân phối bootstrap."""
    d = diffs[~np.isnan(diffs)]
    if len(d) == 0:
        return float("nan")
    n = len(d)
    p_le = (np.sum(d <= 0) + 1) / (n + 1)
    p_ge = (np.sum(d >= 0) + 1) / (n + 1)
    return float(min(1.0, 2 * min(p_le, p_ge)))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Hiệu chỉnh Holm–Bonferroni (kiểm soát FWER) — giữ nguyên thứ tự đầu vào."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)  # p-value NaN (không tính được) giữ nguyên NaN
    valid = np.flatnonzero(~np.isnan(p))
    m = len(valid)
    running_max = 0.0
    for rank, i in enumerate(valid[np.argsort(p[valid])]):
        running_max = max(running_max, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running_max
    return adjusted.tolist()


def compare_models(
    y_true: np.ndarray,
    groups: np.ndarray,
    model_probs: dict[str, Sequence[np.ndarray]],
    metrics: Sequence[str] = ("auc_pr", "auc_roc", "brier_score"),
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """So sánh mọi cặp mô hình bằng paired cluster bootstrap.

    Args:
        y_true: Nhãn thật, đã căn theo cùng thứ tự với mọi mảng xác suất.
        groups: match_id của từng hàng (cụm bootstrap).
        model_probs: {tên mô hình: [xác suất của seed 1, seed 2, ...]}.

    Returns:
        (ci_df, pairs_df)
        ci_df: mỗi (mô hình, độ đo) một dòng — giá trị và KTC 95% theo trận.
        pairs_df: mỗi (cặp A–B, độ đo) một dòng — chênh lệch A − B, KTC 95%,
            p-value thô, p-value Holm, kết luận.
    """
    y_true = np.asarray(y_true).astype(int)
    groups = np.asarray(groups)
    for name, probs in model_probs.items():
        for p in probs:
            if len(p) != len(y_true):
                raise ValueError(f"{name}: số dự đoán ({len(p)}) khác số nhãn ({len(y_true)})")

    names = list(model_probs)
    probs = {name: [np.asarray(p, dtype=float) for p in model_probs[name]] for name in names}

    # Một lượt duyệt qua các mẫu (sinh dần, không giữ trong RAM); mọi mô hình và độ đo dùng
    # CHUNG mẫu b -> so sánh có ghép cặp. Kết quả trùng với bootstrap_metric_matrix.
    boot: dict[tuple[str, str], np.ndarray] = {
        (name, metric): np.full(n_boot, np.nan) for name in names for metric in metrics
    }
    for b, idx in enumerate(iter_cluster_bootstrap_indices(groups, n_boot=n_boot, seed=seed)):
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        for name in names:
            sampled = [p[idx] for p in probs[name]]
            for metric in metrics:
                func = METRIC_FUNCS[metric]
                boot[(name, metric)][b] = float(np.mean([func(y_b, p) for p in sampled]))

    ci_rows = []
    for metric in metrics:
        for name in names:
            mat = boot[(name, metric)]
            ci_rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "n_seeds": len(model_probs[name]),
                    "value": round(point_metric(y_true, model_probs[name], metric), 4),
                    "ci95_low": round(float(np.nanpercentile(mat, 2.5)), 4),
                    "ci95_high": round(float(np.nanpercentile(mat, 97.5)), 4),
                    "n_boot_valid": int(np.sum(~np.isnan(mat))),
                }
            )

    pair_rows = []
    for metric in metrics:
        metric_rows = []
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                diffs = boot[(a, metric)] - boot[(b, metric)]
                point = point_metric(y_true, model_probs[a], metric) - point_metric(
                    y_true, model_probs[b], metric
                )
                metric_rows.append(
                    {
                        "metric": metric,
                        "model_a": a,
                        "model_b": b,
                        "diff_a_minus_b": round(point, 4),
                        "ci95_low": round(float(np.nanpercentile(diffs, 2.5)), 4),
                        "ci95_high": round(float(np.nanpercentile(diffs, 97.5)), 4),
                        "p_value": two_sided_bootstrap_p(diffs),
                        "_diff": point,  # giá trị chưa làm tròn — dùng để xác định chiều kết luận
                    }
                )
        adjusted = holm_adjust([r["p_value"] for r in metric_rows])
        for row, p_adj in zip(metric_rows, adjusted, strict=True):
            row["p_value"] = round(row["p_value"], 4)
            row["p_value_holm"] = round(p_adj, 4)
            significant = bool(p_adj < alpha)  # NaN -> False
            diff = row.pop("_diff")
            if not significant:
                verdict = "không khác biệt có ý nghĩa"
            else:
                a_better = (diff < 0) == (metric in LOWER_IS_BETTER)
                verdict = f"{row['model_a'] if a_better else row['model_b']} tốt hơn"
            row["significant_holm_0.05"] = bool(significant)
            row["verdict"] = verdict
            pair_rows.append(row)

    ci_df = pd.DataFrame(ci_rows)
    pairs_df = pd.DataFrame(pair_rows)
    return ci_df, pairs_df


def align_predictions(
    reference_keys: pd.DataFrame, frame: pd.DataFrame, prob_col: str, label_col: str = "y_true"
) -> np.ndarray:
    """Căn một bảng dự đoán theo thứ tự khóa (match_id, possession_id) tham chiếu.

    Báo lỗi nếu thiếu khóa, thừa khóa hoặc nhãn không khớp — không ghép âm thầm.
    """
    keys = ["match_id", "possession_id"]
    if frame.duplicated(keys).any():
        raise ValueError("Bảng dự đoán có khóa (match_id, possession_id) trùng lặp.")
    other = frame[keys + [label_col, prob_col]].rename(
        columns={label_col: "_label_other", prob_col: "_prob_other"}
    )
    merged = reference_keys[keys + ["y_true"]].merge(
        other, on=keys, how="left", validate="one_to_one"
    )
    if merged["_prob_other"].isna().any() or len(frame) != len(reference_keys):
        raise ValueError(
            f"Dự đoán '{prob_col}' không phủ đúng tập test tham chiếu "
            f"({len(frame)} dòng so với {len(reference_keys)})."
        )
    if not np.array_equal(merged["y_true"].to_numpy(int), merged["_label_other"].to_numpy(int)):
        raise ValueError(f"Nhãn của '{prob_col}' không khớp nhãn tham chiếu.")
    return merged["_prob_other"].to_numpy(float)
