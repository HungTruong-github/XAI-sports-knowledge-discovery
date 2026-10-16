"""Bước 6 — Độ đo đánh giá mô hình dự đoán (Bài toán 1).

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 6 và phản hồi GV ngày 25/08.

Bộ độ đo:
    - Accuracy (chỉ tham khảo)
    - Precision, Recall, F1-score
    - AUC-ROC (độc lập với ngưỡng)
    - AUC-PR (độ đo chính — độc lập với ngưỡng, phù hợp lớp mất cân bằng)
    - Brier score (độ đo độ tin cậy/calibration)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(
    y_true: np.ndarray | list[int | float],
    y_prob: np.ndarray | list[float],
    metric: str = "f1",
    n_thresholds: int = 100,
) -> float:
    """Tìm ngưỡng nhị phân tối ưu trên tập Validation (mặc định tối ưu F1-score).

    Quy tắc:
        - CHỈ ĐƯỢC CHẠY TRÊN TẬP VALIDATION. Tuyệt đối không dùng tập Test để tìm ngưỡng.
        - Ngưỡng tìm được sẽ được cố định và chuyển sang tập Test để đánh giá khách quan.

    Args:
        y_true: Nhãn thực tế (0 hoặc 1).
        y_prob: Xác suất dự đoán thuộc lớp 1.
        metric: Tên độ đo cần tối ưu (mặc định: 'f1').
        n_thresholds: Số lượng ngưỡng quét đều trong khoảng (0.05, 0.95).

    Returns:
        Ngưỡng tối ưu (float, làm tròn 4 chữ số thập phân).
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.clip(np.asarray(y_prob).astype(float), 0.0, 1.0)

    if len(np.unique(y_t)) <= 1:
        return 0.5

    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    best_thresh = 0.5
    best_score = -1.0

    for th in thresholds:
        y_pred = (y_p >= th).astype(int)
        if metric == "f1":
            score = float(f1_score(y_t, y_pred, zero_division=0))
        elif metric == "precision":
            score = float(precision_score(y_t, y_pred, zero_division=0))
        elif metric == "recall":
            score = float(recall_score(y_t, y_pred, zero_division=0))
        elif metric == "accuracy":
            score = float(accuracy_score(y_t, y_pred))
        else:
            raise ValueError(f"Metric {metric} không được hỗ trợ để tìm threshold.")

        if score > best_score:
            best_score = score
            best_thresh = float(th)

    return round(best_thresh, 4)


def compute_binary_metrics(
    y_true: np.ndarray | list[int | float],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Tính toàn bộ bộ độ đo đánh giá cho bài toán phân loại nhị phân.

    Lưu ý:
        - AUC-ROC và AUC-PR được tính trực tiếp từ xác suất liên tục y_prob,
          hoàn toàn độc lập với ngưỡng phân loại `threshold`.
        - Accuracy, Precision, Recall, F1 phụ thuộc vào `threshold`.

    Args:
        y_true: Nhãn thực tế (0 hoặc 1).
        y_prob: Xác suất dự đoán thuộc lớp 1 (trong đoạn [0, 1]).
        threshold: Ngưỡng quyết định nhị phân (mặc định 0.5).

    Returns:
        dict chứa accuracy, precision, recall, f1, auc_roc, auc_pr, brier_score, threshold.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.clip(np.asarray(y_prob).astype(float), 0.0, 1.0)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    # Tránh crash nếu một tập toàn 0 hoặc toàn 1 (ví dụ batch quá nhỏ)
    has_both_classes = len(np.unique(y_true_arr)) > 1

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    auc_roc = float(roc_auc_score(y_true_arr, y_prob_arr)) if has_both_classes else 0.5
    auc_pr = (
        float(average_precision_score(y_true_arr, y_prob_arr))
        if has_both_classes
        else float(np.mean(y_true_arr))
    )
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "brier_score": round(brier, 4),
        "threshold": round(float(threshold), 4),
    }
