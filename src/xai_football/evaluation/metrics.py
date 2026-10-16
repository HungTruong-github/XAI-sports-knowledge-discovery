"""Bước 6 — Độ đo đánh giá mô hình dự đoán (Bài toán 1).

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 6 và phản hồi GV ngày 25/08.

Bộ độ đo:
    - Accuracy (chỉ tham khảo)
    - Precision, Recall, F1-score
    - AUC-ROC
    - AUC-PR (độ đo chính — do dữ liệu mất cân bằng lớp)
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


def compute_binary_metrics(
    y_true: np.ndarray | list[int | float],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Tính toàn bộ bộ độ đo đánh giá cho bài toán phân loại nhị phân.

    Args:
        y_true: Nhãn thực tế (0 hoặc 1).
        y_prob: Xác suất dự đoán thuộc lớp 1 (trong đoạn [0, 1]).
        threshold: Ngưỡng quyết định nhị phân (mặc định 0.5).

    Returns:
        dict chứa accuracy, precision, recall, f1, auc_roc, auc_pr, brier_score.
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
    }
