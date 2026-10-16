"""Baseline PHI ĐỒ THỊ (Tabular Baselines) — Tầng 3(a).

Tham chiếu: configs/model/baseline_tabular.yaml và docs/Quy_trinh_XAI_Football.md, mục 5.2.
Các mô hình:
    - Logistic Regression
    - Random Forest
    - Artificial Neural Network (MLP/ANN)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from xai_football.evaluation.metrics import compute_binary_metrics, find_best_threshold
from xai_football.utils.logging import get_logger

logger = get_logger("tabular_baselines")


def train_and_eval_tabular_baselines(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: np.ndarray,
    X_test: pd.DataFrame | np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    config: dict | None = None,
    threshold_strategy: str = "val_optimal",
    seed: int = 42,
) -> dict[str, Any]:
    """Huấn luyện và đánh giá 3 mô hình baseline phi đồ thị:
    Logistic Regression, Random Forest, ANN.

    Quy tắc phân định Validation - Test:
        - Threshold tối ưu được tìm DUY NHẤT trên tập Validation (tối đa hóa F1).
        - Ngưỡng này sau đó được áp dụng cố định khi đánh giá trên tập Test.
        - AUC-PR và AUC-ROC được tính từ xác suất liên tục, không phụ thuộc threshold.

    Args:
        X_train, y_train: Dữ liệu huấn luyện.
        X_val, y_val: Dữ liệu kiểm định.
        X_test, y_test: Dữ liệu kiểm tra độc lập.
        config: Cấu hình siêu tham số từ baseline_tabular.yaml.
        threshold_strategy: 'val_optimal' (tìm ngưỡng tốt nhất trên val) hoặc 'fixed_0.5'.
        seed: Random seed cho việc khởi tạo mô hình.

    Returns:
        dict gồm:
            "models": dict các model đã huấn luyện,
            "scaler": StandardScaler,
            "val_metrics": dict độ đo trên tập validation,
            "test_metrics": dict độ đo trên tập test (nếu có X_test),
            "val_probs": dict xác suất dự đoán trên tập validation,
            "test_probs": dict xác suất dự đoán trên tập test,
            "thresholds": dict ngưỡng quyết định đã chọn cho từng mô hình.
    """
    cfg = config or {}
    models_cfg = cfg.get("models", {})

    # Chuẩn hóa đặc trưng cho các mô hình nhạy cảm với khoảng giá trị (Logistic Regression, ANN)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_va_scaled = scaler.transform(X_val)
    X_te_scaled = scaler.transform(X_test) if X_test is not None else None

    results: dict[str, Any] = {
        "models": {},
        "scaler": scaler,
        "val_metrics": {},
        "test_metrics": {},
        "val_probs": {},
        "test_probs": {},
        "thresholds": {},
    }

    # 1. Logistic Regression
    lr_cfg = models_cfg.get("logistic_regression", {})
    lr = LogisticRegression(
        class_weight=lr_cfg.get("class_weight", "balanced"),
        max_iter=int(lr_cfg.get("max_iter", 1000)),
        random_state=seed,
    )
    lr.fit(X_tr_scaled, y_train)
    va_prob_lr = lr.predict_proba(X_va_scaled)[:, 1]
    th_lr = (
        find_best_threshold(y_val, va_prob_lr, metric="f1")
        if threshold_strategy == "val_optimal"
        else 0.5
    )
    results["models"]["logistic_regression"] = lr
    results["val_probs"]["logistic_regression"] = va_prob_lr
    results["thresholds"]["logistic_regression"] = th_lr
    results["val_metrics"]["logistic_regression"] = compute_binary_metrics(
        y_val, va_prob_lr, threshold=th_lr
    )

    if X_te_scaled is not None and y_test is not None:
        te_prob_lr = lr.predict_proba(X_te_scaled)[:, 1]
        results["test_probs"]["logistic_regression"] = te_prob_lr
        results["test_metrics"]["logistic_regression"] = compute_binary_metrics(
            y_test, te_prob_lr, threshold=th_lr
        )

    # 2. Random Forest
    rf_cfg = models_cfg.get("random_forest", {})
    rf = RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 500)),
        class_weight=rf_cfg.get("class_weight", "balanced"),
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)  # Tree-based không bắt buộc scale
    va_prob_rf = rf.predict_proba(X_val)[:, 1]
    th_rf = (
        find_best_threshold(y_val, va_prob_rf, metric="f1")
        if threshold_strategy == "val_optimal"
        else 0.5
    )
    results["models"]["random_forest"] = rf
    results["val_probs"]["random_forest"] = va_prob_rf
    results["thresholds"]["random_forest"] = th_rf
    results["val_metrics"]["random_forest"] = compute_binary_metrics(
        y_val, va_prob_rf, threshold=th_rf
    )

    if X_test is not None and y_test is not None:
        te_prob_rf = rf.predict_proba(X_test)[:, 1]
        results["test_probs"]["random_forest"] = te_prob_rf
        results["test_metrics"]["random_forest"] = compute_binary_metrics(
            y_test, te_prob_rf, threshold=th_rf
        )

    # 3. ANN (MLP Classifier)
    # Lưu ý phương pháp luận: sklearn MLPClassifier không hỗ trợ dropout trong kiến trúc.
    # Cơ chế regularization được kiểm soát qua tham số L2 penalty (alpha).
    ann_cfg = models_cfg.get("ann", {})
    hidden_layers = tuple(ann_cfg.get("hidden_layers", [64, 32]))
    max_iter = int(ann_cfg.get("epochs", 200))
    alpha = float(ann_cfg.get("alpha", 0.0001))
    ann = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=max_iter,
        alpha=alpha,
        random_state=seed,
        early_stopping=True,
    )
    ann.fit(X_tr_scaled, y_train)
    va_prob_ann = ann.predict_proba(X_va_scaled)[:, 1]
    th_ann = (
        find_best_threshold(y_val, va_prob_ann, metric="f1")
        if threshold_strategy == "val_optimal"
        else 0.5
    )
    results["models"]["ann"] = ann
    results["val_probs"]["ann"] = va_prob_ann
    results["thresholds"]["ann"] = th_ann
    results["val_metrics"]["ann"] = compute_binary_metrics(y_val, va_prob_ann, threshold=th_ann)

    if X_te_scaled is not None and y_test is not None:
        te_prob_ann = ann.predict_proba(X_te_scaled)[:, 1]
        results["test_probs"]["ann"] = te_prob_ann
        results["test_metrics"]["ann"] = compute_binary_metrics(
            y_test, te_prob_ann, threshold=th_ann
        )

    logger.info("Hoàn tất huấn luyện 3 baseline phi đồ thị (strategy: %s):", threshold_strategy)
    for name, m in results["val_metrics"].items():
        logger.info(
            "  %s | Val PR: %.4f | Val ROC: %.4f | F1: %.4f (th=%.4f)",
            name,
            m["auc_pr"],
            m["auc_roc"],
            m["f1"],
            results["thresholds"][name],
        )

    return results
