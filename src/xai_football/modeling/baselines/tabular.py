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

from xai_football.evaluation.metrics import compute_binary_metrics
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
) -> dict[str, Any]:
    """Huấn luyện và đánh giá 3 mô hình baseline phi đồ thị:
    Logistic Regression, Random Forest, ANN.

    Returns:
        dict gồm:
            "models": dict các model đã huấn luyện,
            "scaler": StandardScaler,
            "val_metrics": dict độ đo trên tập validation,
            "test_metrics": dict độ đo trên tập test (nếu có X_test).
    """
    cfg = config or {}
    models_cfg = cfg.get("models", {})

    # Chuẩn hóa đặc trưng
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_va_scaled = scaler.transform(X_val)
    X_te_scaled = scaler.transform(X_test) if X_test is not None else None

    results = {
        "models": {},
        "scaler": scaler,
        "val_metrics": {},
        "test_metrics": {},
    }

    # 1. Logistic Regression
    lr_cfg = models_cfg.get("logistic_regression", {})
    lr = LogisticRegression(
        class_weight=lr_cfg.get("class_weight", "balanced"),
        max_iter=int(lr_cfg.get("max_iter", 1000)),
        random_state=42,
    )
    lr.fit(X_tr_scaled, y_train)
    va_prob = lr.predict_proba(X_va_scaled)[:, 1]
    results["models"]["logistic_regression"] = lr
    results["val_metrics"]["logistic_regression"] = compute_binary_metrics(y_val, va_prob)
    if X_te_scaled is not None and y_test is not None:
        te_prob = lr.predict_proba(X_te_scaled)[:, 1]
        results["test_metrics"]["logistic_regression"] = compute_binary_metrics(y_test, te_prob)

    # 2. Random Forest
    rf_cfg = models_cfg.get("random_forest", {})
    rf = RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 500)),
        class_weight=rf_cfg.get("class_weight", "balanced"),
        random_state=int(rf_cfg.get("random_state", 42)),
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)  # Tree-based không bắt buộc scale
    va_prob = rf.predict_proba(X_val)[:, 1]
    results["models"]["random_forest"] = rf
    results["val_metrics"]["random_forest"] = compute_binary_metrics(y_val, va_prob)
    if X_test is not None and y_test is not None:
        te_prob = rf.predict_proba(X_test)[:, 1]
        results["test_metrics"]["random_forest"] = compute_binary_metrics(y_test, te_prob)

    # 3. ANN (MLP Classifier)
    ann_cfg = models_cfg.get("ann", {})
    hidden_layers = tuple(ann_cfg.get("hidden_layers", [64, 32]))
    max_iter = int(ann_cfg.get("epochs", 200))
    ann = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
    )
    ann.fit(X_tr_scaled, y_train)
    va_prob = ann.predict_proba(X_va_scaled)[:, 1]
    results["models"]["ann"] = ann
    results["val_metrics"]["ann"] = compute_binary_metrics(y_val, va_prob)
    if X_te_scaled is not None and y_test is not None:
        te_prob = ann.predict_proba(X_te_scaled)[:, 1]
        results["test_metrics"]["ann"] = compute_binary_metrics(y_test, te_prob)

    logger.info("Hoàn tất huấn luyện 3 baseline phi đồ thị:")
    for name, m in results["val_metrics"].items():
        logger.info(
            "  %s | Val AUC-PR: %.4f | Val AUC-ROC: %.4f | F1: %.4f",
            name,
            m["auc_pr"],
            m["auc_roc"],
            m["f1"],
        )

    return results
