"""Bước 7 — Baseline giải thích mức đặc trưng (SHAP Feature Importance) cho mô hình phi đồ thị.

Tham chiếu: configs/xai.yaml -> shap_baseline và docs/Quy_trinh_XAI_Football.md, Tầng 3(b).

NGUYÊN TẮC:
    - SHAP ở đây được áp dụng trên BASELINE PHI ĐỒ THỊ (Random Forest / Logistic Regression),
      giải thích ở MỨC ĐẶC TRƯNG TỔNG HỢP (feature-level attribution).
    - Đây là baseline so sánh, KHÔNG PHẢI giải thích cấu trúc mạng/subgraph của GNN.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from xai_football.utils.logging import get_logger

logger = get_logger("shap_baseline")


def explain_tabular_baseline_shap(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None = None,
    max_samples: int = 200,
) -> dict[str, Any]:
    """Giải thích mô hình baseline phi đồ thị bằng SHAP.

    Args:
        model: Mô hình sklearn đã huấn luyện (RandomForestClassifier, LogisticRegression).
        X: Dữ liệu đặc trưng dạng bảng (features).
        feature_names: Tên các cột đặc trưng.
        max_samples: Số lượng mẫu kiểm tra tối đa để tính toán nhanh.

    Returns:
        dict chứa:
            "shap_values": mảng giá trị SHAP,
            "mean_abs_shap": dict điểm importance trung bình từng đặc trưng,
            "feature_importance_df": DataFrame xếp hạng đặc trưng.
    """
    if isinstance(X, pd.DataFrame):
        feat_names = list(X.columns)
        X_mat = X.values
    else:
        feat_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        X_mat = np.asarray(X)

    sub_X = X_mat[:max_samples]

    model_type = type(model).__name__
    logger.info("Tính toán SHAP cho mô hình %s trên %d mẫu...", model_type, len(sub_X))

    if "Forest" in model_type or "Tree" in model_type:
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(sub_X)
        # Trong shap mới cho binary classification, raw_shap có thể là list 2 phần tử hoặc 3D array
        if isinstance(raw_shap, list):
            shap_matrix = raw_shap[1]  # Lớp dương (shot = 1)
        elif len(raw_shap.shape) == 3:
            shap_matrix = raw_shap[:, :, 1]
        else:
            shap_matrix = raw_shap
    else:
        # Fallback dùng LinearExplainer hoặc Explainer chung
        try:
            explainer = shap.LinearExplainer(model, sub_X)
            shap_matrix = explainer.shap_values(sub_X)
        except Exception:
            explainer = shap.Explainer(model.predict_proba, sub_X)
            res = explainer(sub_X)
            shap_matrix = res.values[:, :, 1] if len(res.values.shape) == 3 else res.values

    # Tính Mean Absolute SHAP value
    mean_abs = np.mean(np.abs(shap_matrix), axis=0)

    rows = []
    for name, score in zip(feat_names, mean_abs):
        rows.append({"feature": name, "mean_abs_shap": round(float(score), 4)})

    df_imp = (
        pd.DataFrame(rows).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    )

    return {
        "model_name": model_type,
        "n_samples": len(sub_X),
        "feature_names": feat_names,
        "shap_values": shap_matrix,
        "mean_abs_shap": dict(zip(feat_names, [round(float(s), 4) for s in mean_abs])),
        "feature_importance_df": df_imp,
    }
