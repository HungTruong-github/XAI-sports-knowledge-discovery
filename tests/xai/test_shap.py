"""Unit tests cho SHAP baseline trên mô hình phi đồ thị."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from xai_football.xai.shap_baseline import explain_tabular_baseline_shap


def test_explain_tabular_baseline_shap():
    rng = np.random.default_rng(42)
    feature_names = ["length", "duration", "pass_count", "circular_angle"]
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=feature_names)
    # Target tương quan mạnh với feature 'length'
    y = (X["length"] > 0).astype(int).values

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    res = explain_tabular_baseline_shap(rf, X, max_samples=20)

    assert res["model_name"] == "RandomForestClassifier"
    assert res["n_samples"] == 20
    assert len(res["feature_names"]) == 4

    df_imp = res["feature_importance_df"]
    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 4
    assert "feature" in df_imp.columns
    assert "mean_abs_shap" in df_imp.columns

    # 'length' phải là feature quan trọng nhất do target được sinh từ 'length'
    top_feature = df_imp.iloc[0]["feature"]
    assert top_feature == "length"
