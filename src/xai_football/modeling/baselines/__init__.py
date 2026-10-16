"""Baseline phi đồ thị — Tầng 3(a) trong chiến lược đối sánh.

Module:
    path_features.py  Trích đặc trưng thống kê từ passing path
    tabular.py        Logistic Regression, Random Forest, ANN
"""

from __future__ import annotations

from xai_football.modeling.baselines.path_features import (
    extract_path_features_from_graph,
    graphs_to_tabular_dataset,
)
from xai_football.modeling.baselines.tabular import train_and_eval_tabular_baselines

__all__ = [
    "extract_path_features_from_graph",
    "graphs_to_tabular_dataset",
    "train_and_eval_tabular_baselines",
]
