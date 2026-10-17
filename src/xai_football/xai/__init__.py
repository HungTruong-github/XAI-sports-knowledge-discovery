"""Bước 7-8 — Bài toán 2: Giải thích và đánh giá chất lượng giải thích.

Module:
    explainers.py    Bọc GNNExplainer / PGExplainer (torch_geometric.explain)
    subgraph.py      Trích subgraph quan trọng + quy điểm importance về từng cầu thủ
    metrics.py       Fidelity+/Fidelity-, Sparsity, Stability, Insertion/Deletion Curves
    shap_baseline.py SHAP trên mô hình phi đồ thị — baseline giải thích mức đặc trưng Tầng 3(b)

NGUYÊN TẮC: điểm importance của XAI phản ánh HÀNH VI CỦA MÔ HÌNH,
không phải quan hệ nhân quả trong bóng đá. Không diễn giải quá mức.
"""

from xai_football.xai.explainers import (
    BaseGraphExplainer,
    ExplanationResult,
    FootballGNNExplainer,
    FootballPGExplainer,
    build_explainer,
)
from xai_football.xai.metrics import (
    compute_fidelity_minus,
    compute_fidelity_plus,
    compute_insertion_deletion_curves,
    compute_sparsity,
    compute_stability,
    evaluate_explanation_quality,
)
from xai_football.xai.shap_baseline import explain_tabular_baseline_shap
from xai_football.xai.subgraph import (
    aggregate_player_importance,
    map_explanation_to_passes,
    select_important_edges,
)

__all__ = [
    "BaseGraphExplainer",
    "ExplanationResult",
    "FootballGNNExplainer",
    "FootballPGExplainer",
    "aggregate_player_importance",
    "build_explainer",
    "compute_fidelity_minus",
    "compute_fidelity_plus",
    "compute_insertion_deletion_curves",
    "compute_sparsity",
    "compute_stability",
    "evaluate_explanation_quality",
    "explain_tabular_baseline_shap",
    "map_explanation_to_passes",
    "select_important_edges",
]
