"""Bước 7-8 — Bài toán 2: Giải thích và đánh giá chất lượng giải thích.

Module dự kiến:
    explainers.py  Bọc GNNExplainer / PGExplainer (torch_geometric.explain)
    shap_baseline.py  SHAP trên mô hình phi đồ thị — baseline giải thích
                      ở mức đặc trưng (feature-level), Tầng 3(b)
    subgraph.py    Trích subgraph quan trọng + quy điểm importance về từng
                   cầu thủ (tổng hợp trọng số các cạnh liên quan) — Bước 7.2
    metrics.py     Fidelity+/Fidelity-, Sparsity, Stability/Robustness,
                   ROAR / Insertion-Deletion curve (mục 1.2b — phản hồi GV)

NGUYÊN TẮC: điểm importance của XAI phản ánh HÀNH VI CỦA MÔ HÌNH,
không phải quan hệ nhân quả trong bóng đá. Không diễn giải quá mức.
"""
