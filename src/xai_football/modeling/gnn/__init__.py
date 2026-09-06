"""Backbone GNN — Tầng 2 trong chiến lược đối sánh 3 tầng.

Module dự kiến:
    gcn.py        GCN  — backbone đề xuất
    mpnn.py       MPNN — backbone đề xuất
    gat.py        GAT / GATv2 — baseline đối sánh (căn cứ: TacticAI)
    graphsage.py  GraphSAGE   — baseline đối sánh (sample-and-aggregate)
    readout.py    Lớp pooling gộp node -> vector đồ thị (mean / attention)
    heads.py      MLP + sigmoid cho dự đoán nhị phân ở mức đồ thị

Tất cả backbone phải dùng CHUNG readout/head và CHUNG cách chia dữ liệu
để việc đối sánh cô lập đúng biến số kiến trúc (mục 1.3 Tầng 2 — phản hồi GV).
"""
