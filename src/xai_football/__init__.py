"""XAI-Football — XAI cho mạng chuyền bóng bằng Graph Neural Network.

Cấu trúc package ánh xạ theo đúng pipeline trong
docs/Quy_trinh_XAI_Football.md:

    data          -> Bước 1-2, 4  (thu thập, tiền xử lý, chia tập)
    graphs        -> Bước 3       (xây dựng đồ thị possession)
    modeling      -> Bước 5       (GNN + baseline phi đồ thị)
    training      -> Bước 5.5     (vòng huấn luyện, loss, early stopping)
    evaluation    -> Bước 6       (độ đo cho Bài toán 1)
    xai           -> Bước 7-8     (GNNExplainer/PGExplainer + độ đo XAI)
    centrality    -> Bước 9-10    (Hybrid Centrality + kiểm chứng)
    visualization -> Bước 10      (sân bóng, mạng chuyền, subgraph)
    utils         -> dùng chung   (seed, đường dẫn, log, I/O)
"""

__version__ = "0.1.0"
