# Báo cáo Đánh giá Hiệu năng & Đối sánh Thống kê (Bước 6)

## 1. Kết quả hiệu năng trên tập Test (Mean ± Std qua 5 seeds):
- **MPNN (GNN tốt nhất)**: AUC-PR `0.5996 ± 0.0042`
- **Random Forest**: AUC-PR `0.6198 ± 0.0017`
- **ANN (MLP)**: AUC-PR `0.6117 ± 0.0097`

## 2. Kiểm định ý nghĩa thống kê (Paired Cluster Bootstrap by Match, Holm Correction):
- Chênh lệch AUC-PR giữa **MPNN và Random Forest / ANN** **không có ý nghĩa thống kê** ($p pprox 0,12 - 0,28$).
- Chênh lệch giữa **MPNN và các GNN khác** (GCN, GraphSAGE, GATv2) **có ý nghĩa thống kê** ($p < 0,05$).

## 3. Khuyến nghị viết bài báo:
- Không tuyên bố GNN đánh bại mô hình bảng truyền thống ở bài toán phân loại nhị phân mức đồ thị.
- Khẳng định MPNN đạt hiệu năng dự đoán tiệm cận mô hình tốt nhất, đồng thời là nền tảng duy nhất cho phép giải thích cấu trúc mạng chuyền bóng (subgraph XAI) và xếp hạng chiến thuật cầu thủ (Hybrid Centrality).
