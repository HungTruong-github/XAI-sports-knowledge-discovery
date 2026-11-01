# Báo cáo Huấn luyện Mô hình GNN & Baselines (Bước 5)

- **Mô hình GNN đã huấn luyện**:
  - GCN (42, 123, 456, 789, 1024)
  - MPNN (42, 123, 456, 789, 1024)
  - GATv2 (42, 123, 456, 789, 1024)
  - GraphSAGE (42, 123, 456, 789, 1024)
- **Mô hình Baseline phi đồ thị**:
  - Logistic Regression (5 seeds)
  - Random Forest (5 seeds)
  - ANN / Multi-Layer Perceptron (5 seeds)
- **Tiêu chuẩn chọn checkpoint chính thức**: Seed có Validation AUC-PR cao nhất được lưu tại `models/official/<model>_best.pt`.
- **Cơ chế chống ô nhiễm kết quả**: Các run dở dang hoặc debug được cách ly hoàn toàn; multi-seed tổng hợp tính mean và std độc lập.
