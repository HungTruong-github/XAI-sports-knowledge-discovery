# results/ — Kết quả cuối cùng cho khóa luận

Khác với `experiments/` (log thô, không commit), thư mục này chứa **bảng và
hình đã chọn lọc** để đưa thẳng vào khóa luận / bài báo. Nội dung **có commit**
vì nhẹ và cần theo dõi phiên bản.

## `tables/`

Các bảng dự kiến (theo Bước 10):

- Bảng so sánh hiệu năng giữa các backbone — **Tầng 2**
  (GCN / MPNN / GATv2 / GraphSAGE)
- Bảng so sánh mô hình đồ thị với baseline phi đồ thị — **Tầng 3(a)**
  (Logistic Regression / Random Forest / ANN)
- Bảng so sánh chất lượng XAI — **Tầng 3(b)**
  (GNNExplainer vs PGExplainer theo Fidelity / Sparsity / Stability)
- Bảng so sánh Hybrid Centrality với centrality truyền thống
- Bảng xếp hạng cầu thủ theo Hybrid Centrality

## `figures/`

- Passing network diagram tô trọng số theo Hybrid Centrality
- Overlay subgraph quan trọng lên possession cụ thể (case study)
- Đường cong Insertion-Deletion / ROAR
- Reliability diagram (calibration)

## Nguyên tắc

Mọi con số trong `results/` phải truy ngược được về một lần chạy cụ thể trong
`experiments/`. Ghi kèm `run_id` trong tên file hoặc trong chú thích bảng.
