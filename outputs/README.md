# outputs/ — Kho Lưu Trữ Minh Chứng Thực Nghiệm (Pipeline Steps 0–6)

Thư mục này chứa toàn bộ các **bằng chứng khoa học có cấu trúc (Evidence Bundles)** được sinh ra tự động từ pipeline Bước 0 đến Bước 6, phục vụ kiểm toán khả năng tái lập và cung cấp số liệu cho bài báo khoa học / khóa luận.

---

## 1. Cấu trúc thư mục chuẩn

```
outputs/
├── README.md                 # Hướng dẫn cấu trúc & quy tắc kiểm toán
├── run_registry.json         # Danh bạ đăng ký trung tâm các run thực nghiệm
├── step_00_environment_and_reproducibility/
│   └── runs/<run_id>/
├── step_01_data_audit_and_collection/
│   └── runs/<run_id>/
├── step_02_preprocessing_and_labels/
│   └── runs/<run_id>/
├── step_03_graph_construction/
│   └── runs/<run_id>/
├── step_04_match_level_split/
│   └── runs/<run_id>/
├── step_05_model_training/
│   └── runs/<run_id>/
└── step_06_model_evaluation/
    └── runs/<run_id>/
```

Trong mỗi thư mục `runs/<run_id>/`:
- `files/`: Bảng số liệu CSV, cấu hình JSON/YAML, danh sách kiểm kê và fingerprints.
- `figures/`: Biểu đồ trực quan hóa xuất đồng thời ở định dạng PNG (300 DPI) và SVG (vector).
- `logs/`: Nhật ký thực thi thực tế.
- `reports/`: Báo cáo kết quả và bảng kiểm định chất lượng (quality checks).
- `manifest.json`: Siêu dữ liệu xuất xứ chuẩn ISO 8601 (SHA-256, commit Git, tham số, inputs, outputs).

---

## 2. Quy tắc Quản lý và Truy vết (Provenance Integrity)

1. **Định danh duy nhất**: Toàn bộ các bước thuộc cùng một lần chạy chia sẻ chung một `run_id`.
2. **Không tự động chọn run mới nhất**: Mọi script phân tích bắt buộc nhận `run_id` rõ ràng; tuyệt đối không quét thư mục theo thứ tự chữ cái hay thời gian sửa file.
3. **Phân loại rõ ràng**:
   - `HISTORICAL_OFFICIAL`: Kết quả thực nghiệm chính thức 5-seed từ commit `fc18a929` trên nhánh `main`, được bảo toàn làm minh chứng đối soát.
   - `HISTORICAL_PROVISIONAL`: Artifact lịch sử quy mô mẫu nhỏ (ví dụ run XAI 5 đồ thị từ commit `9fae6c51`).
   - `SMOKE`: Kết quả kiểm tra đường dẫn pipeline nhanh (1 epoch hoặc subset nhỏ).
   - `OFFICIAL`: Run thực nghiệm đầy đủ, hợp lệ trên mã nguồn hiện tại.
4. **Không đưa file nhị phân nặng lên Git**: Checkpoint `.pt`, đồ thị `.pt`, parquet dung lượng lớn được ghi nhận thông qua mã băm SHA-256 và file pointer `.pointer.json`.
