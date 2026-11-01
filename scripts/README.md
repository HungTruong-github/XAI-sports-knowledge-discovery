# scripts/ — Điểm vào điều phối pipeline

Mỗi script là một **bước trong pipeline nghiên cứu**, chạy được độc lập từ dòng lệnh và nhận cấu hình từ `configs/`. Script chỉ làm nhiệm vụ điều phối và xác thực provenance; toàn bộ logic nghiệp vụ cốt lõi nằm trong package `src/xai_football/`.

Các tiện ích tài liệu và công cụ chuyển đổi định dạng (ví dụ `md2docx.py`) được tổ chức riêng tại `tools/`.

## Danh mục các script hiện có

Đánh số theo đúng thứ tự các bước trong quy trình nghiên cứu:

| Script | Bước | Nhiệm vụ | Đầu ra chính |
|---|---|---|---|
| `00_audit_data.py` | 0 | Kiểm tra kho dữ liệu thô và tạo inventory | `results/tables/dataset_inventory.csv` |
| `00_generate_environment_metadata.py` | 0 | Thu thập metadata môi trường, phiên bản và mã băm cấu hình | `results/metadata/<run>/environment.json` |
| `01_download_data.py` | 1 | Tải dữ liệu sự kiện và đội hình StatsBomb về `data/raw/` | `data/raw/{comp_id}/{season_id}/*.json` |
| `02_descriptive_stats.py` | 2 | Thống kê mô tả dữ liệu thô và gán nhãn | `results/tables/<run>/descriptive_stats.csv` |
| `03_preprocess.py` | 2 | Trích possession, làm sạch, gán nhãn, xuất bảng thống kê | `data/interim/<run>/possessions.parquet` |
| `04_build_graphs.py` | 3 | Dựng possession graph PyG, trích đặc trưng đỉnh/cạnh | `data/processed/<run>/graphs.pt` |
| `05_make_splits.py` | 4 | Chia train/val/test phân tầng theo TRẬN ĐẤU (chống rò rỉ) | `data/splits/<run>/*_matches.csv` |
| `06_train.py` | 5 | Huấn luyện GNN multi-seed (GCN, MPNN, GATv2, GraphSAGE) | `experiments/<run>/<model>/<seed>/` |
| `06_train_baselines.py` | 5 | Huấn luyện baseline phi đồ thị (LR, RF, ANN) | `experiments/<run>/baselines/<seed>/` |
| `06c_baseline_feature_ablation.py` | 5 | Nghiên cứu triệt tiêu đặc trưng (ablation) cho tabular baselines | `results/tables/<run>/baseline_feature_ablation.csv` |
| `07_evaluate.py` | 6 | Tổng hợp đối sánh mô hình, kiểm định bootstrap theo trận | `results/tables/<run>/model_comparison.csv` |
| `08_explain.py` | 7 | Giải thích mô hình bằng GNNExplainer, PGExplainer, SHAP | `results/xai/<run>/<model>/<run_id>/` |
| `09_evaluate_xai.py` | 8 | Đối sánh chất lượng giải thích XAI (Fidelity, Sparsity, Stability) | `results/tables/<run>/xai_comparison.csv` |
| `audit_steps_0_6.py` | Audit | Kiểm toán độc lập tính toán lại từ đầu cho Bước 0-6 | Minh chứng kiểm toán PASS/FAIL |

## Nguyên tắc quản lý thực nghiệm & Chống lấy nhầm số liệu

1. **Yêu cầu chỉ định run rõ ràng**:
   - `scripts/09_evaluate_xai.py` **không tự động chọn run mới nhất** theo thứ tự tên thư mục. Bắt buộc truyền `--run-id <ID>` hoặc `--xai-dir <DIR>`. Nếu thiếu, script sẽ dừng và in danh sách các run khả dụng kèm trạng thái.
2. **Bảo vệ bảng official**:
   - Run `pilot`, `smoke-test`, `partial` hoặc `historical` tuyệt đối không được tự động ghi đè lên các bảng canonical trong `results/tables/official/`.
   - Trong `scripts/07_evaluate.py`, khi chạy với `--allow-partial`, kết quả được lưu riêng vào `partial_model_comparison.csv` để bảo vệ bảng chính thức.
   - Trong `scripts/09_evaluate_xai.py`, run lịch sử (ví dụ run `20261017_091540` gồm 5 đồ thị từ branch `REPRODUCIBILITY`) chỉ được đọc khi có cờ `--allow-historical` và được xuất ra bảng riêng `xai_comparison_historical_<run_id>.csv`.
3. **Ghi file an toàn (Atomic Write)**:
   - Toàn bộ kết quả bảng biểu và manifest đều sử dụng ghi file nguyên tử (`save_atomic_csv`, `save_atomic_json`) để loại bỏ hoàn toàn nguy cơ sinh file dở dang khi gặp sự cố giữa chừng.
