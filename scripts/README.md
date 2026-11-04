# scripts/ — Điểm vào điều phối pipeline

Mỗi script là một **bước trong pipeline nghiên cứu**, chạy được độc lập từ dòng lệnh và nhận cấu hình từ `configs/`. Script chỉ làm nhiệm vụ điều phối và xác thực provenance; toàn bộ logic nghiệp vụ cốt lõi nằm trong package `src/xai_football/`.

Các tiện ích tài liệu và công cụ chuyển đổi định dạng (ví dụ `md2docx.py`) được tổ chức riêng tại `tools/`.

## 1. Danh mục các script trong pipeline (Bước 0 đến 10)

Đánh số theo đúng thứ tự các bước trong quy trình nghiên cứu:

| Script | Bước | Nhiệm vụ chính | Đầu vào | Đầu ra chính |
|---|---|---|---|---|
| `00_audit_data.py` | 0 | Kiểm tra kho dữ liệu thô và tạo inventory | `data/raw/` | `results/tables/dataset_inventory.csv` |
| `00_generate_environment_metadata.py` | 0 | Thu thập metadata môi trường, phiên bản và mã băm cấu hình | Môi trường hệ thống | `results/metadata/official/environment.json` |
| `01_download_data.py` | 1 | Tải dữ liệu sự kiện và đội hình StatsBomb về `data/raw/` | StatsBomb Open Data | `data/raw/2/27/*.json` |
| `02_descriptive_stats.py` | 2 | Thống kê mô tả dữ liệu thô và gán nhãn | `data/raw/` | `results/tables/descriptive_stats.csv` |
| `03_preprocess.py` | 2 | Trích possession, làm sạch, gán nhãn, xuất bảng thống kê | `data/raw/` | `data/interim/official/events.parquet`, `possessions.parquet` |
| `04_build_graphs.py` | 3 | Dựng possession graph PyG, trích 8 đặc trưng đỉnh và 12 đặc trưng cạnh | `data/interim/` | `data/processed/official/graphs.pt` |
| `05_make_splits.py` | 4 | Chia train/val/test phân tầng theo TRẬN ĐẤU (chống rò rỉ) | `data/interim/events.parquet` | `data/splits/official/*_matches.csv` |
| `make_advanced_splits.py` | 4 | Tạo các phân chia nâng cao: 5-fold cross-fitting và phân chia A/B 190/190 trận | `data/splits/` | `data/splits/crossfit_5fold.json`, `hybrid_AB_split.json` |
| `06_train.py` | 5 | Huấn luyện GNN multi-seed (GCN, MPNN, GATv2, GraphSAGE qua 5 seeds) | `graphs.pt`, `configs/model/*.yaml` | `models/official/<model>_seed*.pt` |
| `06_train_baselines.py` | 5 | Huấn luyện baseline phi đồ thị (LR, RF, ANN) | `graphs.pt`, `configs/model/baseline_tabular.yaml` | `results/tables/official/baseline_comparison.csv` |
| `06c_baseline_feature_ablation.py` | 5 | Nghiên cứu triệt tiêu đặc trưng (ablation) cho tabular baselines | `graphs.pt` | `results/tables/official/baseline_feature_ablation.csv` |
| `06d_crossfit_train.py` | 5/9 | Huấn luyện MPNN 5-fold cross-fitting theo trận trên toàn bộ 380 trận | `crossfit_5fold.json` | `models/official/crossfit/fold_*_mpnn_best.pt` |
| `07_evaluate.py` | 6 | Tổng hợp đối sánh mô hình, kiểm định bootstrap theo trận | Checkpoints & Predictions | `results/tables/official/model_comparison.csv`, `significance_tests.csv` |
| `evaluate_readouts.py` | 7 | Khảo sát 4 biến thể Readout Pooling MPNN (`mean`, `max`, `attention`, `mean_lastedge`) | `graphs.pt`, `configs/model/mpnn_*.yaml` | `results/tables/official/readout_comparison.csv` |
| `08_explain.py` | 8 | Giải thích mô hình bằng GNNExplainer, PGExplainer, Random, Last-edge | Checkpoints MPNN, `graphs.pt` | `results/xai/official/mpnn/official_test/` |
| `09_evaluate_xai.py` | 8 | Đánh giá đối sánh chất lượng XAI (Fidelity+, Fidelity-, Sparsity, Gate D17) | Thư mục XAI output | `results/tables/official/xai_comparison.csv` |
| `10_crossfit_explain.py` | 9 | Trích xuất giải thích out-of-fold cho toàn bộ possessions bằng PGExplainer | Checkpoints crossfit | `results/xai/crossfit/player_importance_oof.parquet` |
| `11_run_hybrid_experiment.py` | 9/10 | Thực nghiệm chính thức Hybrid Centrality A/B, OLS ngoại mẫu và phân tích độ nhạy | Set A/B parquets, events, lineups | `results/tables/official/hybrid_centrality_*.csv` |
| `12_generate_all_figures.py` | 10 | Tự động sinh 7 bộ biểu đồ chính thức (PNG 300 DPI + SVG vector) | Các bảng kết quả official | `results/figures/official/*.png`, `*.svg` |
| `run_pipeline_0_6.py` | Runner | Thực thi chuỗi tự động từ Bước 0 đến Bước 6 | Toàn bộ pipeline | Các artifacts bước 0–6 |
| `audit_steps_0_6.py` | Audit | Kiểm toán độc lập tính toán lại từ đầu cho Bước 0–6 | Toàn bộ artifacts | Báo cáo kiểm toán JSON/Markdown |
| `generate_evidence_manifest.py` | Audit | Tổng hợp Evidence Bundles cho các bước thực nghiệm | Thư mục outputs/ | Manifest tổng hợp |

## 2. Thứ tự thực thi pipeline chuẩn mực

1. **Chuẩn bị và tiền xử lý (Bước 0–4):**
   ```bash
   python scripts/00_generate_environment_metadata.py
   python scripts/01_download_data.py
   python scripts/03_preprocess.py
   python scripts/04_build_graphs.py
   python scripts/05_make_splits.py
   python scripts/make_advanced_splits.py
   ```

2. **Huấn luyện và đối sánh mô hình (Bước 5–7):**
   ```bash
   python scripts/06_train.py
   python scripts/06_train_baselines.py
   python scripts/07_evaluate.py
   python scripts/evaluate_readouts.py
   ```

3. **Giải thích XAI & Cổng kiểm soát Gate D17 (Bước 8):**
   ```bash
   python scripts/08_explain.py
   python scripts/09_evaluate_xai.py
   ```

4. **Cross-Fitting & Thực nghiệm Hybrid Centrality (Bước 9–10):**
   ```bash
   python scripts/06d_crossfit_train.py
   python scripts/10_crossfit_explain.py
   python scripts/11_run_hybrid_experiment.py
   python scripts/12_generate_all_figures.py
   ```

## 3. Nguyên tắc quản lý thực nghiệm & Chống lấy nhầm số liệu

1. **Yêu cầu chỉ định run rõ ràng**:
   - `scripts/09_evaluate_xai.py` **không tự động chọn run mới nhất** theo thứ tự tên thư mục. Bắt buộc truyền `--run-id <ID>` hoặc `--xai-dir <DIR>`. Nếu thiếu, script sẽ dừng và in danh sách các run khả dụng kèm trạng thái.
2. **Bảo vệ bảng official**:
   - Run `pilot`, `smoke-test`, `partial` hoặc `historical` tuyệt đối không được tự động ghi đè lên các bảng canonical trong `results/tables/official/`.
   - Trong `scripts/07_evaluate.py`, khi chạy với `--allow-partial`, kết quả được lưu riêng vào `partial_model_comparison.csv` để bảo vệ bảng chính thức.
   - Trong `scripts/09_evaluate_xai.py`, run lịch sử (ví dụ run `20261017_091540` gồm 5 đồ thị từ branch `REPRODUCIBILITY`) chỉ được đọc khi có cờ `--allow-historical` và được xuất ra bảng riêng `xai_comparison_historical_<run_id>.csv`.
3. **Ghi file an toàn (Atomic Write)**:
   - Toàn bộ kết quả bảng biểu và manifest đều sử dụng ghi file nguyên tử (`save_atomic_csv`, `save_atomic_json`) để loại bỏ hoàn toàn nguy cơ sinh file dở dang khi gặp sự cố giữa chừng.

1. **Yêu cầu chỉ định run rõ ràng**:
   - `scripts/09_evaluate_xai.py` **không tự động chọn run mới nhất** theo thứ tự tên thư mục. Bắt buộc truyền `--run-id <ID>` hoặc `--xai-dir <DIR>`. Nếu thiếu, script sẽ dừng và in danh sách các run khả dụng kèm trạng thái.
2. **Bảo vệ bảng official**:
   - Run `pilot`, `smoke-test`, `partial` hoặc `historical` tuyệt đối không được tự động ghi đè lên các bảng canonical trong `results/tables/official/`.
   - Trong `scripts/07_evaluate.py`, khi chạy với `--allow-partial`, kết quả được lưu riêng vào `partial_model_comparison.csv` để bảo vệ bảng chính thức.
   - Trong `scripts/09_evaluate_xai.py`, run lịch sử (ví dụ run `20261017_091540` gồm 5 đồ thị từ branch `REPRODUCIBILITY`) chỉ được đọc khi có cờ `--allow-historical` và được xuất ra bảng riêng `xai_comparison_historical_<run_id>.csv`.
3. **Ghi file an toàn (Atomic Write)**:
   - Toàn bộ kết quả bảng biểu và manifest đều sử dụng ghi file nguyên tử (`save_atomic_csv`, `save_atomic_json`) để loại bỏ hoàn toàn nguy cơ sinh file dở dang khi gặp sự cố giữa chừng.
