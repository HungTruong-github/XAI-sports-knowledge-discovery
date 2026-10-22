# BÁO CÁO KIỂM TOÁN HIỆN TRẠNG BƯỚC 0–6 & TÍNH SẴN SÀNG CHO BÀI BÁO KHOA HỌC (OUTPUTS AND PAPER READINESS AUDIT)

**Repository**: `HungTruong-github/XAI-sports-knowledge-discovery`  
**Nhánh làm việc**: `Clean`  
**Commit HEAD hiện tại**: `7d9e0f8`  
**Thời điểm thực hiện**: 23/10/2026  
**Kiểm toán viên**: Senior Python Engineer & Research Reproducibility Auditor  

---

## 1. Mục tiêu và Phạm vi kiểm toán

Kiểm toán toàn diện hiện trạng của **Bước 0 đến Bước 6** trong pipeline XAI-Football, đối chiếu mã nguồn, scripts, cấu hình, dữ liệu, notebooks và bảng kết quả thực tế trên đĩa. Xác định rõ:
1. Trạng thái triển khai và kiểm chứng thực tế của từng bước.
2. Sự nhất quán xuất xứ (provenance) giữa mã nguồn nhánh `Clean` và các manifest lịch sử.
3. Các vấn đề rủi ro ảnh hưởng tới độ tin cậy khoa học và khả năng tái lập độc lập.
4. Danh mục bằng chứng (evidence bundle) cần thiết phục vụ bảo vệ khóa luận và công bố bài báo.

---

## 2. Bảng tổng hợp hiện trạng Pipeline Bước 0–6

| Bước | Tên bước | Trạng thái | Script Entry Point | Module cốt lõi | Input chính | Output chính hiện tại | Kiểm tra chất lượng (Quality Checks) |
|---|---|---|---|---|---|---|---|
| **0** | **Environment & Audit** | `IMPLEMENTED` / `HISTORICAL_VERIFIED` | `scripts/00_generate_environment_metadata.py` | `src/xai_football/utils/environment.py`, `provenance.py` | Runtime OS, Git, `configs/`, `requirements.lock.txt` | `results/metadata/official/environment.json`, `config_hashes.json` | Khóa phiên bản thư viện, SHA-256 configs, trạng thái git working tree. |
| **1** | **Data Audit & Collection** | `IMPLEMENTED` / `VERIFIED` | `scripts/00_audit_data.py`, `scripts/01_download_data.py` | `src/xai_football/data/collect.py` | StatsBomb Open Data API (`competition_id: 2`, `season_id: 27`) | `data/raw/2/27/` (760 JSONs = 380 events + 380 lineups), `dataset_inventory.csv`, `collection_report.csv`, figures `01_*` | Đủ 380/380 trận Premier League 2015/16, không trận nào rỗng, kiểm tra schema JSON. |
| **2** | **Preprocessing & Labeling** | `IMPLEMENTED` / `VERIFIED` | `scripts/02_descriptive_stats.py`, `scripts/03_preprocess.py` | `src/xai_football/data/canonicalize.py`, `clean.py`, `possession.py`, `labeling.py`, `validation.py` | `data/raw/2/27/*.json`, `configs/data.yaml` | `data/interim/official/*.parquet`, 6 bảng CSV thống kê/làm sạch, figures `02_*` | 45.365 possessions, 261.205 passes; 16 tiêu chí sanity checks đều PASS; cắt sau cú sút đầu. |
| **3** | **Graph Construction** | `IMPLEMENTED` / `VERIFIED` | `scripts/04_build_graphs.py` | `src/xai_football/graphs/builder.py`, `dataset.py`, `features.py` | `data/interim/official/`, `configs/graph.yaml` | `data/processed/official/graphs.pt` (45.047 graphs), `graph_index.parquet`, `feature_names.json`, figures `03_*` | 9/9 graph integrity checks PASS; kẹp biên 1 tọa độ phạt góc x=120.9; không có NaN/Inf. |
| **4** | **Match-Level Split** | `IMPLEMENTED` / `VERIFIED` | `scripts/05_make_splits.py` | `src/xai_football/data/splits.py` | `data/interim/official/possessions.parquet`, `configs/split.yaml` | `data/splits/*.csv` (train/val/test matches, match_splits, split_summary, graph_split_summary), figures `04_*` | 266 Train, 57 Val, 57 Test matches; tập trận rời nhau hoàn toàn (disjoint); không rò rỉ dữ liệu. |
| **5** | **Model Training** | `IMPLEMENTED` / `HISTORICAL_VERIFIED` | `scripts/06_train.py`, `scripts/06_train_baselines.py` | `src/xai_football/training/trainer.py`, `modeling/gnn/`, `modeling/baselines.py` | `data/processed/official/`, `data/splits/`, `configs/train.yaml`, `configs/model/*.yaml` | 4 GNNs × 5 seeds + 3 Baselines × 5 seeds trong `experiments/official/`, `models/official/*_best.pt`, figures `05_*` | Huấn luyện đủ 5 seeds (42, 123, 456, 789, 1024); xác định theo seed; checkpoint lưu đúng mô hình tốt nhất theo Val AUC-PR. |
| **6** | **Model Evaluation & Comparison** | `IMPLEMENTED` / `HISTORICAL_VERIFIED` | `scripts/07_evaluate.py` | `src/xai_football/evaluation/metrics.py`, `significance.py`, `threshold.py` | Checkpoints, `data/splits/`, `experiments/official/` | `model_comparison.csv`, `val_model_comparison.csv`, `baseline_comparison.csv`, `model_metric_ci.csv`, `significance_tests.csv`, figures `06_*` | Đánh giá trên 6.646 đồ thị Test set; bootstrap 10.000 mẫu theo cụm trận đấu; hiệu chỉnh đa giả thuyết Holm. |

---

## 3. Các vấn đề ảnh hưởng trực tiếp tới Độ tin cậy Khoa học và Khả năng Tái lập

### 3.1. Sự không nhất quán giữa Git commit hiện tại và Manifests lịch sử
- **Thực tế kiểm toán**:
  - Nhánh hiện tại: `Clean`, commit HEAD: `7d9e0f8`.
  - Manifest môi trường `results/metadata/official/environment.json` và manifest thực nghiệm ghi nhận commit `fc18a929b82914cea6c9ff1bb54d89550597f12e` trên nhánh `main`, với trạng thái `clean_working_tree: false`.
  - Manifest của artifact XAI lịch sử `results/xai/official/mpnn/20261017_091540/manifest.json` ghi nhận commit `9fae6c5188f986fb866223e71781297eefb3b192` trên nhánh `REPRODUCIBILITY`.
- **Đánh giá rủi ro**: Nếu báo cáo tuyên bố các bảng kết quả hiện có được sinh ra từ commit `Clean` hiện tại thì đây là sai lệch xuất xứ (provenance mismatch).
- **Biện pháp xử lý**: Đánh dấu rõ các kết quả có sẵn trong repo là `HISTORICAL_OFFICIAL` (cho mô hình GNN/Baselines từ `fc18a929`) và `HISTORICAL_PROVISIONAL` (cho run XAI 5 đồ thị từ `9fae6c51`). Khi xuất output bundle chế độ `--mode existing`, hệ thống sẽ bảo toàn chính xác mã commit gốc và đánh dấu kiểm toán rõ ràng.

### 3.2. Tính khả chuyển của Lockfile môi trường (`requirements.lock.txt`)
- **Thực tế kiểm toán**:
  - `requirements.lock.txt` được khóa trên Windows 11 với Python 3.13 và PyTorch CPU (`torch==2.14.1+cpu`).
  - GitHub Actions CI chạy trên Ubuntu-latest với Python 3.11, sử dụng `requirements.txt` (chưa ghim cứng phiên bản chi tiết).
  - Khóa phiên bản trên Windows/Python 3.13 chứa các wheel tag phụ thuộc hệ điều hành. Nếu chạy `pip install -r requirements.lock.txt` trên Linux/Python 3.11 sẽ gặp lỗi wheel không tương thích.
- **Biện pháp xử lý**: Ghi nhận rõ sự khác biệt giữa runtime Windows phục vụ khóa luận và runtime Ubuntu CI. Cập nhật `.github/workflows/ci.yml` để kích hoạt kiểm thử trên nhánh `Clean`.

### 3.3. Các Quyết định Nghiên cứu còn ở trạng thái Chờ (Pending Decisions)
1. **Decision 0001 (Quy tắc gán nhãn)**:
   - Hiện tại: áp dụng `possession_level` với nhãn dương là `shot` (tỷ lệ 14,26% shot vs 1,46% goal).
   - Trạng thái: `Pending` chờ giảng viên hướng dẫn (GVHD) chốt chính thức.
   - Ảnh hưởng: Nếu đổi sang nhãn `goal` hoặc `shot_or_goal`, toàn bộ pipeline từ Bước 2 đến Bước 6 phải chạy lại. Manifest phải lưu rõ nhãn đang dùng là `shot (possession_level)`.
2. **Decision 0002 (Công thức Hybrid Centrality)**:
   - Trạng thái: `Pending`. Chưa chốt trọng số kết hợp giữa độ đo cấu trúc (Degree/PageRank/Betweenness) và trọng số giải thích XAI.
3. **Decision 0003 (Đơn vị đồ thị & Ngưỡng độ dài)**:
   - Trạng thái: `Draft/Pending`. Đã áp dụng `truncate_at_first_shot: true` và lọc possession $< 2$ đường chuyền.
4. **Phạm vi Open-play**:
   - Trong `configs/graph.yaml`, hiện tại `play_patterns_keep: null` (giữ tất cả các pha bóng).

### 3.4. Diễn giải Kết quả Đánh giá Mô hình một cách Thận trọng
- **Thực tế số liệu**:
  - Nhóm GNN (Test AUC-PR): MPNN ($0.5996 \pm 0.0042$) > GATv2 ($0.5925$) > GraphSAGE ($0.5898$) > GCN ($0.5794$). MPNN vượt trội có ý nghĩa thống kê so với các GNN còn lại.
  - Baseline phi đồ thị (Test AUC-PR): Random Forest ($0.6198 \pm 0.0017$), ANN ($0.6117 \pm 0.0097$).
  - Kiểm định Bootstrap ghép cặp theo trận (Holm-Bonferroni): Chênh lệch AUC-PR giữa MPNN và Random Forest / ANN **không có ý nghĩa thống kê** ($p \approx 0.12 - 0.28$).
- **Nguyên tắc viết bài báo**: Không được tuyên bố GNN đánh bại mô hình bảng truyền thống trong bài toán phân loại nhị phân ở mức đồ thị. Thay vào đó, nhấn mạnh rằng GNN đạt hiệu năng dự đoán tương đương các mô hình mạnh nhất, nhưng mang lại giá trị độc nhất mà mô hình bảng không thể có: **khả năng giải thích cấu trúc mạng chuyền bóng (subgraph XAI)** và tính toán vai trò chiến thuật của từng cầu thủ thông qua **Hybrid Centrality**.

---

## 4. Hiện trạng Tổ chức Lưu trữ Artifact & Sự Phân tán Output

Trước đợt kiểm toán này:
- Output của một lần chạy thực nghiệm bị phân tán ở nhiều vị trí: `data/interim/`, `data/processed/`, `data/splits/`, `experiments/official/`, `models/official/`, `results/tables/official/`, `results/figures/official/`, `results/metadata/official/`.
- File biểu đồ `results/figures/official/` được sinh ra khi chạy Jupyter Notebooks, chưa được tích hợp trực tiếp vào script điều phối pipeline.
- Thư mục `.gitignore` cũ loại bỏ toàn bộ `outputs/`, khiến cho các minh chứng tái lập có nguy cơ không thể đưa lên kho lưu trữ trực tuyến để người đọc bài báo đối soát.

---

## 5. Kế hoạch Hoàn thiện Output Evidence Bundle

Để phục vụ bài báo và đồ án tốt nghiệp, toàn bộ minh chứng của Bước 0–6 sẽ được tổ chức thành cấu trúc chuẩn hoá `outputs/`:

```
outputs/
├── README.md
├── run_registry.json
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

Mỗi thư mục `runs/<run_id>/` bắt buộc có đủ:
1. `files/`: Bảng CSV, JSON cấu hình, fingerprint.
2. `figures/`: Biểu đồ PNG (300 DPI) và SVG.
3. `logs/`: Nhật ký thực thi thực tế.
4. `reports/`: Báo cáo kết quả và bảng kiểm định chất lượng (quality checks).
5. `manifest.json`: Siêu dữ liệu xuất xứ đầy đủ theo chuẩn portable.
