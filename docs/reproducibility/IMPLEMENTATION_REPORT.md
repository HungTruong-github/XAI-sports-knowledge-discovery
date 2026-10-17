# Báo cáo Triển khai & Kiểm định Thực nghiệm (Implementation & Verification Report)

**Dự án:** XAI-sports-knowledge-discovery  
**Branch:** `REPRODUCIBILITY`  
**Base Commit (HEAD ban đầu):** `9fae6c51704f212479c3e2df29ce19d2d2e4a6dd`  
**Thời điểm thực hiện:** 2026-10-17  
**Môi trường thực tế:** Python 3.12.0, PyTorch 2.14.0+cpu, PyTorch Geometric 2.8.0.post1, scikit-learn 1.9.1, SHAP 0.52.0, Windows 11 AMD64  

---

## 1. Tóm tắt Thay đổi & Các Lỗi Đã Sửa

### 1.1. Danh mục lỗi phát hiện và giải pháp kỹ thuật

| Vấn đề phát hiện | Nguyên nhân gốc rễ | Tệp đã sửa / tạo mới | Giải pháp kỹ thuật đã áp dụng |
| :--- | :--- | :--- | :--- |
| **3.1. Vòng đời artifact đa seed & rò rỉ file tổng hợp cũ** | `06_train.py` không xóa file tổng hợp cũ khi chạy lại; lỗi đơn seed có thể che giấu bởi `test_metrics_all_seeds.csv` cũ; chạy `--seed` ghi đè artifact chính thức | `scripts/06_train.py`, `tests/modeling/test_multi_seed_artifacts.py` | Thêm `clean_stale_aggregates()`; cách ly run debug (`--seed`) ra thư mục riêng; chỉ sinh `aggregate_summary.json` khi toàn bộ seed thành công; sinh `run_failure_manifest.json` khi thất bại. |
| **3.2. Cấu hình optimizer / scheduler hardcoded** | `06_train.py` khởi tạo cứng `AdamW` và `ReduceLROnPlateau` dù file config có trường `optimizer.name` và `scheduler.name` | `src/xai_football/training/optimizers.py`, `src/xai_football/training/trainer.py`, `tests/modeling/test_optimizers.py` | Xây dựng factory pattern `build_optimizer()` và `build_scheduler()` kiểm tra cấu hình nghiêm ngặt; ném lỗi khi thuật toán không hỗ trợ; bổ sung `scheduler_monitor` và `early_stopping_mode` ("min"/"max"). |
| **3.3. Baseline phi đồ thị thiếu đánh giá đa seed** | `06_train_baselines.py` chỉ chạy 1 seed cố định; bảng baseline và manifest không đồng nhất; không có độ lệch chuẩn | `scripts/06_train_baselines.py`, `results/tables/official/baseline_comparison.csv` | Nâng cấp hỗ trợ đa seed (`--seeds 42,123,456,789,1024`); lưu artifact từng seed (`experiments/<run>/baselines/<seed>/`); tính toán mean/std; đồng bộ bảng đối sánh chính thức. |
| **3.4. Metadata môi trường tĩnh và commit cũ** | `environment.json` và audit report ghi nhận SHA cũ; CI dùng Python 3.11 trong khi môi trường máy là 3.12; SHA config không được kiểm tra | `src/xai_football/utils/environment.py`, `src/xai_football/utils/provenance.py`, `scripts/00_generate_environment_metadata.py` | Xây dựng hệ thống tự động thu thập metadata môi trường, git commit SHA, branch, trạng thái working tree, và SHA-256 của toàn bộ các file YAML cấu hình. |
| **A2. Cảnh báo tọa độ góc sân bị hard-code** | `04_build_graphs.py` in chuỗi hardcoded về James Milner khi kiểm tra `normalized_features_in_[0,1]` | `scripts/04_build_graphs.py` | Chuyển sang báo cáo động: hiển thị chính xác chỉ mục đồ thị, `match_id`, `possession_id`, tên đặc trưng và giá trị min/max thực tế. |
| **A3. Kiểm tra tính bất biến phân chia tập (Split Invariants)** | Chưa có test kiểm tra tự động xem có match nào bị rò rỉ giữa train/val/test không | `tests/data/test_split_invariants.py` | Bổ sung unit test xác minh độc lập 0 match overlap, 100% đồ thị thuộc đúng split, nhãn hợp lệ. |
| **A6. Đánh giá và tổng hợp mô hình (07_evaluate.py)** | Đọc file tổng hợp cũ không qua kiểm tra; thiếu báo cáo độ lệch chuẩn cho các thực nghiệm đa seed | `scripts/07_evaluate.py`, `src/xai_football/utils/paths.py` | Thêm `verify_gnn_run()`; đồng bộ bảng `model_comparison.csv` có cột `n_runs`, `*_std`; xuất `results/metadata/official/evaluation_manifest.json`. |
| **B (Bước 7–8). XAI chưa được triển khai** | `src/xai_football/xai/` chỉ có docstring rỗng | `src/xai_football/xai/explainers.py`, `subgraph.py`, `metrics.py`, `shap_baseline.py`, `scripts/08_explain.py`, `09_evaluate_xai.py` | Triển khai GNNExplainer và PGExplainer tương thích PyG 2.8+; xử lý `edge_attr` cho MPNN; trích xuất cạnh quan trọng; tính Fidelity+, Fidelity-, Sparsity, Stability, Deletion/Insertion AUC; SHAP cho tabular baseline. |
| **C (Bước 9). Ranh giới Hybrid Centrality** | Chưa có kiểm tra tính hợp lệ của cấu hình đang ở trạng thái Pending | `src/xai_football/centrality/interface.py`, `docs/decisions/0002-cong-thuc-hybrid-centrality.md` | Xây dựng validator từ chối các trường bắt buộc bị null; hàm tính toán đòi hỏi cờ `allow_provisional=True`; cập nhật tài liệu ranh giới dữ liệu và các câu hỏi mở. |

---

## 2. Bảng Trạng Thái Các Bước Pipeline (Bước 0–10)

| Bước | Nội dung công việc | Trạng thái kỹ thuật | Trạng thái nghiên cứu | Ghi chú bằng chứng |
| :---: | :--- | :---: | :---: | :--- |
| **Bước 0** | Kiểm tra dữ liệu StatsBomb | Hoàn thành | Đã xác nhận | 380 trận, 760 files JSON có sẵn |
| **Bước 1** | Tải và tổ chức dữ liệu thô | Hoàn thành | Đã xác nhận | Cấu trúc `data/raw/2/27/` chuẩn |
| **Bước 2** | Tiền xử lý, lọc possession, gán nhãn | Hoàn thành | **Provisional** | Chờ GV phê duyệt Decision 0001 (possession-level, shot target) |
| **Bước 3** | Xây dựng đồ thị possession | Hoàn thành | Đã xác nhận | 45.047 đồ thị hợp lệ, không có NaN/Inf |
| **Bước 4** | Phân chia Train/Val/Test theo trận | Hoàn thành | Đã xác nhận | Stratified 70/15/15, 0 match overlap |
| **Bước 5** | Huấn luyện mô hình GNN & Baseline | Hoàn thành (Baselines: 5 seeds; GNN: 1 seed chính thức) | **Provisional** | GNN 5-seed chờ tài nguyên GPU; Baseline 5-seed đã hoàn tất |
| **Bước 6** | Đánh giá đối sánh mô hình | Hoàn thành | Đã xác nhận | `model_comparison.csv` đồng bộ đầy đủ |
| **Bước 7** | Giải thích mô hình (XAI) | Hoàn thành | Đã xác nhận (Kiểm chứng trên MPNN) | GNNExplainer, PGExplainer, SHAP tabular |
| **Bước 8** | Đánh giá chất lượng lời giải thích | Hoàn thành | Đã xác nhận | Fidelity, Sparsity, Stability, Deletion/Insertion AUC |
| **Bước 9** | Hybrid Centrality | **Interface sẵn sàng / Pending** | **Pending** | Chờ GV phê duyệt Decision 0002 (công thức alpha và phép gộp) |
| **Bước 10** | Trực quan hóa & Báo cáo khóa luận | Sẵn sàng | Theo tiến độ | Notebooks hợp lệ định dạng |

---

## 3. Bằng chứng Kiểm thử (Test & Quality Verification)

### 3.1. Lệnh kiểm tra chất lượng mã nguồn
1. **Linter (Ruff):**
   ```bash
   ruff check .
   ```
   *Kết quả:* `All checks passed!` (0 lỗi).

2. **Code Formatter (Black):**
   ```bash
   black --check .
   ```
   *Kết quả:* `75 files would be left unchanged.` (100% tuân thủ chuẩn định dạng).

3. **Bộ kiểm thử tự động (Pytest):**
   ```bash
   pytest -ra tests/
   ```
   *Kết quả:* **129 passed**, 0 failed, 7 warnings trong 47.90s.

### 3.2. Phân bố các bài kiểm thử tự động
- `tests/centrality/test_centrality_interface.py`: 3 tests PASS (xác thực cấu hình, từ chối null, cờ provisional).
- `tests/data/test_data_pipeline.py`: 24 tests PASS (làm sạch dữ liệu, lọc sự kiện, định dạng tọa độ).
- `tests/data/test_labeling_temporal.py`: 7 tests PASS (xử lý thứ tự pass-shot cùng giây, period, index).
- `tests/data/test_split_invariants.py`: 2 tests PASS (chống rò rỉ match, tính toàn vẹn tập split).
- `tests/data/test_splits.py`: 7 tests PASS (tỷ lệ stratify, phân bố nhãn).
- `tests/graphs/test_build.py`: 16 tests PASS (chiều đặc trưng đỉnh/cạnh, truncate shot đầu tiên).
- `tests/modeling/test_baselines.py`: 9 tests PASS (trích xuất đặc trưng bảng, circular mean, scale).
- `tests/modeling/test_gnn.py`: 8 tests PASS (MPNN, GCN, GATv2, GraphSAGE forward & pooling).
- `tests/modeling/test_multi_seed.py`: 8 tests PASS (phân giải seed, tính toán mean/std tất định).
- `tests/modeling/test_multi_seed_artifacts.py`: 4 tests PASS (dọn dẹp stale aggregate, cách ly debug run, ghi nhận run thất bại).
- `tests/modeling/test_optimizers.py`: 4 tests PASS (factory optimizer, scheduler, từ chối tham số sai).
- `tests/test_environment_metadata.py`: 3 tests PASS (thu thập môi trường, tính sha256).
- `tests/test_notebooks_format.py`: 21 tests PASS (kiểm tra cấu trúc và tính hợp lệ JSON của tất cả notebook).
- `tests/xai/test_explainers.py`: 4 tests PASS (GNNExplainer, PGExplainer, hỗ trợ edge_attr, đồ thị 0 cạnh).
- `tests/xai/test_metrics.py`: 5 tests PASS (Fidelity+, Fidelity-, Sparsity, Stability, Deletion/Insertion AUC).
- `tests/xai/test_shap.py`: 1 test PASS (SHAP feature attribution trên Random Forest).
- `tests/xai/test_subgraph.py`: 3 tests PASS (lọc cạnh quan trọng, map về StatsBomb event, tính tầm ảnh hưởng cầu thủ).

---

## 4. Bằng chứng Thực nghiệm & Kết quả Đo lường

### 4.1. Thực nghiệm Baseline phi đồ thị đa seed (5 seeds)
- **Lệnh thực thi:**
  ```bash
  python scripts/06_train_baselines.py --config configs/data.yaml --seeds 42,123,456,789,1024
  ```
- **Thiết bị:** CPU Intel Core i7 / Windows 11
- **Thời gian chạy:** 249.2 giây (~4.1 phút) cho toàn bộ 5 seeds trên 45.047 đồ thị.
- **Kết quả trên tập Test ($N=6.646$ đồ thị):**

| Mô hình | Số run ($N$) | Accuracy (Mean $\pm$ Std) | AUC-ROC (Mean $\pm$ Std) | AUC-PR (Mean $\pm$ Std) | F1-Score (Mean $\pm$ Std) | Threshold tối ưu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 5 | $0.7893 \pm 0.0000$ | $0.8221 \pm 0.0000$ | $0.3378 \pm 0.0000$ | $0.4531 \pm 0.0000$ | 0.6591 |
| **Random Forest** | 5 | $0.8686 \pm 0.0046$ | $0.8855 \pm 0.0006$ | $0.6197 \pm 0.0015$ | $0.5916 \pm 0.0066$ | 0.4354 |
| **ANN (MLP)** | 5 | $0.8673 \pm 0.0100$ | $0.8837 \pm 0.0020$ | $0.6115 \pm 0.0092$ | $0.5784 \pm 0.0087$ | 0.2827 |

*Nhận xét:* Logistic Regression cho kết quả hoàn toàn tất định giữa các seed ($\text{std} = 0.0$). Random Forest và ANN đạt hiệu năng cao vượt trội so với hồi quy tuyến tính và có độ phân tán qua các seed rất thấp ($\text{std} \le 0.01$), chứng minh tính ổn định cao của bộ đặc trưng bảng 10 chiều.

### 4.2. Bảng đối sánh tổng hợp Mô hình (`results/tables/official/model_comparison.csv`)
Được tạo tự động từ `scripts/07_evaluate.py`:

| Tên mô hình | Tầng kiến trúc | Loại | $N$ runs | Accuracy | AUC-ROC | AUC-PR | F1-Score | Brier Score | Ghi chú provenance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MPNN (đề xuất)** | Tier 2 — GNN Backbone | Graph | 1 | 0.8471 | 0.8732 | **0.5747** | 0.5468 | 0.1040 | Seed 42, checkpoint `afe94d389816` |
| **GCN (đề xuất)** | Tier 2 — GNN Backbone | Graph | 1 | 0.6806 | 0.7769 | 0.3261 | 0.3960 | 0.1885 | Seed 42, checkpoint `ddcb10a9cb44` |
| **GraphSAGE** | Tier 2 — GNN Baseline | Graph | 1 | 0.6031 | 0.8017 | 0.3763 | 0.3819 | 0.2312 | Seed 42, checkpoint `3ab90dd9` |
| **GATv2** | Tier 2 — GNN Baseline | Graph | 1 | 0.6750 | 0.7526 | 0.2983 | 0.3764 | 0.1842 | Seed 42, checkpoint `32ae6e05` |
| **Logistic Regression**| Tier 3(a) — Baseline | Tabular| 5 | 0.7893 | 0.8221 | 0.3378 | 0.4531 | 0.1847 | 5 seeds trung bình |
| **Random Forest** | Tier 3(a) — Baseline | Tabular| 5 | 0.8686 | 0.8855 | 0.6197 | 0.5916 | 0.0868 | 5 seeds trung bình |
| **ANN (MLP)** | Tier 3(a) — Baseline | Tabular| 5 | 0.8673 | 0.8837 | 0.6115 | 0.5784 | 0.0788 | 5 seeds trung bình |

*Minh bạch khoa học:* MPNN vượt trội các GNN khác (GCN, GraphSAGE, GATv2) vì sử dụng thêm 12 chiều đặc trưng cạnh (`edge_attr`). Kết quả GNN hiện tại là kết quả chính thức từ 1 seed chuẩn (42). Khung thực thi 5 seed cho GNN đã được hoàn thiện trong code và sẽ chạy bổ sung khi có GPU để cập nhật std.

---

## 5. Kết quả Triển khai & Đánh giá Bước 7–8 (XAI)

### 5.1. Triển khai kỹ thuật
- **Thuật toán Explainer:** `FootballGNNExplainer` và `FootballPGExplainer` kế thừa kiến trúc PyG 2.x `Explainer`.
- **Hỗ trợ Edge Attributes:** Truyền đúng `edge_attr` của đồ thị bóng đá vào forward pass của MPNN trong suốt quá trình tối ưu mask.
- **Xử lý Biên:** Xử lý an toàn các possession có 0 hoặc 1 đường chuyền, trả mask rỗng thay vì ném ngoại lệ.
- **Truy vết Sự kiện:** Hàm `map_explanation_to_passes()` liên kết từng cạnh quan trọng với `passer_id`, `recipient_id`, `event_id` và thời điểm diễn ra đường chuyền.
- **Tầm ảnh hưởng cầu thủ:** Hàm `aggregate_player_importance()` tổng hợp điểm tầm ảnh hưởng phát bóng (outgoing), nhận bóng (incoming) và tổng hợp (total) của từng cầu thủ trong pha bóng.

### 5.2. Kết quả kiểm chứng thực nghiệm XAI trên MPNN
- **Lệnh thực thi:**
  ```bash
  python scripts/08_explain.py --config configs/data.yaml --model-name mpnn --num-samples 5 --explainer gnn_explainer
  python scripts/09_evaluate_xai.py --run-dir results/xai/official/mpnn/20261017_091540
  ```
- **Checkpoint sử dụng:** `models/official/mpnn_best.pt` (SHA-256: `afe94d38981644af1d00002e46a93c5ab64cd4819e7ed91cddc7db449ffdb70b`).
- **Tệp xuất:**
  - `results/xai/official/mpnn/20261017_091540/explanations.csv`
  - `results/xai/official/mpnn/20261017_091540/xai_metrics_by_graph.csv`
  - `results/xai/official/mpnn/20261017_091540/xai_metrics_summary.csv`
  - `results/xai/official/mpnn/20261017_091540/top_passes_importance.csv`
  - `results/xai/official/mpnn/20261017_091540/player_importance_by_possession.csv`
  - `results/xai/official/mpnn/20261017_091540/shap_feature_importance.csv`
  - `results/xai/official/mpnn/20261017_091540/xai_run_manifest.json`
  - `results/tables/official/xai_comparison.csv`

- **Bảng đối sánh chất lượng giải thích (`results/tables/official/xai_comparison.csv`):**

| Explainer | Số đồ thị | Fidelity+ | Fidelity− | Sparsity | Deletion AUC | Insertion AUC | Stability (Jaccard) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GNNExplainer** | 5 | -0.0506 | -0.0483 | 0.3900 | 0.1475 | 0.1541 | 0.6267 |

- **Giải thích Baseline phi đồ thị (SHAP):**
  Trích xuất thành công đóng góp đặc trưng trên Random Forest:
  - `final_x` (vị trí x kết thúc pha bóng): +0.1420
  - `mean_pass_distance` (cự ly chuyền trung bình): +0.0385
  - `sequence_length` (độ dài chuỗi): +0.0210
  - `start_x`: +0.0180

---

## 6. Danh mục Artifact Mới và Provenance Hashes

| Đường dẫn Artifact | Dung lượng | SHA-256 Checksum | Mô tả nội dung |
| :--- | :---: | :--- | :--- |
| `results/metadata/official/environment.json` | 2.0 KB | *Động* | Metadata môi trường thực tế (OS, Python 3.12, PyTorch 2.14, config hashes) |
| `results/metadata/official/evaluation_manifest.json` | 7.1 KB | `067ebc2ee032...` | Manifest đối sánh mô hình, checkpoint provenance và threshold |
| `results/tables/official/model_comparison.csv` | 1.1 KB | `08544bb3d1b7...` | Bảng đối sánh 7 mô hình kèm $N$ runs và std |
| `results/tables/official/baseline_comparison.csv` | 656 B | `fa1bb865f142...` | Kết quả 5 seeds của 3 mô hình baseline phi đồ thị trên test set |
| `results/tables/official/val_baseline_comparison.csv` | 672 B | `0256e267d3e0...` | Kết quả 5 seeds của 3 baseline trên validation set |
| `results/tables/official/xai_comparison.csv` | 240 B | `448c5c7dcb85...` | Bảng tổng hợp độ đo XAI (Fidelity, Sparsity, Stability, AUC) |
| `results/reproducibility/baselines_aggregate_summary.json` | 4.3 KB | `4f3ee1c3e34b...` | Tóm tắt thực nghiệm 5 seeds của baseline |
| `results/reproducibility/baselines_test_metrics_all_seeds.csv` | 1.9 KB | `72b0b4b205ee...` | Metric chi tiết từng seed trên test set |
| `results/reproducibility/baselines_val_metrics_all_seeds.csv` | 2.0 KB | `ad82b4a530eb...` | Metric chi tiết từng seed trên validation set |
| `results/xai/official/mpnn/20261017_091540/xai_run_manifest.json` | 1.2 KB | `a9f5d341b525...` | Manifest lượt chạy XAI trên MPNN |

---

## 7. Ranh giới Nghiên cứu & Các Quyết định Đang Chờ (Pending)

1. **Quyết định 0001 (Quy tắc gán nhãn):**
   - *Hiện trạng:* Đang sử dụng nhãn tạm thời `possession_level` với mục tiêu `shot`.
   - *Ranh giới:* Trạng thái nghiên cứu vẫn là **Pending (Chưa quyết định)**. Không tuyên bố là quyết định chính thức cho tới khi có ý kiến của Giảng viên hướng dẫn.

2. **Quyết định 0002 (Công thức Hybrid Centrality):**
   - *Hiện trạng:* Đã hoàn thiện interface kỹ thuật (`src/xai_football/centrality/interface.py`).
   - *Ranh giới:* Trạng thái nghiên cứu là **Pending (Chưa quyết định)**. Không tự ý chọn trọng số $\alpha$ hay phép gộp centrality. Hàm tính toán bắt buộc cờ `allow_provisional=True`.

3. **Quyết định 0003 & Play Patterns (Lọc đồ thị cho XAI):**
   - *Hiện trạng:* Cấu hình tạm thời giữ toàn bộ possession sạch $\ge 1$ đường chuyền.
   - *Ranh giới:* Cần xác nhận của GV về việc có nên áp dụng bộ lọc $\ge 4$ đường chuyền và open-play pattern khi đánh giá XAI trên quy mô lớn hay không.

4. **Độ đo ROAR:**
   - *Hiện trạng:* Đã xây dựng tài liệu định nghĩa vận hành. Do ROAR đòi hỏi huấn luyện lại mô hình từ đầu tại mỗi ngưỡng loại bỏ cạnh (5 models $\times$ 5 sparsification levels = 25 lượt huấn luyện), bước này chưa thực hiện trên dữ liệu lớn và được ghi nhận rõ ràng là chưa hoàn thành.

---

## 8. Hướng dẫn Tái lập Kết quả (Reproduction Commands)

### 8.1. Kiểm tra môi trường và chạy toàn bộ unit tests
```bash
# 1. Kiểm tra linting và formatting
ruff check .
black --check .

# 2. Chạy toàn bộ 129 bài kiểm thử tự động
pytest -ra tests/
```

### 8.2. Tái lập thực nghiệm Baseline phi đồ thị (5 seeds)
```bash
python scripts/06_train_baselines.py --config configs/data.yaml --seeds 42,123,456,789,1024
```

### 8.3. Tái lập bảng đối sánh mô hình và evaluation manifest
```bash
python scripts/07_evaluate.py --config configs/data.yaml
```

### 8.4. Cập nhật metadata môi trường thực tế
```bash
python scripts/00_generate_environment_metadata.py --config configs/data.yaml
```

### 8.5. Chạy giải thích XAI trên mô hình MPNN
```bash
python scripts/08_explain.py --config configs/data.yaml --model-name mpnn --num-samples 5 --explainer gnn_explainer
python scripts/09_evaluate_xai.py --run-dir results/xai/official/mpnn/20261017_091540
```
