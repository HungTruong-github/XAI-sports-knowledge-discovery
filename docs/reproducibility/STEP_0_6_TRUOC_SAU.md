# Rà soát & hoàn thiện Bước 0–6 — Đối chiếu TRƯỚC / SAU

> [!NOTE]
> **TÀI LIỆU LỊCH SỬ (HISTORICAL AUDIT LOG — 21–22/10/2026):**
> Tài liệu này ghi lại nhật ký đối chiếu kỹ thuật đợt rà soát đầu tiên (commit `8d82bf8`). Trạng thái "chưa commit" tại dòng 8 phản ánh thời điểm trước khi commit `8d82bf8`.
> Báo cáo kiểm toán chính thức, toàn diện và có hiệu lực cho nhánh `Clean` hiện tại được lưu trữ tại:
> [`docs/reproducibility/PIPELINE_0_6_COMPLETION_AUDIT.md`](file:///d:/Project/XAI-sports-knowledge-discovery/docs/reproducibility/PIPELINE_0_6_COMPLETION_AUDIT.md)
> và hồ sơ máy đọc được tại [`results/metadata/official/pipeline_audit.json`](file:///d:/Project/XAI-sports-knowledge-discovery/results/metadata/official/pipeline_audit.json).

- **Ngày thực hiện:** 21–22/10/2026
- **Phạm vi:** chỉ Bước 0–6 (môi trường → thu thập → tiền xử lý → đồ thị → chia tập → huấn luyện → đánh giá).
  Bước 7–10 (XAI, Hybrid Centrality, demo) **không bị sửa**.
- **Commit gốc khi bắt đầu:** `fc18a92` (nhánh `main`, working tree sạch).
- **Commit kết quả đợt 1:** `8d82bf8` (đã commit và đẩy lên nhánh `Clean`).
- **Môi trường chạy lại:** Windows 11, Python 3.13.0 (`.venv`), CPU; phiên bản thư viện khóa trong `requirements.lock.txt`.

Mỗi mục bên dưới có: **Trước** (hiện trạng + bằng chứng), **Sau** (đã đổi gì), **File**, và **Cách hoàn tác**
(để dễ sửa lại nếu giảng viên/nhóm không đồng ý).

---

## 0. Tóm tắt 1 phút

| # | Vấn đề tìm thấy | Mức độ | Đã xử lý |
|---|---|---|---|
| 1 | 4 mô hình GNN chỉ chạy **1 seed** (baseline chạy 5 seed) → so sánh không công bằng, không có độ lệch chuẩn | Nặng | Chạy lại đủ **5 seed** cho cả 4 GNN |
| 2 | Số validation của GCN trong `experiment_manifest.json` là số của **smoke test** (best_epoch=1, AUC-ROC 0,46, recall 1,0); file này **viết tay**, không script nào sinh ra | Nặng | `07_evaluate.py` nay **tự sinh** manifest từ kết quả thật |
| 3 | `07_evaluate.py` có cờ `--allow-partial` nhưng **không dùng**; run thiếu seed vẫn được đưa vào bảng chính thức | Nặng | Run thiếu seed bị **từ chối** (mã lỗi 1) trừ khi bật `--allow-partial` |
| 4 | Không có **kiểm định thống kê** khi kết luận mô hình A hơn B | Trung bình | Thêm paired cluster bootstrap theo trận + hiệu chỉnh Holm |
| 5 | File dự đoán không có khóa possession; baseline chỉ lưu dự đoán của seed đầu | Trung bình | Lưu dự đoán **mọi seed**, kèm `match_id`, `possession_id` |
| 6 | Notebook 00–06 **không còn output**; phần chữ ghi sai tên file, nhắc cờ không tồn tại, mô tả đặc trưng baseline **không có trong code**; notebook 06 tính AUC-PR khác công thức của bảng; có `assert` ép "MPNN phải tốt nhất" | Trung bình | Sửa nội dung, chạy lại và lưu output |
| 7 | 1 tọa độ phạt góc x = 120,9 làm đặc trưng chuẩn hóa vượt 1 (WARNING) | Nhẹ | Kẹp tọa độ về khung sân, có đếm số lượng |
| 8 | Phân tầng Bước 4 luôn dùng cột `has_shot`, bỏ qua `labeling.target` | Nhẹ (chưa gây sai vì target = shot) | Đọc đúng target trong config |
| 9 | Đường dẫn `data/splits`, `experiments/`, `models/` tính theo **thư mục đang đứng** | Nhẹ | Tính từ gốc repo |
| 10 | Lỗi đọc trận ở pha trích possession bị **nuốt** trong run chính thức | Nhẹ | Dừng với mã lỗi 1 |
| 11 | `train.yaml` ghi `loss: null # TODO` dù code vẫn chạy Weighted BCE; `pos_weight` thực tế **không được ghi lại** | Nhẹ | Ghi tường minh; lưu `pos_weight` vào manifest |
| 12 | `requirements.txt` không khóa phiên bản | Nhẹ | Thêm `requirements.lock.txt` |

---

## 1. Bước 0 — Môi trường & tái lập

### 1.1 Khóa phiên bản thư viện
- **Trước:** `requirements.txt` không ghi phiên bản; chính file này khuyên "trước khi nộp hãy `pip freeze`" nhưng chưa làm.
  `environment.json` cũ ghi Python 3.12, máy hiện tại không có môi trường nào cài sẵn
  (chạy `pytest` báo `ModuleNotFoundError: shap`, `xai_football`).
- **Sau:** tạo `.venv` riêng cho dự án; thêm `requirements.lock.txt` (pip freeze, 190 gói);
  `environment.json` ghi thêm `requirements_lock_sha256`.
- **File:** `requirements.lock.txt` (mới), `src/xai_football/utils/environment.py`.
- **Hoàn tác:** xóa `requirements.lock.txt` và dòng `requirements_lock_sha256` trong `environment.py`.

### 1.2 Khóa đường dẫn trong `config_hashes`
- **Trước:** khóa có dạng `configs\\data.yaml` (dấu `\` của Windows) → không so sánh được với CI Linux.
- **Sau:** dùng `Path.as_posix()` → `configs/data.yaml`. Có test `test_config_hash_keys_use_posix_separator`.
- **File:** `src/xai_football/utils/environment.py`, `tests/test_environment_metadata.py`.

### 1.3 Notebook 00 tự viết `environment.json` theo cách riêng
- **Trước:** notebook tự thu thập thông tin (bỏ sót `shap`, ghi `NOT_INSTALLED` thay vì `not installed`)
  rồi **ghi đè** file do script sinh → hai nguồn khác định dạng.
- **Sau:** notebook gọi chung `collect_environment_metadata()` với `scripts/00_generate_environment_metadata.py`;
  kiểm kê thêm checkpoint của cả 4 GNN, bảng kiểm định, file lock.

---

## 2. Bước 1 — Thu thập dữ liệu

- **Trước:** code đúng. Máy hiện tại **không có dữ liệu** (`data/raw` trống).
- **Sau:** không sửa code. Đã tải lại toàn bộ: **380/380 trận, 1.313.773 sự kiện, 0 trận lỗi** — trùng khớp
  hoàn toàn `collection_report.csv` cũ. `dataset_inventory.csv`: 380 trận, 20 đội, 08/08/2015 → 17/05/2016.
- Notebook 01: sửa tên file sai trong phần chữ (`collection_summary.csv`, `collection_checks.csv` không tồn tại →
  `collection_report.csv`, `dataset_inventory.csv`); đếm file raw **không lẫn** file `*.lineups.json`
  (trước đây đếm ra 760 thay vì 380); thêm kiểm tra "số file trên đĩa = số trận đã tải".

---

## 3. Bước 2 — Tiền xử lý & gán nhãn

### 3.1 Lỗi đọc trận bị nuốt
- **Trước** (`scripts/03_preprocess.py`, pha 2):
  ```python
  except Exception as exc:
      logger.error("Trận %s: lỗi load/clean possession — %s", mid, exc)   # rồi chạy tiếp
  ```
- **Sau:** ghi lại `failed_possession_match_ids`; run `official` gặp lỗi → `return 1` (giống pha canonicalize).

### 3.2 Cấu hình "ma" trong `data.yaml`
- **Trước:** `event_types_keep` và `impute_coordinates` có trong config nhưng **code không đọc**.
- **Sau:** ghi chú rõ `event_types_keep` chỉ mang tính mô tả; `impute_coordinates: true` nay **báo lỗi**
  (`NotImplementedError`) thay vì im lặng bỏ qua. Giá trị hiện tại (`false`) không đổi hành vi.
- **File:** `configs/data.yaml`, `src/xai_football/data/clean.py`.

### 3.3 Kết quả chạy lại
Toàn bộ bảng Bước 2 **trùng khớp từng giá trị** với bản cũ (so bằng `diff`, bỏ qua khác biệt ký tự xuống dòng
CRLF/LF do Windows ghi file):
`data_pipeline_summary.csv`, `cleaning_report.csv`, `label_comparison.csv`, `sanity_checks.csv` (16/16 PASS),
`play_pattern_distribution.csv`, `descriptive_stats.csv`.
45.365 possession sạch, 261.205 đường chuyền, tỉ lệ có sút 14,26%.

> **Vẫn chờ GV:** Decision 0001 (quy tắc gán nhãn) — pipeline vẫn dùng **tạm** `possession_level / shot`.

---

## 4. Bước 3 — Dựng đồ thị

### 4.1 Tọa độ ngoài sân
- **Trước:** 1 đường chuyền phạt góc (trận 3754087, possession 46, thuộc tập **train**) có `start_x = 120.9 > 120`
  → `start_x` chuẩn hóa = 1,0075; kiểm tra `normalized_features_in_[0,1]` = **WARNING**.
- **Sau:** `features.clip_to_pitch()` kẹp tọa độ về khung 120×80 **trước** khi chuẩn hóa; độ dài/góc tính từ tọa độ
  đã kẹp. `graph_build_report.csv` có cột mới `coords_clipped` (= 1). Kiểm tra `[0,1]` nay **PASS**, và nếu sau này
  có giá trị ngoài [0,1] thì là lỗi code → **FAIL** (không còn WARNING).
- **File:** `src/xai_football/graphs/features.py`, `graphs/dataset.py`, `scripts/04_build_graphs.py`,
  test `test_corner_kick_coordinate_is_clipped_to_pitch` (thay test cũ khẳng định giá trị 1,0075).
- **Hoàn tác:** trong `features.py` bỏ các lời gọi `clip_to_pitch(...)` (dùng lại `p["start_x"]`…); trong
  `04_build_graphs.py` đổi `"FAIL"` về `"WARNING"` và xóa check `coords_clipped_to_pitch`.

### 4.2 Kết quả chạy lại
`graph_build_report.csv` và `graph_checks.csv` giống bản cũ **trừ đúng** phần chủ động thay đổi ở trên:
45.047 đồ thị, 6.153 dương, 318 possession bị loại sau khi cắt tại cú sút, 0 cạnh sau cú sút, 9/9 kiểm tra PASS.

Notebook 03: hình minh họa trước đây vẽ **cả các đường chuyền sau cú sút** (không cắt như đồ thị thật) và chỉ vẽ
vị trí người chuyền; nay vẽ đúng các cạnh của đồ thị (có `assert` số cạnh = `n_edges`) và vị trí đỉnh = trung bình
mọi lần chạm bóng (khớp đặc trưng `mean_x/mean_y`). Mục 7.2 viết lại theo cơ chế kẹp tọa độ.

---

## 5. Bước 4 — Chia tập theo trận

- **Trước:** `compute_match_stats()` luôn phân tầng theo `has_shot`; nếu đổi `labeling.target: goal` thì phân tầng
  vẫn theo sút → sai nhãn. Thư mục split lấy `Path("data/splits")` theo thư mục đang đứng.
- **Sau:** `label_column_for(data_cfg)` chọn `has_shot` / `has_goal` theo config; `get_split_dir()` tính từ gốc repo
  (dùng chung cho 05, 06_train, 06_train_baselines).
- **File:** `src/xai_football/data/splits.py`, `src/xai_football/utils/paths.py`, `scripts/05_make_splits.py`,
  3 test mới trong `tests/data/test_splits.py`.
- **Kết quả:** danh sách trận `train/val/test_matches.csv`, `match_splits.csv`, `split_summary.csv`,
  `split_checks.csv`, `graph_split_summary.csv` **trùng khớp hoàn toàn** bản cũ (266/57/57 trận, 0 trận trùng).

---

## 6. Bước 5 — Huấn luyện

### 6.1 Chỉ 1 seed cho GNN
- **Trước:** `model_comparison.csv` cũ: GCN/MPNN/GATv2/GraphSAGE `n_runs = 1`, mọi cột `*_std = 0`;
  baseline `n_runs = 5`. `configs/train.yaml` đã cam kết `n_runs: 5`.
- **Sau:** chạy `scripts/06_train.py --seeds 42,123,456,789,1024` cho cả 4 GNN (không đổi kiến trúc, không đổi
  siêu tham số). Checkpoint `models/official/<model>_best.pt` = seed có **Val** AUC-PR cao nhất.

### 6.2 Artifact thiếu cho kiểm định / truy vết
- **Trước:** `test_predictions.parquet` chỉ có `y_true, y_prob` (không biết dòng nào là possession nào);
  không lưu dự đoán validation; `run_manifest.json` không ghi `pos_weight` thực tế; `aggregate_summary.json`
  không ghi seed được chọn; baseline chỉ lưu dự đoán của seed đầu.
- **Sau:** `trainer.prediction_frame()` lưu `match_id, possession_id, y_true, y_prob` (có kiểm tra thứ tự khớp nhãn)
  cho **test và val**; manifest có `pos_weight`, `n_parameters`, số đồ thị từng tập, `epochs_run`;
  `aggregate_summary.json` có `best_seed_by_val_auc_pr`; baseline lưu `test/val_predictions.parquet` cho **mọi seed**.
- **File:** `src/xai_football/training/trainer.py`, `scripts/06_train.py`, `scripts/06_train_baselines.py`,
  test `tests/modeling/test_trainer_outputs.py`.

### 6.3 Cấu hình ghi chưa đúng thực tế
- **Trước:** `train.yaml` → `loss.name: null # TODO`, `pos_weight: null # TODO`; `mpnn.yaml` → `edge_dim: null # TODO`;
  `06_train.py` mặc định cứng `edge_in_dim = 12` nếu thiếu `edge_attr`.
- **Sau:** `loss.name: "weighted_bce"` (đúng hành vi code cũ — **không đổi kết quả**), ghi chú `pos_weight: null` =
  tự tính trên train; `edge_dim` ghi chú lấy từ dữ liệu; thiếu `edge_attr` → báo lỗi thay vì đoán 12.

### 6.4 Baseline chạy lại
Logistic Regression trùng khớp tuyệt đối bản cũ. Random Forest và ANN lệch ở chữ số thứ 3–4
(RF test AUC-PR 0,6197 → 0,6198; ANN 0,6115 → 0,6117). **Đã chứng minh nguyên nhân:** đặt lại đúng 1 tọa độ
chưa kẹp (mục 4.1, mẫu thuộc tập train) rồi chạy lại seed 42 → cả 3 baseline cho số **trùng khớp tuyệt đối** với
bản cũ (LR 0,3378; RF 0,6188; ANN 0,6187). Vậy chênh lệch chỉ do thay đổi có chủ đích ở Bước 3.

---

## 7. Bước 6 — Đánh giá

### 7.1 `scripts/07_evaluate.py` viết lại
| | Trước | Sau |
|---|---|---|
| Run thiếu seed | vẫn đưa vào bảng (fallback đọc `test_metrics.csv` 1 seed) | từ chối, `return 1`; `--allow-partial` mới cho qua (cột `complete = False`) |
| Bảng validation | không có cho GNN | `val_model_comparison.csv` |
| Kiểm định | không có | `model_metric_ci.csv` (KTC 95%), `significance_tests.csv` (21 cặp × 3 độ đo, Holm) |
| `experiment_manifest.json` | **viết tay**, chứa số smoke test của GCN | sinh tự động từ kết quả + SHA cấu hình/checkpoint |
| `results/reproducibility/` | chép tay, chỉ baseline | sinh tự động, thêm `all_models_{test,val}_metrics_all_seeds.csv` |

### 7.2 Phương pháp kiểm định (`src/xai_football/evaluation/significance.py`, mới)
> **Nguồn gốc:** tài liệu dự án chỉ yêu cầu chung "khoảng tin cậy bootstrap cho mọi chỉ số" để tránh diễn giải nhiễu
> như khác biệt thật (`docs/Rui_ro_va_phuong_an_XAI_Football.md`, rủi ro E3/E5). **Thiết kế cụ thể bên dưới (lấy mẫu
> theo trận, ghép cặp, Holm, số vòng) là lựa chọn bổ sung trong đợt rà soát này** — cần nhóm/GV duyệt. Số vòng chính
> thức **10.000** do nhóm chốt ngày 22/10/2026.

Paired cluster bootstrap theo **trận** (10.000 vòng, seed 42): lấy mẫu có hoàn lại 57 trận test; trên mỗi mẫu tính
độ đo của **từng seed** rồi lấy trung bình qua seed cho từng mô hình; chênh lệch tính trên **cùng** mẫu. KTC 95%
theo phân vị; p-value hai phía; hiệu chỉnh Holm theo từng độ đo. Lấy mẫu theo trận (không theo possession) vì các
possession cùng trận tương quan — cùng lý do Bước 4 chia theo trận. Có 10 unit test (`tests/evaluation/test_significance.py`).
Mẫu bootstrap được sinh dần từng vòng (không giữ cả 10.000 mẫu trong RAM).

### 7.3 Notebook 05–06
- 05: bỏ câu nhắc cờ `RUN_SMOKE_TEST` (không tồn tại); đọc **cả 5 seed** (trước lấy `glob(...)[0]` tùy ý);
  đường cong loss của MPNN dùng seed tốt nhất theo validation; kiểm tra đủ 5 seed.
- 06: AUC-PR trên hình tính bằng `average_precision_score` (trước dùng `auc(recall, precision)` → số trên hình
  **lệch** số trong bảng); vẽ đủ 7 mô hình; thanh lỗi ± std; thêm bảng KTC và kiểm định; **xóa** câu khẳng định
  baseline dùng `end_dist_to_goal`, `directness`, `territory_gained` (code **không có** các đặc trưng này — thực tế là
  10 đặc trưng trong `configs/model/baseline_tabular.yaml`); **xóa** `assert` "MPNN là best GNN" (notebook không được
  ép kết quả theo giả thuyết).

---

## 8. Kết quả TRƯỚC / SAU (tập test)

Nguồn: TRƯỚC = `results/tables/official/model_comparison.csv` ở commit `fc18a92`;
SAU = cùng file sau khi chạy lại (`scripts/07_evaluate.py`). Ngưỡng của F1 chọn trên validation.

| Mô hình | Seed (trước) | AUC-PR | AUC-ROC | F1 | Brier | Seed (sau) | AUC-PR | AUC-ROC | F1 | Brier | Val AUC-PR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GCN | 1 | 0.3261 | 0.7769 | 0.3960 | 0.1885 | 5 | 0.4526 ± 0.0048 | 0.8310 ± 0.0013 | 0.4722 ± 0.0065 | 0.1688 ± 0.0058 | 0.4168 |
| **MPNN** | 1 | 0.5747 | 0.8732 | 0.5468 | 0.1040 | 5 | **0.5996 ± 0.0042** | 0.8757 ± 0.0034 | 0.5762 ± 0.0036 | 0.1302 ± 0.0114 | 0.5847 |
| GATv2 | 1 | 0.2983 | 0.7526 | 0.3764 | 0.1842 | 5 | 0.4620 ± 0.0048 | 0.8407 ± 0.0014 | 0.4934 ± 0.0048 | 0.1534 ± 0.0070 | 0.4373 |
| GraphSAGE | 1 | 0.3763 | 0.8017 | 0.3819 | 0.2312 | 5 | 0.5159 ± 0.0022 | 0.8504 ± 0.0009 | 0.5139 ± 0.0066 | 0.1460 ± 0.0049 | 0.4903 |
| Logistic Regression | 5 | 0.3378 | 0.8221 | 0.4531 | 0.1847 | 5 | 0.3378 ± 0.0000 | 0.8221 ± 0.0000 | 0.4531 ± 0.0000 | 0.1847 ± 0.0000 | 0.3517 |
| Random Forest | 5 | 0.6197 | 0.8855 | 0.5916 | 0.0868 | 5 | 0.6198 ± 0.0017 | 0.8855 ± 0.0006 | 0.5892 ± 0.0047 | 0.0868 ± 0.0002 | 0.6079 |
| ANN (MLP) | 5 | 0.6115 | 0.8837 | 0.5784 | 0.0788 | 5 | 0.6117 ± 0.0097 | 0.8839 ± 0.0018 | 0.5822 ± 0.0099 | 0.0787 ± 0.0013 | 0.6087 |

**Bằng chứng số cũ của GCN bị lẫn smoke test:** `experiment_manifest.json` cũ ghi validation của GCN:
`best_epoch = 1, AUC-ROC = 0.4643, AUC-PR = 0.159, recall = 1.0` (kém hơn đoán ngẫu nhiên). Chạy lại 5 seed:
Val AUC-PR trung bình 0.4168, epoch tốt nhất 72–127. GATv2 và GraphSAGE cũng tăng mạnh (+0.16 và +0.14 AUC-PR),
nhất quán với việc các số 1-seed cũ không đáng tin.

**Checkpoint chính thức mới** (`models/official/`, chọn theo Val AUC-PR): GCN seed 789, MPNN seed 789,
GATv2 seed 1024, GraphSAGE seed 1024 (SHA-256 ghi trong `results/metadata/official/experiment_manifest.json`).

### Kiểm định (10.000 vòng bootstrap theo trận, 57 trận test, Holm) — các cặp có MPNN

| So sánh | Độ đo | Chênh lệch | KTC 95% | p thô | p (Holm) | Kết luận |
|---|---|---|---|---|---|---|
| MPNN − GraphSAGE | AUC-PR | +0.0836 | [+0.0645, +0.1035] | 0.0002 | 0.004 | MPNN tốt hơn |
| MPNN − GATv2 | AUC-PR | +0.1375 | [+0.1132, +0.1617] | 0.0002 | 0.004 | MPNN tốt hơn |
| MPNN − GCN | AUC-PR | +0.1470 | [+0.1244, +0.1698] | 0.0002 | 0.004 | MPNN tốt hơn |
| MPNN − Random Forest | AUC-PR | −0.0203 | [−0.0418, +0.0016] | 0.068 | 0.273 | **không khác biệt có ý nghĩa** |
| MPNN − ANN | AUC-PR | −0.0121 | [−0.0264, +0.0018] | 0.089 | 0.273 | **không khác biệt có ý nghĩa** |
| MPNN − Random Forest | AUC-ROC | −0.0099 | [−0.0181, −0.0021] | 0.013 | 0.0504 | **mức biên** (100.000 vòng: p Holm 0,048; mục 12.5b) |
| MPNN − ANN | AUC-ROC | −0.0083 | [−0.0153, −0.0015] | 0.015 | 0.0504 | **mức biên** (100.000 vòng: p Holm 0,048; mục 12.5b) |
| MPNN − Random Forest | Brier | +0.0434 | [+0.0391, +0.0475] | 0.0002 | 0.004 | RF tốt hơn (xác suất GNN bị đẩy lên do `pos_weight`) |

KTC 95% của AUC-PR từng mô hình: MPNN [0.566, 0.632], RF [0.583, 0.655], ANN [0.578, 0.645],
GraphSAGE [0.481, 0.551], GATv2 [0.427, 0.497], GCN [0.417, 0.489], LR [0.313, 0.366]. Đầy đủ 21 cặp × 3 độ đo:
`results/tables/official/significance_tests.csv` (AUC-PR: 17/21 cặp có ý nghĩa; AUC-ROC 17/21; Brier 21/21).

> **Ghi chú đọc p-value.** Với 10.000 vòng, p thô nhỏ nhất đo được là 2/10001 ≈ 0.0002; nhân hệ số Holm (21 cặp)
> thành **0.004** — tức "0.004" nghĩa là "nhỏ nhất đo được".
>
> **Độ ổn định của kết luận (đã kiểm tra riêng).** Chạy lại 5 lần × 10.000 vòng với 5 seed bootstrap khác nhau:
> - MPNN − RF, AUC-PR: cận trên KTC luôn trong [+0.0011, +0.0016], p thô 0.063–0.069 → kết luận "không khác biệt"
>   **ổn định** (không có ý nghĩa ngay cả trước hiệu chỉnh Holm). MPNN − ANN tương tự (p thô 0.087–0.095).
> - MPNN − RF/ANN, AUC-ROC: p thô 0.009–0.015, KTC (chưa hiệu chỉnh) luôn **không** chứa 0, nhưng sau Holm
>   p ≈ 0.04–0.06 — nằm **ngay ngưỡng 0.05**. Kết luận trung thực: MPNN có AUC-ROC thấp hơn RF/ANN khoảng 0.01,
>   bằng chứng **ở mức biên**, không nên khẳng định chắc chắn theo hướng nào.
> - Lần chạy đầu (2.000 vòng) cho p Holm AUC-ROC = 0.068 vì sàn p thô 0.001 làm hệ số Holm của các cặp phía sau
>   lớn hơn; đó là lý do chuyển sang 10.000 vòng.

**Thay đổi so với kết luận cũ:** báo cáo 11/10 viết baseline dạng bảng "cao hơn" MPNN. Với 5 seed và kiểm định theo
trận, chênh lệch AUC-PR giữa MPNN và RF/ANN **không có ý nghĩa thống kê**; MPNN tốt hơn có ý nghĩa so với cả 3 GNN
còn lại. AUC-ROC của MPNN thấp hơn RF/ANN khoảng 0.01, ở mức **biên** ý nghĩa (p Holm 0,048–0,050, mục 12.5b). Về Brier score (độ tin cậy xác suất)
baseline vẫn tốt hơn rõ.

---

## 9. Kiểm thử

| | Trước | Sau |
|---|---|---|
| `pytest` (trong `.venv`) | 128 passed, 1 skipped (skip vì thiếu `graph_index.parquet`) | **148 passed, 0 skipped** |
| `ruff check .` (gồm cả notebook) | sạch | sạch |
| `black --check .` | sạch | sạch |
| Notebook 00–06 chạy bằng `nbconvert --execute` | không có output | 7/7 chạy hết, 0 lỗi, 14 hình; `nbformat.validate` OK |

148 = 128 cũ + 1 test trước bị skip nay chạy được (`test_graph_index_matches_belong_to_splits`, vì đã có dữ liệu)
+ 19 test mới: `tests/evaluation/test_significance.py` (10, gồm đối chiếu độ đo numpy với sklearn),
`tests/evaluation/test_evaluate_script.py` (4), `tests/modeling/test_trainer_outputs.py` (1),
`tests/data/test_splits.py` (+3), `tests/test_environment_metadata.py` (+1), `tests/graphs/test_build.py`
(1 test thay thế test cũ về tọa độ 120.9).

Ghi chú hiệu năng: lần đầu chạy `07_evaluate.py` mất > 16 phút (sklearn kiểm tra đầu vào ở mỗi lần gọi, ~210.000 lần)
và bị hệ thống dừng vì máy thiếu RAM. AUC-PR / AUC-ROC trong bootstrap được viết lại bằng numpy
(`fast_average_precision`, `fast_roc_auc`), có test khớp sklearn tới 1e-12 → 2.000 vòng mất 4 phút, 10.000 vòng ~10 phút.

Notebook trước đây dùng `REPO_ROOT = Path(".")` nên **chạy bằng nbconvert (kernel đứng ở `notebooks/`) sẽ lỗi**, kể
cả lệnh tái chạy ghi trong chính notebook. Nay tự dò thư mục chứa `configs/`.

---

## 10. Việc KHÔNG làm (ngoài phạm vi 0–6 hoặc cần GV quyết)

- **Không sửa Bước 7–10.** Lưu ý: checkpoint `models/official/*_best.pt` đã được huấn luyện lại, nên kết quả XAI
  cũ trong `results/xai/official/mpnn/20261017_091540/` và `results/tables/official/xai_comparison.csv` được sinh từ
  checkpoint **cũ** (SHA khác). Chạy lại Bước 7–8 hay không là quyết định của nhóm.
- **Không tự chốt** Decision 0001 (nhãn), 0003 (đơn vị đồ thị cho XAI), phạm vi open-play (`play_patterns_keep: null`).
- **Không đổi** kiến trúc, siêu tham số, đặc trưng của mô hình (ngoài việc kẹp 1 tọa độ).
- **Không sửa** các báo cáo tuần cũ trong `docs/meetings/` (là hồ sơ lịch sử); số liệu mới nằm ở file này.

## 11. Cách chạy lại toàn bộ Bước 0–6

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install -e . --no-deps
python scripts/00_audit_data.py --config configs/data.yaml
python scripts/01_download_data.py --config configs/data.yaml
python scripts/00_generate_environment_metadata.py --config configs/data.yaml
python scripts/03_preprocess.py --config configs/data.yaml
python scripts/02_descriptive_stats.py --config configs/data.yaml
python scripts/04_build_graphs.py --config configs/data.yaml
python scripts/05_make_splits.py --config configs/data.yaml
python scripts/06_train_baselines.py --config configs/data.yaml --seeds 42,123,456,789,1024
for m in mpnn gcn gatv2 graphsage; do
  python scripts/06_train.py --config configs/data.yaml --model-config configs/model/$m.yaml --seeds 42,123,456,789,1024
done
python scripts/07_evaluate.py --config configs/data.yaml
jupyter nbconvert --to notebook --execute --inplace notebooks/0[0-6]_*.ipynb
```

Thời gian tham khảo (CPU 12 luồng, 4 GNN chạy song song, mỗi tiến trình 3 luồng): tải dữ liệu ~20 phút; Bước 2–4
~3 phút; baseline 5 seed ~2 phút; GNN 5 seed: MPNN ~1,4 giờ, GCN/GraphSAGE ~2 giờ, GATv2 ~4,5 giờ; Bước 6 ~10 phút.

---

## 12. Rà soát lần 2 (22/10/2026)

### 12.1 Kiểm tra độc lập toàn bộ số liệu — `scripts/audit_steps_0_6.py` (mới)
Script tự tính lại mọi con số **từ dữ liệu gốc bằng sklearn/pandas** (không dùng code tính độ đo của dự án) rồi
đối chiếu với bảng kết quả. Chạy: `python scripts/audit_steps_0_6.py .` → **88/88 PASS** (mã thoát 0; ~15 phút). Các nhóm kiểm tra:
- Bước 1–2: 380 + 380 file thô; đếm lại 1.313.773 sự kiện; 45.365 possession; 261.205 đường chuyền; 6.471 possession có
  sút; khóa duy nhất; chỉ hiệp 1–2; mọi đường chuyền đủ tọa độ, có người nhận, thuộc đội kiểm soát bóng.
- Bước 3: 45.047 đồ thị; nhãn đồ thị = `has_shot`; tính lại: 0 cạnh sau cú sút; đặc trưng chuẩn hóa trong [0, 1].
- Bước 4: 266/57/57 trận, không trùng, phủ 380 trận; số đồ thị test/val khớp bảng.
- Bước 5 (35 run = 7 mô hình × 5 seed): khóa dự đoán đúng thứ tự tập test/val; **ngưỡng tính lại trên validation trùng
  ngưỡng đã dùng**; độ đo test tính lại khớp file; seed được chọn = seed có Val AUC-PR cao nhất; `models/official/*_best.pt`
  trùng SHA-256 với checkpoint của seed đó; `pos_weight` = n_âm/n_dương của tập train (6,311); seed của mọi mô hình
  = đúng danh sách trong `configs/train.yaml`.
- Bước 6: mean/std trong `model_comparison.csv` = tính lại từ dự đoán; giá trị điểm trong `model_metric_ci.csv` khớp
  sklearn; 63 dòng kiểm định; Holm hợp lệ; bootstrap độc lập sklearn 10.000 vòng khớp KTC, kết luận và p-value
  (mục 12.4); SHA trong manifest đúng.

### 12.2 Soát code độc lập (agent riêng)
Không phát hiện lỗi sai. Đã xác minh lại: hàm numpy khớp sklearn (3.000 bộ dữ liệu, có điểm trùng), Holm/p-value/chiều
kết luận, ghép dự đoán theo khóa, chọn seed theo validation, kẹp tọa độ, không rò rỉ dữ liệu. Các điểm **đã sửa** theo góp ý:

| Điểm | Trước | Sau | File |
|---|---|---|---|
| Kiểm định bị bỏ qua | file KTC/kiểm định **cũ** vẫn nằm lại, script vẫn trả 0 | xóa file cũ trước khi chạy; run chính thức trả **1** nếu không kiểm định được | `scripts/07_evaluate.py` |
| Mã băm cấu hình trong manifest | băm config **hiện tại** | thêm băm config **lúc huấn luyện** (từ `config_snapshot.yaml` / `aggregate_summary.json`) + cờ `current_config_matches_training`; cảnh báo nếu khác | `scripts/07_evaluate.py` |
| `--allow-partial` | vẫn ghi đè `results/reproducibility/` | chỉ ghi khi mọi mô hình đủ seed | `scripts/07_evaluate.py` |
| Cờ "đủ seed" | chỉ đếm số lượng; baseline ghi cứng `>= 5` | phải chứa **đúng** danh sách seed trong `train.yaml` và đủ `n_runs` | `scripts/06_train.py`, `06_train_baselines.py` |
| Hình PR/ROC | GNN lấy seed tốt nhất theo val, baseline lấy seed đầu (thiên vị GNN) | cả hai đều lấy seed tốt nhất theo val | `notebooks/06_model_evaluation.ipynb` |
| Holm với p = NaN | thành 1.0 | giữ NaN, không tính vào hệ số | `significance.py` (+1 test) |
| Chiều kết luận | dựa trên chênh lệch đã làm tròn | dựa trên giá trị chưa làm tròn | `significance.py` |
| Cột `best_seed_by_val` | ghi `789.0` | ghi `789` | `scripts/07_evaluate.py` |

### 12.3 Số vòng bootstrap 2.000 → 10.000
Kiểm tra độc lập cho thấy kết luận AUC-ROC MPNN−RF nhạy với sai số Monte Carlo khi chỉ có 2.000 vòng (xem ghi chú ở
mục 8). Đã chuyển mặc định sang 10.000 vòng và viết lại để sinh mẫu dần (không tốn RAM). Đã kiểm chứng: code mới chạy
với 2.000 vòng, seed 42 cho **từng con số trùng khớp** bảng cũ → thay đổi chỉ về hiệu năng, không đổi phép tính.

Hai điểm hiển thị cũng đã sửa: notebook 03 vẽ tọa độ **đã kẹp** (đúng như đặc trưng đồ thị); bảng kiểm tra ở
notebook 05 lấy "Best epoch" từ `run_manifest.json` (đúng epoch trainer lưu checkpoint, thay vì `idxmax`) và kiểm tra số
dòng `training_history.csv` khớp `epochs_run` của manifest.

### 12.4 Kiểm tra bootstrap độc lập — tiêu chí chặt
Lần đầu, script kiểm tra dùng bootstrap sklearn 500 vòng và trượt tiêu chí "cùng phía số 0" (500 vòng quá ít cho một KTC
sát 0). Thay vì nới tiêu chí, nay dùng **sklearn 10.000 vòng, seed 2026** (khác seed 42 của bản báo cáo), cho cả AUC-PR
và AUC-ROC của MPNN−RF, với tiêu chí đặt trước khi chạy: hai cận KTC lệch < 0,003, **cùng kết luận về số 0**, p thô lệch
< 0,01. Kết quả: **cả hai PASS** — AUC-PR: độc lập [−0,0419; +0,0012], p = 0,064 so với báo cáo [−0,0418; +0,0016], p = 0,068 (cùng chứa 0); AUC-ROC: độc lập [−0,0184; −0,0023], p = 0,010 so với báo cáo [−0,0181; −0,0021], p = 0,013 (cùng không chứa 0).

### 12.5 Hai điểm đã kiểm chứng bằng thực nghiệm bổ sung (22/10/2026)

**(a) GATv2 có bị cắt ở giới hạn 200 epoch không? — KHÔNG (đã kiểm chứng).**
- Dấu hiệu: seed 456 chạy đủ 200 epoch (epoch tốt nhất 195); seed 42/123 chạy 193/199 epoch.
- Thử nghiệm độ nhạy: huấn luyện lại GATv2 seed 456 với **tối đa 400 epoch** (`06_train.py --seed 456 --epochs 400`,
  thư mục riêng `experiments/official_debug/gatv2/456/` — không ghi đè kết quả chính thức).
- Kết quả: early stopping **tự dừng ở epoch 200** (epoch tốt nhất 180), Val AUC-PR tăng thêm nhờ epoch > 200 = **0,000**;
  test AUC-PR 0,461 (run chính thức 0,463), AUC-ROC 0,840 (chính thức 0,842). → GATv2 đã hội tụ trong giới hạn 200 epoch;
  giới hạn không làm thiệt GATv2. Kết hợp với khoảng cách test AUC-PR MPNN − GATv2 = +0,138 (KTC [+0,113; +0,162]),
  thứ hạng các GNN là chắc chắn.
- Ghi chú tái lập: lần chạy này dùng 8 luồng CPU (run chính thức 3 luồng) nên kết quả từng epoch lệch nhẹ do sai số làm
  tròn số thực (val AUC-PR lệch trung bình 0,004/epoch; test AUC-PR cuối lệch 0,002, nhỏ hơn độ lệch giữa các seed
  0,0048). Huấn luyện GNN tái lập được **tới mức sai số làm tròn**, không trùng từng bit khi đổi số luồng CPU; muốn trùng
  tuyệt đối cần giữ nguyên số luồng (`OMP_NUM_THREADS=3` như lần chạy chính thức).

**(b) AUC-ROC MPNN − RF/ANN sát ngưỡng — đã chốt bằng 100.000 vòng.**
- Bảng chính thức (10.000 vòng): p Holm = 0,0504 cho cả hai cặp (sai số Monte Carlo ≈ ±0,006) → chưa phân định được.
- Kiểm tra bổ sung **100.000 vòng** (cùng phương pháp, seed 7; chỉ độ đo AUC-ROC, 21 cặp, Holm):

  | Cặp | Chênh lệch | KTC 95% | p thô | p Holm (± 1 sai số chuẩn Monte Carlo) |
  |---|---|---|---|---|
  | MPNN − Random Forest | −0,0099 | [−0,0183; −0,0021] | 0,0120 | **0,048 ± 0,002** |
  | MPNN − ANN | −0,0083 | [−0,0153; −0,0016] | 0,0142 | **0,048 ± 0,002** |

- **Kết luận:** p Holm ≈ 0,048–0,050 — **đúng ở ngưỡng 0,05** (cách ngưỡng chưa tới 1 sai số chuẩn Monte Carlo).
  RF/ANN có AUC-ROC cao hơn MPNN khoảng 0,01; bằng chứng ở **mức biên**. Chênh lệch này rất nhỏ về thực tế.
  Kết luận theo **AUC-PR (độ đo chính)** không đổi và chắc chắn: MPNN và RF/ANN **không khác biệt có ý nghĩa**.
- Cách viết trong khóa luận: "Về AUC-PR, MPNN tương đương RF/ANN; về AUC-ROC, RF/ANN nhỉnh hơn MPNN khoảng 0,01
  (ở mức biên ý nghĩa thống kê, p Holm ≈ 0,05); về Brier score, RF/ANN tốt hơn rõ."
- **Quyết định (22/10/2026):** bảng chính thức giữ **10.000 vòng** (mức phổ biến, đủ cho 61/63 so sánh). Vì vậy
  `significance_tests.csv` ghi hai cặp AUC-ROC này là "không khác biệt có ý nghĩa" (p Holm 0,0504); phép kiểm tra
  bổ sung 100.000 vòng cho p Holm 0,048. Hai con số nằm hai phía ngưỡng 0,05 trong phạm vi sai số Monte Carlo →
  khi viết khóa luận, trình bày hai cặp này là **chênh lệch ~0,01 ở mức biên ý nghĩa**, không khẳng định theo hướng nào.

### 12.6 Thứ tự thời gian (chứng minh kết quả sinh từ code cuối cùng)
- Lần sửa cuối của code tính toán Bước 6: `significance.py` 23:58:02, `07_evaluate.py` 23:59:51 (21/10). Bảng
  `model_comparison.csv`, `model_metric_ci.csv`, `significance_tests.csv`, `experiment_manifest.json` sinh lúc
  00:00–00:08 (22/10) → **sau** lần sửa cuối.
- `06_train.py` / `06_train_baselines.py` sửa lúc 23:34 (sau khi huấn luyện xong 17:34–22:11) — chỉ đổi cách tính cờ
  "đủ seed". Đã kiểm chứng: áp code mới lên 5 nhóm run hiện có cho đúng giá trị đã ghi (True) → không cần huấn luyện lại.
- Notebook 00–06, script kiểm tra độc lập và bộ test được chạy lại **sau cùng**, sau mọi chỉnh sửa (mục 9 và 12.1).
