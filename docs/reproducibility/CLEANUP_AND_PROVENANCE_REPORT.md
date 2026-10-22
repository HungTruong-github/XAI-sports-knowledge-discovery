# BÁO CÁO KIỂM TOÁN TÁI LẬP, LÀM SẠCH MÃ NGUỒN VÀ BẢO ĐẢM XUẤT XỨ SỐ LIỆU (PROVENANCE AUDIT & CLEANUP REPORT)

**Dự án**: `HungTruong-github/XAI-sports-knowledge-discovery`  
**Nhánh kiểm toán & làm sạch**: `Clean`  
**Thời điểm hoàn tất**: 23/10/2026  
**Vai trò**: Senior Python Engineer & Research Reproducibility Auditor  

---

## 1. Tóm tắt điều hành (Executive Summary)

Báo cáo này ghi nhận toàn bộ các công việc làm sạch mã nguồn, loại bỏ code dư thừa, hợp nhất các module/test trùng lặp, đồng thời thiết lập cơ chế **Bảo đảm Xuất xứ Số liệu (Provenance Integrity)** trên toàn bộ pipeline nghiên cứu.

Toàn bộ các nguyên tắc bắt buộc đã được tuân thủ nghiêm ngặt:
1. **Không thay đổi định nghĩa khoa học**: Quy tắc gán nhãn, cấu hình split theo trận, giao thức đánh giá (AUC-PR, AUC-ROC, Fidelity+, Fidelity-, Sparsity, Stability) và ý nghĩa metric giữ nguyên vẹn 100%.
2. **Không xóa dữ liệu và minh chứng**: Toàn bộ split manifests, kết quả thực nghiệm trong `results/tables/official/`, biểu đồ trong `results/figures/official/`, artifact XAI lịch sử trong `results/xai/official/mpnn/20261017_091540/` đều được bảo toàn.
3. **Không tạo hay sửa số liệu**: Không làm số liệu khớp nhau một cách giả tạo; ghi nhận chính xác nguồn gốc và hạn chế của từng artifact.
4. **Không dùng lệnh Git phá hủy**: Không dùng `git reset --hard`, `git clean -fdx`, force push; không tự động push lên remote.

---

## 2. Bảng kiểm kê hành động (Action Inventory)

| File / Nhóm file | Hành động | Lý do | Tác động pipeline | Test / Bằng chứng |
|---|---|---|---|---|
| `scripts/md2docx.py` | **Di chuyển** sang `tools/md2docx.py` | Là tiện ích chuyển đổi định dạng tài liệu Markdown sang Word, không thuộc quy trình nghiên cứu khoa học 10 bước. | Giữ `scripts/` sạch sẽ, chỉ chứa đúng các pipeline entry points từ Bước 00 đến Bước 09. | `git status`, `python tools/md2docx.py --help` hoạt động bình thường. |
| `tests/modeling/test_multi_seed_artifacts.py` | **Xóa sau khi hợp nhất** | File test bị trùng lặp mục đích với `tests/modeling/test_multi_seed.py` (cùng kiểm tra việc dọn artifact và cấu trúc multi-seed). | Không làm giảm độ phủ test; giảm phân mảnh mã nguồn kiểm thử. | Đã hợp nhất toàn bộ assertions vào `tests/modeling/test_multi_seed.py`. 11/11 tests PASS. |
| `src/xai_football/data/possession.py` | **Hợp nhất / Khử trùng lặp** | Định nghĩa trùng lặp các hàm trợ giúp trích xuất JSON `_name`, `_id`, `_nested` vốn đã có trong `canonicalize.py`. | Chuẩn hóa điểm truy cập dữ liệu; tái sử dụng hàm chuẩn `extract_name`, `extract_id`, `extract_nested` từ `canonicalize.py`. | `tests/data/test_data_pipeline.py` PASS. |
| `scripts/06_train.py` & `scripts/06_train_baselines.py` | **Hợp nhất vào package `src/`** | Cả hai script độc lập tự triển khai logic seed (`resolve_seeds`, `resolve_baseline_seeds`), tổng hợp metric (`compute_metrics_aggregation`) và dọn dẹp artifact (`clean_stale_aggregates`). | Logic cốt lõi được chuyển về `src/xai_football/training/trainer.py` và export qua `training/__init__.py`. Scripts chỉ đóng vai trò điều phối. | `tests/modeling/test_multi_seed.py` PASS toàn bộ 11 tests. |
| `src/xai_football/utils/io.py` | **Bổ sung / Củng cố** | Việc ghi file kết quả trực tiếp bằng `to_csv` hoặc `open('w')` có nguy cơ làm hỏng bảng số liệu chính thức nếu tiến trình bị dừng đột ngột (crash/SIGINT). | Thêm `save_atomic_csv`, `save_atomic_json`, `save_atomic_yaml` ghi qua file tạm `.tmp` rồi thay thế nguyên tử bằng `os.replace`. | Đã áp dụng trên toàn bộ `scripts/07_evaluate.py` và `scripts/09_evaluate_xai.py`. |
| `src/xai_football/utils/provenance.py` | **Mở rộng chức năng** | Thiếu cơ chế tự động kiểm tra tính hợp lệ của manifest run XAI và quét các run hợp lệ. | Cung cấp `validate_run_manifest`, `scan_available_xai_runs`, `format_run_table`, `compute_dataset_fingerprint`, và bảng hằng số `RUN_TYPES`, `RUN_STATUSES`. | `tests/xai/test_evaluate_xai_provenance.py` PASS toàn bộ 7 regression tests. |
| `scripts/09_evaluate_xai.py` | **Sửa lỗi nghiêm trọng (Lỗi 5.1)** | Script cũ tự động chọn thư mục mới nhất theo thứ tự bảng chữ cái (`sorted(dirs)[-1]`), dẫn đến nguy cơ lấy nhầm run pilot/debug ghi đè lên kết quả chính thức. | **BẮT BUỘC** chỉ định `--run-id` hoặc `--xai-dir`. Nếu không có, hiển thị bảng danh sách các run hợp lệ kèm trạng thái rồi dừng lại. Kiểm tra nghiêm ngặt manifest; chặn run pilot/smoke/partial ghi đè bảng official. Cô lập run lịch sử bằng `--allow-historical`. | `tests/xai/test_evaluate_xai_provenance.py` (7 tests kiểm tra mọi trường hợp biên). |
| `scripts/07_evaluate.py` | **Sửa lỗi nghiêm trọng (Lỗi 5.2)** | Cờ `--allow-partial` trước đây ghi đè trực tiếp vào `results/tables/official/model_comparison.csv`, làm mất tính toàn vẹn của kết quả 5-seed đầy đủ. | Phân luồng an toàn: Run partial bắt buộc ghi ra `results/tables/official/partial_model_comparison.csv`, bảo vệ tuyệt đối file canonical `model_comparison.csv`. Chặn run debug/smoke. | `tests/evaluation/test_evaluate_script.py::test_evaluate_script_partial_does_not_overwrite_official` PASS. |
| `results/metadata/RUN_REGISTRY.json` | **Tạo mới** | Chưa có danh bạ tập trung phân loại các run thực nghiệm trong repo. | Đăng ký rõ ràng: Run official GNN/Baselines (`fc18a929`), Run pilot (`configs/data_pilot.yaml`), và Run XAI lịch sử 5 đồ thị (`20261017_091540`). | Tra cứu xuất xứ nhất quán trên toàn bộ pipeline. |
| `results/tables/official/model_comparison.csv` & `significance_tests.csv` | **Giữ lại nguyên vẹn** | Là số liệu chính thức của 4 GNN + 3 Baselines qua 5 seeds trên toàn bộ Test set (6.646 đồ thị). | Nguồn sự thật cho Bài toán 1 (MPNN đạt AUC-PR 0.5996 ± 0.0042). | Đã đối soát khớp với `results/metadata/official/config_hashes.json`. |
| `results/xai/official/mpnn/20261017_091540/` | **Giữ lại nguyên vẹn** | Artifact lịch sử giải thích 5 đồ thị mẫu thực hiện trên branch `REPRODUCIBILITY` (commit `9fae6c51`). | Bảo toàn minh chứng lịch sử, phân loại rõ là `historical_provisional`. Xuất bảng riêng biệt `xai_comparison_historical_20261017_091540.csv`, không ghi đè official. | Đã đăng ký trong `RUN_REGISTRY.json`. |
| `data/splits/*` | **Giữ lại nguyên vẹn** | Split manifests phân chia 380 trận đấu thành 266 Train, 57 Val, 57 Test. | Bảo đảm tính bất giao (disjoint) và ngăn ngừa rò rỉ dữ liệu (leakage). | `tests/data/test_splits.py` và `test_split_invariants.py` PASS. |
| `docs/decisions/*` | **Giữ lại nguyên vẹn** | Toàn bộ 4 Decision Records về gán nhãn, Hybrid Centrality, đơn vị đồ thị và dataset. | Bảo tồn hồ sơ khoa học và tiến trình làm việc với GVHD. | Hồ sơ nguyên vẹn. |

---

## 3. Thống kê số lượng File

- **Số file đã xóa**: `1` (`tests/modeling/test_multi_seed_artifacts.py` sau khi đã hợp nhất toàn bộ assertions).
- **Số file đã di chuyển**: `1` (`scripts/md2docx.py` -> `tools/md2docx.py`).
- **Số file đã hợp nhất / tái cấu trúc**: `4` (`canonicalize.py`, `possession.py`, `06_train.py`, `06_train_baselines.py`).
- **Số file tạo mới / nâng cấp bảo vệ**: `4` (`results/metadata/RUN_REGISTRY.json`, `tests/xai/test_evaluate_xai_provenance.py`, `src/xai_football/utils/provenance.py`, `src/xai_football/utils/io.py`).
- **Số file minh chứng nghiên cứu được giữ lại**: `100%` (Toàn bộ 84 file trong `results/`, toàn bộ configs trong `configs/`, toàn bộ splits trong `data/splits/`, toàn bộ decisions trong `docs/decisions/`).

---

## 4. Các lỗi Provenance & Aggregation đã khắc phục

### 4.1. Lỗi tự động chọn thư mục mới nhất theo thứ tự chữ cái (`09_evaluate_xai.py`)
- **Hành vi cũ**: Khi người dùng không truyền `--xai-dir`, mã nguồn cũ gọi `dirs = sorted(...)` và lấy `dirs[-1]`. Hành vi này cực kỳ rủi ro: chỉ cần một lần chạy thử nghiệm nhỏ hoặc một thư mục có tên bắt đầu bằng số/chữ cái lớn hơn thì script sẽ tự động lấy kết quả đó làm kết quả chính thức.
- **Cách khắc phục**:
  1. Loại bỏ hoàn toàn khối lệnh tự động chọn thư mục.
  2. Bắt buộc người dùng chỉ định rõ ràng `--run-id` hoặc `--xai-dir`.
  3. Nếu thiếu tham số, script quét toàn bộ thư mục XAI, in ra bảng tổng hợp gồm: `Run ID`, `Run Name`, `Model`, `Split`, `Graphs`, `Commit`, `Status` và hướng dẫn câu lệnh cụ thể, sau đó dừng chương trình với `exit(1)`.
  4. Đọc và kiểm tra tính toàn vẹn của file `manifest.json` bên trong run (kiểm tra `run_id`, `run_name`, `model_name`, `split`, `n_graphs_explained`, `git.commit_sha`, `checkpoint_sha256`).

### 4.2. Lỗi run Pilot/Smoke/Partial ghi đè bảng Official
- **Hành vi cũ**:
  - `09_evaluate_xai.py` cho phép mọi run ghi đè vào `results/tables/official/xai_comparison.csv`.
  - `07_evaluate.py` khi nhận cờ `--allow-partial` vẫn ghi trực tiếp vào `results/tables/official/model_comparison.csv`.
- **Cách khắc phục**:
  1. Trong `09_evaluate_xai.py`: Nếu run có `is_pilot: true`, `is_smoke: true`, hoặc trạng thái khác `complete`, hệ thống lập tức từ chối tổng hợp vào bảng chính thức (`ValueError: Cannot promote pilot/smoke run ...`).
  2. Trong `07_evaluate.py`: Khi chạy với `--allow-partial` (đánh giá tập mô hình chưa hoàn thành đủ 5 seeds), script chuyển hướng xuất kết quả sang `results/tables/official/partial_model_comparison.csv`, tuyệt đối không chạm vào file canonical `model_comparison.csv`. Các run debug/smoke cũng bị chặn hoàn toàn.

### 4.3. Quản lý an toàn cho Artifact XAI lịch sử (`20261017_091540`)
- **Thực trạng**: Artifact tại `results/xai/official/mpnn/20261017_091540/` được tạo trên branch `REPRODUCIBILITY` (commit `9fae6c51`), chỉ giải thích trên 5 đồ thị mẫu (Test subset). Nếu đưa số liệu này vào bảng official thì sẽ gây hiểu nhầm rằng đây là kết quả đánh giá trên toàn bộ 6.646 đồ thị Test set.
- **Cách khắc phục**:
  1. Bảo toàn toàn bộ dữ liệu thô và manifest của run lịch sử.
  2. Đăng ký rõ trong `results/metadata/RUN_REGISTRY.json` với trạng thái `historical_provisional`.
  3. Trong `09_evaluate_xai.py`, khi xử lý run này mà không có cờ `--allow-historical`, hệ thống sẽ từ chối tổng hợp. Khi truyền cờ `--allow-historical`, hệ thống sẽ xuất kết quả ra file riêng: `results/tables/official/xai_comparison_historical_20261017_091540.csv` kèm cột `status = HISTORICAL_PROVISIONAL`. Bảng chính thức `xai_comparison.csv` được giữ nguyên vẹn.

### 4.4. Cơ chế ghi file nguyên tử (Atomic Writing)
- **Rủi ro**: Nếu quá trình ghi đè CSV/JSON bị ngắt giữa chừng (mất điện, lỗi bộ nhớ, ngắt tiến trình), file đích sẽ bị rỗng hoặc cụt dữ liệu.
- **Giải pháp**: Xây dựng `save_atomic_csv`, `save_atomic_json`, `save_atomic_yaml` trong `src/xai_football/utils/io.py`. Dữ liệu được ghi hoàn tất vào file tạm `.tmp_<filename>` trước khi hoán đổi tức thời bằng lệnh `os.replace`.

---

## 5. Quy tắc phân biệt các loại Run (Official vs Pilot vs Historical vs Partial)

| Loại Run | Định nghĩa & Điều kiện | File cấu hình | Vị trí lưu Artifact | Quy tắc ghi Bảng tổng hợp |
|---|---|---|---|---|
| **Official** | Đủ 5 seeds (42, 123, 456, 789, 1024), toàn bộ tập Test (6.646 đồ thị), không lỗi, manifest đầy đủ hash. | `configs/data.yaml` | `results/tables/official/` | Được phép ghi vào canonical tables: `model_comparison.csv`, `significance_tests.csv`, `xai_comparison.csv`. |
| **Pilot** | Chạy kiểm tra nhanh trên tập con (2 trận, ~200 đồ thị) để kiểm thử pipeline. | `configs/data_pilot.yaml` | `results/tables/pilot/` | Chỉ ghi vào `results/tables/pilot/`. Nghiêm cấm ghi đè vào thư mục `official/`. |
| **Partial** | Run chưa chạy đủ 5 seeds hoặc chỉ chạy subset mô hình (khi dùng `--allow-partial`). | Bất kỳ | `results/tables/official/partial_*` | Bắt buộc ghi ra `results/tables/official/partial_model_comparison.csv`. Không được ghi đè canonical. |
| **Historical Provisional** | Run lịch sử thực hiện trên các branch trước đó (ví dụ: `20261017_091540` trên `REPRODUCIBILITY`), số mẫu nhỏ (5 đồ thị). | Đã ghi nhận trong manifest lịch sử | `results/xai/official/...` | Lưu riêng biệt ra file `xai_comparison_historical_<run_id>.csv` kèm cờ `--allow-historical`. |
| **Smoke / Debug** | Chạy 1 epoch, vài mẫu để xác nhận mã nguồn không crash. | Cờ `--smoke-test` | Thư mục cách ly `official_smoke` | Tuyệt đối không được tổng hợp số liệu vào bất kỳ bảng báo cáo nào. |

---

## 6. Đối soát Manifest lịch sử và Hiện trạng Branch Clean

Qua kiểm toán `results/metadata/RUN_REGISTRY.json` và các file manifest hiện có trong repo:
1. **Model Evaluation Manifest (`results/metadata/official/config_hashes.json`)**:
   - Ghi nhận commit `fc18a92911b339fca6bf5113d8036ea245224fa9` trên nhánh `main`.
   - Kết quả này phản ánh đúng thực nghiệm 4 GNN × 5 seeds và 3 Baselines × 5 seeds trên 6.646 đồ thị Test set.
   - Trạng thái: **Verified Official Baseline & GNN Benchmark**.
2. **XAI Historical Manifest (`results/xai/official/mpnn/20261017_091540/manifest.json`)**:
   - Ghi nhận commit `9fae6c5188f986fb866223e71781297eefb3b192` trên nhánh `REPRODUCIBILITY`.
   - Đánh giá trên 5 đồ thị mẫu (Test subset) cho MPNN.
   - Trạng thái: **Historical Provisional Artifact** (Không coi là đánh giá hoàn chỉnh toàn bộ Test set; đã cô lập an toàn).

---

## 7. Kết quả kiểm thử thực tế (Test Execution Results)

Toàn bộ các lệnh kiểm tra sau đã được thực thi trên môi trường Python 3.12:

### 7.1. Chạy toàn bộ Test Suite
```bash
python -m pytest tests/ -v
```
- **Kết quả**: `156 passed, 0 failed` (100% thành công).
- **Chi tiết các nhóm test**:
  - `tests/data/`: 4 test suites (splits, split invariants, temporal labeling, data pipeline) -> **PASS**
  - `tests/graphs/`: 1 test suite (PyG graph construction & feature invariants) -> **PASS**
  - `tests/modeling/`: 5 test suites (GNN backbones, baselines, multi-seed aggregation & cleanup, optimizers, trainer outputs) -> **PASS**
  - `tests/evaluation/`: 2 test suites (significance tests, evaluate script partial-isolation) -> **PASS**
  - `tests/xai/`: 5 test suites (explainers, SHAP, metrics, subgraph, provenance evaluation regression) -> **PASS**
  - `tests/centrality/`: 1 test suite (centrality interfaces) -> **PASS**
  - `tests/test_environment_metadata.py`: Môi trường và lockfile -> **PASS**
  - `tests/test_notebooks_format.py`: Chuẩn định dạng 7 notebooks -> **PASS**

### 7.2. Kiểm tra Regression Provenance & Anti-Stale
```bash
python -m pytest tests/xai/test_evaluate_xai_provenance.py tests/evaluation/test_evaluate_script.py -v
```
- **Kết quả**: `8 passed, 0 failed`.
- **Nội dung chứng minh**:
  - Bắt buộc phải truyền `--xai-dir` hoặc `--run-id`; không tự động chọn thư mục theo tên.
  - Từ chối run thiếu manifest hoặc manifest bị hỏng.
  - Chặn run pilot và smoke ghi đè vào official table.
  - Từ chối ghi đè bảng official khi xử lý run lịch sử; yêu cầu cờ `--allow-historical` và xuất file riêng.
  - Đảm bảo tính nguyên tử khi ghi file (atomic write).
  - Đảm bảo `--allow-partial` trong `07_evaluate.py` không làm bẩn `model_comparison.csv`.

### 7.3. Kiểm tra Lint và Format
```bash
python -m ruff check .
```
- **Kết quả**: `All checks passed!`

```bash
python -m black --check scripts/ src/ tests/ tools/
```
- **Kết quả**: Toàn bộ các file được sửa đổi tuân thủ nghiêm ngặt chuẩn định dạng PEP 8 / Black.

---

## 8. Hướng dẫn chạy Pipeline và Tổng hợp một Run cụ thể

### 8.1. Đánh giá Mô hình GNN & Baseline (Bước 6)
```bash
# Đánh giá chính thức đầy đủ (chỉ chạy khi cả 4 GNN và 3 Baselines đã huấn luyện đủ 5 seeds):
python scripts/07_evaluate.py --config configs/data.yaml

# Đánh giá tạm thời khi đang thử nghiệm dở dang (không làm ảnh hưởng bảng official):
python scripts/07_evaluate.py --config configs/data.yaml --allow-partial
```

### 8.2. Giải thích XAI (Bước 7)
```bash
# Chạy giải thích GNNExplainer trên tập Test với mô hình MPNN:
python scripts/08_run_xai.py --model-config configs/model/mpnn.yaml --method gnnexplainer --split test
```

### 8.3. Tổng hợp và Đánh giá Chất lượng XAI (Bước 8)
```bash
# Liệt kê tất cả các runs XAI khả dụng và trạng thái:
python scripts/09_evaluate_xai.py

# Tổng hợp một run chính thức cụ thể theo run_id (bắt buộc run phải hoàn chỉnh và đầy đủ manifest):
python scripts/09_evaluate_xai.py --run-id <RUN_ID>

# Tổng hợp artifact lịch sử (ví dụ run 20261017_091540) một cách an toàn:
python scripts/09_evaluate_xai.py --run-id 20261017_091540 --allow-historical
```

---

## 9. Các hạng mục chưa thể xác minh & Lỗi còn tồn tại (Known Limitations)

1. **Thực nghiệm XAI quy mô đầy đủ (Full Test Set XAI Benchmark)**:
   - Hiện tại, artifact XAI duy nhất trong repo là run lịch sử `20261017_091540` gồm 5 đồ thị kiểm tra trên branch `REPRODUCIBILITY`.
   - Do chi phí tính toán tối ưu hóa mặt nạ giải thích (GNNExplainer và PGExplainer) trên toàn bộ 6.646 đồ thị Test set đòi hỏi nhiều giờ tính toán trên GPU, đợt kiểm toán này **không tự ý kích hoạt chạy lại toàn bộ Test set** nhằm tránh lãng phí tài nguyên và làm sai lệch môi trường khi chưa có yêu cầu từ người dùng.
   - Toàn bộ hạ tầng code, hàm đo đạc và script tổng hợp đã được kiểm chứng bằng unit test và mock data.
2. **Quyết định nghiên cứu (Decision Records 0001, 0002, 0003)**:
   - Các quyết định về công thức Hybrid Centrality cuối cùng và phạm vi open-play vẫn đang ở trạng thái Pending chờ trao đổi chuyên môn với GVHD; mã nguồn đã chuẩn bị sẵn giao diện chờ nạp công thức.
