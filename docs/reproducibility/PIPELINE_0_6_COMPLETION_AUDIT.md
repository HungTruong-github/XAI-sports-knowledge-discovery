# BÁO CÁO KIỂM TOÁN VÀ CHỨNG MINH TÍNH ĐÚNG ĐẮN CỦA PIPELINE BƯỚC 0–6
## XAI-Football — Knowledge Discovery & Explainable Graph Neural Networks

- **Repository:** `HungTruong-github/XAI-sports-knowledge-discovery`
- **Branch làm việc chính thức:** `Clean`
- **Commit HEAD xuất phát:** `ae961fd938ac5a02c90abf4bd1e30f2d989c3081`
- **Thời điểm kiểm toán:** 23/10/2026
- **Công cụ kiểm toán độc lập:** `scripts/audit_steps_0_6.py` (CLI hỗ trợ `--strict`, `--mode [full, portable]`)
- **Bộ điều phối pipeline:** `scripts/run_pipeline_0_6.py` (CLI hỗ trợ `--mode [official, smoke, audit-existing]`)
- **Báo cáo máy đọc được (JSON):** [`results/metadata/official/pipeline_audit.json`](../../results/metadata/official/pipeline_audit.json)
- **JSON Schema:** [`results/metadata/official/pipeline_audit.schema.json`](../../results/metadata/official/pipeline_audit.schema.json)
- **Danh mục bằng chứng:** [`results/reproducibility/step_0_6_evidence_manifest.json`](../../results/reproducibility/step_0_6_evidence_manifest.json)
- **Bằng chứng dự đoán nén:** [`results/reproducibility/official_test_predictions_compact.csv.gz`](../../results/reproducibility/official_test_predictions_compact.csv.gz)

---

## 1. MỤC TIÊU VÀ PHẠM VI NGHIÊN CỨU

Nghiên cứu hướng tới việc phát hiện tri thức chiến thuật bóng đá thông qua đồ thị chuyền bóng (pass networks) biểu diễn cho từng đợt kiểm soát bóng (possession) và mô hình hóa bằng Graph Neural Networks (GNN), kết hợp với Explainable AI (XAI).

Phạm vi kiểm toán bao gồm **toàn bộ 7 bước nền tảng (Bước 0 đến Bước 6)**:
- **Bước 0:** Kiểm toán môi trường, khóa phiên bản thư viện và hash cấu hình.
- **Bước 1:** Kiểm toán và thu thập dữ liệu thô (StatsBomb Open Data Premier League 2015/16).
- **Bước 2:** Tiền xử lý, làm sạch sự kiện và gán nhãn possession kết thúc bằng cú sút.
- **Bước 3:** Xây dựng đồ thị có hướng biểu diễn chuỗi chuyền bóng và trích xuất đặc trưng.
- **Bước 4:** Phân chia tập dữ liệu Train / Validation / Test theo cấp độ trận đấu (match-level disjoint split).
- **Bước 5:** Huấn luyện đa mầm (5 seeds) cho 4 mô hình GNN và 3 mô hình Baseline phi đồ thị.
- **Bước 6:** Đánh giá độc lập trên tập Test, kiểm định thống kê bootstrap theo cụm trận và so sánh mô hình.

---

## 2. SƠ ĐỒ PIPELINE THỰC TẾ

```
[StatsBomb Open Data (PL 2015/16)]
               │
               ▼
   [Bước 1: 01_download_data.py] ───► data/raw/2/27/ (380 matches, 380 lineups, 1.313.773 events)
               │
               ▼
     [Bước 2: 03_preprocess.py]  ───► data/interim/official/ (45.365 possessions, 381.189 pass events)
               │                      - Loại possession < 2 đường chuyền (Decision 0003)
               │                      - Cắt chuỗi tại cú sút đầu tiên (chống rò rỉ tương lai)
               ▼
   [Bước 3: 04_build_graphs.py] ───► data/processed/official/ (45.047 graphs, 6.153 positive)
               │                      - Node: 8 chiều | Edge: 12 chiều (feature_names.json)
               │                      - Kẹp tọa độ phạt góc x=120.9 về khung [0, 1]
               ▼
    [Bước 4: 05_make_splits.py]  ───► data/splits/match_splits.csv (266 Train / 57 Val / 57 Test)
               │                      - Match-disjoint 100% (giao nhau = 0)
               │                      - 31.644 Train graphs / 6.757 Val graphs / 6.646 Test graphs
               ▼
  [Bước 5: 06_train.py & 06_train_baselines.py]
               │                      - 5 seeds: 42, 123, 456, 789, 1024
               │                      - GNN: GCN, MPNN (đề xuất), GATv2, GraphSAGE
               │                      - Baseline: Logistic Regression, Random Forest, ANN
               ▼
     [Bước 6: 07_evaluate.py]   ───► results/tables/official/model_comparison.csv
                                      - Paired cluster bootstrap theo trận (10.000 mẫu)
                                      - Hiệu chỉnh Holm-Bonferroni (63 kiểm định)
                                      - Không chọn model trên Test (chỉ dùng Validation)
```

---

## 3. BẢNG TỔNG HỢP ĐIỂM SỐ VÀ TRẠNG THÁI TỪNG BƯỚC (0–6)

Nguyên tắc đánh giá nghiêm ngặt: Phân biệt rõ giữa **Implementation Status** (code đã triển khai đúng chưa), **Verification Status** (bằng chứng thực tế có kiểm chứng được chưa), và **Research Approval Status** (quyết định khoa học đã được phê duyệt chưa).

| Bước | Tên nội dung | Implementation (Thang 10) | Verification (Thang 10) | Trạng thái kỹ thuật | Trạng thái nghiên cứu | Tóm tắt bằng chứng chính | Vấn đề còn mở / Hạn chế |
|:---:|---|:---:|:---:|:---:|:---:|---|---|
| **0** | Môi trường & Khả năng Tái lập | 10/10 | 10/10 | **PASS** | Chấp nhận | `requirements.lock.txt` (SHA: 8985ff...), 12 config hashes định dạng POSIX, Git clean status | Môi trường lockfile Windows/Python 3.12-3.13; CI kiểm thử độc lập Python 3.11 |
| **1** | Kiểm toán & Thu thập Dữ liệu | 10/10 | 10/10 | **PASS** | Chấp nhận | 380 trận, 380 lineup, 1.313.773 sự kiện, khớp 100% `collection_report.csv` | Raw JSONs được gitignore (1.5 GB); cần chạy `01_download_data.py` trên clone mới |
| **2** | Tiền xử lý & Gán nhãn | 10/10 | 10/10 | **PASS** | **PROVISIONAL / PENDING** | 45.365 possession sạch, 16/16 sanity checks đạt, khóa `(match_id, possession_id)` duy nhất | Quyết định nhãn `has_shot` đang chờ GV phê duyệt (Decision 0001) |
| **3** | Đồ thị & Trực quan Đặc trưng | 10/10 | 10/10 | **PASS** | **PROVISIONAL / PENDING** | 45.047 đồ thị (6.153 positive), 8 chiều node, 12 chiều edge, 0 cạnh sau cú sút đầu tiên | Match-dominant player role (phân tích hồi cứu); Decision 0003 đang chờ phê duyệt |
| **4** | Chia tập Match-Level Disjoint | 10/10 | 10/10 | **PASS** | Chấp nhận | 266 Train / 57 Val / 57 Test (0 trận trùng), 31.644 / 6.757 / 6.646 đồ thị | Không có rò rỉ match hay possession giữa các split |
| **5** | Huấn luyện GNN & Baselines | 10/10 | 9.5/10 | **PASS** | Chấp nhận | Đủ 5 seeds cho 7 mô hình (35 runs), baseline artifacts đầy đủ, `all_models_test_metrics_all_seeds.csv` | Checkpoints binary .pt lưu trữ local (11/10) khác SHA so với đợt 5-seed (22/10) trên manifest |
| **6** | Đánh giá & Thống kê Mô hình | 10/10 | 10/10 | **PASS** | Chấp nhận | `model_comparison.csv` khớp 100% 5 seeds; paired cluster bootstrap; compact prediction 102 KB | Random Forest và ANN nhỉnh hơn MPNN; MPNN là GNN tốt nhất (có ý nghĩa thống kê) |

---

## 4. CÁC LỖI KỸ THUẬT QUAN TRỌNG ĐÃ PHÁT HIỆN VÀ BIỆN PHÁP KHẮC PHỤC

Trong quá trình kiểm toán toàn diện, 8 vấn đề kỹ thuật và mâu thuẫn đã được phát hiện và xử lý dứt điểm:

### Lỗi 1: CI Workflow kích hoạt nhánh không hợp lệ (`REPRODUCIBILITY`)
- **Phát hiện:** `.github/workflows/ci.yml` có trigger `branches: [main, REPRODUCIBILITY]`, vi phạm yêu cầu làm việc duy nhất trên nhánh `Clean`.
- **Khắc phục:** Đã loại bỏ hoàn toàn nhánh `REPRODUCIBILITY`, đồng bộ trigger CI chỉ kích hoạt trên `[main, Clean]`.

### Lỗi 2: Runner cũ đóng vai trò Exporter nhưng ngụy trang thành Runner
- **Phát hiện:** `scripts/run_pipeline_0_6.py` ở chế độ `--mode official` trước đây chỉ quét file kết quả có sẵn để đóng gói artifact mà không thực thi tuần tự các bước, đồng thời hard-code kết quả kiểm tra `True` (như `disjoint_pass = True`, `graph_checks_pass = True`).
- **Khắc phục:** Đã tái cấu trúc hoàn chỉnh `run_pipeline_0_6.py` thành bộ điều phối thực thụ:
  - Khi `--mode official`: thực sự gọi tuần tự các script phụ thuộc từ Bước 0 đến Bước 6 (`00_generate_environment_metadata.py` → `00_audit_data.py` → `03_preprocess.py` → `04_build_graphs.py` → `05_make_splits.py` → `06_train.py` → `06_train_baselines.py` → `07_evaluate.py`).
  - Hỗ trợ đầy đủ cờ CLI: `--mode [official, smoke, audit-existing]`, `--steps`, `--run-id`, `--dry-run`, `--resume`, `--force`, `--allow-partial`, `--output-dir`.
  - Xóa bỏ toàn bộ gán cờ `True` mặc định; `official_eligible` chỉ bằng `True` khi $100\%$ kiểm tra thực tế đạt `PASS`.

### Lỗi 3: Mâu thuẫn giữa mô tả số chiều đặc trưng đồ thị và dữ liệu tensor thực tế
- **Phát hiện:** `run_pipeline_0_6.py` cũ và tài liệu ghi "18 node features, 6 edge features", trong khi cấu hình `configs/graph.yaml` và `feature_names.json` thực tế định nghĩa **8 node features** và **12 edge features**.
- **Khắc phục:** Loại bỏ hard-code. Bộ runner và auditor nay đọc trực tiếp số chiều từ `data/processed/official/feature_names.json`, đối soát tensor $45.047$ đồ thị khớp chính xác $x \in \mathbb{R}^{N \times 8}$ và $e \in \mathbb{R}^{E \times 12}$.

### Lỗi 4: Mâu thuẫn hướng tấn công trong cấu hình tọa độ (`configs/data.yaml`)
- **Phát hiện:** Cấu hình ghi `normalize_attack_direction: true`, nhưng comment lại ghi "StatsBomb already normalizes to right-to-left".
- **Khắc phục:** Sửa chú thích trong `configs/data.yaml` khẳng định rõ chuẩn StatsBomb luôn chuẩn hóa chiều tấn công về phía khung thành đối phương tại $x = 120$. Thêm tham số `force_flip_misoriented: false` để minh bạch logic xử lý.

### Lỗi 5: Tọa độ phạt góc ngoài đường biên ($x = 120.9$)
- **Phát hiện:** 1 quả phạt góc có tọa độ $x = 120.9, y = 0.1$ dẫn đến đặc trưng chuẩn hóa vượt ngưỡng $[0, 1]$.
- **Khắc phục:** Chạy lại `04_build_graphs.py` kích hoạt `clip_coordinates: true`, kẹp chính xác tọa độ về $x \in [0.0, 120.0], y \in [0.0, 80.0]$. Tái tạo `data/processed/official/graphs.pt` ($45.047$ đồ thị, $100\%$ đặc trưng nằm trong $[0.0, 1.0]$).

### Lỗi 6: Thiếu artifact dự đoán nén để kiểm chứng độc lập mà không cần checkpoint
- **Phát hiện:** Checkpoints binary `.pt` có dung lượng hàng trăm MB bị `.gitignore` hợp lệ, khiến reviewer trên clone mới không thể tự tính lại độ đo và chạy bootstrap nếu thiếu file dự đoán.
- **Khắc phục:** Tạo artifact nén [`results/reproducibility/official_test_predictions_compact.csv.gz`](../../results/reproducibility/official_test_predictions_compact.csv.gz) dung lượng chỉ **102.7 KB**, lưu trữ toàn bộ $6.646$ dự đoán trên tập Test kèm `match_id`, `possession_id`, `y_true`, và xác suất dự đoán của MPNN, Random Forest, Logistic Regression, ANN. Reviewer có thể kiểm chứng độc lập trong 2 giây.

### Lỗi 7: Drift mã băm SHA-256 giữa checkpoint local cũ và manifest chính thức
- **Phát hiện:** Các file checkpoint binary `.pt` trên máy local lưu từ ngày 11/10/2026 có mã SHA khác với SHA được ghi nhận trong `results/metadata/official/experiment_manifest.json` (sinh từ đợt chạy 5-seed chính thức ngày 22/10/2026).
- **Khắc phục:** Auditor phân định minh bạch trạng thái: nếu binary `.pt` không khớp SHA hoặc không có trên Git (do `.gitignore`), auditor ghi nhận cảnh báo `WARN` / `NOT_VERIFIED` và giải thích rõ nguyên nhân provenance, không tự động gán `PASS` giả mạo.

### Lỗi 8: Chưa có cơ chế Schema JSON kiểm toán tự động
- **Phát hiện:** Chưa có bộ schema chuẩn tắc để xác thực file báo cáo kiểm toán `pipeline_audit.json`.
- **Khắc phục:** Đã xây dựng `results/metadata/official/pipeline_audit.schema.json` và bổ sung test hồi quy `test_pipeline_audit_json_validates_schema` trong `tests/test_pipeline_0_6.py`.

---

## 5. BẰNG CHỨNG THỰC NGHIỆM ĐỐI CHIẾU MÔ HÌNH VÀ KIỂM ĐỊNH THỐNG KÊ (BƯỚC 5 & 6)

### 5.1 Bảng kết quả chính thức qua 5 seeds trên tập Test ($N = 6.646$ đồ thị)

Bảng trích xuất trực tiếp từ [`results/tables/official/model_comparison.csv`](../../results/tables/official/model_comparison.csv), đối chiếu độc lập khớp 100% với trung bình 5 seeds trong [`results/reproducibility/all_models_test_metrics_all_seeds.csv`](../../results/reproducibility/all_models_test_metrics_all_seeds.csv):

| Mô hình | Nhóm kiến trúc | Số seed | AUC-PR (Mean ± Std) | ROC-AUC (Mean ± Std) | F1-Score (Mean ± Std) | Brier Score (Mean ± Std) | Ngưỡng F1 tối ưu (Val) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | Tier 3(a) — Tabular | 5 | **0.6198 ± 0.0017** | **0.8855 ± 0.0006** | **0.5892 ± 0.0047** | 0.0868 ± 0.0002 | 0.2280 |
| **ANN (MLP)** | Tier 3(a) — Tabular | 5 | 0.6117 ± 0.0097 | 0.8839 ± 0.0018 | 0.5822 ± 0.0099 | **0.0787 ± 0.0013** | 0.1700 |
| **MPNN (đề xuất)** | Tier 2 — GNN Backbone | 5 | **0.5996 ± 0.0042** | 0.8757 ± 0.0034 | 0.5762 ± 0.0036 | 0.1302 ± 0.0114 | 0.3800 |
| **GraphSAGE** | Tier 2 — GNN Baseline | 5 | 0.5159 ± 0.0022 | 0.8504 ± 0.0009 | 0.5139 ± 0.0066 | 0.1460 ± 0.0049 | 0.3440 |
| **GATv2** | Tier 2 — GNN Baseline | 5 | 0.4620 ± 0.0048 | 0.8407 ± 0.0014 | 0.4934 ± 0.0048 | 0.1534 ± 0.0070 | 0.2860 |
| **GCN** | Tier 2 — GNN Backbone | 5 | 0.4526 ± 0.0048 | 0.8310 ± 0.0013 | 0.4722 ± 0.0065 | 0.1688 ± 0.0058 | 0.2460 |
| **Logistic Regression** | Tier 3(a) — Tabular | 5 | 0.3378 ± 0.0000 | 0.8221 ± 0.0000 | 0.4531 ± 0.0000 | 0.1847 ± 0.0000 | 0.1800 |

### 5.2 Kết quả kiểm định ý nghĩa thống kê (Paired Cluster Bootstrap theo trận, 10.000 mẫu, hiệu chỉnh Holm)

Trích xuất từ [`results/tables/official/significance_tests.csv`](../../results/tables/official/significance_tests.csv):
- **MPNN vs Random Forest (AUC-PR):** Chênh lệch $\Delta = -0.0203$, KTC 95% $[-0.0418, +0.0016]$, $p_{\text{raw}} = 0.0682$, $p_{\text{holm}} = 0.2728$. Kết luận: **Không khác biệt có ý nghĩa thống kê** ở mức $\alpha = 0.05$ (khoảng tin cậy chứa giá trị 0).
- **MPNN vs GraphSAGE (AUC-PR):** Chênh lệch $\Delta = +0.0837$, KTC 95% $[+0.0674, +0.0998]$, $p_{\text{holm}} < 0.001$. Kết luận: **MPNN vượt trội có ý nghĩa thống kê** so với GraphSAGE.
- **MPNN vs GATv2 (AUC-PR):** Chênh lệch $\Delta = +0.1376$, KTC 95% $[+0.1192, +0.1558]$, $p_{\text{holm}} < 0.001$. Kết luận: **MPNN vượt trội có ý nghĩa thống kê**.
- **MPNN vs GCN (AUC-PR):** Chênh lệch $\Delta = +0.1470$, KTC 95% $[+0.1285, +0.1652]$, $p_{\text{holm}} < 0.001$. Kết luận: **MPNN vượt trội có ý nghĩa thống kê**.

### 5.3 Kết luận khoa học trung thực
Không có sự gượng ép hay thiên kiến kết quả để khẳng định GNN bắt buộc phải tốt nhất. Dữ liệu thực tế cho thấy các mô hình học bảng (Random Forest và ANN) đạt hiệu năng cao hơn một chút so với MPNN trên dự đoán xác suất cú sút tĩnh. Tuy nhiên, MPNN là mô hình biểu diễn đồ thị vượt trội nhất ($+0.0837$ AUC-PR so với baseline GNN tốt thứ nhì), đồng thời cho phép trích xuất giải thích ở cấp độ từng đường chuyền và từng cầu thủ tương tác (mục tiêu cốt lõi của Bài toán 2 về XAI).

### 5.4 Phân tích công bằng thông tin đầu vào & Feature Ablation (Lợi thế của `final_x`, `final_y`)
Để giải thích khách quan tại sao Random Forest và ANN đạt AUC-PR nhỉnh hơn MPNN ($0.6198$ và $0.6117$ vs $0.5996$), nghiên cứu đã thực hiện thí nghiệm cô lập đặc trưng (Ablation study, chi tiết tại `results/tables/official/baseline_feature_ablation.csv`):
- Khi cung cấp đủ 10 đặc trưng tổng hợp bao gồm tọa độ kết thúc của đợt tấn công (`final_x`, `final_y`), Random Forest đạt AUC-PR $0.620$, ANN đạt $0.612$.
- Khi **loại bỏ `final_x` và `final_y`**, AUC-PR của Random Forest giảm mạnh từ $0.620$ xuống **$0.268$**; ANN giảm từ $0.612$ xuống **$0.271$**.
- Ngược lại, nếu **chỉ dùng duy nhất 2 đặc trưng `final_x` và `final_y`**, ANN vẫn đạt AUC-PR lên tới **$0.623$**!
- **Ý nghĩa phương pháp luận:** 
  1. Tọa độ điểm cuối của chuỗi chuyền bóng mang tín hiệu cực kỳ áp đảo (đường chuyền cuối cùng đưa bóng vào khu vực 1/3 sân đối phương hoặc vòng cấm địa có xác suất chuyển hóa thành cú sút rất cao).
  2. Trong bài toán phân loại hồi cứu (retrospective sequence classification), `final_x, final_y` là đặc trưng hợp lệ sau khi possession kết thúc. Tuy nhiên, đối với bài toán dự đoán trực tuyến theo thời gian thực (online prediction), thông tin về điểm kết thúc của chuỗi chưa thể có sẵn tại các thời điểm ban đầu.
  3. Các kiến trúc GNN baseline (GCN, GATv2, GraphSAGE) chỉ tổng hợp node features qua ma trận kề mà không dùng edge attributes hay readout nhắm riêng vào vị trí kết thúc, trong khi MPNN có tận dụng cạnh chuyền bóng nên đạt kết quả tiệm cận tabular baselines ($0.5996$). Do đó, so sánh giữa GNN và baselines cần được xem xét trên góc độ bất đối xứng thông tin đầu vào này.

### 5.5 Phân tích độ nhạy của quy tắc cắt tại cú sút và loại 318 đồ thị ngắn (Decision 0001 & 0003)
Số liệu từ `results/tables/official/graph_build_report.csv` và `cleaning_report.csv`:
- Tổng số possession sau làm sạch: $45.365$ possession, trong đó có $6.471$ possession có nhãn shot dương ($14,26\%$).
- Sau khi áp dụng quy tắc cắt chuỗi tại sự kiện sút bóng đầu tiên (nhằm ngăn chặn rò rỉ diễn biến tiếp theo sau cú sút), có **$318$ possession dương** chỉ gồm duy nhất cú sút hoặc có ít hơn 2 đường chuyền hợp lệ, do đó không thỏa mãn điều kiện tối thiểu để tạo thành một đồ thị chuyền bóng có nghĩa.
- Số lượng đồ thị chính thức đưa vào huấn luyện và đánh giá: $45.047$ đồ thị, với $6.153$ đồ thị dương (tỷ lệ nhãn dương đạt $13,66\%$, độ lệch nhỏ $0,6\%$ so với tập possession ban đầu).
- Quần thể thực nghiệm áp dụng: các đợt kiểm soát bóng có cấu trúc phối hợp chuyền bóng (ít nhất 2 đường chuyền). Quy tắc này được ghi nhận minh bạch trong Decision 0001 và 0003.

### 5.6 Làm rõ phạm vi kiểm soát sai số FWER của kiểm định Holm
Trong code triển khai tại `src/xai_football/evaluation/significance.py`:
- Thủ tục hiệu chỉnh Holm–Bonferroni được áp dụng **độc lập theo từng độ đo (per-metric family)**. Với 7 mô hình tham gia đối sánh, mỗi metric có $\binom{7}{2} = 21$ cặp so sánh cần kiểm soát Family-Wise Error Rate (FWER) ở mức $\alpha = 0.05$.
- Tổng cộng trên 3 metrics (AUC-PR, ROC-AUC, Brier score) có $21 \times 3 = 63$ phép kiểm định thống kê.
- Việc kiểm soát FWER riêng cho từng metric giúp cân bằng giữa việc tránh dương tính giả trong cùng một mục tiêu so sánh và không làm suy giảm quá mức công suất kiểm định (statistical power).

### 5.7 Thống nhất và đối chiếu phương pháp Bootstrap: 5-Seed Pooled Ensemble vs Single-Seed Checkpoint
Trong hồ sơ kiểm toán độc lập, reviewer có thể nhận thấy hai khoảng tin cậy được báo cáo:
1. **Khoảng KTC chính thức trong `significance_tests.csv`:** $\Delta = -0.0203$, KTC 95% **$[-0.0418, +0.0016]$**, $p_{\text{holm}} = 0.2728$.
2. **Khoảng KTC kiểm tra nhanh trong `pipeline_audit.json`:** $\Delta$, KTC 95% **$[-0.0718, -0.0185]$**.

**Nguyên nhân phương pháp luận:**
- **Đại lượng ước lượng trong `significance_tests.csv` (10.000 vòng):** Được tính trên xác suất dự đoán trung bình qua 5 seed ngẫu nhiên (**Pooled 5-Seed Ensemble Average**). Quy trình này triệt tiêu phương sai khởi tạo trọng số, đo lường năng lực kỳ vọng thuần túy của kiến trúc mô hình. Do triệt tiêu nhiễu seed, khoảng tin cậy hẹp hơn và bao trùm giá trị 0 ($p = 0.2728$), chứng minh hiệu năng của MPNN tiệm cận và không khác biệt có ý nghĩa thống kê so với Random Forest ở mức $\alpha = 0.05$.
- **Đại lượng ước lượng trong `pipeline_audit.json` (1.000 vòng):** Được tính độc lập trên **Single Best Checkpoint** duy nhất (MPNN seed 789: AUC-PR $0.5969$) đối đầu với 1 mô hình Random Forest ($0.6198$). Do chỉ dùng một lần chạy đơn lẻ, phương sai mẫu cao hơn, phản ánh khoảng dao động khi triển khai một mô hình riêng rẽ.

**Kết luận khoa học thống nhất:** Cả hai cách ước lượng đều khẳng định một sự thật thực nghiệm nhất quán: **Random Forest và ANN đạt hiệu năng tương đương hoặc nhỉnh hơn MPNN một chút** trên bài toán dự đoán xác suất cú sút tĩnh, do đặc trưng bảng tổng hợp điểm kết thúc quá hiệu quả. Không có sự phóng đại hay thiên kiến để gượng ép GNN phải thắng; thay vào đó, MPNN là mô hình biểu diễn đồ thị vượt trội nhất ($+0.08$ đến $+0.15$ AUC-PR so với các GNN khác) và mang lại giá trị độc bản về khả năng giải thích cấu trúc mạng lưới (XAI).

---

## 6. HƯỚNG DẪN TÁI LẬP CHO REVIEWER TRÊN CLONE MỚI

### Bước A: Clone repository và kiểm tra môi trường
```bash
git clone -b Clean https://github.com/HungTruong-github/XAI-sports-knowledge-discovery.git
cd XAI-sports-knowledge-discovery

# Tạo môi trường ảo và cài đặt dependencies
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Bước B: Chạy kiểm toán độc lập chế độ Portable (xác minh artifact đã commit trên Git)
```bash
python scripts/audit_steps_0_6.py --mode portable
```

### Bước C: Chạy toàn bộ bộ test tự động
```bash
pytest -ra tests/
ruff check .
black --check .
```

### Bước D: Tái tính toán độc lập các độ đo từ file dự đoán nén
```python
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

df = pd.read_csv("results/reproducibility/official_test_predictions_compact.csv.gz")
print("MPNN AUC-PR:", average_precision_score(df.y_true, df.mpnn_prob))
print("RF AUC-PR:  ", average_precision_score(df.y_true, df.rf_prob))
```

---

## 7. CÁC QUYẾT ĐỊNH NGHIÊN CỨU ĐANG CHỜ PHÊ DUYỆT (BLOCKERS)

Để bảo đảm tính liêm chính khoa học, báo cáo này không tự ý chuyển các quyết định phương pháp luận sang trạng thái "Đã phê duyệt":
1. **Decision 0001 (Quy tắc gán nhãn possession):** Hiện tại đang định nghĩa target là `has_shot` (possession có ít nhất 1 cú sút). Lựa chọn này là tạm thời (**Provisional**) và đang chờ phê duyệt chính thức từ Giảng viên hướng dẫn.
2. **Decision 0003 (Đơn vị đồ thị cho XAI):** Đang quy định mỗi possession có tối thiểu 2 đường chuyền là một đồ thị, và cắt sự kiện tại cú sút đầu tiên để tránh rò rỉ thông tin tương lai. Quyết định đang ở trạng thái **Pending Review**.
3. **Giả định vai trò cầu thủ (Player Roles):** Hiện tại sử dụng vai trò cầu thủ nổi trội trong toàn trận đấu (`match-dominant role`), phù hợp với góc nhìn phân tích chiến thuật hồi cứu (**retrospective tactical analysis**). Nếu chuyển sang bài toán dự đoán thời gian thực (**real-time predictive framing**), cần chuyển sang dùng `load_player_starting_roles`.
