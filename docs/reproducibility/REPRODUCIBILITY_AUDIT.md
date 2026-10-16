# Báo cáo Đánh giá Tính Tái Lập (Reproducibility Audit Report)
**Dự án:** XAI-sports-knowledge-discovery  
**Thời điểm đánh giá:** 2026-10-16  
**Môi trường:** Python 3.12, PyTorch 2.5+, PyTorch Geometric 2.6+, Windows 11  

---

## 1. Thông tin Phiên bản & Branch kiểm tra

- **Repository:** `HungTruong-github/XAI-sports-knowledge-discovery`
- **Branch làm việc:** `main` (đồng bộ trực tiếp với `origin/REPRODUCIBILITY` và `origin/main`)
- **HEAD Commit SHA:** `a581cee2268f803bfffcdd0437c4afcaf6393200`
- **Tình trạng git workspace:** Clean (không có commit lẻ chưa được đồng bộ trước đợt audit).

---

## 2. Rà soát Các bước Pipeline Đã Triển Khai Mã Nguồn

| Bước | Tên bước trong quy trình | Mã nguồn chính (`src/`) | Script thực thi (`scripts/`) | Trạng thái mã nguồn |
| :---: | :--- | :--- | :--- | :---: |
| **00/01** | Kiểm tra & Thu thập dữ liệu StatsBomb | `src/xai_football/data/collect.py` | `00_audit_data.py`, `01_download_data.py` | Đầy đủ, hoạt động tốt |
| **02** | Tiền xử lý & Khảo sát nhãn | `src/xai_football/data/possession.py`, `clean.py`, `canonicalize.py`, `labeling.py` | `02_descriptive_stats.py`, `03_preprocess.py` | Đầy đủ, hoạt động tốt |
| **03** | Xây dựng đồ thị possession | `src/xai_football/graphs/build.py`, `features.py`, `dataset.py` | `04_build_graphs.py` | Đầy đủ, hoạt động tốt |
| **04** | Phân chia Train/Val/Test theo trận | `src/xai_football/data/splits.py` | `05_make_splits.py` | Đầy đủ, chống rò rỉ 100% |
| **05** | Huấn luyện GNN & Baseline | `src/xai_football/modeling/gnn/`, `training/`, `baselines/` | `06_train.py`, `06_train_baselines.py` | Đầy đủ 4 GNN + 3 baseline |
| **06** | Đánh giá & Đối sánh 3 tầng | `src/xai_football/evaluation/metrics.py` | `07_evaluate.py` | Đầy đủ bảng đối sánh |
| **07–10**| XAI, Đánh giá XAI, Hybrid Centrality | *Chưa triển khai* | *Chưa triển khai* | **PENDING (Đúng lộ trình)** |

---

## 3. Kiểm kê Artifact Thực Tế Trong Workspace

### 3.1. Dữ liệu thô (Raw Data)
- **Đường dẫn:** `data/raw/2/27/` (Premier League 2015/16).
- **Hiện trạng:** Đầy đủ 760 tệp tin JSON (380 trận events + 380 trận lineups). Dung lượng xấp xỉ 3,4 GB.
- **Tính khả dụng:** Đầy đủ 100%, không cần tải lại từ internet.

### 3.2. Dữ liệu trung gian (Interim Data)
- **Đường dẫn:** `data/interim/official/`
- **Hiện trạng:**
  - `events.parquet`: 91,2 MB (toàn bộ 1.313.773 sự kiện hợp lệ).
  - `possession_events.parquet`: 15,4 MB (261.205 đường chuyền).
  - `possessions.parquet`: 292 KB (45.365 pha bóng sạch).
- **Tính khả dụng:** Đầy đủ 100%.

### 3.3. Dữ liệu đồ thị đã xử lý (Processed Graphs)
- **Đường dẫn:** `data/processed/official/`
- **Hiện trạng:**
  - `graphs.pt`: 98,6 MB (45.047 đối tượng đồ thị `torch_geometric.data.Data`).
  - `graph_index.parquet`: 420 KB (chỉ mục truy xuất nhanh 45.047 đồ thị kèm nhãn, match_id, possession_id).
  - `feature_names.json`: Mô tả 8 đặc trưng đỉnh và 12 đặc trưng cạnh.
- **Tính khả dụng:** Đầy đủ 100%.

### 3.4. Phân chia tập dữ liệu (Splits)
- **Đường dẫn:** `data/splits/`
- **Hiện trạng:** `train_matches.csv` (266 trận), `val_matches.csv` (57 trận), `test_matches.csv` (57 trận), `match_splits.csv` (380 dòng), `split_summary.csv`.
- **Tính khả dụng:** Đầy đủ 100%.

### 3.5. Bảng kết quả & Trọng số mô hình
- **Đường dẫn bảng:** `results/tables/official/` (15 tệp CSV kết quả).
- **Đường dẫn mô hình:** `models/official/` (`mpnn_best.pt`, `gcn_best.pt`, `gatv2_best.pt`, `graphsage_best.pt`).
- **Đường dẫn lịch sử thực nghiệm:** `experiments/official/` (thư mục riêng cho từng mô hình GNN).

### 3.6. Artifact còn thiếu cần bổ sung trong task này
- Báo cáo phân chia ở cấp độ đồ thị: `results/tables/official/graph_split_summary.csv`.
- Tách bạch lưu trữ `validation_metrics.csv` và `run_manifest.json` trong từng lượt chạy mô hình.
- Thư mục hình ảnh chuẩn hóa: `results/figures/official/` để lưu biểu đồ sinh ra từ notebooks.
- Metadata môi trường: `results/metadata/official/environment.json`.
- Decision Record `0003` đang được trích dẫn trong tài liệu nhưng chưa có tệp lưu trữ.

## 4. Các Lỗi và Điểm Không Nhất Quán Tìm Được (Discrepancy Audit)

1. **Quyết định 0001 (Quy tắc gán nhãn):**
   - *Phát hiện:* Tệp `docs/decisions/0001-quy-tac-gan-nhan.md` vẫn lưu bảng số liệu pilot từ 5 trận (585 possession), chưa cập nhật bằng chứng định lượng mùa giải chính thức (380 trận, 45.365 possession).
   - *Thuật ngữ:* Chưa phân biệt rõ ràng giữa `has_shot` (possession có ít nhất 1 cú sút) và `ends_with_shot` (sự kiện kết thúc pha bóng là cú sút).
   - *Giải pháp:* Cập nhật bảng bằng chứng 380 trận từ `label_comparison.csv`, chuẩn hóa câu chữ, và giữ nguyên trạng thái `Chưa quyết định` (chờ người dùng/GV phê duyệt).

2. **Schema đặc trưng baseline phi đồ thị:**
   - *Phát hiện:* Cấu hình `configs/model/baseline_tabular.yaml` khai báo trường `mean_pass_angle`, nhưng module `path_features.py` không trích xuất trường này. Ngược lại, module trích xuất `start_x, start_y` nhưng config không liệt kê. Hơn nữa, `06_train_baselines.py` không lọc hay sắp xếp cột theo config.
   - *Giải pháp:* Thiết lập schema thống nhất; tính `mean_pass_angle` bằng công thức circular mean chuẩn cho góc ($\text{atan2}(\bar{S}, \bar{C})$); trích xuất và sắp xếp cột đầu vào đúng theo danh sách khai báo trong config; ném lỗi tường minh nếu gặp đặc trưng không hợp lệ; bổ sung unit test.

3. **Cấu hình `n_runs` và Seed thực nghiệm:**
   - *Phát hiện:* `configs/train.yaml` ghi `n_runs: 5`, nhưng script huấn luyện trước đây chỉ chạy 1 lần đơn lẻ (`seed: 42`).
   - *Giải pháp:* Nâng cấp `scripts/06_train.py` hỗ trợ lặp nhiều seed độc lập (`experiments/<run>/<model>/seed_<seed>/`), ghi nhận manifest chi tiết cho từng seed, tính toán mean/std qua các run đã chạy thực tế; bổ sung cờ `--smoke-test` phục vụ CI/CD nhanh. Không tuyên bố số liệu là trung bình 5 lần nếu chưa chạy đủ 5 lần.

4. **Tách biệt Validation và Test Protocol & Chính sách Threshold:**
   - *Phát hiện:* Quá trình huấn luyện trước đây lưu metric test nhưng chưa lưu bảng `validation_metrics.csv` độc lập tại epoch tốt nhất; ngưỡng phân loại mặc định cố định ở 0.5 mà chưa lưu ngưỡng tối ưu xác định từ validation.
   - *Giải pháp:* Tách riêng `validation_metrics.csv` và `test_metrics.csv`. Cài đặt cơ chế tìm threshold tối ưu trên tập Validation (tối đa hóa F1), sau đó áp dụng cố định threshold này khi suy luận trên tập Test; ghi lại threshold trong `run_manifest.json`.

5. **Cảnh báo trong kiểm tra đồ thị (`graph_checks.csv`):**
   - *Phát hiện:* Kiểm tra `normalized_features_in_[0,1]` có 1 WARNING: "1 đồ thị có tọa độ/độ dài ngoài [0, 1]".
   - *Nguyên nhân điều tra thực tế:* Đồ thị thứ 13555 (`match_id = 3754087`, `possession_id = 46`) chứa sự kiện đường chuyền phạt góc của James Milner tại tọa độ thực tế $x = 120.9, y = 2.8$ (đặt bóng trên vòng cung phạt góc ngoài đường biên ngang $x = 120.0$). Khi chuẩn hóa chia cho 120.0, giá trị là $1.0075$.
   - *Kết luận:* Đây là giá trị vật lý hoàn toàn hợp lệ trong dữ liệu bóng đá thực tế của StatsBomb, không phải lỗi suy diễn. Giữ nguyên cảnh báo và ghi rõ giải trình chuyên môn trong báo cáo kiểm tra; bổ sung unit test kiểm tra việc xử lý tình huống bóng ngoài biên hợp lệ.

6. **Thiếu tệp Decision 0003:**
   - *Phát hiện:* Nhiều tài liệu tham chiếu tới `0003-don-vi-do-thi-cho-xai.md` nhưng tệp này chưa tồn tại trong `docs/decisions/`.
   - *Giải pháp:* Tạo bản thảo Decision 0003 mô tả bài toán lọc ngưỡng độ dài possession cho phần XAI, đặt trạng thái `Chưa quyết định`.

7. **Thứ tự thời gian khi gán nhãn đường chuyền (Priority 2):**
   - *Phát hiện:* Hàm `label_passes()` trước đây so sánh `0 <= (st - pass_time) <= window_seconds`. Nếu đường chuyền và cú sút xảy ra trong cùng một giây (`st == pass_time`), nhưng cú sút diễn ra TRƯỚC đường chuyền (`s.index < p.index`), đường chuyền sau cú sút vẫn bị gán nhãn dương 1.
   - *Giải pháp:* Tích hợp hàm `is_shot_after_pass()`, sử dụng `index` tuần tự của sự kiện trong trận từ StatsBomb (`s.index > p.index`); xử lý chặt chẽ trường hợp thiếu index (chính sách bảo thủ không gán nhãn dương khi cùng giây), kiểm tra tính nhất quán timestamp (`dt >= 0`) và kiểm tra chu kỳ thi đấu (`period`).
   - *Kiểm chứng thực tế:* Trên 380 trận chính thức, toàn bộ 135 cặp pass-shot cùng giây đều diễn ra trước cú sút (`index_pass < index_shot`), nên bảng số liệu chính thức `label_comparison.csv` không bị sai lệch; hàm mới khắc phục triệt để lỗi biên lý thuyết và đã vượt qua 7 unit tests thời gian.

8. **Đặc trưng cạnh trong baseline phi đồ thị (Priority 3):**
   - *Phát hiện:* `path_features.py` trước đây truy xuất `edge_attr` bằng các chỉ số cột cố định (0, 1, 2, 3, 4, 5, 6, 11). Nếu thứ tự đặc trưng cạnh trong `graph.yaml` thay đổi, dữ liệu bảng sẽ bị trích xuất sai lệch âm thầm.
   - *Giải pháp:* Thiết lập mapping động từ tên đặc trưng sang chỉ mục cột dựa trên `DEFAULT_EDGE_FEATURE_NAMES` hoặc `edge_schema` của đồ thị; kiểm tra tính hữu hạn (`np.isfinite`), kiểm tra độ dài chiều và ném lỗi rõ ràng nếu thiếu cột bắt buộc. Đã bổ sung 4 unit tests (bao gồm test tráo đổi thứ tự cột cho kết quả tương đương).

9. **Minh bạch hóa đối sánh GNN & Chuẩn hóa cấu hình Baseline (Priority 4):**
   - *Phát hiện:* MPNN sử dụng cả `edge_attr` (12 chiều), trong khi GCN, GraphSAGE và GATv2 chỉ sử dụng cấu trúc liên kết và đặc trưng đỉnh; cấu hình `ann` trước đây ghi `dropout: 0.3` dù `sklearn.neural_network.MLPClassifier` không hỗ trợ dropout.
   - *Giải pháp:* Tài liệu hóa minh bạch đầu vào của từng backbone; xóa tham số `dropout` gây hiểu lầm trong `baseline_tabular.yaml` và thay bằng tham số $L_2$ regularization `alpha: 0.0001`; khẳng định rõ trong báo cáo việc MPNN có hiệu năng cao hơn một phần do được cung cấp thêm thông tin cạnh.

10. **Giả định thời gian và Tính sẵn sàng của dữ liệu vai trò cầu thủ (Priority 5):**
    - *Phát hiện:* `load_player_roles` xác định vai trò cầu thủ dựa trên vị trí xuất hiện nhiều nhất trong toàn bộ 90 phút của trận đấu. Trong kịch bản dự đoán trực tiếp (real-time stream), các sự kiện cuối trận chưa khả dụng tại các pha bóng đầu trận.
    - *Giải pháp:* Phân định rõ giả định bài toán: Đây là bài toán Đánh giá chuỗi đường chuyền hồi cứu (Retrospective Possession Evaluation / Post-match Knowledge Discovery). Đồng thời, bổ sung hàm `load_player_starting_roles` chỉ lấy vị trí xuất hiện đầu tiên của cầu thủ (`event_index` nhỏ nhất) phục vụ kịch bản dự đoán không rò rỉ thông tin tương lai, kèm unit test kiểm chứng.

11. **Tính bền vững của Notebooks (Priority 7):**
    - *Phát hiện:* Một số notebook có chứa giá trị fallback tĩnh (`len(raw_files) or 380`, `else 662`, `b.get_height()/380*100`) có nguy cơ che giấu việc thiếu dữ liệu thực tế.
    - *Giải pháp:* Loại bỏ toàn bộ các fallback tĩnh; đọc dữ liệu trực tiếp từ các bảng kết quả chính thức; bổ sung thông báo hướng dẫn lệnh CLI khi thiếu artifact thay vì ném lỗi không rõ ràng; kiểm định toàn bộ 7 notebook qua `test_notebooks_format.py` (21/21 PASS).

12. **Metadata thực nghiệm và Định danh phiên bản (Priority 8):**
    - *Giải pháp:* Tạo tệp `results/metadata/official/experiment_manifest.json` liên kết toàn bộ kết quả với cấu hình, mã SHA-256, checkpoint, và split; cập nhật `environment.json` tự động ghi nhận branch, commit SHA, trạng thái git tree thực tế và config hashes.

---

## 5. Kết Quả Có Thể Tái Lập Ngay

1. **Toàn bộ bước kiểm tra dữ liệu và tiền xử lý (Bước 0–2):** Dữ liệu thô và Parquet đã có sẵn trên đĩa; các script `00_audit_data.py`, `02_descriptive_stats.py` có thể chạy lại ngay lập tức và cho kết quả trùng khớp 100%.
2. **Phân chia tập dữ liệu (Bước 4):** Seed 42 và danh sách trận đấu cố định cho ra kết quả chia tập 70/15/15 đồng nhất hoàn toàn với bảng `split_summary.csv`, `graph_split_summary.csv` và `split_checks.csv` (5/5 PASS, 0 match overlap).
3. **Huấn luyện và đánh giá Baseline phi đồ thị (Bước 5–6):** Chạy lại `06_train_baselines.py` trên dữ liệu bảng trích xuất từ 45.047 đồ thị mất dưới 1 phút, tái lập chính xác hiệu năng của Logistic Regression, Random Forest và ANN theo schema đặc trưng đã chuẩn hóa.
4. **Đánh giá và tổng hợp mô hình (Bước 6):** Script `07_evaluate.py` hỗ trợ thu thập kết quả đa seed và tái lập bảng `model_comparison.csv`.
5. **Bộ kiểm thử tự động (Unit Tests & Format):** 100% tests thực thi thành công (59 tests gốc + 15 regression tests mới = 74 tests PASS).

---

## 6. Kết Quả Chưa Thể Tái Lập Đầy Đủ Ngay Lập Tức

1. **Chạy lặp lại 5 seed đầy đủ (200 epochs) cho cả 4 backbone GNN:**
   - Việc huấn luyện 4 mô hình $\times$ 5 seeds = 20 lượt huấn luyện trên toàn bộ 45.047 đồ thị với CPU đòi hỏi từ 10–15 giờ tính toán liên tục.
   - Trạng thái hiện tại: Đã huấn luyện và kiểm chứng đầy đủ 1 seed chuẩn (`seed = 42`) cho cả 4 backbone GNN (MPNN, GCN, GATv2, GraphSAGE). Khung thực thi đa seed (`n_runs`) đã được hoàn thiện trong mã nguồn; các seed tiếp theo được đánh dấu là `pending execution` và sẽ được bổ sung khi có tài nguyên GPU phù hợp.

---

## 7. Các Việc Cần Giải Quyết Trước Khi Bắt Đầu Bước 7 (XAI)

1. **Chốt chính thức Decision 0001 (Quy tắc gán nhãn):** Nhận phản hồi/phê duyệt của Giảng viên hướng dẫn về việc giữ nhãn `possession_level` với mục tiêu `shot` (hiện đang là đề xuất kỹ thuật).
2. **Chốt chính thức Decision 0003 (Đơn vị đồ thị cho XAI):** Quyết định việc đánh giá XAI trên toàn bộ possession sạch hay chỉ lọc tập possession có $\ge 4$ đường chuyền để đảm bảo đồ thị đủ dày cho việc đo Fidelity/Sparsity.
3. **Chốt phạm vi Open-play trong `configs/graph.yaml`:** Xác định danh sách `play_patterns_keep` (giữ toàn bộ play pattern hay chỉ giữ các pha bóng mở).
4. **Khóa checkpoint MPNN chính thức:** Mô hình MPNN (Best Val AUC-PR = 0.5831) đã sẵn sàng làm backbone cho GNNExplainer và PGExplainer.

---
*Báo cáo được cập nhật sau khi hoàn thành đợt sửa lỗi và kiểm định toàn diện tính tái lập.*
