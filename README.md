# XAI-sports-knowledge-discovery

**Hệ thống trí tuệ nhân tạo có thể giải thích (XAI) trong bài toán đánh giá và khám phá tri thức thể thao**

Hướng nghiên cứu cụ thể: *Mạng chuyền bóng và phân tích chiến thuật bóng đá bằng Graph Neural Network (GNN) kết hợp XAI.*

Repository: `https://github.com/HungTruong-github/XAI-sports-knowledge-discovery`  
Branch hiện tại: `feature/08-xai-completion`

---

## 1. Bài toán nghiên cứu

Đề tài gồm **hai bài toán con nối tiếp nhau**, bám sát định hướng học thuật và phản hồi từ giảng viên hướng dẫn (xem `docs/meetings/`):

1. **Bài toán 1 — Dự đoán (Prediction)**: Biểu diễn một chuỗi kiểm soát bóng (possession) thành đồ thị chuyền bóng có hướng không gian - thời gian (spatio-temporal passing graph). Đỉnh biểu diễn cầu thủ, cạnh biểu diễn đường chuyền. Mô hình GNN dự đoán xác suất possession đó dẫn đến cơ hội dứt điểm (shot). Đây là bài toán phân loại nhị phân ở **mức đồ thị (graph-level binary classification)**.
2. **Bài toán 2 — Giải thích (Explanation & Knowledge Discovery)**: Trên mô hình GNN có hiệu năng dự đoán tốt nhất, áp dụng phương pháp XAI cho đồ thị để:
   - (a) Trích xuất giải thích cục bộ (subgraph explanation) chỉ ra các đường chuyền then chốt.
   - (b) Tổng hợp trọng số giải thích qua các possession thành chỉ số **Hybrid Centrality** nhằm xếp hạng vai trò chiến thuật của từng cầu thủ và so sánh đối chiếu với các mô hình bóng đá truyền thống (xT, xA).

> [!IMPORTANT]
> Bài toán 2 phụ thuộc hoàn toàn vào Bài toán 1: Mô hình GNN dự đoán phải đạt chất lượng tốt, học đúng biểu diễn cấu trúc mạng chuyền bóng và không bị rò rỉ dữ liệu thì kết quả giải thích XAI mới có giá trị thực tiễn.

---

## 2. Cấu trúc Repository

```
.
├── configs/              # Toàn bộ cấu hình thực nghiệm định dạng YAML
│   ├── data.yaml             Bước 1-2: thu thập, tiền xử lý, gán nhãn
│   ├── graph.yaml            Bước 3:   định nghĩa đồ thị possession
│   ├── split.yaml            Bước 4:   chia train/val/test theo trận
│   ├── train.yaml            Bước 5:   cấu hình huấn luyện dùng chung
│   ├── xai.yaml              Bước 7-8: cấu hình explainer (GNNExplainer, PGExplainer)
│   ├── hybrid_centrality.yaml Bước 9:  công thức Hybrid Centrality
│   └── model/                Kiến trúc từng backbone GNN và baselines
│
├── data/                 # Dữ liệu (KHÔNG commit, trừ splits/)
│   ├── raw/                  Dữ liệu JSON thô StatsBomb
│   ├── interim/              Possessions và sự kiện đã làm sạch (.parquet)
│   ├── processed/            Đồ thị PyG đã đóng gói (.pt) và index
│   └── splits/               Danh sách match_id và tóm tắt phân chia ← CÓ commit
│
├── src/xai_football/     # Package cốt lõi (pip install -e .)
│   ├── data/                 Bước 1-2, 4 (thu thập, làm sạch, split)
│   ├── graphs/               Bước 3 (dựng đồ thị PyG, trích xuất feature)
│   ├── modeling/             Bước 5 (GNN backbones + baseline feature extractor)
│   ├── training/             Bước 5 (trainer, loss, multi-seed aggregation, atomic I/O)
│   ├── evaluation/           Bước 6 (metrics, threshold, significance tests)
│   ├── xai/                  Bước 7-8 (GNNExplainer, PGExplainer, SHAP, XAI metrics)
│   ├── centrality/           Bước 9-10 (giao diện Hybrid Centrality)
│   └── utils/                logging, paths, seed, serialization, provenance
│
├── scripts/              # Entry points chạy từng bước pipeline từ CLI
├── tools/                # Tiện ích bổ trợ tài liệu (md2docx.py)
├── notebooks/            # 7 Jupyter Notebooks phân tích, kiểm toán & tái lập
├── tests/                # 20 unit & regression test suites
├── experiments/          # Log thô từng lần chạy (bị gitignore)
├── models/               # Trọng số mô hình đã huấn luyện (bị gitignore)
├── results/              # Bảng số liệu, metadata và biểu đồ phục vụ báo cáo
│   ├── figures/official/     Biểu đồ định dạng PNG + SVG
│   ├── metadata/             RUN_REGISTRY.json & metadata official
│   ├── tables/official/      Bảng kết quả đối sánh và kiểm tra chất lượng
│   └── xai/                  Artifact và mask giải thích XAI
└── docs/                 # Tài liệu quy trình, báo cáo tái lập và Decision Records
```

---

## 3. Quy trình Pipeline & Trạng thái triển khai

```
[0] Kiểm toán môi trường & Tính tái lập          ──> notebooks/00
[1] Thu thập dữ liệu (StatsBomb Open Data)        ──> scripts/00, 01 ──> notebooks/01
[2] Tiền xử lý & gán nhãn                         ──> scripts/03     ──> notebooks/02
[3] Xây dựng đồ thị possession                     ──> scripts/04     ──> notebooks/03
[4] Phân chia train/val/test theo TRẬN ĐẤU         ──> scripts/05     ──> notebooks/04
[5] Huấn luyện mô hình GNN (Bài toán 1)           ──> scripts/06, 06b, 06d ──> notebooks/05
[6] Đánh giá mô hình & Baseline 3 tầng            ──> scripts/07     ──> notebooks/06
[7] Khảo sát Readout Pooling MPNN (Quyết định D4) ──> scripts/evaluate_readouts.py
[8] Áp dụng & Đối sánh XAI (Cổng Gate D17)       ──> scripts/08, 09
[9] Cross-Fitting & Tính Hybrid Centrality        ──> scripts/06d, 10, 11
[10] Kiểm chứng chuyên môn & Trực quan hóa Khóa luận ──> scripts/12, docs/thesis/
```

### Trạng thái chi tiết từng bước (100% Hoàn thành trên nhánh `feature/08-xai-completion`)

> Báo cáo kiểm toán tái lập đầy đủ 5 giai đoạn: [`docs/reproducibility/`](docs/reproducibility/)

| Bước | Tên bước | Trạng thái | Nguồn sự thật & Ghi chú |
|---|---|---|---|
| **0** | Environment Audit | **Đã hoàn thành** | `results/metadata/official/environment.json`, `requirements.lock.txt`, `notebooks/00` |
| **1** | Thu thập dữ liệu | **Đã hoàn thành** | Đủ 380/380 trận Premier League 2015/16 (`data/raw/2/27/`) |
| **2** | Tiền xử lý & Gán nhãn | **Đã hoàn thành** | 45.365 clean possessions, 261.205 đường chuyền. Bằng chứng định lượng 380 trận: possession-level shot 14,26% vs goal 1,46% |
| **3** | Xây dựng đồ thị | **Đã hoàn thành** | 45.047 đồ thị PyG, cắt trước cú sút đầu tiên (`truncate_at_first_shot: true`). 9/9 kiểm tra PASS |
| **4** | Phân chia tập dữ liệu | **Đã hoàn thành** | Chia theo TRẬN ĐẤU (266 Train, 57 Val, 57 Test), không rò rỉ dữ liệu. Bổ sung `crossfit_5fold.json` và `hybrid_AB_split.json` |
| **5** | Huấn luyện GNN | **Đã hoàn thành** | 4 GNN × 5 seed (42, 123, 456, 789, 1024) + 3 baseline × 5 seed. MPNN tốt nhất: Test AUC-PR 0.5996 ± 0.0042 |
| **6** | Đánh giá & Đối sánh | **Đã hoàn thành** | MPNN AUC-PR 0.5996; RF 0.6198, ANN 0.6117. Xem `results/tables/official/significance_tests.csv` |
| **7** | Đối sánh Readout Pooling | **Đã hoàn thành** | Huấn luyện 5 seeds cho `max`, `attention`, `mean_lastedge`. Biến thể `mean_lastedge` đạt Val AUC-PR 0.5921 (+0.0074 so với mean 0.5847 < ngưỡng 0.0100 của D4) -> Giữ `mpnn` mean pooling làm backbone. Xem `PHASE_1_REPORT.md` |
| **8** | Đối sánh XAI & Gate D17 | **Đã hoàn thành** | Đối sánh 4 explainers trên 1.222 test graphs (57 trận). Gate D17 (Paired Clustered 95% CI theo match_id): GNNExplainer PASS (+0.1111, CI [0.0939, 0.1286]), PGExplainer FAIL (-0.0128, Stability 0.9872), Last-edge Heuristic PASS (+0.2282, CI [0.2077, 0.2502]). Trạng thái `OFFICIAL_VERIFIED`. Xem `PHASE_3_REPORT.md` |
| **9** | Cross-Fitting & Hybrid Centrality | **Đã hoàn thành** | Giao thức A/B chống rò rỉ mẫu triệt để (Set A cross-fit nội bộ, Set B chuyển giao); co rút Bayesian $m=20$, $n_i \ge 30$; phân tách mạng độc lập theo đội; trung bình có trọng số phút thi đấu; loại trừ thủ môn; tối ưu α* = 0.70 trên Set A (190 trận). Xem `PHASE_4_REPORT.md` |
| **10**| Kiểm chứng & Khóa luận | **Đã hoàn thành** | Đánh giá ngoài mẫu Out-of-Sample trên Set B (190 trận, 314 cầu thủ): Tương quan Spearman của Hybrid Centrality đạt xT/90 = 0.5804 (tăng +0.0514 so với Betweenness thuần), Key Passes/90 = 0.2885 (tăng +0.2510), xA/90 = 0.1816 (tăng +0.2354), Assists/90 = 0.1753 (tăng +0.1446). Hồi quy OLS xT/90 ngoài mẫu: $\Delta R^2_{\text{OOS}} = +0.0786$ (95% Bootstrap CI [0.0144, 0.1268] > 0, unclipped). Bổ sung trích xuất StatsBomb key_pass_id chính thức, lọc trước percentile, phân tích độ nhạy thời lượng (90m, 450m, 900m) và bảng xếp hạng Creative Contribution. Xuất 7 bộ biểu đồ chính thức và hoàn thiện `docs/thesis/ket_qua_va_thao_luan.md` |

---

## 4. Trạng thái các Quyết định Nghiên cứu (Research Decisions)

Tất cả các quyết định nghiên cứu được hoàn thiện và chuẩn hóa đầy đủ tại `docs/decisions/`:

- **[Decision 0001: Quy tắc gán nhãn](docs/decisions/0001-quy-tac-gan-nhan.md)**: `Đã chốt (Finalized)`. Chọn bài toán phân loại nhị phân mức possession với mục tiêu cú sút (`shot`), tỷ lệ dương 14,26%.
- **[Decision 0002: Công thức Hybrid Centrality](docs/decisions/0002-cong-thuc-hybrid-centrality.md)**: `Đã chốt (Finalized)`. Chốt công thức kết hợp tuyến tính với chuẩn hóa Percentile Rank và tham số α* = 0.70.
- **[Decision 0003: Đơn vị đồ thị và Ngưỡng độ dài](docs/decisions/0003-don-vi-do-thi-cho-xai.md)**: `Đã chốt (Finalized)`. Đồ thị possession cắt trước cú sút đầu tiên, ngưỡng lọc tối thiểu $\ge 4$ cạnh cho XAI.
- **[Decision 0004: Chọn Premier League 2015/2016](docs/decisions/0004-main-dataset-premier-league-2015-16.md)**: `Đã chấp thuận (Accepted)`. 380 trận đấu hoàn chỉnh.
- **[Decision 0005: Readout Pooling cho MPNN](docs/decisions/0005-readout-mpnn.md)**: `Đã chốt (Finalized)`. Duy trì MPNN Mean Pooling nguyên bản theo quy tắc D4 (margin +0.0074 < 0.0100).
- **[Decision 0006: Giao thức thực nghiệm XAI](docs/decisions/0006-giao-thuc-xai.md)**: `Đã chốt (Finalized)`. Chốt giao thức Cross-Fitting 5-fold, Cổng Gate D17 và phân chia Set A / Set B (190 / 190 trận).

---

## 5. Hướng dẫn cài đặt và Tái lập Thực nghiệm

### 5.1. Thiết lập môi trường

```bash
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
.venv\Scripts\activate          # Trên Windows
# source .venv/bin/activate     # Trên Linux / macOS

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# Cài đặt package xai_football ở chế độ editable
pip install -e .
```

### 5.2. Chạy pipeline bằng Scripts

```bash
# Bước 1: Kiểm toán dữ liệu thô
python scripts/00_audit_data.py

# Bước 2: Tiền xử lý và gán nhãn
python scripts/03_preprocess.py --config configs/data.yaml

# Bước 3: Xây dựng đồ thị possession
python scripts/04_build_graphs.py --config configs/data.yaml

# Bước 4: Chia tập dữ liệu theo trận (xuất graph_split_summary.csv)
python scripts/05_make_splits.py --config configs/data.yaml

# Bước 5: Huấn luyện mô hình
# (a) Smoke test kiểm tra nhanh (1 epoch, subset nhỏ, thư mục cách ly official_smoke):
python scripts/06_train.py --model-config configs/model/mpnn.yaml --smoke-test

# (b) Huấn luyện single-seed (debug / kiểm tra nhanh):
python scripts/06_train.py --model-config configs/model/mpnn.yaml --seed 42

# (c) Huấn luyện multi-seed đầy đủ (5 seeds theo cấu hình configs/train.yaml):
python scripts/06_train.py --model-config configs/model/mpnn.yaml --seeds 42,123,456,789,1024

# (d) Huấn luyện baseline phi đồ thị (Logistic Regression, Random Forest, ANN):
python scripts/06_train_baselines.py --config configs/data.yaml

# Bước 6: Đánh giá mô hình và đối sánh (tự động phân luồng: partial -> partial_model_comparison.csv)
python scripts/07_evaluate.py --config configs/data.yaml

# Bước 7: Chạy giải thích XAI (ví dụ trên mô hình tốt nhất MPNN)
python scripts/08_explain.py --model-config configs/model/mpnn.yaml --method gnnexplainer --split test

# Bước 8: Tổng hợp và đánh giá chất lượng XAI (BẮT BUỘC chỉ định run_id hoặc xai-dir)
python scripts/09_evaluate_xai.py --run-id 20261017_091540 --allow-historical
```

### 5.3. Khảo sát bằng Jupyter Notebooks

7 notebook được lưu trữ trong `notebooks/`, tuân thủ chuẩn `nbformat` v4:
- `notebooks/00_environment_and_reproducibility.ipynb`
- `notebooks/01_data_audit_and_collection.ipynb`
- `notebooks/02_preprocessing_and_labels.ipynb`
- `notebooks/03_graph_construction.ipynb`
- `notebooks/04_match_level_split.ipynb`
- `notebooks/05_model_training.ipynb`
- `notebooks/06_model_evaluation.ipynb`

### 5.4. Kiểm toán độc lập và Chứng minh tính đúng đắn (Pipeline 0–6)

Pipeline 0–6 đã được rà soát và kiểm toán độc lập bằng công cụ tự động:

```bash
# Chạy kiểm toán độc lập toàn diện (yêu cầu strict exit code)
python scripts/audit_steps_0_6.py --strict

# Kiểm toán ở chế độ portable (dành cho reviewer checkout clone mới từ GitHub)
python scripts/audit_steps_0_6.py --mode portable

# Kiểm tra bộ test tích hợp và unit tests
pytest -ra tests/
```

- Báo cáo chi tiết: [`docs/reproducibility/PIPELINE_0_6_COMPLETION_AUDIT.md`](docs/reproducibility/PIPELINE_0_6_COMPLETION_AUDIT.md)
- Báo cáo máy đọc được: [`results/metadata/official/pipeline_audit.json`](results/metadata/official/pipeline_audit.json)
- Bằng chứng dự đoán nén: [`results/reproducibility/official_test_predictions_compact.csv.gz`](results/reproducibility/official_test_predictions_compact.csv.gz)
- Danh mục bằng chứng: [`results/reproducibility/step_0_6_evidence_manifest.json`](results/reproducibility/step_0_6_evidence_manifest.json)

Toàn bộ biểu đồ định dạng PNG (300 DPI) và SVG được tự động kết xuất vào `results/figures/official/`.

---

## 6. Kiểm thử và CI

Dự án tích hợp kiểm thử tự động với Pytest, Ruff và Black:

```bash
# Kiểm tra code style và import
ruff check .

# Kiểm tra định dạng code
black --check .

# Chạy toàn bộ unit tests
pytest -ra tests/
```

Workflow GitHub Actions được thiết lập tại `.github/workflows/ci.yml`.
