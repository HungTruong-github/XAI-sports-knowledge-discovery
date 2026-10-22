# XAI-sports-knowledge-discovery

**Hệ thống trí tuệ nhân tạo có thể giải thích (XAI) trong bài toán đánh giá và khám phá tri thức thể thao**

Hướng nghiên cứu cụ thể: *Mạng chuyền bóng và phân tích chiến thuật bóng đá bằng Graph Neural Network (GNN) kết hợp XAI.*

Repository: `https://github.com/HungTruong-github/XAI-sports-knowledge-discovery`  
Branch hiện tại: `Clean`

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
[5] Huấn luyện mô hình GNN (Bài toán 1)           ──> scripts/06     ──> notebooks/05
[6] Đánh giá mô hình & Baseline 3 tầng            ──> scripts/07     ──> notebooks/06
[7] Áp dụng XAI (Bài toán 2)                      ──> scripts/08
[8] Đánh giá chất lượng giải thích XAI            ──> scripts/09
[9] Tính Hybrid Centrality                        ──> [Interface hoàn tất, chờ quyết định công thức]
[10] Kiểm chứng chuyên môn & Trực quan hóa        ──> [Đang triển khai]
```

### Trạng thái chi tiết từng bước

> Rà soát & chạy lại Bước 0–6 ngày 21/10/2026: xem đối chiếu trước/sau tại [`docs/reproducibility/STEP_0_6_TRUOC_SAU.md`](docs/reproducibility/STEP_0_6_TRUOC_SAU.md).

| Bước | Tên bước | Trạng thái | Nguồn sự thật & Ghi chú |
|---|---|---|---|
| **0** | Environment Audit | **Đã hoàn thành** | `results/metadata/official/environment.json`, `requirements.lock.txt`, `notebooks/00` |
| **1** | Thu thập dữ liệu | **Đã hoàn thành** | Đủ 380/380 trận Premier League 2015/16 (`data/raw/2/27/`) |
| **2** | Tiền xử lý & Gán nhãn | **Đã hoàn thành** | 45.365 clean possessions, 261.205 đường chuyền. Bằng chứng định lượng 380 trận: possession-level shot 14,26% vs goal 1,46% |
| **3** | Xây dựng đồ thị | **Đã hoàn thành** | 45.047 đồ thị PyG, cắt trước cú sút đầu tiên (`truncate_at_first_shot: true`). 9/9 kiểm tra PASS; 1 tọa độ ngoài sân (phạt góc $x=120.9$) được kẹp về biên sân (`coords_clipped`) |
| **4** | Phân chia tập dữ liệu | **Đã hoàn thành** | Chia theo TRẬN ĐẤU (266 Train, 57 Val, 57 Test), không rò rỉ dữ liệu. Có 2 bảng tóm tắt: `split_summary.csv` và `graph_split_summary.csv` |
| **5** | Huấn luyện GNN | **Đã hoàn thành** | 4 GNN × 5 seed (42, 123, 456, 789, 1024) + 3 baseline × 5 seed. Checkpoint `models/official/<model>_best.pt` = seed có Val AUC-PR cao nhất |
| **6** | Đánh giá & Đối sánh | **Đã hoàn thành** | Test set, 5 seed: MPNN AUC-PR 0.5996 ± 0.0042 (tốt nhất nhóm GNN, có ý nghĩa thống kê); RF 0.6198 ± 0.0017, ANN 0.6117 ± 0.0097. Xem `results/tables/official/significance_tests.csv` |
| **7–8**| XAI cho đồ thị | **Code hoàn tất; Chờ chạy toàn bộ Test set** | Module `xai_football.xai`, scripts `08_run_xai.py` và `09_evaluate_xai.py` đã triển khai và có test đầy đủ. Run lịch sử `20261017_091540` (5 đồ thị) được lưu trữ làm bằng chứng đối soát (`HISTORICAL_PROVISIONAL`) |
| **9–10**| Hybrid Centrality & UI | *Interface hoàn thành* | Interface trong `xai_football.centrality` đã triển khai. Quyết định công thức trọng số và kiểm chứng chuyên môn đang chờ hoàn tất thực nghiệm |

---

## 4. Trạng thái các Quyết định Nghiên cứu (Research Decisions)

Tất cả các quyết định nghiên cứu được lưu trữ tại `docs/decisions/` và tuân thủ nguyên tắc không tự ý chốt quyết định thay cho giảng viên/người dùng:

- **[Decision 0001: Quy tắc gán nhãn](docs/decisions/0001-quy-tac-gan-nhan.md)**: `Chưa quyết định (Pending)`. Đã cập nhật bằng chứng định lượng từ toàn bộ 380 trận (shot 14,26%, goal 1,46%). Tạm thời áp dụng `possession_level` (mục tiêu `shot`) để chạy pipeline.
- **[Decision 0002: Công thức Hybrid Centrality](docs/decisions/0002-cong-thuc-hybrid-centrality.md)**: `Chưa quyết định (Pending)`. Chưa triển khai code.
- **[Decision 0003: Đơn vị đồ thị và Ngưỡng độ dài](docs/decisions/0003-don-vi-do-thi-cho-xai.md)**: `Chưa quyết định (Draft/Pending)`. Làm rõ việc cắt đồ thị tại cú sút đầu tiên và loại các possession < 2 đường chuyền.
- **[Decision 0004: Chọn Premier League 2015/2016](docs/decisions/0004-main-dataset-premier-league-2015-16.md)**: `Đã chấp thuận (Accepted)`.
- **Phạm vi Open-play**: Trong `configs/graph.yaml`, hiện tại `play_patterns_keep: null` (giữ mọi play pattern) và vẫn đang ở trạng thái chờ xác nhận.

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
python scripts/08_run_xai.py --model-config configs/model/mpnn.yaml --method gnnexplainer --split test

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
