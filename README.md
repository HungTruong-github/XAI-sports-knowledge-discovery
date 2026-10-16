# XAI-sports-knowledge-discovery

**Hệ thống trí tuệ nhân tạo có thể giải thích (XAI) trong bài toán đánh giá và khám phá tri thức thể thao**

Hướng nghiên cứu cụ thể: *Mạng chuyền bóng và phân tích chiến thuật bóng đá bằng Graph Neural Network (GNN) kết hợp XAI.*

Repository: `https://github.com/HungTruong-github/XAI-sports-knowledge-discovery`  
Branch hiện tại: `REPRODUCIBILITY`

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
│   ├── xai.yaml              Bước 7-8: cấu hình explainer (chưa triển khai)
│   ├── hybrid_centrality.yaml Bước 9:  công thức Hybrid Centrality (chưa chốt)
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
│   ├── training/             Bước 5 (trainer, loss functions, early stopping)
│   ├── evaluation/           Bước 6 (metrics, threshold optimization)
│   ├── xai/                  Bước 7-8 (chưa triển khai)
│   ├── centrality/           Bước 9-10 (chưa triển khai)
│   └── utils/                logging, paths, seed, serialization
│
├── scripts/              # Entry points chạy từng bước pipeline từ CLI
├── notebooks/            # 7 Jupyter Notebooks phân tích, kiểm toán & tái lập
├── tests/                # Unit tests (chống rò rỉ dữ liệu, kiểm tra schema)
├── experiments/          # Log thô từng lần chạy (bị gitignore)
├── models/               # Trọng số mô hình đã huấn luyện (bị gitignore)
├── results/              # Bảng số liệu, metadata và biểu đồ phục vụ báo cáo
│   ├── figures/official/     Biểu đồ định dạng PNG + SVG do notebook sinh ra
│   ├── metadata/official/    Môi trường, config hashes, commit metadata
│   └── tables/official/      Bảng kết quả đối sánh và kiểm tra chất lượng
└── docs/                 # Tài liệu quy trình, báo cáo và Decision Records
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
[7] Áp dụng XAI (Bài toán 2)                      ──> [Chưa triển khai]
[8] Đánh giá chất lượng giải thích XAI            ──> [Chưa triển khai]
[9] Tính Hybrid Centrality                        ──> [Chưa triển khai]
[10] Kiểm chứng chuyên môn & Trực quan hóa        ──> [Chưa triển khai]
```

### Trạng thái chi tiết từng bước

| Bước | Tên bước | Trạng thái | Nguồn sự thật & Ghi chú |
|---|---|---|---|
| **0** | Environment Audit | **Đã hoàn thành** | `results/metadata/official/environment.json`, `notebooks/00` |
| **1** | Thu thập dữ liệu | **Đã hoàn thành** | Đủ 380/380 trận Premier League 2015/16 (`data/raw/2/27/`) |
| **2** | Tiền xử lý & Gán nhãn | **Đã hoàn thành** | 45.365 clean possessions, 261.205 đường chuyền. Bằng chứng định lượng 380 trận: possession-level shot 14,26% vs goal 1,46% |
| **3** | Xây dựng đồ thị | **Đã hoàn thành** | 45.047 đồ thị PyG, cắt trước cú sút đầu tiên (`truncate_at_first_shot: true`). **Cảnh báo**: 1 WARNING ở `normalized_features_in_[0,1]` do quả phạt góc tại $x=120.9$ (hợp lệ theo miền bóng đá) |
| **4** | Phân chia tập dữ liệu | **Đã hoàn thành** | Chia theo TRẬN ĐẤU (266 Train, 57 Val, 57 Test), không rò rỉ dữ liệu. Có 2 bảng tóm tắt: `split_summary.csv` (possession) và `graph_split_summary.csv` (post-graph) |
| **5** | Huấn luyện GNN | **Đã triển khai code** | Hỗ trợ cấu hình multi-seed, smoke test, lưu snapshot config và tách validation/test. Đã hoàn thành 1 run chính thức (seed 42) cho GCN, MPNN, GATv2, GraphSAGE |
| **6** | Đánh giá & Đối sánh | **Đã hoàn thành** | MPNN dẫn đầu nhóm GNN (AUC-PR 0.5747, AUC-ROC 0.8732). Baselines phi đồ thị (RF AUC-PR 0.6234, ANN AUC-PR 0.6141) đang cao hơn do tận dụng đặc trưng toàn cục |
| **7–8**| XAI cho đồ thị | *Chưa triển khai* | Dừng lại chờ xác nhận phương án giải thích |
| **9–10**| Hybrid Centrality & UI | *Chưa triển khai* | Công thức Hybrid Centrality và phạm vi open-play vẫn đang chờ quyết định |

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
# Smoke test kiểm tra nhanh (1 epoch):
python scripts/06_train.py --model-config configs/model/mpnn.yaml --smoke-test
# Huấn luyện baseline phi đồ thị:
python scripts/06_train_baselines.py --config configs/data.yaml

# Bước 6: Đánh giá mô hình
python scripts/07_evaluate.py --config configs/data.yaml
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
