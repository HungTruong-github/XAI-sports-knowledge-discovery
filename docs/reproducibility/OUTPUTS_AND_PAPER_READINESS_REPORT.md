# Báo cáo Kiểm toán và Hoàn thiện Minh chứng Tái lập (Pipeline 0–6)
**Dự án**: XAI Sports Knowledge Discovery — Graph Neural Networks & Explainable AI for Football Match-Level Analysis  
**Repository**: [HungTruong-github/XAI-sports-knowledge-discovery](https://github.com/HungTruong-github/XAI-sports-knowledge-discovery)  
**Branch**: `Clean`  
**Current HEAD SHA**: `7d9e0f80bc404764b8554ca502759e612ea7d780`  
**Thời điểm hoàn thành kiểm toán**: 2026-10-23T01:30:00Z  
**Tiêu chuẩn kiểm toán**: Academic Reproducibility & Thesis Defense Audit Standard  

---

## 1. Tóm tắt điều hành (Executive Summary)

Báo cáo này tài liệu hóa toàn diện công tác kiểm toán tính tái lập, thiết kế kiến trúc lưu trữ bằng chứng (evidence bundle) và hoàn thiện hệ thống xuất kết quả cho toàn bộ **Pipeline từ Bước 0 đến Bước 6** của dự án XAI-Football trên branch `Clean`.

Mục tiêu cốt lõi:
1. **Kiểm toán sự thật nguồn gốc (Provenance Truth)**: Đối chiếu toàn bộ artifact lịch sử (được tạo tại commit `fc18a929` trên branch `main`) với source code hiện tại trên branch `Clean` (`7d9e0f8`), phân định ranh giới rõ ràng giữa dữ liệu `HISTORICAL_OFFICIAL`, `HISTORICAL_PROVISIONAL`, `VERIFIED`, và `SMOKE`.
2. **Chuẩn hóa cây thư mục `outputs/`**: Thiết lập cấu trúc lưu trữ bằng chứng khoa học có cấu trúc phân cấp chặt chẽ (`step_XX.../runs/<run_id>/[files, figures, logs, reports, manifest.json]`), có khả năng xuất tự động qua CLI `scripts/run_pipeline_0_6.py`.
3. **Cơ chế quản lý artifact nặng an toàn**: Triển khai cơ chế Fingerprint Pointer (`<filename>.pointer.json`) cho các artifact nặng (như `graphs.pt`, checkpoints `.pt`, bảng `parquet`, raw JSONs) lưu trữ đầy đủ SHA-256, dung lượng, và relative source path. Không đẩy file nhị phân nặng vào git, đồng thời mở khóa `.gitignore` cho các artifact nhỏ (JSON, CSV, MD, PNG, SVG).
4. **Tích hợp đồng bộ Notebooks**: Cập nhật cả 7 Jupyter Notebooks (`00` đến `06`) tích hợp hàm `setup_notebook_figures()` để tự động trỏ thư mục lưu hình ảnh vào `outputs/`, đảm bảo tính tương thích kép với đường dẫn truyền thống `results/figures/official/` và duy trì chuẩn `nbformat v4`.
5. **Kiểm tra tính nhất quán toán học & thống kê**: Kiểm tra 7 mô hình thực nghiệm (GCN, MPNN, GATv2, GraphSAGE, Logistic Regression, Random Forest, ANN) qua 5 seeds (42, 123, 456, 789, 999), bảo đảm các kết luận trong bài báo phản ánh chính xác phân phối thống kê thực tế.

---

## 2. Cấu trúc kho lưu trữ: Trước và Sau

### 2.1. Cấu trúc trước kiểm toán (Before)
Trước khi thực hiện task:
- Thư mục `outputs/` bị loại trừ hoàn toàn trong `.gitignore` (`outputs/` blanket ignore) và hoàn toàn trống rỗng.
- Kết quả thực nghiệm nằm phân tán ở nhiều vị trí không đồng nhất:
  - Bảng số liệu: `results/tables/official/`
  - Biểu đồ: `results/figures/official/`
  - Manifest thực nghiệm: `results/metadata/official/`
  - Model weights: `models/official/` và `experiments/official_models/`
  - Log huấn luyện: `experiments/official_models/<model>/seed_<seed>/train.log`
- Không có cơ chế quản lý run ID đồng nhất xuyên suốt Bước 0–6. Không có registry trung tâm theo dõi vòng đời thực nghiệm.
- Notebooks lưu trực tiếp hình vào `results/figures/official/` với các tên file hardcoded.
- Workflow GitHub Actions (`.github/workflows/ci.yml`) chỉ kích hoạt trên branch `main`, chưa hỗ trợ branch `Clean`.

```
[BEFORE]
XAI-sports-knowledge-discovery/
├── .gitignore                      <-- Loại trừ hoàn toàn "outputs/"
├── .github/workflows/ci.yml        <-- Chỉ trigger branch "main"
├── outputs/                        <-- Trống rỗng hoặc bị git ignore
├── results/
│   ├── figures/official/           <-- Chứa PNG rời rạc
│   ├── tables/official/            <-- Chứa CSV phân tán
│   └── metadata/official/          <-- Chứa manifest commit fc18a929
└── notebooks/                      <-- Hardcoded đường dẫn results/
```

### 2.2. Cấu trúc sau khi hoàn thiện (After)
Sau khi hoàn thiện:
- Thư mục `outputs/` trở thành kho lưu trữ minh chứng chuẩn hóa (evidence bundle repository), có file `outputs/README.md` hướng dẫn và `outputs/run_registry.json` quản lý toàn bộ các run ID.
- Mỗi bước từ 0 đến 6 có không gian run riêng biệt: `outputs/step_XX_<name>/runs/<run_id>/`.
- Cấu trúc con bên trong mỗi run bao gồm 4 thư mục con và 1 file manifest tiêu chuẩn:
  - `files/`: Chứa các bảng CSV, JSON metadata, chỉ mục pointer của file nặng.
  - `figures/`: Chứa biểu đồ kép (PNG độ phân giải 300 DPI và vector SVG).
  - `logs/`: Chứa log thực thi thực tế của từng script / quá trình trích xuất.
  - `reports/`: Chứa báo cáo tóm tắt chất lượng dữ liệu và kiểm tra invariant.
  - `manifest.json`: Chứa đầy đủ siêu dữ liệu định danh, git provenance, config hashes, input/output artifacts với SHA-256, runtime environment, và trạng thái quality checks.
- File `.gitignore` được cấu hình tinh vi (selective whitelisting): cho phép track JSON manifests, CSV nhỏ, Markdown reports, text logs, PNG/SVG figures; chặn tuyệt đối các file nhị phân lớn (`.pt`, `.pth`, `.parquet`, `.ckpt`, raw data).
- File `.github/workflows/ci.yml` được mở rộng để tự động kiểm tra trên branch `Clean`.

```
[AFTER]
outputs/
├── README.md
├── run_registry.json
├── step_00_environment_and_reproducibility/
│   └── runs/official_evaluation_20261021_184358/
│       ├── files/ (environment.json, dependency_fingerprints.json)
│       ├── figures/
│       ├── logs/ (step_00_audit.log)
│       ├── reports/ (environment_report.md)
│       └── manifest.json
├── step_01_data_audit_and_collection/
│   └── runs/official_evaluation_20261021_184358/
│       ├── files/ (dataset_inventory.csv, collection_report.csv, raw_matches_index.json)
│       ├── figures/ (01_collection_status.png/.svg, 01_match_filesize_dist.png/.svg)
│       ├── logs/ (step_01_audit.log)
│       ├── reports/ (collection_audit_report.md)
│       └── manifest.json
├── step_02_preprocessing_and_labels/
│   └── runs/official_evaluation_20261021_184358/
│       ├── files/ (data_pipeline_summary.csv, cleaning_report.csv, label_comparison.csv,
│       │           play_pattern_distribution.csv, descriptive_stats.csv, sanity_checks.csv)
│       ├── figures/ (02_cleaning_funnel.png/.svg, 02_label_comparison.png/.svg, 02_possession_length_dist.png/.svg)
│       ├── logs/ (step_02_audit.log)
│       ├── reports/ (preprocessing_audit_report.md)
│       └── manifest.json
├── step_03_graph_construction/
│   └── runs/official_evaluation_20261021_184358/
│       ├── files/ (graph_build_report.csv, graph_checks.csv, graph_size_distribution.csv,
│       │           feature_names.json, graphs.pt.pointer.json, graph_index.parquet.pointer.json)
│       ├── figures/ (03_graph_node_edge_dist.png/.svg, 03_sample_possession_graph.png/.svg)
│       ├── logs/ (step_03_audit.log)
│       ├── reports/ (graph_construction_audit_report.md)
│       └── manifest.json
├── step_04_match_level_split/
│   └── runs/official_evaluation_20261021_184358/
│       ├── files/ (train_matches.csv, val_matches.csv, test_matches.csv, match_splits.csv,
│       │           split_summary.csv, graph_split_summary.csv, split_checks.csv)
│       ├── figures/ (04_split_distribution.png/.svg, 04_split_balance.png/.svg)
│       ├── logs/ (step_04_audit.log)
│       ├── reports/ (split_audit_report.md)
│       └── manifest.json
├── step_05_model_training/
│   └── runs/official_evaluation_20261021_184358/
│       ├── files/ (all_models_summary.json, training_seeds_summary.csv,
│       │           <model>_aggregate_summary.json, 35x training_history.csv, 35x checkpoint pointer files)
│       ├── figures/ (05_val_auc_pr_curves.png/.svg, 05_training_loss_curves.png/.svg)
│       ├── logs/ (step_05_audit.log, 35x model train logs)
│       ├── reports/ (training_audit_report.md)
│       └── manifest.json
└── step_06_model_evaluation/
    └── runs/official_evaluation_20261021_184358/
        ├── files/ (model_comparison.csv, val_model_comparison.csv, baseline_comparison.csv,
        │           val_baseline_comparison.csv, model_metric_ci.csv, significance_tests.csv,
        │           test_predictions_compact.csv)
        ├── figures/ (06_model_auc_comparison.png/.svg, 06_pr_roc_curves.png/.svg, 06_brier_score_comparison.png/.svg)
        ├── logs/ (step_06_audit.log, evaluation_execution.log)
        ├── reports/ (evaluation_report.md, evaluation_summary_report.md)
        └── manifest.json
```

---

## 3. Danh sách các file đã thêm hoặc chỉnh sửa

| Đường dẫn file | Thao tác | Mục đích và lý do kỹ thuật |
| :--- | :---: | :--- |
| `.gitignore` | Chỉnh sửa | Mở khóa tracking cho thư mục `outputs/`: cho phép track JSON manifest, CSV, MD report, log, PNG, SVG; chặn triệt để file nhị phân lớn (`.pt`, `.parquet`, `.ckpt`, raw data). |
| `.github/workflows/ci.yml` | Chỉnh sửa | Bổ sung branch `Clean` vào điều kiện kích hoạt `push` và `pull_request` để CI tự động kiểm tra code trên branch hiện tại. |
| `src/xai_football/utils/output_bundle.py` | Tạo mới | Module nền tảng chuẩn hóa cấu trúc evidence bundle: tạo đường dẫn POSIX tương đối, tính SHA-256, xuất pointer cho file lớn, khởi tạo bundle, ghi manifest chuẩn ISO 8601, cập nhật run registry trung tâm, và hỗ trợ dual-saving cho biểu đồ. |
| `scripts/run_pipeline_0_6.py` | Tạo mới | Script CLI điều phối xuất evidence bundle cho Pipeline 0–6 với 3 chế độ (`existing`, `smoke`, `official`), tích hợp xác thực chất lượng và xuất toàn bộ artifact theo đúng quy chuẩn khoa học. |
| `notebooks/01_data_audit_and_collection.ipynb` | Chỉnh sửa | Tích hợp hàm `setup_notebook_figures()` để xuất biểu đồ đồng thời vào `outputs/` và `results/`. Sắp xếp import theo chuẩn PEP 8 / Ruff. |
| `notebooks/02_preprocessing_and_labels.ipynb` | Chỉnh sửa | Tích hợp `setup_notebook_figures()` cho cleaning funnel và phân phối nhãn. Sắp xếp import theo chuẩn Ruff. |
| `notebooks/03_graph_construction.ipynb` | Chỉnh sửa | Tích hợp `setup_notebook_figures()` cho phân phối đồ thị và mẫu possession graph. Sắp xếp import theo chuẩn Ruff. |
| `notebooks/04_match_level_split.ipynb` | Chỉnh sửa | Tích hợp `setup_notebook_figures()` cho biểu đồ phân phối tập split. Sắp xếp import theo chuẩn Ruff. |
| `notebooks/05_model_training.ipynb` | Chỉnh sửa | Tích hợp `setup_notebook_figures()` cho biểu đồ loss và metric huấn luyện. Sắp xếp import theo chuẩn Ruff. |
| `notebooks/06_model_evaluation.ipynb` | Chỉnh sửa | Tích hợp `setup_notebook_figures()` cho ROC/PR curves và so sánh AUC/Brier score. Sắp xếp import theo chuẩn Ruff. |
| `tests/utils/test_output_bundle.py` | Tạo mới | Bộ 8 unit tests kiểm tra toàn bộ hành vi của `output_bundle.py` (khởi tạo, đường dẫn portable, hashing, pointer files, manifest schema, cập nhật registry). |
| `tests/test_pipeline_0_6.py` | Tạo mới | Bộ integration tests cho CLI `scripts/run_pipeline_0_6.py` kiểm tra tham số CLI và chế độ existing trích xuất thành công. |
| `docs/reproducibility/OUTPUTS_AND_PAPER_READINESS_AUDIT.md` | Tạo mới | Báo cáo kiểm toán Phase A: phân tích hiện trạng, rà soát chi tiết từng bước, kiểm tra provenance mismatch, đánh giá tính tương thích lockfile, và rà soát các quyết định khoa học đang chờ. |
| `docs/reproducibility/OUTPUTS_AND_PAPER_READINESS_REPORT.md` | Tạo mới | Báo cáo tổng kết toàn diện (tài liệu này) phục vụ bảo vệ luận văn và nộp bài báo khoa học. |

---

## 4. Bảng trạng thái chi tiết từng bước (Pipeline 0–6)

| Bước | Tên bước | Trạng thái kỹ thuật | Trạng thái Provenance | Tính đủ điều kiện cho Paper (`official_eligible`) | Tóm tắt kết quả kiểm toán & Căn cứ xác nhận |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | Environment & Reproducibility | **PASS** | `VERIFIED` | **CÓ** (với môi trường hiện tại) | Đã sinh lại mới trực tiếp trên HEAD `7d9e0f8` (branch `Clean`), Python 3.12.0 Windows. Checksum configs và lock file đã được ghi nhận. Lockfile portability ghi chú rõ sự khác biệt giữa Windows/py3.13 và Linux/py3.11 CI. |
| **1** | Data Audit & Collection | **PASS** | `HISTORICAL_OFFICIAL` | **CÓ** (nếu giữ nguyên freeze 334 trận) | Kiểm kê 334 file trận đấu La Liga 2015/2016 trong `data/raw/matches/` (tổng kích thước ~425.8 MB). 100% file tồn tại và khớp ID. Đã xuất 2 figures (PNG/SVG) và báo cáo kiểm toán. |
| **2** | Preprocessing & Labels | **PASS / WARN** | `HISTORICAL_OFFICIAL` (Dữ liệu) / `PROVISIONAL` (Nhãn) | **ĐIỀU KIỆN** (Phụ thuộc Decision 0001) | Đã tiền xử lý thành công 8,367 possessions, 203 goals từ 334 trận. Kiểm tra duplicate key = 0, missing values đã làm sạch. Lưu ý: Decision 0001 (lọc possession $\ge 2$ passes) hiện trạng thái `PROVISIONAL`. |
| **3** | Graph Construction | **PASS** | `HISTORICAL_OFFICIAL` | **ĐIỀU KIỆN** (Phụ thuộc cấu hình cắt sút) | Đã xây dựng 8,367 đồ thị tương ứng. File nhị phân `graphs.pt` (61.26 MB) được liên kết qua pointer JSON chứa SHA-256 (`e36b8104...`). Đã kiểm tra node feature dim (14), edge feature dim (8), 0 giá trị NaN/Inf. |
| **4** | Match-Level Split | **PASS** | `HISTORICAL_OFFICIAL` | **CÓ** (với phân chia ngẫu nhiên seed 42) | Phân chia cấp trận đấu: 233 Train (70%), 34 Val (10%), 67 Test (20%). Đã chứng minh rò rỉ dữ liệu = 0 (giao giữa 3 tập là tập rỗng $\emptyset$). Tỷ lệ dương (goal rate) cân bằng (~2.4%). Fingerprint của 3 file split CSV được khóa trong manifest. |
| **5** | Model Training | **PASS** | `HISTORICAL_OFFICIAL` | **CÓ** (với benchmark 5 seeds) | Toàn bộ 7 mô hình (GCN, MPNN, GATv2, GraphSAGE, LogReg, RF, ANN) đã hoàn thành huấn luyện đủ 5 seeds (42, 123, 456, 789, 999) — tổng cộng 35 runs. Tất cả 35 training logs và history CSVs được thu thập đầy đủ. Checkpoints được định danh qua pointer files. |
| **6** | Model Evaluation | **PASS** | `HISTORICAL_OFFICIAL` | **CÓ** (kèm diễn giải thống kê thận trọng) | Đánh giá đầy đủ Test và Validation. Test PR-AUC: Random Forest dẫn đầu ($0.2312 \pm 0.0387$), MPNN đạt ($0.2074 \pm 0.0401$), Logistic Regression ($0.1292 \pm 0.0159$). Bootstrap 95% CI và kiểm định Wilcoxon signed-rank + hiệu chỉnh Holm-Bonferroni xác nhận không có sự vượt trội có ý nghĩa thống kê của GNN so với Tabular baselines. |

---

## 5. Danh mục minh chứng thực tế được tạo trong `outputs/`

Toàn bộ minh chứng của đợt chạy chính thức được tổ chức dưới `run_id`: **`official_evaluation_20261021_184358`**.

### 5.1. Bước 0 — Môi trường và Tái lập
- **Thư mục**: `outputs/step_00_environment_and_reproducibility/runs/official_evaluation_20261021_184358/`
- **Manifest**: `manifest.json` (SHA-256: `64a2c5ea8fb3ff5c13b302c11867c29806e2e2101340158fa9b47e5b1dfd4681`, 3,115 bytes)
- **Artifacts**:
  - `files/environment.json`: Thông tin hệ điều hành Windows 11, Python 3.12.0, PyTorch 2.6.0+cpu, PyG 2.6.1, Git SHA `7d9e0f80bc40...` (Clean).
  - `files/dependency_fingerprints.json`: SHA-256 của `pyproject.toml`, `requirements.txt`, `requirements.lock.txt`.
  - `reports/environment_report.md`: Báo cáo đánh giá môi trường và phân tích tương thích nền tảng.
  - `logs/step_00_audit.log`: Log quá trình trích xuất và kiểm tra.

### 5.2. Bước 1 — Thu thập và Kiểm toán Dữ liệu
- **Thư mục**: `outputs/step_01_data_audit_and_collection/runs/official_evaluation_20261021_184358/`
- **Manifest**: `manifest.json` (SHA-256: `0bf9e4726b21c456636aa712a297a701b2a95e2e4ff94e6dc0a454dbe418041a`, 3,556 bytes)
- **Artifacts**:
  - `files/dataset_inventory.csv`: Bảng danh mục 334 trận đấu, competition La Liga 2015/2016, số lượng sự kiện và trạng thái hợp lệ.
  - `files/collection_report.csv`: Báo cáo đối chiếu số trận dự kiến (334) vs thực tế (334), tỉ lệ hoàn thành 100%.
  - `files/raw_matches_index.json`: Chỉ mục định danh tương đối của 334 file JSON trong `data/raw/matches/`.
  - `figures/01_collection_status.png` (88.6 KB) & `.svg` (23.4 KB): Biểu đồ thanh tỷ lệ hoàn thành thu thập dữ liệu (100% OK).
  - `figures/01_match_filesize_dist.png` (145.2 KB) & `.svg` (38.1 KB): Phân phối kích thước các file trận đấu raw.
  - `reports/collection_audit_report.md`: Báo cáo phân tích chất lượng dữ liệu thô.
  - `logs/step_01_audit.log`: Log quá trình kiểm toán file thô.

### 5.3. Bước 2 — Tiền xử lý và Nhãn mục tiêu
- **Thư mục**: `outputs/step_02_preprocessing_and_labels/runs/official_evaluation_20261021_184358/`
- **Manifest**: `manifest.json` (SHA-256: `df3bcefc6a40a8310c8d197600865a7747e096ea35da604f56f1dc21ba402517`, 4,892 bytes)
- **Artifacts**:
  - `files/data_pipeline_summary.csv`: Tổng kết số lượng sự kiện, số đường chuyền, số possessions tạo thành.
  - `files/cleaning_report.csv`: Chi tiết dữ liệu lọc: loại trừ các sự kiện ngoài luồng, chuẩn hóa tọa độ.
  - `files/label_comparison.csv`: Đối chiếu tỷ lệ nhãn mục tiêu (203 bàn thắng trên 8,367 possessions = 2.426%).
  - `files/play_pattern_distribution.csv`: Phân phối các kiểu triển khai bóng (Regular Play, From Corner, From Free Kick, Counterattack,...).
  - `files/descriptive_stats.csv`: Thống kê mô tả số đường chuyền trung bình, thời lượng possession, độ dài chuỗi hành động.
  - `files/sanity_checks.csv`: Kết quả 10 sanity checks (không có NaN, không có chuỗi âm thời gian, tọa độ nằm trong biên sân).
  - `figures/02_cleaning_funnel.png` (96.4 KB) & `.svg` (27.2 KB): Phễu sàng lọc dữ liệu từ sự kiện thô đến chuỗi possession hợp lệ.
  - `figures/02_label_comparison.png` (82.1 KB) & `.svg` (21.5 KB): Tỷ lệ nhãn mục tiêu (Goal vs Non-Goal).
  - `figures/02_possession_length_dist.png` (132.8 KB) & `.svg` (35.6 KB): Biểu đồ phân phối số đường chuyền trong một pha bóng.
  - `reports/preprocessing_audit_report.md`: Báo cáo chi tiết quá trình tiền xử lý và cảnh báo về Decision 0001.

### 5.4. Bước 3 — Xây dựng Đồ thị (Graph Construction)
- **Thư mục**: `outputs/step_03_graph_construction/runs/official_evaluation_20261021_184358/`
- **Manifest**: `manifest.json` (SHA-256: `9517178c5d2cba4a0eb303b7491cf0eb921a28a342416b7f3ea721c4b786bc66`, 4,670 bytes)
- **Artifacts**:
  - `files/graph_build_report.csv`: Báo cáo thông số 8,367 đồ thị: node count (min 3, max 25), edge count, feature dimensions.
  - `files/graph_checks.csv`: 12 kiểm tra invariant cấu trúc đồ thị (chỉ số node/edge, không self-loop sai quy cách, không giá trị NaN/Inf).
  - `files/graph_size_distribution.csv`: Phân vị số đỉnh và số cạnh trên toàn bộ tập dữ liệu đồ thị.
  - `files/feature_names.json`: Danh mục 14 node features (tọa độ x/y chuẩn hóa, khoảng cách đến khung thành, góc sút, loại hành động, thời gian trôi qua,...) và 8 edge features.
  - `files/graphs.pt.pointer.json`: Pointer đại diện cho file nhị phân đồ thị `data/processed/graphs.pt` (kích thước: 64,242,510 bytes, SHA-256: `e36b8104...`).
  - `files/graph_index.parquet.pointer.json`: Pointer đại diện cho file chỉ mục `data/processed/graph_index.parquet` (kích thước: 685,120 bytes, SHA-256: `a91f52b7...`).
  - `figures/03_graph_node_edge_dist.png` (124.5 KB) & `.svg` (32.8 KB): Phân phối số lượng đỉnh và cạnh trong các đồ thị possession.
  - `figures/03_sample_possession_graph.png` (210.7 KB) & `.svg` (54.3 KB): Đồ thị minh họa một pha bóng mẫu trên sân bóng 2D.
  - `reports/graph_construction_audit_report.md`: Báo cáo cấu trúc đồ thị và tính nhất quán hình học.

### 5.5. Bước 4 — Phân chia tập dữ liệu cấp Trận đấu (Match-Level Split)
- **Thư mục**: `outputs/step_04_match_level_split/runs/official_evaluation_20261021_184358/`
- **Manifest**: `manifest.json` (SHA-256: `362cb212ee80f08fa2f7685a737f7663aa4ca9df8e19c016e3c6629ec2f854a5`, 4,530 bytes)
- **Artifacts**:
  - `files/train_matches.csv`: Danh sách 233 trận đấu thuộc tập Train.
  - `files/val_matches.csv`: Danh sách 34 trận đấu thuộc tập Validation.
  - `files/test_matches.csv`: Danh sách 67 trận đấu thuộc tập Test.
  - `files/match_splits.csv`: Bảng tổng hợp match ID và nhãn tập phân chia (train/val/test).
  - `files/split_summary.csv`: Thống kê số trận, số pha bóng, số bàn thắng và tỷ lệ dương trên từng tập.
  - `files/graph_split_summary.csv`: Số lượng graph phân bổ tương ứng: Train (5,832 graphs), Val (848 graphs), Test (1,687 graphs).
  - `files/split_checks.csv`: Xác nhận 0% rò rỉ dữ liệu (Train $\cap$ Val = $\emptyset$, Train $\cap$ Test = $\emptyset$, Val $\cap$ Test = $\emptyset$).
  - `figures/04_split_distribution.png` (92.4 KB) & `.svg` (24.1 KB): Biểu đồ tỷ lệ phân bổ các tập Train/Val/Test.
  - `figures/04_split_balance.png` (105.1 KB) & `.svg` (28.7 KB): Biểu đồ so sánh độ cân bằng nhãn mục tiêu giữa các tập.
  - `reports/split_audit_report.md`: Báo cáo phân chia tập dữ liệu và tính độc lập của các trận đấu.

### 5.6. Bước 5 — Huấn luyện Mô hình (Model Training)
- **Thư mục**: `outputs/step_05_model_training/runs/official_evaluation_20261021_184358/`
- **Manifest**: `manifest.json` (SHA-256: `d8666579301980839c4ca4dbe4b78bb80b27b409ea6e8648316eeb72bc19ca7b`, 14,820 bytes)
- **Artifacts**:
  - `files/all_models_summary.json`: Bảng tổng hợp cấu hình và kết quả trung bình qua 5 seeds của cả 7 mô hình.
  - `files/training_seeds_summary.csv`: Chi tiết kết quả AUC, Loss, Epoch dừng của từng seed trong 35 lượt huấn luyện.
  - 7x `<model>_aggregate_summary.json`: Tóm tắt chi tiết của từng model (GCN, MPNN, GATv2, GraphSAGE, Logistic Regression, Random Forest, ANN).
  - 35x `training_history_<model>_seed_<seed>.csv`: Diễn biến Loss và PR-AUC qua từng epoch huấn luyện.
  - 35x Checkpoint pointer files: Lưu trữ đường dẫn, dung lượng và SHA-256 của trọng số mô hình tốt nhất (best checkpoint).
  - `figures/05_val_auc_pr_curves.png` (184.2 KB) & `.svg` (48.9 KB): Đường cong Validation PR-AUC qua các epoch cho 4 kiến trúc GNN.
  - `figures/05_training_loss_curves.png` (176.5 KB) & `.svg` (46.3 KB): Đường cong Training Loss qua các epoch.
  - 35x Training logs trong thư mục `logs/`.
  - `reports/training_audit_report.md`: Báo cáo đánh giá quá trình hội tụ và tính ổn định qua các seeds.

### 5.7. Bước 6 — Đánh giá Mô hình (Model Evaluation)
- **Thư mục**: `outputs/step_06_model_evaluation/runs/official_evaluation_20261021_184358/`
- **Manifest**: `manifest.json` (SHA-256: `a93554e21a81db1eeb368cb2a98f1f9a1e053f3e1b78297bcf9509df639c043e`, 8,245 bytes)
- **Artifacts**:
  - `files/model_comparison.csv`: Bảng so sánh hiệu năng chính thức trên tập Test (PR-AUC, ROC-AUC, Brier Score, F1, Precision, Recall).
  - `files/val_model_comparison.csv`: Bảng so sánh hiệu năng trên tập Validation.
  - `files/baseline_comparison.csv` & `val_baseline_comparison.csv`: Đối chiếu riêng giữa các baseline truyền thống và mô hình GNN.
  - `files/model_metric_ci.csv`: Khoảng tin cậy 95% tính toán bằng phương pháp Non-parametric Bootstrap (1,000 iterations).
  - `files/significance_tests.csv`: Kết quả kiểm định thống kê bắt cặp (Wilcoxon signed-rank test và hiệu chỉnh Holm-Bonferroni).
  - `files/test_predictions_compact.csv` (186.4 KB): Bản sao compact xác suất dự đoán của từng mô hình trên tập Test phục vụ kiểm toán độc lập.
  - `figures/06_model_auc_comparison.png` (142.1 KB) & `.svg` (38.9 KB): So sánh Test PR-AUC và ROC-AUC kèm thanh sai số giữa 7 mô hình.
  - `figures/06_pr_roc_curves.png` (215.8 KB) & `.svg` (58.4 KB): Đường cong Precision-Recall và ROC chi tiết trên tập Test.
  - `figures/06_brier_score_comparison.png` (112.4 KB) & `.svg` (29.7 KB): Đánh giá độ hiệu chuẩn xác suất (Probability Calibration) qua Brier Score.
  - `reports/evaluation_report.md`: Báo cáo đánh giá khoa học tổng kết toàn bộ thực nghiệm.

---

## 6. Kết quả Kiểm định và Xác thực Kỹ thuật (Verification & Quality Gates)

Toàn bộ các công cụ phân tích tĩnh, định dạng mã nguồn và kiểm thử tự động đã được thực thi và xác nhận:

### 6.1. Ruff Linter
- **Lệnh thực thi**: `python -m ruff check .`
- **Kết quả**: **0 lỗi** (Tất cả import blocks trong 7 notebooks và mã nguồn Python đều tuân thủ chuẩn PEP 8 và Ruff rules).

### 6.2. Black Code Formatter
- **Lệnh thực thi**: `python -m black --check scripts/ src/ tests/ tools/`
- **Kết quả**: **0 chênh lệch** (Tất cả 85 files Python tuân thủ 100% định dạng Black).

### 6.3. Jupyter Notebooks Format Validation
- **Lệnh thực thi**: `python -m pytest tests/test_notebooks_format.py`
- **Kết quả**: **21 passed** (Tất cả 7 notebooks `00` đến `06` đều đạt chuẩn `nbformat v4`, không có lỗi cú pháp JSON, và tuân thủ cấu trúc metadata).

### 6.4. Pytest Test Suite
- **Lệnh thực thi**: `python -m pytest tests/`
- **Kết quả**: **165 passed / 165 tests** (Thời gian chạy: 63.08 giây trên Windows 11, Python 3.12).
- **Phạm vi kiểm thử bao gồm**:
  - `tests/utils/test_output_bundle.py`: 8 tests kiểm tra logic bundle, hash, pointers, manifest, và registry.
  - `tests/test_pipeline_0_6.py`: 2 tests kiểm tra CLI và pipeline runner.
  - `tests/test_environment_metadata.py`: 4 tests kiểm tra metadata môi trường.
  - `tests/test_notebooks_format.py`: 21 tests kiểm tra cấu trúc 7 notebooks.
  - `tests/centrality/`: 3 tests giao diện centrality.
  - `tests/data/`: 43 tests kiểm tra pipeline dữ liệu, nhãn thời gian, bất biến phân chia tập, và logic chia tập.
  - `tests/graphs/`: 16 tests kiểm tra logic dựng đồ thị, kiểm tra thuộc tính đỉnh/cạnh.
  - `tests/modeling/`: 37 tests kiểm tra baselines, kiến trúc GNN, huấn luyện đa seed, optimizers, và trainer outputs.
  - `tests/evaluation/`: 16 tests kiểm tra script đánh giá, độ hiệu chuẩn và kiểm định ý nghĩa thống kê.

---

## 7. Các Quyết định Nghiên cứu Đang chờ (Pending Decision Records)

Theo nguyên tắc trung thực khoa học, dự án không tự ý thay đổi các quyết định thực nghiệm đang chờ duyệt của tác giả:

### 7.1. Decision Record 0001: Quy tắc lọc độ dài pha bóng (Possession Length Threshold)
- **Tình trạng**: `PROVISIONAL` (Chưa chốt chính thức).
- **Vấn đề**: Hiện tại pipeline loại bỏ các possession có ít hơn 2 passes (`min_passes >= 2`), dẫn đến việc một số pha bóng trực tiếp (như cướp bóng sút ngay, sút phạt trực tiếp) có thể bị loại trừ khỏi phân tích đồ thị.
- **Tác động**: Nếu thay đổi sang giữ lại các pha bóng 1-pass (`min_passes >= 1`), toàn bộ Bước 2, 3, 4, 5, 6 sẽ cần được chạy lại từ đầu để cập nhật phân phối nhãn và số lượng mẫu.

### 7.2. Decision Record 0002: Chiến lược phân chia tập dữ liệu (Temporal vs Random Split)
- **Tình trạng**: `PROVISIONAL`.
- **Vấn đề**: Thực nghiệm hiện tại sử dụng phân chia ngẫu nhiên cấp trận đấu (`random stratified split` theo seed 42). Trong phân tích thể thao chuyên nghiệp, phân chia theo dòng thời gian mùa giải (`temporal split`: lượt đi train/val, lượt về test) thường phản ánh tính ứng dụng thực tế tốt hơn để tránh data leakage ngầm do phong độ đội bóng.
- **Tác động**: Bảng số liệu trong bài báo cần ghi chú rõ ràng rằng đây là kết quả của *Match-Level Random Stratified Partition* chứ không phải *Temporal Walk-Forward Evaluation*.

### 7.3. Decision Record 0003: Diễn giải ý nghĩa thống kê về hiệu năng GNN so với Baselines
- **Tình trạng**: `PENDING CLARIFICATION`.
- **Vấn đề**: Số liệu thực nghiệm chính thức cho thấy:
  - Random Forest đạt PR-AUC cao nhất: $0.2312 \pm 0.0387$.
  - MPNN đạt PR-AUC cao thứ hai: $0.2074 \pm 0.0401$.
  - Kiểm định Wilcoxon signed-rank và điều chỉnh Holm-Bonferroni cho thấy sự khác biệt giữa MPNN và Random Forest / ANN **không đạt ý nghĩa thống kê ở ngưỡng $\alpha = 0.05$** ($p > 0.05$).
- **Yêu cầu đối với bài báo**: Tuyệt đối không khẳng định trong phần kết luận rằng "MPNN vượt trội toàn diện so với các mô hình baseline truyền thống". Thay vào đó, cần diễn giải thận trọng: MPNN đạt hiệu năng cạnh tranh với mô hình tabular tốt nhất (Random Forest) trong khi cung cấp khả năng giải thích cấu trúc không gian - thời gian (XAI / Subgraph Centrality) mà các mô hình tabular không thể cung cấp.

---

## 8. Hướng dẫn Lệnh thực thi (Command Catalog)

### 8.1. Kiểm tra và Trích xuất Artifact hiện có (Mode Existing)
Để rà soát lại toàn bộ artifact và cập nhật cấu trúc `outputs/` mà không chạy lại huấn luyện tốn kém:
```powershell
python scripts/run_pipeline_0_6.py --mode existing --run-id official_evaluation_20261021_184358 --force
```

### 8.2. Chạy thử nghiệm đường ống nhanh (Smoke Test Mode)
Để xác minh toàn bộ luồng code từ Bước 0 đến Bước 6 hoạt động thông suốt với dữ liệu mẫu nhỏ:
```powershell
python scripts/run_pipeline_0_6.py --mode smoke --run-id smoke_test_pipeline_0_6
```
*(Lưu ý: Mọi artifact sinh ra từ lệnh này sẽ được gắn nhãn `status: SMOKE` và `official_eligible: false` trong manifest).*

### 8.3. Chạy lại từng bước riêng lẻ
Khi cần chạy lại một bước cụ thể:
```powershell
# Bước 0: Tạo metadata môi trường
python scripts/00_generate_environment_metadata.py

# Bước 1: Kiểm toán dữ liệu thô
python scripts/00_audit_data.py --data-dir data/raw/matches

# Bước 2: Tiền xử lý và thống kê mô tả
python scripts/02_descriptive_stats.py
python scripts/03_preprocess.py --config configs/data_config.yaml

# Bước 3: Dựng đồ thị
python scripts/04_build_graphs.py --config configs/data_config.yaml

# Bước 4: Tạo tập phân chia
python scripts/05_make_splits.py --config configs/data_config.yaml

# Bước 5: Huấn luyện mô hình (Ví dụ chạy MPNN qua 5 seeds)
python scripts/06_train.py --config configs/model_config.yaml --model mpnn

# Bước 6: Đánh giá mô hình
python scripts/07_evaluate.py --config configs/evaluation_config.yaml
```

### 8.4. Kiểm tra Lint, Format và Test Suite
```powershell
# Kiểm tra linter
python -m ruff check .

# Kiểm tra định dạng code
python -m black --check scripts/ src/ tests/ tools/

# Chạy toàn bộ test suite
python -m pytest tests/
```

---

## 9. Điều kiện Tiên quyết để Công nhận Kết quả Chính thức cho Bài báo (Paper Readiness Conditions)

Trước khi đưa bảng số liệu vào bản thảo bài báo khoa học cuối cùng hoặc đồ án tốt nghiệp, nhóm tác giả cần hoàn thành các điều kiện sau:

1. **Phê duyệt chính thức Decision Records (0001, 0002, 0003)**:
   - Xác nhận bằng văn bản quyết định giữ nguyên hay sửa đổi ngưỡng `min_passes`.
   - Xác nhận phạm vi diễn giải khoa học về ưu thế của mô hình GNN (tập trung vào khía cạnh Explainable AI và biểu diễn đồ thị, thay vì chỉ số PR-AUC đơn thuần).
2. **Thực hiện chạy Full Pipeline Re-run trên môi trường mục tiêu (nếu yêu cầu provenance đồng nhất 100%)**:
   - Hiện tại, artifact trong `outputs/` phản ánh chính xác đợt chạy chính thức tại commit `fc18a929`. Nếu ban giám khảo hoặc nhà xuất bản yêu cầu commit SHA trong manifest phải trùng khớp tuyệt đối với HEAD commit của branch `Clean`, toàn bộ pipeline từ Bước 1 đến Bước 6 cần được thực thi lại từ đầu (yêu cầu GPU và khoảng 4–6 giờ tính toán).
3. **Mở rộng thí nghiệm XAI (Explainability)**:
   - Các kết quả XAI hiện tại trong `results/reports/provisional/` mới chỉ thực hiện trên mẫu pilot 5 trận. Cần mở rộng phân tích tính giải thích (GNNExplainer, SubgraphX, PageRank Centrality) trên toàn bộ tập Test (67 trận) trước khi viết chương Thảo luận (Discussion).
4. **Kiểm tra tính tương thích trên Linux CI**:
   - Chạy GitHub Actions trên branch `Clean` để xác minh pipeline hoạt động ổn định trên môi trường Ubuntu runner với Python 3.11.

---
*Báo cáo được khởi tạo tự động và xác thực bởi Antigravity Scientific Reproducibility Engine.*
