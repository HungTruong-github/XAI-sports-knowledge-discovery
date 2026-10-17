# notebooks/ — Jupyter Notebooks Khảo sát & Tái lập (Reproducibility)

Thư mục này chứa 7 Jupyter Notebooks chính thức phục vụ mục tiêu giải thích, phân tích định lượng và trực quan hóa từng bước trong pipeline của dự án **XAI-Football**.

## 1. Nguyên tắc thiết kế
1. **Không nhân đôi logic nghiệp vụ**: Mọi logic tiền xử lý, dựng đồ thị, huấn luyện và đánh giá được import từ `src/xai_football/` và gọi qua các entry point trong `scripts/`.
2. **Không tự tạo số liệu giả**: Toàn bộ số liệu và bảng đối sánh được đọc từ các artifact thực tế trong `data/`, `experiments/`, và `results/`.
3. **Traceability & Reproducibility**: Mỗi notebook tuân thủ cấu trúc 10 mục chuẩn hóa:
   - Mục tiêu
   - Vị trí của bước trong pipeline
   - Input và nguồn dữ liệu
   - Config / script / module sử dụng
   - Các bước xử lý
   - Kết quả và cách diễn giải
   - Kiểm tra chất lượng (Quality Checks)
   - Output được lưu ở đâu
   - Cách tái chạy
   - Giới hạn và giả định

---

## 2. Danh mục Notebook & Thứ tự thực thi

| Thứ tự | File Notebook | Bước Pipeline | Nội dung & Đầu ra chính | Trạng thái |
|---|---|---|---|---|
| **00** | `00_environment_and_reproducibility.ipynb` | Bước 0 — Tiên quyết | Kiểm toán phiên bản Python, OS, PyTorch, PyG, CUDA, git commit HEAD, hash cấu hình SHA-256. Xuất `results/metadata/official/environment.json`. | **Đã hoàn thành** |
| **01** | `01_data_audit_and_collection.ipynb` | Bước 1 — Thu thập | Kiểm toán dữ liệu StatsBomb Open Data (Premier League 2015/2016, 380 trận). Trực quan hóa trạng thái thu thập và kích thước file. | **Đã hoàn thành** |
| **02** | `02_preprocessing_and_labels.ipynb` | Bước 2 — Tiền xử lý | Phễu làm sạch dữ liệu possession, phân tích phân bố độ dài, play patterns. Bằng chứng định lượng 380 trận cho Decision Record 0001 (possession-level shot 14.26% vs goal 1.46%). | **Đã hoàn thành** |
| **03** | `03_graph_construction.ipynb` | Bước 3 — Dựng đồ thị | Chuyển đổi possession thành spatio-temporal graphs (45,047 đồ thị). Cắt tại cú sút đầu tiên (`truncate_at_first_shot: true`). Điều tra graph warning tại chấm phạt góc. Minh họa đồ thị trên sân cỏ. | **Đã hoàn thành** |
| **04** | `04_match_level_split.ipynb` | Bước 4 — Phân chia tập | Phân tầng theo TRẬN ĐẤU (Train 70%, Val 15%, Test 15%). Báo cáo phân bố đồ thị sau khi dựng graph (`graph_split_summary.csv`). Kiểm tra chống rò rỉ dữ liệu (leakage checks). | **Đã hoàn thành** |
| **05** | `05_model_training.ipynb` | Bước 5 — Huấn luyện GNN | Trình bày cấu hình siêu tham số chung, theo dõi quá trình hội tụ loss và validation AUC-PR của MPNN, GCN, GATv2, GraphSAGE. Cơ chế multi-seed và early stopping. | **Đã hoàn thành** |
| **06** | `06_model_evaluation.ipynb` | Bước 6 — Đánh giá | Đối sánh công bằng 7 mô hình: GNN (Tầng 2) vs Baselines phi đồ thị (Tầng 3). Trực quan hóa PR curve, ROC curve, Brier score từ prediction scores thật. | **Đã hoàn thành** |
| **07–10** | *Chưa triển khai* | Bước 7–10 — XAI & Khám phá tri thức | Giải thích bằng GNNExplainer/PGExplainer, công thức Hybrid Centrality, so sánh với xT, phát hiện tri thức chiến thuật. | **Chưa triển khai** |

---

## 3. Cách cài đặt và chạy

Chạy một lần tại thư mục gốc repository để cài đặt package ở chế độ editable:

```bash
pip install -e .
```

Sau khi cài đặt, chạy Jupyter Lab hoặc Jupyter Notebook từ thư mục gốc:

```bash
jupyter lab
```

### Chạy tự động toàn bộ notebook từ dòng lệnh:
```bash
python -c "
import nbformat
from nbclient import NotebookClient
from pathlib import Path

for nb_file in sorted(Path('notebooks').glob('*.ipynb')):
    print(f'Running {nb_file}...')
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    client = NotebookClient(nb, timeout=180, kernel_name='python3')
    client.execute()
    with open(nb_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f'Done {nb_file}')
"
```

---

## 4. Thư mục Output

- **Biểu đồ**: Được xuất đồng thời dưới định dạng PNG (300 DPI) và SVG (vector) vào `results/figures/official/`:
  - `01_collection_status.png`, `.svg`
  - `01_match_filesize_dist.png`, `.svg`
  - `02_cleaning_funnel.png`, `.svg`
  - `02_label_comparison.png`, `.svg`
  - `02_possession_length_dist.png`, `.svg`
  - `03_graph_node_edge_dist.png`, `.svg`
  - `03_sample_possession_graph.png`, `.svg`
  - `04_split_distribution.png`, `.svg`
  - `04_split_balance.png`, `.svg`
  - `05_training_loss_curves.png`, `.svg`
  - `05_val_auc_pr_curves.png`, `.svg`
  - `06_model_auc_comparison.png`, `.svg`
  - `06_pr_roc_curves.png`, `.svg`
  - `06_brier_score_comparison.png`, `.svg`
- **Metadata**: `results/metadata/official/environment.json`.
- **Bảng số liệu**: `results/tables/official/`.

---

## 5. Chính sách quản lý đầu ra (Output Policy)
- Biểu đồ và hình ảnh dùng cho khóa luận/báo cáo được lưu trực tiếp vào `results/figures/official/`.
- Không lưu các file trung gian nặng (checkpoint model, raw events) vào git. Thư mục `experiments/` và `models/` được quản lý bởi `.gitignore`.
