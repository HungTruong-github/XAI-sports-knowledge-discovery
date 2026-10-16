# scripts/ — Điểm vào chạy pipeline

Mỗi script là một **bước trong pipeline**, chạy được độc lập từ dòng lệnh và
nhận cấu hình từ `configs/`. Script chỉ làm nhiệm vụ điều phối; toàn bộ logic
nằm trong `src/xai_football/`.

## Các script dự kiến

Đánh số theo đúng thứ tự các bước trong `docs/Quy_trinh_XAI_Football.md`:

| Script | Bước | Nhiệm vụ |
|---|---|---|
| `00_audit_data.py` | 1 | Kiểm tra kho dữ liệu thô và tạo inventory |
| `01_download_data.py` | 1 | Tải dữ liệu StatsBomb về `data/raw/` |
| `02_descriptive_stats.py` | 2 | Thống kê mô tả dữ liệu thô và gán nhãn |
| `03_preprocess.py` | 2 | Trích possession, làm sạch, gán nhãn, xuất bảng |
| `04_build_graphs.py` | 3 | Dựng possession graph PyG, lưu `data/processed/` |
| `05_make_splits.py` | 4 | Chia train/val/test theo trận, ghi `data/splits/` |
| `06_train.py` | 5 | Huấn luyện mô hình GNN (GCN, MPNN, GATv2, GraphSAGE) |
| `06_train_baselines.py` | 5 | Huấn luyện baseline phi đồ thị (LR, RF, ANN) |
| `07_evaluate.py` | 6 | Tính bộ độ đo, xuất bảng đối sánh 3 tầng |
| `08_run_xai.py` | 7 | Chạy GNNExplainer / PGExplainer (chưa triển khai) |
| `09_eval_xai.py` | 8 | Fidelity, Sparsity, Stability, ROAR (chưa triển khai) |
| `10_hybrid_centrality.py` | 9 | Tính và kiểm chứng Hybrid Centrality (chưa triển khai) |
| `11_make_figures.py` | 10 | Sinh hình và bảng vào `results/` (chưa triển khai) |

## Quy ước

```bash
python scripts/05_train.py --config configs/model/gcn.yaml --seed 42
```

Mỗi script phải ghi lại bản chụp cấu hình vào thư mục run tương ứng trong
`experiments/` để kết quả tái lập được.
