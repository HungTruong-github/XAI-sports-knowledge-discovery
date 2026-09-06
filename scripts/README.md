# scripts/ — Điểm vào chạy pipeline

Mỗi script là một **bước trong pipeline**, chạy được độc lập từ dòng lệnh và
nhận cấu hình từ `configs/`. Script chỉ làm nhiệm vụ điều phối; toàn bộ logic
nằm trong `src/xai_football/`.

## Các script dự kiến

Đánh số theo đúng thứ tự các bước trong `docs/Quy_trinh_XAI_Football.md`:

| Script | Bước | Nhiệm vụ |
|---|---|---|
| `01_collect_data.py` | 1 | Tải dữ liệu StatsBomb về `data/raw/` |
| `02_preprocess.py` | 2 | Trích possession, làm sạch, gán nhãn |
| `03_build_graphs.py` | 3 | Dựng possession graph, lưu `data/processed/` |
| `04_make_splits.py` | 4 | Chia train/val/test theo trận, ghi `data/splits/` |
| `05_train.py` | 5 | Huấn luyện một backbone theo config |
| `06_evaluate.py` | 6 | Tính bộ độ đo, xuất bảng đối sánh |
| `07_run_xai.py` | 7 | Chạy GNNExplainer / PGExplainer |
| `08_eval_xai.py` | 8 | Fidelity, Sparsity, Stability, ROAR |
| `09_hybrid_centrality.py` | 9 | Tính và kiểm chứng Hybrid Centrality |
| `10_make_figures.py` | 10 | Sinh hình và bảng vào `results/` |

## Quy ước

```bash
python scripts/05_train.py --config configs/model/gcn.yaml --seed 42
```

Mỗi script phải ghi lại bản chụp cấu hình vào thư mục run tương ứng trong
`experiments/` để kết quả tái lập được.
