# tests/ — Kiểm thử hệ thống & Tái lập thực nghiệm

Chạy toàn bộ test suite:

```bash
pytest -ra tests/
```

Chạy từng nhóm test cụ thể:

```bash
pytest tests/data/ -v           # Kiểm tra pipeline dữ liệu, split và chống rò rỉ
pytest tests/graphs/ -v         # Kiểm tra dựng đồ thị PyG và feature invariants
pytest tests/modeling/ -v       # Kiểm tra GNN, baselines, multi-seed và trainer
pytest tests/evaluation/ -v     # Kiểm tra metrics, significance testing và evaluate script
pytest tests/xai/ -v            # Kiểm tra explainers, SHAP, metrics và provenance evaluation
pytest tests/centrality/ -v     # Kiểm tra interface Hybrid Centrality
pytest tests/test_notebooks_format.py -v  # Kiểm tra chuẩn nbformat v4
```

---

## 1. Mục tiêu và Nguyên tắc kiểm thử

Trong một dự án nghiên cứu kết hợp GNN và XAI:
- **Lỗi thầm lặng (silent failure)** trong pipeline nguy hiểm hơn lỗi dừng chương trình: chương trình vẫn ra số liệu, nhưng là số liệu sai hoặc bị rò rỉ (leakage).
- **Tính toàn vẹn xuất xứ (provenance integrity)**: Kiểm tra các cơ chế bảo vệ ngăn chặn việc lấy nhầm kết quả pilot/debug/historical ghi đè vào các bảng kết quả chính thức (`results/tables/official/`).

---

## 2. Danh mục kiểm thử chi tiết

| Thư mục / File | Trọng tâm kiểm thử | Mục đích & Invariants được bảo vệ |
|---|---|---|
| `tests/data/test_splits.py` | Chia tập dữ liệu | Đảm bảo tập `match_id` của Train/Val/Test rời nhau hoàn toàn (disjoint split), không rò rỉ giữa các trận. |
| `tests/data/test_split_invariants.py` | Invariants phân chia | Tỷ lệ phân bố nhãn (shot rate), số lượng possessions và số trận đấu trên từng split. |
| `tests/data/test_labeling_temporal.py` | Gán nhãn & Thời gian | Thứ tự thời gian sự kiện, cắt chuỗi trước cú sút đầu tiên (`truncate_at_first_shot`), không rò rỉ tương lai vào đồ thị. |
| `tests/data/test_data_pipeline.py` | Pipeline dữ liệu | Chuẩn hóa tọa độ sân (120x80), kẹp biên sân (`coords_clipped`), trích xuất thông tin định danh và thuộc tính sự kiện. |
| `tests/graphs/test_build.py` | Dựng đồ thị PyG | Node feature (vị trí, vai trò), edge feature (chiều dài, góc, thời gian), tính kết nối của đồ thị chuỗi chuyền bóng. |
| `tests/modeling/test_gnn.py` | Mô hình GNN | Kiểm tra các kiến trúc GNN (GCN, MPNN, GATv2, GraphSAGE) với đầu ra xác suất nhị phân hợp lệ trong $[0, 1]$. |
| `tests/modeling/test_baselines.py` | Mô hình Baseline | Trích xuất đặc trưng bảng và huấn luyện Logistic Regression, Random Forest, ANN. |
| `tests/modeling/test_multi_seed.py` | Multi-seed & Trainer | Tính xác định theo seed, tổng hợp trung bình/độ lệch chuẩn qua các seed, làm sạch artifact cũ khi cấu hình seed thay đổi. *(Đã hợp nhất các kiểm tra artifact)* |
| `tests/modeling/test_optimizers.py` | Optimizer & Scheduler | Khởi tạo optimizer AdamW, learning rate scheduling và weight decay. |
| `tests/modeling/test_trainer_outputs.py` | Đầu ra huấn luyện | Định dạng checkpoint `.pt`, cấu trúc `history.json`, lưu trữ metadata huấn luyện. |
| `tests/evaluation/test_evaluate_script.py` | Script đánh giá mô hình | Ngăn chặn việc ghi đè bảng chính thức khi chạy `--allow-partial` (kết quả được chuyển hướng sang `partial_model_comparison.csv`), bảo vệ `model_comparison.csv`. |
| `tests/evaluation/test_significance.py` | Ý nghĩa thống kê | Paired t-test, Wilcoxon signed-rank test, Bootstrap theo cụm trận đấu, hiệu chỉnh đa giả thuyết Holm-Bonferroni. |
| `tests/xai/test_explainers.py` | Thuật toán XAI | GNNExplainer, PGExplainer, RandomExplainer tạo mask trọng số cạnh hợp lệ trong $[0, 1]$. |
| `tests/xai/test_shap.py` | Baseline giải thích | TreeSHAP và KernelSHAP giải thích các baseline phi đồ thị. |
| `tests/xai/test_metrics.py` | Độ đo XAI | Tính toán Fidelity+ (Fidelity), Fidelity- (Infidelity), Sparsity và Stability. |
| `tests/xai/test_subgraph.py` | Trích xuất đồ thị con | Ánh xạ node/edge quan trọng và trực quan hóa subgraph giải thích. |
| `tests/xai/test_evaluate_xai_provenance.py` | Kiểm toán Provenance XAI | **Regression test bộ tổng hợp XAI**: Ngăn tự động chọn run mới nhất theo tên thư mục; bắt buộc chỉ định run; từ chối run thiếu manifest/sai hash; chặn run pilot/smoke/partial ghi đè official; cô lập run lịch sử bằng cờ `--allow-historical`. |
| `tests/centrality/test_centrality_interface.py` | Hybrid Centrality | Giao diện chuẩn hóa Degree Centrality, Betweenness, PageRank và tích hợp trọng số giải thích XAI. |
| `tests/test_notebooks_format.py` | Định dạng Notebook | Toàn bộ 7 notebook trong `notebooks/` tuân thủ chuẩn `nbformat` v4. |
| `tests/test_environment_metadata.py` | Môi trường thực nghiệm | Kiểm tra phiên bản Python, PyTorch, PyG và tính toàn vẹn của lockfile. |

---

## 3. Các thay đổi dọn dẹp và củng cố (Cleanup & Hardening)

1. **Hợp nhất Multi-seed Tests**:
   - Chuyển toàn bộ assertions từ `tests/modeling/test_multi_seed_artifacts.py` vào `tests/modeling/test_multi_seed.py`.
   - Loại bỏ file trùng lặp và xác minh đầy đủ 11 test cases trong `test_multi_seed.py`.
2. **Bổ sung Provenance Regression Tests**:
   - `tests/xai/test_evaluate_xai_provenance.py`: Đảm bảo `scripts/09_evaluate_xai.py` tuân thủ 100% nguyên tắc xuất xứ, không chấp nhận chạy mập mờ hoặc tự động chọn folder theo thứ tự bảng chữ cái.
   - `tests/evaluation/test_evaluate_script.py`: Bổ sung regression test `test_evaluate_script_partial_does_not_overwrite_official` bảo đảm run dở dang không làm bẩn bảng official.
