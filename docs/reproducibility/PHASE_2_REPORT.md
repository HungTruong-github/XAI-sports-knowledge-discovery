# Báo cáo Giai đoạn 2 — Chuẩn hoá và Tái cấu trúc Pipeline XAI

**Thời gian thực hiện:** 2026-11-02  
**Nhánh thực hiện:** `feature/08-xai-completion`  
**Trạng thái kiểm thử:** PASS 100% (pytest `tests/xai/`, ruff 100% xanh)  

---

## 1. Mục tiêu và Phạm vi Giai đoạn 2

Giai đoạn 2 tập trung chuẩn hoá toàn diện giao thức giải thích mô hình GNN (XAI), hoàn thiện các lớp Explainer theo các quyết định kỹ thuật đã chốt (D5, D6, D7, D8, D10, D12, D17), đảm bảo tính tái lập tuyệt đối và sẵn sàng cho các thực nghiệm quy mô lớn ở Giai đoạn 3.

---

## 2. Các công việc và Cải tiến kỹ thuật đã thực hiện

### 2.1. Chuẩn hoá GNNExplainer và PGExplainer
- **GNNExplainer (Quyết định D5):** Cố định `explanation_type="phenomenon"` và `target=1` (giải thích tại sao đồ thị dẫn đến cú sút). Khắc phục triệt để lỗi giải thích nhầm lớp dự đoán (model phenomenon).
- **PGExplainer (Quyết định D8):** Cố định huấn luyện với `target=1`. Loại bỏ hoàn toàn khối `except Exception: pass` nuốt lỗi; thêm kiểm đếm số bước tối ưu thành công (`success_steps`) và phát sinh ngoại lệ nếu toàn bộ các bước tối ưu thất bại. Thực nghiệm kiểm chứng: 30 epochs trên 100 đồ thị đạt 3.000/3.000 bước thành công (0 lỗi).

### 2.2. Xây dựng Baselines đối chứng (Quyết định D17)
- **RandomExplainer:** Tạo mặt nạ cạnh ngẫu nhiên đều trong đoạn $[0, 1]$, tính trung bình qua 5 seed độc lập trên mỗi đồ thị để giảm phương sai ngẫu nhiên.
- **LastEdgeExplainer:** Baseline quy tắc nghiệp vụ gán trọng số 1.0 cho đường chuyền cuối cùng (chuẩn bị dứt điểm) và 0.0 cho tất cả các đường chuyền trước đó.
- Cả 4 explainers (`gnn_explainer`, `pg_explainer`, `random`, `last_edge`) đã được tích hợp vào hàm xuất xưởng `build_explainer()`.

### 2.3. Lấy mẫu phân tầng tất định (Quyết định D7)
- Tạo module `src/xai_football/xai/sampling.py` với hàm `stratified_sample()`:
  - Lọc ngưỡng chặt chẽ: chỉ giữ các đồ thị có từ `min_edges` cạnh trở lên (mặc định D2: `min_edges = 4`).
  - Phân tầng cân bằng theo lớp (dương/âm) và theo quy mô đồ thị: 4 cạnh, 5–6 cạnh, $\ge 7$ cạnh.
  - Phủ đều tất cả 57 trận đấu thuộc tập kiểm thử (Test set).
  - Tái lập tất định tuyệt đối theo `seed=42`.
  - Tự động xuất manifest kiểm kê `sample_manifest.csv` lưu lại `match_id`, `possession_id`, `size_bucket`, `play_pattern`.

### 2.4. Tổng hợp điểm cầu thủ & Khử thiên lệch (Quyết định D12)
- Cập nhật hàm `aggregate_player_importance()`:
  - Bổ sung cột `mean_importance = total_importance / participations` để triệt tiêu thiên lệch tích luỹ điểm cho các cầu thủ chạm bóng nhiều lần.
  - Bổ sung 2 cột bóc tách tác động cạnh cuối (last-edge ablation): `total_importance_no_last` và `mean_importance_no_last`.
  - Cập nhật hàm `map_explanation_to_passes()` bổ sung cờ `is_last_edge` để truy vết đường chuyền quyết định.

### 2.5. Hoàn thiện Bộ độ đo chất lượng XAI (Quyết định D6, D10)
- **Ngưỡng Top-K (D6):** Cài đặt công thức thích ứng $k = \max(1, \text{round}(0.3 \times E))$ trong `resolve_top_k()`.
- **Fidelity:** Bổ sung kiểm tra đảm bảo khi xoá hoặc giữ các cạnh trong đồ thị, ma trận đặc trưng nút `graph.x` hoàn toàn được bảo toàn nguyên vẹn.
- **Đường cong Sparsity-Fidelity:** Bổ sung hàm `compute_sparsity_fidelity_curve()` quét $k \in [1, E-1]$.
- **ROAR (D10):** Đặt trạng thái `"not_run"` và ghi nhận lý do chi phí tính toán ngoài phạm vi đề tài vào tài liệu.

### 2.6. Nâng cấp Script thực thi `scripts/08_explain.py` và `scripts/09_evaluate_xai.py`
- Hỗ trợ tham số `--sample-mode` (`pilot`, `official`, `head`, `custom`).
- Hỗ trợ chia mảnh xử lý song song `--shard-idx` và `--num-shards`, cùng cờ `--resume`.
- Phân nhóm kết quả đo đạc theo ma trận nhầm lẫn (`confusion_type`: TP, FP, FN, TN), nhóm quy mô (`size_bucket`), và hình thức phối hợp (`play_pattern`).
- Bổ sung kiểm định Cổng Gate D17 trong `scripts/09_evaluate_xai.py`: kiểm tra khoảng tin cậy bootstrap 95% của $(\text{Fidelity}^+ - \text{Fidelity}^-) - \text{Random}$ có hoàn toàn lớn hơn 0 hay không.

---

## 3. Kết quả Kiểm thử Đơn vị (Unit Tests)

Bộ kiểm thử mới `tests/xai/test_xai_phase2.py` đã vượt qua 8/8 bài kiểm tra:
1. `test_stratified_sample_determinism_and_buckets`: PASS
2. `test_random_explainer`: PASS
3. `test_last_edge_explainer`: PASS
4. `test_resolve_top_k`: PASS
5. `test_aggregate_player_importance_unbiased`: PASS
6. `test_map_explanation_to_passes`: PASS
7. `test_fidelity_preserves_node_features`: PASS
8. `test_sparsity_fidelity_curve`: PASS

Toàn bộ 27 bài kiểm thử trong thư mục `tests/xai/` đạt tỷ lệ thành công 100% (27 passed, 0 failed).

Smoke test đầu-cuối trên 10 đồ thị mẫu với cả 4 thuật toán giải thích (`gnn_explainer`, `pg_explainer`, `random`, `last_edge`) cùng mô hình SHAP baseline chạy thành công hoàn toàn và xuất đầy đủ artifacts.
