# BÁO CÁO TÁI LẬP GIAI ĐOẠN 3: THỰC THI VÀ KIỂM TOÁN PIPELINE GIẢI THÍCH (XAI) VÀ CROSS-FITTING

**Dự án:** XAI Football — Graph Neural Networks & Explainable AI on Passing Networks  
**Nhánh:** `feature/08-xai-completion`  
**Ngày hoàn thành:** 2026-11-02  
**Mục tiêu:** Thực thi kiểm toán thực nghiệm giải thích đồ thị trên mô hình MPNN backbone chính thức, đối sánh 4 phương pháp (`gnn_explainer`, `pg_explainer`, `random`, `last_edge`), thẩm định Cổng chất lượng Gate D17 theo Quyết định D17, và thực hiện Cross-Fitting 5-fold để trích xuất 100% out-of-fold player importance sạch rò rỉ trên 380 trận đấu.

---

## 1. Thiết kế thực nghiệm và Đối sánh 4 Phương pháp XAI

Thực nghiệm được thực hiện trên tập Test độc lập (57 trận, 3.860 đồ thị hợp lệ có $\ge 4$ cạnh theo Quyết định D2):
- Phương thức lấy mẫu: Stratified Sampling theo Quyết định D7 trên toàn bộ 57 trận test. Trong 3.860 đồ thị hợp lệ có đúng 611 đồ thị dương (có cú sút) và 3.249 đồ thị âm; do đó kích thước mẫu cân bằng tối đa chính thức là **1.222 đồ thị** (611 dương, 611 âm, seed = 42).
- Các thuật toán giải thích:
  1. `gnn_explainer`: Tối ưu hóa mặt nạ cạnh theo cơ chế tối đa thông tin tương hỗ (phenomenon mode, target class = 1).
  2. `pg_explainer`: Mạng tham số hóa MLP học phân phối mặt nạ cạnh có trọng số, tối ưu hóa trên tập huấn luyện (target class = 1).
  3. `random`: Baseline mặt nạ ngẫu nhiên đồng đều (Uniform[0, 1]).
  4. `last_edge`: Heuristic bóng đá gán trọng số 1.0 cho đường chuyền cuối cùng (chuỗi bóng dẫn tới dứt điểm) và 0.0 cho các đường chuyền trước.
  5. Baseline phi đồ thị: SHAP TreeExplainer trên mô hình Random Forest tabular.

---

## 2. Kết quả đối sánh chi tiết và Đánh giá Cổng kiểm soát Gate D17

Nguồn dữ liệu: `results/tables/official/xai_comparison.csv` và `results/tables/official/xai_comparison_manifest.json`  
Trạng thái xác thực: **`OFFICIAL_VERIFIED`** (1.222 đồ thị trên 57 trận test)

| Thuật toán | Số đồ thị | Fidelity+ (mean ± std) | Fidelity- (mean ± std) | $\Delta\text{Fidelity}$ | Sparsity | Stability | Deletion AUC | Insertion AUC | Trạng thái Gate D17 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gnn_explainer` | 1,222 | 0.0492 ± 0.1563 | -0.1116 ± 0.2296 | +0.1608 | 0.6939 | 0.2851 | 0.4041 | 0.4901 | **PASS** (CI hoàn toàn > 0) |
| `pg_explainer` | 1,222 | -0.0125 ± 0.1266 | -0.0493 ± 0.2429 | +0.0368 | 0.6939 | **0.9872** | 0.4892 | 0.4009 | **FAIL** (CI vắt qua 0) |
| `random` | 1,222 | 0.0000 ± 0.1518 | -0.0496 ± 0.2219 | +0.0496 | 0.6939 | — | 0.4451 | 0.4447 | Baseline |
| `last_edge` | 1,222 | **0.1237 ± 0.2329** | **-0.1542 ± 0.1407** | **+0.2779** | 0.6939 | — | **0.4054** | **0.4800** | **PASS** (CI hoàn toàn > 0) |

### Kiểm định thống kê Paired Clustered Bootstrap theo `match_id` cho Gate D17:
- **Phương pháp kiểm định:** Lấy mẫu lại 1.000 lần có hoàn lại theo cụm trận đấu (`cluster_by: match_id`, 57 cụm trận). Trên từng đồ thị được lấy mẫu, tính trực tiếp hiệu số theo cặp:
  $$\Delta_i = (\Delta\text{Fid}_{i, \text{expl}} - \Delta\text{Fid}_{i, \text{rand}})$$
- **Tiêu chí Gate D17:** Khoảng tin cậy 95% cụm của $\text{Mean}(\Delta_i)$ phải nằm hoàn toàn phía trên 0.
- **Kết quả kiểm định:**
  - `gnn_explainer`: Paired Mean = **$+0.1111$**, 95% Cluster CI = **$[+0.0939, +0.1286]$** $\rightarrow$ **PASS**.
  - `pg_explainer`: Paired Mean = **$-0.0128$**, 95% Cluster CI = **$[-0.0306, +0.0049]$** $\rightarrow$ **FAIL**.
  - `last_edge`: Paired Mean = **$+0.2282$**, 95% Cluster CI = **$[+0.2077, +0.2502]$** $\rightarrow$ **PASS**.
- **Kết luận Cổng Gate D17:** **ĐẠT (PASS)**. Cả `last_edge` và `gnn_explainer` đều vượt qua Random Baseline có ý nghĩa thống kê ở mức tin cậy 95% khi xét cụm trận đấu.

### Phân tích ý nghĩa khoa học (RQ2):
1. **Sức mạnh vượt trội của cấu trúc pha bóng cuối (`last_edge`):** Đạt mức tăng $\Delta\text{Fidelity}$ lên tới $+0.2282$ so với ngẫu nhiên. Điều này tái khẳng định trong mạng chuyền bóng, hành động then chốt đưa bóng vào khu vực dứt điểm chi phối dự đoán của mô hình.
2. **Hiệu năng của GNNExplainer trên mẫu lớn:** Khi mở rộng lên quy mô toàn bộ 1.222 đồ thị, GNNExplainer thể hiện năng lực tối ưu mặt nạ cạnh vượt trội rõ rệt so với Random Baseline ($+0.1111$ chênh lệch $\Delta\text{Fid}$, CI $[0.0939, 0.1286]$ hoàn toàn tách rời 0).
3. **Độ ổn định (Stability):** PGExplainer tiếp tục thể hiện tính ổn định gần như tuyệt đối (**0.9872** so với 0.2851 của GNNExplainer), do mạng nơ-ron tham số hóa học được biểu diễn trọng số chung thay vì tối ưu hóa lặp lại dễ rơi vào cực tiểu cục bộ khác nhau của GNNExplainer.

---

## 3. Huấn luyện và Giải thích Out-of-Fold bằng Cross-Fitting 5-Fold (D11)

Để phục vụ bài toán xây dựng chỉ số Hybrid Centrality cho toàn bộ 380 trận mà không vi phạm quy tắc rò rỉ dữ liệu (data leakage), mô hình MPNN được huấn luyện và giải thích theo cơ chế Cross-Fitting 5-fold:

### Kết quả huấn luyện 5 Folds:

| Fold | Trận Train | Trận Test OOF | Đồ thị OOF | Val AUC-PR | Test AUC-PR | Test AUC-ROC | Test Brier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 304 | 76 | 9,132 | 0.5771 | 0.5781 | 0.8806 | 0.1416 |
| Fold 2 | 304 | 76 | 8,970 | 0.5975 | 0.5981 | 0.8875 | 0.1450 |
| Fold 3 | 304 | 76 | 8,888 | 0.5956 | 0.5716 | 0.8710 | 0.1639 |
| Fold 4 | 304 | 76 | 9,078 | 0.5873 | 0.5812 | 0.8814 | 0.1462 |
| Fold 5 | 304 | 76 | 8,979 | 0.6219 | 0.5688 | 0.8806 | 0.1352 |
| **Trung bình** | — | — | — | **0.5959** | **0.5796 ± 0.0116** | **0.8802 ± 0.0059** | **0.1464** |

### Kết quả giải thích Out-of-Fold:
- Tổng số đồ thị OOF được giải thích: **11,400 đồ thị** (đảm bảo $\ge 4$ cạnh và tối đa 30 đồ thị/trận trên toàn bộ 380 trận).
- Tổng số bản ghi điểm quan trọng cầu thủ: **71,126 dòng** (`player_importance_oof.parquet`).
- Các cột dữ liệu theo Quyết định D12: `player_id`, `mean_importance`, `total_importance`, `mean_importance_no_last`, `total_importance_no_last`.

---

## 4. Danh mục Artifacts đã lưu

1. **Bảng đối sánh XAI và Manifest chính thức:**
   - `results/tables/official/xai_comparison.csv`
   - `results/tables/official/xai_comparison_manifest.json`
2. **Dữ liệu thực nghiệm XAI chính thức (1.222 đồ thị trên 57 trận test):**
   - `results/xai/official/mpnn/official_test/xai_run_manifest.json`
   - `results/xai/official/mpnn/official_test/sample_manifest.csv`
   - `results/xai/official/mpnn/official_test/xai_metrics_by_graph.csv`
   - `results/xai/official/mpnn/official_test/xai_metrics_summary.csv`
   - `results/xai/official/mpnn/official_test/sparsity_fidelity_curves.csv`
   - `results/xai/official/mpnn/official_test/shap_feature_importance.csv`
3. **Mô hình Cross-Fitting Checkpoints & Manifest:**
   - `models/crossfit/crossfit_train_manifest.json`
   - `models/crossfit/fold_1/best_model.pt` ... `models/crossfit/fold_5/best_model.pt`
4. **Dữ liệu giải thích Out-of-Fold:**
   - `results/xai/crossfit/crossfit_explain_manifest.json`
   - `results/xai/crossfit/explanations_oof.parquet` (11,400 đồ thị)
   - `results/xai/crossfit/player_importance_oof.parquet` (71,126 bản ghi)
   - `results/xai/crossfit/top_passes_oof.parquet`

---

## 5. Lệnh tái lập (Reproducibility Commands)

```bash
# 1. Chạy thực nghiệm XAI chính thức 1.222 đồ thị (test split) với 4 explainers và SHAP baseline
python scripts/08_explain.py --config configs/data.yaml --model-name mpnn --explainer all --sample-mode official --output-dir results/xai/official/mpnn/official_test

# 2. Đánh giá Cổng kiểm soát Gate D17 với Paired Clustered Bootstrap và xuất bảng đối sánh chính thức
python scripts/09_evaluate_xai.py --xai-dir results/xai/official/mpnn/official_test

# 3. Huấn luyện Cross-Fitting 5-fold
python scripts/06d_crossfit_train.py --fold all --epochs 10

# 4. Trích xuất Out-of-Fold Explanations trên 380 trận
python scripts/10_crossfit_explain.py --explainer pg_explainer --max-graphs-per-match 30 --min-edges 4
```
