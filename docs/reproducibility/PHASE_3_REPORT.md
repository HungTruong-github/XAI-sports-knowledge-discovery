# BÁO CÁO TÁI LẬP GIAI ĐOẠN 3: THỰC THI VÀ KIỂM TOÁN PIPELINE GIẢI THÍCH (XAI) VÀ CROSS-FITTING

**Dự án:** XAI Football — Graph Neural Networks & Explainable AI on Passing Networks  
**Nhánh:** `feature/08-xai-completion`  
**Ngày hoàn thành:** 2026-11-02  
**Mục tiêu:** Thực thi kiểm toán thực nghiệm giải thích đồ thị trên mô hình MPNN backbone chính thức, đối sánh 4 phương pháp (`gnn_explainer`, `pg_explainer`, `random`, `last_edge`), thẩm định Cổng chất lượng Gate D17 theo Quyết định D17, và thực hiện Cross-Fitting 5-fold để trích xuất 100% out-of-fold player importance sạch rò rỉ trên 380 trận đấu.

---

## 1. Thiết kế thực nghiệm và Đối sánh 4 Phương pháp XAI

Thực nghiệm được thực hiện trên tập Test độc lập (57 trận, 3,860 đồ thị hợp lệ có $\ge 4$ cạnh theo Quyết định D2):
- Phương thức lấy mẫu: Stratified Sampling theo Quyết định D7 (cân bằng 75 possession có cú sút và 75 possession không có cú sút, seed = 42).
- Các thuật toán giải thích:
  1. `gnn_explainer`: Tối ưu hóa mặt nạ cạnh theo cơ chế tối đa thông tin tương hỗ (phenomenon mode, target class = 1).
  2. `pg_explainer`: Mạng tham số hóa MLP học phân phối mặt nạ cạnh có trọng số, tối ưu hóa trên tập huấn luyện (target class = 1).
  3. `random`: Baseline mặt nạ ngẫu nhiên đồng đều (Uniform[0, 1]).
  4. `last_edge`: Heuristic bóng đá gán trọng số 1.0 cho đường chuyền cuối cùng (chuỗi bóng dẫn tới dứt điểm) và 0.0 cho các đường chuyền trước.
  5. Baseline phi đồ thị: SHAP TreeExplainer trên mô hình Random Forest tabular.

---

## 2. Kết quả đối sánh chi tiết và Đánh giá Cổng kiểm soát Gate D17

Nguồn dữ liệu: `results/tables/official/xai_comparison.csv`

| Thuật toán | Số đồ thị | Fidelity+ (mean ± std) | Fidelity- (mean ± std) | $\Delta\text{Fidelity}$ | Sparsity | Stability | Deletion AUC | Insertion AUC | Trạng thái Gate D17 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gnn_explainer` | 150 | 0.0105 ± 0.1313 | -0.0496 ± 0.2556 | +0.0601 | 0.6987 | 0.2912 | 0.4481 | 0.5043 | **FAIL** (CI vắt qua 0) |
| `pg_explainer` | 150 | -0.0078 ± 0.1168 | -0.0352 ± 0.2527 | +0.0274 | 0.6987 | **0.9900** | 0.5206 | 0.4216 | **FAIL** (CI vắt qua 0) |
| `random` | 150 | 0.0077 ± 0.1671 | -0.0349 ± 0.2363 | +0.0426 | 0.6987 | — | 0.4735 | 0.4730 | Baseline |
| `last_edge` | 150 | **0.1490 ± 0.2343** | **-0.1419 ± 0.1372** | **+0.2909** | 0.6987 | — | **0.4238** | **0.5127** | **PASS** (CI hoàn toàn > 0) |

### Kiểm định thống kê Bootstrap 95% CI cho Gate D17:
- Tiêu chí D17: Chênh lệch $\Delta\text{Fidelity}_{\text{expl}} - \Delta\text{Fidelity}_{\text{rand}}$ phải có khoảng tin cậy 95% bootstrap (1,000 resamples) nằm hoàn toàn phía trên 0.
- Kết quả kiểm định:
  - `gnn_explainer`: Mean = $+0.0169$, 95% CI = $[-0.0262, +0.0593]$ $\rightarrow$ **FAIL**.
  - `pg_explainer`: Mean = $-0.0143$, 95% CI = $[-0.0597, +0.0263]$ $\rightarrow$ **FAIL**.
  - `last_edge`: Mean = **$+0.2486$**, 95% CI = **$[+0.2095, +0.2883]$** $\rightarrow$ **PASS**.

### Phân tích ý nghĩa khoa học (RQ2):
1. **Tính chất nhạy cảm của pha dứt điểm:** Trong bài toán dự đoán possession có cú sút từ mạng chuyền bóng, hành động then chốt nhất là đường chuyền cuối cùng (đưa bóng vào vị trí dứt điểm). Khi cô lập hoặc loại bỏ đường chuyền này (`last_edge`), mô hình lập tức sụt giảm độ tự tin mạnh nhất ($\text{Fidelity}^+ = 0.1490$), trong khi việc giữ lại nó bảo toàn tối đa dự đoán ($\text{Fidelity}^- = -0.1419$).
2. **Hạn chế của phương pháp perturbation tổng quát:** Đồ thị possession trong bóng đá có quy mô nhỏ (trung bình 4–12 đường chuyền). Các thuật toán tối ưu nhiễu cạnh cục bộ như GNNExplainer và PGExplainer dễ làm thay đổi cấu trúc liên thông của chuỗi bóng tương tự như nhiễu ngẫu nhiên, dẫn đến việc không vượt qua được random baseline ở mức ý nghĩa thống kê 95%.
3. **Độ ổn định (Stability):** PGExplainer thể hiện tính ổn định vượt trội ($0.9900$) so với GNNExplainer ($0.2912$), chứng minh mạng tham số hóa tránh được tính ngẫu hứng trong quá trình tối ưu riêng lẻ từng đồ thị.

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

1. **Bảng đối sánh XAI chính thức:**
   - `results/tables/official/xai_comparison.csv`
2. **Dữ liệu Pilot XAI 150 đồ thị:**
   - `results/xai/pilot/mpnn/pilot_150/xai_run_manifest.json`
   - `results/xai/pilot/mpnn/pilot_150/sample_manifest.csv`
   - `results/xai/pilot/mpnn/pilot_150/xai_metrics_by_graph.csv`
   - `results/xai/pilot/mpnn/pilot_150/xai_metrics_summary.csv`
   - `results/xai/pilot/mpnn/pilot_150/sparsity_fidelity_curves.csv`
   - `results/xai/pilot/mpnn/pilot_150/shap_feature_importance.csv`
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
# 1. Chạy pilot XAI 150 đồ thị với 4 explainers và SHAP baseline
python scripts/08_explain.py --config configs/data.yaml --model-name mpnn --explainer all --sample-mode pilot --output-dir results/xai/pilot/mpnn/pilot_150

# 2. Đánh giá Cổng kiểm soát Gate D17 và xuất bảng đối sánh
python scripts/09_evaluate_xai.py --xai-dir results/xai/pilot/mpnn/pilot_150 --allow-historical

# 3. Huấn luyện Cross-Fitting 5-fold
python scripts/06d_crossfit_train.py --fold all --epochs 10

# 4. Trích xuất Out-of-Fold Explanations trên 380 trận
python scripts/10_crossfit_explain.py --explainer pg_explainer --max-graphs-per-match 30 --min-edges 4
```
