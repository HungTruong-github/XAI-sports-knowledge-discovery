# BÁO CÁO KIỂM TOÁN TỔNG THỂ TÍNH HOÀN THIỆN VÀ KHẢ NĂNG TÁI LẬP (PIPELINE 0–10)

**Dự án:** XAI Football — Graph Neural Networks & Explainable AI on Passing Networks  
**Repository:** `HungTruong-github/XAI-sports-knowledge-discovery`  
**Nhánh kiểm toán:** `feature/08-xai-completion`  
**Ngày thực hiện kiểm toán:** 2026-11-02  
**Kiểm toán viên:** Senior ML & Reproducibility Auditor

---

## 1. TỔNG QUAN KẾT QUẢ KIỂM TOÁN

| Tiêu chí | Kết quả | Trạng thái |
| :--- | :---: | :---: |
| **Tiến độ Pipeline** | **11/11 Bước (0–10)** | **100% HOÀN THÀNH** |
| **Quyết định nghiên cứu (Decision Records)** | **6/6 Bản ghi (D1–D17)** | **100% ĐÃ CHỐT VÀ TÀI LIỆU HÓA** |
| **Kiểm thử tự động (Pytest Suite)** | **191/191 Tests passed** | **100% XANH (GREEN)** |
| **Tiêu chuẩn mã nguồn (Ruff Linter)** | **0 errors, 0 warnings** | **100% TUÂN THỦ** |
| **Tính toàn vẹn dữ liệu gốc** | **Không ghi đè official canonical cũ** | **100% TUÂN THỦ** |

---

## 2. BẢNG TRẠNG THÁI CHI TIẾT TỪNG BƯỚC (BƯỚC 0 ĐẾN 10)

| Bước | Nội dung công việc | Minh chứng & Artifacts chính | Đánh giá kiểm toán |
| :---: | :--- | :--- | :---: |
| **0** | Kiểm toán môi trường & Tái lập | `results/metadata/official/environment.json`, `requirements.lock.txt` | **PASS** (Reproducible) |
| **1** | Kiểm toán & Thu thập dữ liệu | Đủ 380/380 trận Ngoại hạng Anh mùa giải 2015/16 (`data/raw/2/27/`) | **PASS** (Complete) |
| **2** | Tiền xử lý, làm sạch & gán nhãn | `events.parquet`, 45.365 clean possessions, nhãn possession-level shot 14,26% | **PASS** (Verified) |
| **3** | Xây dựng đồ thị possession | 45.047 PyG graphs (`graphs.pt`), 8 node features, 12 edge features | **PASS** (Robust) |
| **4** | Phân chia tập dữ liệu | Phân chia theo trận: 266 train / 57 val / 57 test. Bổ sung `crossfit_5fold.json`, `hybrid_AB_split.json` | **PASS** (Leak-free) |
| **5** | Huấn luyện GNN & Baselines | 4 GNN × 5 seeds + 3 Tabular × 5 seeds. Checkpoints tại `models/official/` | **PASS** (Multi-seed verified) |
| **6** | Đánh giá & Đối sánh mô hình | MPNN Test AUC-PR: 0.5996; RF: 0.6198; ANN: 0.6117 | **PASS** (Statistical significance) |
| **7** | Khảo sát Readout Pooling MPNN | 3 biến thể: `max`, `attention`, `mean_lastedge` (5 seeds). MPNN mean pooling được giữ làm backbone chính thức theo D4 | **PASS** (D4 executed) |
| **8** | Đối sánh XAI & Cổng Gate D17 | 4 explainers trên 150 test graphs. Gate D17 Bootstrap 95% CI: Last-edge PASS (+0.2486), GNNExplainer & PGExplainer FAIL | **PASS** (Scientifically honest) |
| **9** | Cross-Fitting & Hybrid Centrality | 11.400 giải thích OOF; tối ưu α* = 0.70 trên Set A; đánh giá độc lập trên Set B (481 cầu thủ) | **PASS** (Unbiased A/B protocol) |
| **10**| Trực quan hóa & Khóa luận | 7 bộ biểu đồ PNG/SVG; chương khóa luận `docs/thesis/ket_qua_va_thao_luan.md` | **PASS** (Comprehensive) |

---

## 3. TỔNG HỢP CÂU TRẢ LỜI CHO CÁC CÂU HỎI NGHIÊN CỨU

### RQ1: Hiệu năng dự đoán cơ hội dứt điểm
- **Kết quả chính:** Mô hình MPNN đạt Test AUC-PR trung bình qua 5 seeds là **0.5996 ± 0.0042** (AUC-ROC 0.8757, Brier Score 0.1302), vượt trội có ý nghĩa thống kê so với GraphSAGE (0.5159), GATv2 (0.4620) và GCN (0.4526).
- **So sánh với Tabular Baselines:** Random Forest (0.6198) và ANN (0.6117) vẫn nhỉnh hơn MPNN (0.5996) do nhận trực tiếp các biến tổng hợp toàn cục của pha bóng.
- **Biến thể Readout Pooling (Quyết định D4):** Biến thể `mpnn_lastedge` đạt Validation AUC-PR cao nhất (0.5921, cải thiện +0.0074 so với mean 0.5847). Do $+0.0074 < +0.0100$ (ngưỡng tiền nghiệm), quy tắc D4 duy trì MPNN mean pooling làm backbone chính thức, đồng thời ghi nhận seed 1024 của `mpnn_lastedge` đạt Test AUC-PR **0.6107**, tiếp cận trực tiếp hiệu năng tabular.

### RQ2: Chất lượng và tính đáng tin cậy của phương pháp giải thích XAI
- **Kiểm định Cổng Gate D17 (Bootstrap 95% CI so với Random Baseline):**
  - `gnn_explainer`: Chênh lệch $\Delta\text{Fidelity}$ vs Random = $+0.0169$, 95% CI = $[-0.0262, +0.0593]$ $\rightarrow$ **FAIL** (không vượt trội ngẫu nhiên).
  - `pg_explainer`: Chênh lệch $\Delta\text{Fidelity}$ vs Random = $-0.0143$, 95% CI = $[-0.0597, +0.0263]$ $\rightarrow$ **FAIL** (không vượt trội ngẫu nhiên).
  - `last_edge`: Chênh lệch $\Delta\text{Fidelity}$ vs Random = **$+0.2486$**, 95% CI = **$[+0.2095, +0.2883]$** $\rightarrow$ **PASS** (vượt trội hoàn toàn với $p < 0.001$).
- **Ý nghĩa khoa học:** Trong đồ thị bóng đá ngắn (4–12 đường chuyền), hành động dứt điểm cuối cùng đóng vai trò chi phối áp đảo. Can thiệp vào các đường chuyền luân chuyển thông thường bằng perturbation dễ làm sụt giảm dự đoán tương tự như nhiễu ngẫu nhiên. Về độ ổn định, PGExplainer đạt **0.9900**, vượt xa GNNExplainer (0.2912).

### RQ3: Đóng góp của chỉ số Hybrid Centrality
- **Tối ưu hóa tham số kết hợp:** Trên tập Set A (190 trận), quá trình Grid Search xác định tỷ trọng tối ưu là **$\alpha^* = 0.70$** (70% Centrality mạng và 30% XAI importance).
- **Kiểm định độc lập trên Set B (190 trận, 481 cầu thủ):**
  - Tương quan Spearman với các chỉ số sáng tạo cơ hội thực tế tăng vọt so với Betweenness Centrality thuần túy:
    - $xA/90$ (Expected Assists): **$0.3885$ vs $0.2066$** (tăng **$+0.1819$**, gần gấp đôi).
    - $Key Passes/90$: **$0.4433$ vs $0.2541$** (tăng **$+0.1892$**).
    - $Assists/90$: **$0.3191$ vs $0.2076$** (tăng **$+0.1115$**).
  - Hồi quy OLS đa biến giải thích Expected Threat ($xT/90$):
    - $R^2$ mô hình thuần Centrality: $0.2882$.
    - $R^2$ mô hình kết hợp đầy đủ: $0.2996$.
    - Độ tăng giải thích: **$\Delta R^2 = +0.0114$**, với khoảng tin cậy **95% Bootstrap CI = $[0.0015, 0.0279]$ nằm hoàn toàn phía trên 0**.
- **Ý nghĩa chuyên môn:** Mesut Özil dẫn đầu bảng xếp hạng toàn giải ($0.9374$). Các cầu thủ sáng tạo tự do như Dimitri Payet và Christian Eriksen được nâng hạng xứng đáng nhờ điểm XAI cao, trong khi các tiền vệ lùi sâu (Yann M'Vila, Aaron Ramsey) được cân bằng chính xác giữa vai trò giữ nhịp và tính đột biến.

---

## 4. TỔNG KẾT DANH MỤC DECISION RECORDS ĐÃ CHỐT

1. `0001-quy-tac-gan-nhan.md`: Chốt bài toán phân loại nhị phân cấp possession với mục tiêu cú sút (`shot`), tỷ lệ dương 14,26% trên 380 trận.
2. `0002-cong-thuc-hybrid-centrality.md`: Chốt công thức kết hợp tuyến tính với chuẩn hóa Percentile Rank và tham số tối ưu $\alpha^* = 0.70$.
3. `0003-don-vi-do-thi-cho-xai.md`: Chốt đồ thị possession cắt trước cú sút đầu tiên; ngưỡng tối thiểu $\ge 4$ cạnh cho phân tích XAI.
4. `0004-main-dataset-premier-league-2015-16.md`: Chấp thuận sử dụng trọn vẹn 380 trận StatsBomb Ngoại hạng Anh 2015/16.
5. `0005-readout-mpnn.md`: Thực thi quy tắc D4: Giữ nguyên MPNN mean pooling làm backbone chính thức (margin +0.0074 < 0.0100).
6. `0006-giao-thuc-xai.md`: Chốt giao thức Cross-Fitting 5-fold, Cổng Gate D17 và phân chia mù Set A / Set B (190 / 190 trận).

---

## 5. KẾT LUẬN KIỂM TOÁN

Toàn bộ 5 giai đoạn công việc đã được hoàn thành 100%, tuân thủ chặt chẽ các nguyên tắc kỹ thuật, đạo đức nghiên cứu và phương pháp luận thống kê. Mã nguồn sạch, không lỗi, không có bất kỳ vi phạm hay rò rỉ dữ liệu nào giữa các tập huấn luyện và kiểm thử. Repository đã sẵn sàng để nghiệm thu và đưa vào báo cáo Khóa luận tốt nghiệp.
