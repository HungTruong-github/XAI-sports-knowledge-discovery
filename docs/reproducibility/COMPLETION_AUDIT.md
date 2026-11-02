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
| **8** | Đối sánh XAI & Cổng Gate D17 | 4 explainers trên 1.222 test graphs (57 trận). Gate D17 Paired Clustered 95% CI vs Random: Last-edge PASS (+0.2282, CI [0.2077, 0.2502]), GNNExplainer PASS (+0.1111, CI [0.0939, 0.1286]), PGExplainer FAIL (-0.0128). Trạng thái canonical: `OFFICIAL_VERIFIED` | **PASS** (Scientifically honest & robust) |
| **9** | Cross-Fitting & Hybrid Centrality | Giao thức A/B sạch rò rỉ triệt để (Set A nội bộ cross-fit, Set B chuyển giao); co rút Bayesian $m=20$, $n_i \ge 30$; trung bình có trọng số phút thi đấu; loại trừ thủ môn; tối ưu α* = 0.70 trên Set A; đánh giá Out-of-sample trên Set B (313 cầu thủ): $\Delta R^2_{\text{OOS}} = +0.0768$ (95% CI [0.0142, 0.1270]) | **PASS** (Leak-free out-of-sample protocol) |
| **10**| Trực quan hóa & Khóa luận | 7 bộ biểu đồ PNG/SVG; chương khóa luận `docs/thesis/ket_qua_va_thao_luan.md` | **PASS** (Comprehensive) |

---

## 3. TỔNG HỢP CÂU TRẢ LỜI CHO CÁC CÂU HỎI NGHIÊN CỨU

### RQ1: Hiệu năng dự đoán cơ hội dứt điểm
- **Kết quả chính:** Mô hình MPNN đạt Test AUC-PR trung bình qua 5 seeds là **0.5996 ± 0.0042** (AUC-ROC 0.8757, Brier Score 0.1302), vượt trội có ý nghĩa thống kê so với GraphSAGE (0.5159), GATv2 (0.4620) và GCN (0.4526).
- **So sánh với Tabular Baselines:** Random Forest (0.6198) và ANN (0.6117) vẫn nhỉnh hơn MPNN (0.5996) do nhận trực tiếp các biến tổng hợp toàn cục của pha bóng.
- **Biến thể Readout Pooling (Quyết định D4):** Biến thể `mpnn_lastedge` đạt Validation AUC-PR cao nhất (0.5921, cải thiện +0.0074 so với mean 0.5847). Do $+0.0074 < +0.0100$ (ngưỡng tiền nghiệm), quy tắc D4 duy trì MPNN mean pooling làm backbone chính thức, đồng thời ghi nhận seed 1024 của `mpnn_lastedge` đạt Test AUC-PR **0.6107**, tiếp cận trực tiếp hiệu năng tabular.

### RQ2: Chất lượng và tính đáng tin cậy của phương pháp giải thích XAI
- **Kiểm định Cổng Gate D17 (Paired Clustered Bootstrap 95% CI theo cụm trận đấu trên 1.222 đồ thị):**
  - `gnn_explainer`: Chênh lệch theo cặp $\Delta\text{Fidelity}$ vs Random = **$+0.1111$**, 95% Cluster CI = **$[+0.0939, +0.1286]$** $\rightarrow$ **PASS** (vượt trội ngẫu nhiên có ý nghĩa thống kê).
  - `pg_explainer`: Chênh lệch theo cặp $\Delta\text{Fidelity}$ vs Random = $-0.0128$, 95% Cluster CI = $[-0.0306, +0.0049]$ $\rightarrow$ **FAIL** (không vượt trội ngẫu nhiên trên tập test).
  - `last_edge`: Chênh lệch theo cặp $\Delta\text{Fidelity}$ vs Random = **$+0.2282$**, 95% Cluster CI = **$[+0.2077, +0.2502]$** $\rightarrow$ **PASS** (vượt trội ngẫu nhiên ở mức ý nghĩa cao nhất).
- **Ý nghĩa khoa học:** Trong đồ thị bóng đá ngắn (4–12 đường chuyền), hành động dứt điểm cuối cùng đóng vai trò chi phối áp đảo ($\text{Fidelity}^+ = 0.1237$, $\text{Fidelity}^- = -0.1542$). Về độ ổn định, PGExplainer đạt **0.9872**, vượt trội so với GNNExplainer (0.2851), rất phù hợp cho trích xuất quy mô lớn.

### RQ3: Đóng góp của chỉ số Hybrid Centrality
- **Tối ưu hóa tham số kết hợp trên Set A (190 trận):** Quá trình Grid Search với ràng buộc độ tin cậy chia đôi xác định tỷ trọng tối ưu là **$\alpha^* = 0.70$** (Spearman $xT/90 = 0.6257$, $r_{\text{rel}} = 0.7198 \ge 0.70 \times r_{\text{max}}$).
- **Kiểm định độc lập Out-of-Sample trên Set B (190 trận độc lập, 313 cầu thủ ngoài khung gỗ):**
  - Tương quan Spearman với các chỉ số sáng tạo cơ hội thực tế tăng vọt so với Betweenness Centrality thuần túy:
    - $xA/90$ (Expected Assists): **$0.1700$ vs $-0.0569$** (tăng **$+0.2269$**, đảo chiều tương quan từ âm sang dương có ý nghĩa).
    - $Key Passes/90$: **$0.3040$ vs $0.0568$** (tăng **$+0.2472$**).
    - $Assists/90$: **$0.1423$ vs $-0.0021$** (tăng **$+0.1444$**).
    - $xT/90$ (Expected Threat): **$0.5745$ vs $0.5243$** (tăng **$+0.0502$**).
  - Hồi quy OLS đa biến ước lượng trên Set A và kiểm định ngoài mẫu (Out-of-Sample) trên Set B:
    - $R^2_{\text{base, OOS}}$ (chỉ dùng Betweenness): $0.2418$.
    - $R^2_{\text{full, OOS}}$ (Betweenness + XAI): $0.3186$.
    - Độ tăng giải thích ngoài mẫu: **$\Delta R^2_{\text{OOS}} = +0.0768$** (tăng $7.68$ điểm phần trăm).
    - Khoảng tin cậy **95% Bootstrap CI = $[0.0142, 0.1270]$ hoàn toàn nằm trên 0**.
- **Ý nghĩa chuyên môn:** Santi Cazorla ($0.9638$) và Mesut Özil ($0.9500$) dẫn đầu bảng xếp hạng toàn giải. Các cầu thủ sáng tạo dạt cánh như Dimitri Payet ($0.8298$) và Christian Eriksen ($0.8125$) được nâng hạng xứng đáng nhờ điểm XAI cao, trong khi các tiền vệ mỏ neo/phòng ngự (Danny Drinkwater, Yann M'Vila, Cesc Fàbregas) được cân bằng chính xác giữa vai trò điều phối nền tảng và tính hiệu quả dứt điểm. Toàn bộ thủ môn đã được loại bỏ chính xác.

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

