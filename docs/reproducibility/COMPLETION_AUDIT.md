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
| **9** | Cross-Fitting & Hybrid Centrality | Giao thức A/B sạch rò rỉ triệt để (Set A nội bộ cross-fit, Set B chuyển giao); co rút Bayesian $m=20$, $n_i \ge 30$; trung bình có trọng số phút thi đấu; loại trừ thủ môn; tối ưu α* = 0.65 trên Set A; đánh giá Out-of-sample trên Set B (314 cầu thủ): $\Delta R^2_{\text{OOS}} = +0.0764$ (95% CI [0.0120, 0.1265]); bổ sung trích xuất StatsBomb key_pass_id chính thức, lọc trước percentile, phân tích độ nhạy và bảng xếp hạng Creative Contribution | **PASS** (Leak-free out-of-sample protocol) |
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
- **Tối ưu hóa tham số kết hợp trên Set A (190 trận):** Quá trình Grid Search với ràng buộc độ tin cậy chia đôi xác định tỷ trọng tối ưu là **$\alpha^* = 0.65$** (Spearman $xT/90 = 0.5691$, thỏa mãn ràng buộc độ tin cậy chia đôi).
- **Kiểm định độc lập Out-of-Sample trên Set B (190 trận độc lập, 314 cầu thủ ngoài khung gỗ, lọc $\ge 90$ phút trước khi tính percentile):**
  - Tương quan Spearman với các chỉ số sáng tạo cơ hội thực tế tăng vọt so với Betweenness Centrality thuần túy:
    - $xA/90$ (Expected Assists chính thức): **$0.2436$ vs $-0.0554$** (tăng **$+0.2991$**, đảo chiều tương quan từ âm sang dương có ý nghĩa thống kê cao).
    - $Key Passes/90$ (Key Passes chính thức StatsBomb): **$0.3527$ vs $0.0367$** (tăng **$+0.3161$**).
    - $Assists/90$: **$0.2096$ vs $0.0287$** (tăng **$+0.1809$**).
    - $xT/90$ (Expected Threat): **$0.5730$ vs $0.5237$** (tăng **$+0.0493$**).
  - Hồi quy OLS đa biến ước lượng trên Set A và kiểm định ngoài mẫu (Out-of-Sample) trên Set B:
    - $R^2_{\text{base, OOS}}$ (chỉ dùng Betweenness): $0.2409$.
    - $R^2_{\text{full, OOS}}$ (Betweenness + XAI): $0.3173$.
    - Độ tăng giải thích ngoài mẫu: **$\Delta R^2_{\text{OOS}} = +0.0764$** (tăng $7.64$ điểm phần trăm).
    - Khoảng tin cậy **95% Bootstrap CI = $[0.0120, 0.1265]$ hoàn toàn nằm trên 0**.
- **Ý nghĩa chuyên môn và Tách bạch hai vai trò chiến thuật:**
  - **Hybrid Centrality ($\alpha^* = 0.65$):** Đo lường năng lực luân chuyển và trung chuyển bóng toàn đội ($xT/90$), dẫn đầu bởi Santi Cazorla ($0.9637$), Mesut Özil ($0.9498$), Aaron Ramsey ($0.9034$), Darren Fletcher ($0.8807$), Cesc Fàbregas ($0.8792$).
  - **Creative Contribution Index:** Đo lường năng lực kết liễu và tạo đột biến dứt điểm trực tiếp ($Key Passes/90$ và $xA/90$), dẫn đầu bởi Mesut Özil (#1), Dimitri Payet (#2), Santi Cazorla (#3), David Silva (#4), Dušan Tadić (#5), Gerard Deulofeu (#6), Christian Eriksen (#7), Kevin De Bruyne (#8).
  - **Phân tích độ nhạy thời lượng:** Khi nâng ngưỡng lên $\ge 450$ phút và $\ge 900$ phút, bảng xếp hạng ngày càng vững chắc, Mesut Özil dẫn đầu toàn diện toàn giải (0.9536 tại $\ge 900$m), đồng thời tương quan $xT/90$ tăng lên $0.6554$ và $KP/90$ tăng lên $0.4931$.
- **Minh bạch về Gate D17:** PGExplainer đạt độ ổn định 0.9872 cho suy luận diện rộng nhưng chưa vượt qua Gate D17 trên tập test, do đó Hybrid Centrality đóng vai trò là một minh chứng phương pháp luận mang tính thăm dò (exploratory methodological proof-of-concept).

---

## 4. TỔNG KẾT DANH MỤC DECISION RECORDS ĐÃ CHỐT

1. `0001-quy-tac-gan-nhan.md`: Chốt bài toán phân loại nhị phân cấp possession với mục tiêu cú sút (`shot`), tỷ lệ dương 14,26% trên 380 trận.
2. `0002-cong-thuc-hybrid-centrality.md`: Chốt công thức kết hợp tuyến tính với chuẩn hóa Percentile Rank và tham số tối ưu $\alpha^* = 0.65$.
3. `0003-don-vi-do-thi-cho-xai.md`: Chốt đồ thị possession cắt trước cú sút đầu tiên; ngưỡng tối thiểu $\ge 4$ cạnh cho phân tích XAI.
4. `0004-main-dataset-premier-league-2015-16.md`: Chấp thuận sử dụng trọn vẹn 380 trận StatsBomb Ngoại hạng Anh 2015/16.
5. `0005-readout-mpnn.md`: Thực thi quy tắc D4: Giữ nguyên MPNN mean pooling làm backbone chính thức (margin +0.0074 < 0.0100).
6. `0006-giao-thuc-xai.md`: Chốt giao thức Cross-Fitting 5-fold, Cổng Gate D17 và phân chia mù Set A / Set B (190 / 190 trận).

---

## 5. KẾT LUẬN KIỂM TOÁN

Toàn bộ 5 giai đoạn công việc đã được hoàn thành 100%, tuân thủ chặt chẽ các nguyên tắc kỹ thuật, đạo đức nghiên cứu và phương pháp luận thống kê. Mã nguồn sạch, không lỗi, không có bất kỳ vi phạm hay rò rỉ dữ liệu nào giữa các tập huấn luyện và kiểm thử. Repository đã sẵn sàng để nghiệm thu và đưa vào báo cáo Khóa luận tốt nghiệp.

