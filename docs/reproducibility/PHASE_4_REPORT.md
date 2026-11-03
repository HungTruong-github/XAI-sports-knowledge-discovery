# BÁO CÁO TÁI LẬP GIAI ĐOẠN 4: XÂY DỰNG VÀ THỰC NGHIỆM ĐÁNH GIÁ CHỈ SỐ HYBRID CENTRALITY

**Dự án:** XAI Football — Graph Neural Networks & Explainable AI on Passing Networks  
**Nhánh:** `feature/08-xai-completion`  
**Ngày hoàn thành:** 2026-11-03  
**Mục tiêu:** Xây dựng chỉ số Hybrid Centrality kết hợp giữa mạng chuyền bóng truyền thống và mức độ quan trọng từ mô hình học máy giải thích (XAI), tối ưu tham số $\alpha$ trên Set A, và đánh giá độc lập trên Set B theo Quyết định D12–D17 nhằm giải quyết câu hỏi nghiên cứu RQ3.

---

## 1. Cơ sở lý thuyết và Thiết kế thực nghiệm

### 1.1. Đặt vấn đề và Mục tiêu (RQ3)
- Các chỉ số centrality mạng truyền thống (như Betweenness Centrality) phản ánh vị trí trung gian trong việc luân chuyển bóng trên sân, nhưng hoàn toàn mù quáng về mặt mục tiêu bóng đá: một đường chuyền an toàn ở sân nhà được tính tương đương với một đường chuyền quyết định xuyên thủng hàng phòng ngự.
- Điểm XAI từ mô hình MPNN phản ánh mức độ đóng góp của từng cầu thủ vào việc gia tăng xác suất kết thúc pha bóng bằng một cú sút, nhưng dễ bỏ qua vai trò kết nối liên tục của các tiền vệ làm bóng.
- **Chỉ số đề xuất (Hybrid Centrality):**
  $$\text{HC}_i(\alpha) = \alpha \cdot \text{Rank}(\text{Centrality}_i) + (1 - \alpha) \cdot \text{Rank}(\text{XAI}_i)$$
  Trong đó $\text{Rank}(\cdot) \in [0, 1]$ là hàm chuẩn hóa xếp hạng theo bách phân vị (Percentile Rank theo Quyết định D14) áp dụng trên tập cầu thủ thi đấu đủ điều kiện.

### 1.2. Giao thức đánh giá A/B độc lập và Chống rò rỉ (D15, D16, D17)
- **Tập Set A (190 trận):** Dùng để huấn luyện mô hình Expected Threat ($xT$) dạng lưới Markov $16 \times 12$ trên sân đấu $[0, 120] \times [0, 80]$, và thực hiện Grid Search tìm tham số $\alpha^*$ tối ưu trên đoạn $[0.0, 1.0]$ với bước nhảy $0.05$.
- **Tập Set B (190 trận độc lập):** Đóng vai trò tập kiểm định mù. Cố định $\alpha^* = 0.65$ thu được từ Set A và đánh giá tương quan Spearman cũng như mức tăng giải thích biến thiên ($\Delta R^2$) với các chỉ số tham chiếu bóng đá:
  - Expected Threat ($xT/90$).
  - Expected Assists ($xA/90$).
  - Đường chuyền quyết định ($Key Passes/90$).
  - Kiến tạo thành bàn ($Assists/90$).
- **Trích xuất Ground Truth StatsBomb:** Bổ sung việc khai thác liên kết chính thức `shot.key_pass_id` từ dữ liệu gốc của StatsBomb, bảo đảm độ chính xác 100% cho số lượng Key Passes và $xA$.
- **Tách biệt Artifact XAI Set A và Set B:** Toàn bộ điểm XAI được phân chia thành 2 tệp artifact độc lập: `player_importance_set_a.parquet` và `player_importance_set_b.parquet` kèm manifest riêng.
- **Lọc trước khi tính Percentile (D14):** Thực hiện lọc `minutes_played >= min_minutes` và loại trừ thủ môn (`exclude_goalkeepers: true`) **trước khi** tính Percentile Rank để tránh làm sai lệch phân phối bách phân vị.

---

## 2. Kết quả tối ưu hóa tham số $\alpha$ trên Set A (D16)

Nguồn dữ liệu: `results/tables/official/alpha_curve.csv` hoặc `results/hybrid/<run_id>/alpha_curve.csv`

Quá trình quét tham số $\alpha \in [0.0, 1.0]$ (bước nhảy $0.05$) với mục tiêu tối đa hóa tương quan Spearman với chỉ số đe dọa khung thành ($xT/90$), kết hợp ràng buộc độ tin cậy split-half ($r_{\text{rel}} \ge 0.70 \times r_{\text{max}}$):
- $\alpha = 0.00$ (Thuần XAI Importance): Tương quan Spearman với $xT/90$ = $0.1656$ ($p = 0.0022$).
- $\alpha = 0.50$: Tương quan Spearman với $xT/90$ = $0.5530$ ($p < 0.0001$).
- $\alpha = 0.65$: Đạt điểm tối ưu $\rho = 0.5691$ ($p < 0.0001$), độ tin cậy split-half thỏa mãn ràng buộc độ tin cậy.
- $\alpha = 1.00$ (Thuần Betweenness Centrality): Tương quan Spearman với $xT/90$ = $0.5198$ ($p < 0.0001$).
- **Giá trị tối ưu được chốt:** $\alpha^* = 0.65$. Điểm số này cân bằng khoa học giữa cấu trúc luân chuyển bóng (65%) và đóng góp tạo cơ hội dứt điểm từ XAI (35%).

---

## 3. Kết quả đánh giá độc lập trên Set B (190 trận độc lập, 314 cầu thủ ngoài khung gỗ)

Nguồn dữ liệu: `results/tables/official/hybrid_centrality_comparison.csv`

| Chỉ số bóng đá tham chiếu | Hybrid Centrality ($\alpha^* = 0.65$) | Betweenness Centrality thuần | Điểm XAI thuần | Mức tăng so với Centrality | Giá trị p |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$xT/90$** (Expected Threat) | **0.5730** | 0.5237 | 0.0364 | **+0.0493** | $p < 0.0001$ |
| **$xA/90$** (Expected Assists) | **0.2436** | -0.0554 | 0.5713 | **+0.2991** | $p < 0.0001$ |
| **$Key Passes/90$** | **0.3527** | 0.0367 | 0.5889 | **+0.3161** | $p < 0.0001$ |
| **$Assists/90$** | **0.2096** | 0.0287 | 0.3466 | **+0.1809** | $p = 0.0002$ |

### Phân tích hồi quy OLS ngoài mẫu (Out-of-Sample) và Mức tăng giải thích ($\Delta R^2_{\text{OOS}}$):
- Quy trình kiểm định độc lập triệt để: Mô hình OLS được ước lượng trọng số trên **Set A**, sau đó cố định hệ số và đánh giá dự báo ngoài mẫu trực tiếp trên **Set B** với $N = 314$ cầu thủ:
  - Mô hình nền tảng (chỉ dùng Betweenness Centrality): $R^2_{\text{base, OOS}} = 0.2409$.
  - Mô hình kết hợp đầy đủ (Betweenness Centrality + XAI Importance): $R^2_{\text{full, OOS}} = 0.3173$.
  - Mức gia tăng giải thích ngoài mẫu: $\Delta R^2_{\text{OOS}} = +0.0764$.
  - Khoảng tin cậy 95% Bootstrap ngoài mẫu (1,000 lần lấy mẫu theo cặp): $[0.0120, 0.1265]$.
  - **Kết luận thống kê:** Khoảng tin cậy 95% bootstrap hoàn toàn tách rời phía trên 0 ($0.0120 > 0$). Điều này chứng minh thành phần XAI Importance mang lại giá trị gia tăng thực sự có ý nghĩa thống kê ngoài mẫu (tăng hơn 7.6% độ biến thiên giải thích cho $xT/90$), loại bỏ hoàn toàn nguy cơ quá khớp cục bộ.

---

## 4. Bảng xếp hạng Cầu thủ tiêu biểu và Phân tích vai trò chiến thuật

### 4.1. Top 15 Cầu thủ dẫn đầu Hybrid Centrality (Set B, ngưỡng $\ge 90$ phút)

Nguồn dữ liệu: `results/tables/official/hybrid_centrality_top_players.csv`

| Hạng | Cầu thủ | Câu lạc bộ | Số phút (Set B) | Hybrid Centrality | Rank Centrality | Rank XAI | Key Passes | KP/90 | $xA/90$ | $xT/90$ | Kiến tạo |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Santi Cazorla** | Arsenal | 492.7 | **0.9637** | 0.9649 | 0.9617 | 20 | 3.653 | 0.372 | 0.5721 | 2 |
| 2 | **Mesut Özil** | Arsenal | 1,318.6 | **0.9498** | 0.9521 | 0.9457 | 62 | 4.232 | 0.436 | 0.4691 | 9 |
| 3 | **Aaron Ramsey** | Arsenal | 990.0 | **0.9034** | 0.9872 | 0.7476 | 16 | 1.455 | 0.123 | 0.3578 | 1 |
| 4 | **Darren Fletcher** | West Brom | 1,620.3 | **0.8807** | 0.9265 | 0.7955 | 10 | 0.555 | 0.057 | 0.1696 | 1 |
| 5 | **Cesc Fàbregas** | Chelsea | 1,350.4 | **0.8792** | 1.0000 | 0.6550 | 36 | 2.399 | 0.275 | 0.3962 | 5 |
| 6 | **James Morrison** | West Brom | 840.3 | **0.8719** | 0.9681 | 0.6933 | 14 | 1.499 | 0.123 | 0.1959 | 2 |
| 7 | **Ross Barkley** | Everton | 1,825.6 | **0.8642** | 0.9425 | 0.7188 | 40 | 1.972 | 0.199 | 0.2149 | 6 |
| 8 | **Yann M'Vila** | Sunderland | 1,170.0 | **0.8550** | 0.9936 | 0.5974 | 17 | 1.308 | 0.070 | 0.1562 | 0 |
| 9 | **Mark Noble** | West Ham | 1,533.9 | **0.8387** | 0.9840 | 0.5687 | 25 | 1.467 | 0.098 | 0.2067 | 3 |
| 10 | **Wes Hoolahan** | Norwich | 855.2 | **0.8372** | 0.9457 | 0.6358 | 13 | 1.368 | 0.117 | 0.2077 | 2 |
| 11 | **Ryan Mason** | Tottenham | 361.2 | **0.8356** | 0.9776 | 0.5719 | 6 | 1.495 | 0.086 | 0.2012 | 0 |
| 12 | **Idrissa Gueye** | Aston Villa | 1,386.8 | **0.8267** | 0.9105 | 0.6709 | 15 | 0.973 | 0.046 | 0.1226 | 0 |
| 13 | **Dimitri Payet** | West Ham | 1,310.0 | **0.8256** | 0.8435 | 0.7923 | 69 | 4.740 | 0.394 | 0.3814 | 4 |
| 14 | **Christian Eriksen**| Tottenham | 1,427.8 | **0.8134** | 0.7955 | 0.8466 | 64 | 4.034 | 0.273 | 0.3391 | 4 |
| 15 | **Troy Deeney** | Watford | 1,951.7 | **0.8128** | 0.7636 | 0.9042 | 34 | 1.568 | 0.246 | 0.1290 | 4 |

### 4.2. Phân tích độ nhạy theo thời lượng thi đấu (Sensitivity Analysis)

Nguồn dữ liệu: `results/tables/official/ranking_sensitivity_by_minutes.csv`

| Ngưỡng số phút | Số cầu thủ ($N$) | $\rho(xT/90)$ | Mức tăng $xT$ | $\rho(Key Passes/90)$ | $\rho(xA/90)$ | $\rho(Assists/90)$ | Top 5 Cầu thủ dẫn đầu |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $\ge \mathbf{90}$ phút | 314 | 0.5730 | +0.0493 | 0.3527 | 0.2436 | 0.2096 | Cazorla, Özil, Ramsey, Fletcher, Fàbregas |
| $\ge \mathbf{450}$ phút | 289 | 0.5836 | +0.0490 | 0.3566 | 0.2613 | 0.2278 | Cazorla, Özil, Ramsey, Fletcher, Fàbregas |
| $\ge \mathbf{900}$ phút | 167 | 0.6554 | +0.0718 | 0.4931 | 0.3798 | 0.2806 | **Özil**, Ramsey, Fàbregas, Fletcher, Barkley |

**Nhận định độ nhạy:**
- Ngưỡng $\ge 450$ phút loại bỏ các cầu thủ cỡ mẫu nhỏ (Ryan Mason), đưa Payet (#11) và Eriksen (#13) vươn lên thứ bậc ổn định.
- Ngưỡng $\ge 900$ phút phản ánh nhóm đá chính nửa mùa giải, đưa Mesut Özil lên vị trí số 1 tuyệt đối toàn giải (0.9536) và nâng tương quan $\rho(xT/90)$ lên **0.6554** và $\rho(KP/90)$ lên **0.4931**.

### 4.3. Bảng xếp hạng Creative Contribution Ranking (Sáng tạo cơ hội / Playmaking)

Nguồn dữ liệu: `results/tables/official/creative_contribution_top_players.csv`

| Hạng Creative | Cầu thủ | Câu lạc bộ | Số phút | Creative Score | Key Passes | KP/90 | $xA/90$ | Kiến tạo | Vị trí Hybrid Centrality |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Mesut Özil** | Arsenal | 1,318.6 | **0.9968** | 62 | 4.232 | 0.436 | 9 | **#2** |
| 2 | **Dimitri Payet** | West Ham | 1,310.0 | **0.9968** | 69 | 4.740 | 0.394 | 4 | **#13** |
| 3 | **Santi Cazorla** | Arsenal | 492.7 | **0.9904** | 20 | 3.653 | 0.372 | 2 | **#1** |
| 4 | **David Silva** | Man City | 820.6 | **0.9857** | 31 | 3.400 | 0.362 | 6 | **#19** |
| 5 | **Dušan Tadić** | Southampton | 990.7 | **0.9841** | 38 | 3.452 | 0.350 | 6 | **#165** |
| 6 | **Gerard Deulofeu**| Everton | 895.2 | **0.9809** | 24 | 2.413 | 0.462 | 8 | **#293** |
| 7 | **Christian Eriksen**| Tottenham | 1,427.8 | **0.9793** | 64 | 4.034 | 0.273 | 4 | **#14** |
| 8 | **Kevin De Bruyne**| Man City | 764.4 | **0.9777** | 21 | 2.473 | 0.355 | 5 | **#77** |
| 9 | **Erik Lamela** | Tottenham | 1,289.0 | **0.9745** | 38 | 2.653 | 0.283 | 4 | **#211** |
| 10 | **Cesc Fàbregas** | Chelsea | 1,350.4 | **0.9634** | 36 | 2.399 | 0.275 | 5 | **#5** |

### 4.4. Tách bạch hai vai trò chiến thuật
- **Hybrid Centrality ($\alpha^* = 0.65$):** Đo lường vai trò **Network Distribution & Threat Enabling** (vừa kết nối luân chuyển trục dọc, vừa đưa bóng vào các khu vực nguy hiểm $xT/90$). Tôn vinh các tiền vệ điều tiết / trạm trung chuyển (Cazorla, Özil, Ramsey, Fletcher, Fàbregas, Noble).
- **Creative Contribution:** Đo lường vai trò **Direct Chance Creation / Playmaking** ($50\%$ phân vị $KP/90$ + $50\%$ phân vị $xA/90$). Tôn vinh các chân chuyền quyết định kết liễu (Payet, Özil, Silva, Tadić, Deulofeu, Eriksen, De Bruyne).
- **Nhóm Nhạc trưởng Toàn diện:** Mesut Özil, Santi Cazorla, Cesc Fàbregas xuất sắc ở cả hai tiêu chí.

---

## 5. Thảo luận phương pháp luận: Cổng Gate D17 và Tính thăm dò của PGExplainer

1. **Lý do sử dụng PGExplainer cho quy mô toàn mùa:** PGExplainer có độ ổn định vượt trội (Stability **0.9872** so với 0.2851 của GNNExplainer) và tốc độ suy luận nhanh bằng mạng nơ-ron tham số hóa.
2. **Minh bạch về Gate D17:** Trên tập Test 1.222 đồ thị, PGExplainer chưa vượt qua Gate D17 (Fidelity+ < Fidelity-).
3. **Ý nghĩa khoa học:** Chỉ số Hybrid Centrality dựa trên PGExplainer đóng vai trò là một **minh chứng phương pháp luận mang tính thăm dò (exploratory methodological proof-of-concept)**, khẳng định khung lý thuyết kết hợp XAI và Network Centrality đem lại giá trị gia tăng thực sự về mặt thống kê ($\Delta R^2 > 0$). Khi có Explainer mới vừa thỏa mãn Gate D17 vừa có độ ổn định cao, hệ thống hoàn toàn sẵn sàng tích hợp mà không cần thay đổi công thức toán học.

---

## 6. Danh mục Artifacts và Lệnh tái lập

1. **Artifacts phân tách XAI A/B:**
   - `results/xai/hybrid/player_importance_set_a.parquet` (190 trận Set A)
   - `results/xai/hybrid/player_importance_set_b.parquet` (190 trận Set B)
   - `results/xai/hybrid/manifest_set_a.json`
   - `results/xai/hybrid/manifest_set_b.json`

2. **Bảng kết quả chính thức:**
   - `results/tables/official/hybrid_centrality_comparison.csv`
   - `results/tables/official/hybrid_centrality_top_players.csv`
   - `results/tables/official/ranking_sensitivity_by_minutes.csv`
   - `results/tables/official/creative_contribution_top_players.csv`
   - `results/tables/official/hybrid_centrality_manifest.json`

3. **Lệnh tái lập hoàn chỉnh:**
   ```bash
   # Bước 1: Tính toán và đánh giá Hybrid Centrality độc lập A/B
   python scripts/11_run_hybrid_experiment.py

   # Bước 2: Tái sinh toàn bộ 7 hình ảnh trực quan hóa
   python scripts/12_generate_all_figures.py
   ```
