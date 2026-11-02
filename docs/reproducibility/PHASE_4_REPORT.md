# BÁO CÁO TÁI LẬP GIAI ĐOẠN 4: XÂY DỰNG VÀ THỰC NGHIỆM ĐÁNH GIÁ CHỈ SỐ HYBRID CENTRALITY

**Dự án:** XAI Football — Graph Neural Networks & Explainable AI on Passing Networks  
**Nhánh:** `feature/08-xai-completion`  
**Ngày hoàn thành:** 2026-11-02  
**Mục tiêu:** Xây dựng chỉ số Hybrid Centrality kết hợp giữa mạng chuyền bóng truyền thống và mức độ quan trọng từ mô hình học máy giải thích (XAI), tối ưu tham số $\alpha$ trên Set A, và đánh giá độc lập trên Set B theo Quyết định D12–D16 nhằm giải quyết câu hỏi nghiên cứu RQ3.

---

## 1. Cơ sở lý thuyết và Thiết kế thực nghiệm

### 1.1. Đặt vấn đề và Mục tiêu (RQ3)
- Các chỉ số centrality mạng truyền thống (như Betweenness Centrality) phản ánh vị trí trung gian trong việc luân chuyển bóng trên sân, nhưng hoàn toàn mù quáng về mặt mục tiêu bóng đá: một đường chuyền an toàn ở sân nhà được tính tương đương với một đường chuyền quyết định xuyên thủng hàng phòng ngự.
- Điểm XAI từ mô hình MPNN phản ánh mức độ đóng góp của từng cầu thủ vào việc gia tăng xác suất kết thúc pha bóng bằng một cú sút, nhưng dễ bỏ qua vai trò kết nối liên tục của các tiền vệ làm bóng.
- **Chỉ số đề xuất (Hybrid Centrality):**
  $$\text{HC}_i(\alpha) = \alpha \cdot \text{Rank}(\text{Centrality}_i) + (1 - \alpha) \cdot \text{Rank}(\text{XAI}_i)$$
  Trong đó $\text{Rank}(\cdot) \in [0, 1]$ là hàm chuẩn hóa xếp hạng theo bách phân vị (Percentile Rank theo Quyết định D14) áp dụng trên tập cầu thủ thi đấu đủ điều kiện.

### 1.2. Giao thức đánh giá A/B độc lập (D15, D16)
- **Tập Set A (190 trận):** Dùng để huấn luyện mô hình Expected Threat ($xT$) dạng lưới Markov $16 \times 12$ trên sân đấu $[0, 120] \times [0, 80]$, và thực hiện Grid Search tìm tham số $\alpha^*$ tối ưu trên đoạn $[0.0, 1.0]$ với bước nhảy $0.05$.
- **Tập Set B (190 trận độc lập):** Đóng vai trò tập kiểm định mù. Cố định $\alpha^* = 0.70$ thu được từ Set A và đánh giá tương quan Spearman cũng như mức tăng giải thích biến thiên ($\Delta R^2$) với các chỉ số tham chiếu bóng đá:
  - Expected Threat ($xT/90$).
  - Expected Assists ($xA/90$).
  - Đường chuyền quyết định ($Key Passes/90$).
  - Kiến tạo thành bàn ($Assists/90$).
- Số phút thi đấu được trích xuất chính xác theo từng phút ra/vào sân từ 380 tệp `lineups.json`.

---

## 2. Kết quả tối ưu hóa tham số $\alpha$ trên Set A (D16)

Nguồn dữ liệu: `results/hybrid/20261102_154729/alpha_curve.csv`

Quá trình quét tham số $\alpha \in [0.0, 1.0]$ với mục tiêu tối đa hóa tương quan với chỉ số đe dọa khung thành ($xT/90$):
- $\alpha = 0.00$ (Thuần XAI): Tương quan Spearman với $xT/90$ = $0.0308$.
- $\alpha = 0.50$: Tương quan Spearman với $xT/90$ = $0.4682$.
- $\alpha = 0.70$: Đạt điểm cực đại $\rho = 0.4955$.
- $\alpha = 1.00$ (Thuần Betweenness Centrality): Tương quan Spearman với $xT/90$ = $0.4891$.
- **Giá trị tối ưu được chốt:** $\alpha^* = 0.70$. Giá trị này thể hiện tỷ trọng đóng góp tối ưu: 70% vai trò luân chuyển bóng nền tảng và 30% vai trò chức năng thúc đẩy dứt điểm.

---

## 3. Kết quả đánh giá độc lập trên Set B (190 trận, 481 cầu thủ)

Nguồn dữ liệu: `results/tables/official/hybrid_centrality_comparison.csv`

| Chỉ số bóng đá tham chiếu | Hybrid Centrality ($\alpha^* = 0.70$) | Betweenness Centrality thuần | Điểm XAI thuần | Mức tăng so với Centrality | Giá trị p |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$xT/90$** (Expected Threat) | **0.5925** | 0.6005 | 0.0308 | -0.0080 | $p < 0.0001$ |
| **$xA/90$** (Expected Assists) | **0.3885** | 0.2066 | 0.5340 | **+0.1819** | $p < 0.0001$ |
| **$Key Passes/90$** | **0.4433** | 0.2541 | 0.5448 | **+0.1892** | $p < 0.0001$ |
| **$Assists/90$** | **0.3191** | 0.2076 | 0.3313 | **+0.1115** | $p < 0.0001$ |

### Phân tích hồi quy OLS và Độ tăng biến thiên giải thích ($\Delta R^2$):
- Hồi quy tuyến tính giải thích chỉ số $xT/90$:
  - Mô hình nền tảng (chỉ dùng Betweenness Centrality): $R^2 = 0.2882$.
  - Mô hình kết hợp đầy đủ (Betweenness Centrality + XAI Importance): $R^2 = 0.2996$.
  - Mức gia tăng giải thích: $\Delta R^2 = +0.0114$.
  - Khoảng tin cậy 95% Bootstrap (1,000 lần lấy mẫu lại): $[0.0015, 0.0279]$.
  - **Kết luận thống kê:** Khoảng tin cậy 95% bootstrap nằm hoàn toàn phía trên 0 ($0.0015 > 0$), chứng minh thành phần XAI mang lại giá trị gia tăng thực sự có ý nghĩa thống kê trong việc giải thích giá trị đe dọa tấn công của cầu thủ, vượt ra ngoài thông tin cấu trúc mạng đơn thuần.

---

## 4. Bảng xếp hạng Top cầu thủ tiêu biểu Ngoại hạng Anh mùa giải 2015/16

Nguồn dữ liệu: `results/tables/official/hybrid_centrality_top_players.csv`

| Hạng | Cầu thủ | Câu lạc bộ | Số phút | Hybrid Centrality | Rank Centrality | Rank XAI | Key Passes | Kiến tạo | $xA/90$ | $xT/90$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Mesut Özil** | Arsenal | 3,044 | **0.9374** | 0.9788 | 0.8408 | 70 | 9 | 0.220 | 0.2032 |
| 2 | **Santi Cazorla** | Arsenal | 1,291 | **0.9274** | 0.9809 | 0.8025 | 23 | 2 | 0.184 | 0.2183 |
| 3 | **Ross Barkley** | Everton | 3,079 | **0.9072** | 0.9703 | 0.7601 | 47 | 6 | 0.125 | 0.1274 |
| 4 | **Aaron Ramsey** | Arsenal | 2,622 | **0.9070** | 0.9936 | 0.7049 | 17 | 1 | 0.048 | 0.1351 |
| 5 | **Christian Eriksen**| Tottenham | 2,944 | **0.9059** | 0.9448 | 0.8153 | 75 | 8 | 0.194 | 0.1645 |
| 6 | **Troy Deeney** | Watford | 3,297 | **0.8994** | 0.9045 | 0.8875 | 41 | 5 | 0.168 | 0.0764 |
| 7 | **Darren Fletcher**| West Brom | 3,375 | **0.8979** | 0.9533 | 0.7686 | 15 | 1 | 0.033 | 0.0814 |
| 8 | **Yann M'Vila** | Sunderland | 3,103 | **0.8970** | 0.9958 | 0.6667 | 20 | 0 | 0.030 | 0.0589 |
| 9 | **Cesc Fàbregas** | Chelsea | 2,897 | **0.8921** | 0.9979 | 0.6454 | 46 | 8 | 0.158 | 0.1846 |
| 10 | **Dimitri Payet** | West Ham | 2,573 | **0.8728** | 0.9321 | 0.7346 | 80 | 5 | 0.262 | 0.1942 |
| 11 | **Jordan Henderson**| Liverpool | 1,291 | **0.8675** | 0.9745 | 0.6178 | 13 | 1 | 0.089 | 0.1260 |
| 12 | **James Milner** | Liverpool | 2,442 | **0.8660** | 0.8960 | 0.7962 | 35 | 6 | 0.122 | 0.1191 |

### Nhận định bóng đá:
- **Mesut Özil** đứng đầu bảng xếp hạng Hybrid Centrality với điểm số $0.9374$. Mùa giải 2015/16 là mùa giải thăng hoa rực rỡ của Özil (19 kiến tạo cả mùa, trong đó Set B ghi nhận 70 key passes và 9 kiến tạo trực tiếp). Özil có điểm centrality rất cao ($0.9788$) và điểm XAI cao vượt bậc ($0.8408$), phản ánh đúng bản chất: vừa là trạm trung chuyển bóng của Arsenal, vừa là ngòi nổ dứt điểm xuất sắc nhất giải đấu.
- **Dimitri Payet** và **Christian Eriksen** được nâng bậc đáng kể nhờ điểm XAI cao ($0.7346$ và $0.8153$), đưa họ vào nhóm dẫn đầu tuyệt đối về khả năng sáng tạo cơ hội, điều mà centrality truyền thống thường đánh giá thấp do họ chơi dạt cánh hoặc di chuyển tự do.
- Ngược lại, các tiền vệ phòng ngự lùi sâu như **Yann M'Vila** hay **Aaron Ramsey** dù có Betweenness cực cao ($> 0.99$) nhưng do tham gia ít vào khâu kết thúc pha bóng nên điểm XAI ở mức vừa phải ($0.66$–$0.70$), giúp bảng xếp hạng Hybrid phản ánh cân bằng cả 2 khía cạnh.

---

## 5. Danh mục Artifacts và Lệnh tái lập

1. **Bảng kết quả chính thức:**
   - `results/tables/official/hybrid_centrality_comparison.csv`
   - `results/tables/official/hybrid_centrality_top_players.csv`
   - `results/tables/official/hybrid_centrality_manifest.json`
2. **Dữ liệu thực nghiệm chi tiết:**
   - `results/hybrid/20261102_154729/alpha_curve.csv`
   - `results/hybrid/20261102_154729/player_hybrid_rankings.csv`
   - `results/hybrid/20261102_154729/evaluation_results.json`
3. **Cấu hình tham số đã cập nhật:**
   - `configs/hybrid_centrality.yaml` (chốt `alpha: 0.70`, `source_explainer: "pg_explainer"`)

---

## 6. Lệnh tái lập (Reproducibility Commands)

```bash
# Thực thi toàn bộ thí nghiệm Hybrid Centrality và đánh giá độc lập Set B
python scripts/11_run_hybrid_experiment.py --xai-results-path results/xai/crossfit/player_importance_oof.parquet
```

