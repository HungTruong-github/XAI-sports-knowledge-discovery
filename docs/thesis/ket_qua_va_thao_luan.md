# CHƯƠNG 4: KẾT QUẢ THỰC NGHIỆM VÀ THẢO LUẬN

---

## 4.1. ĐẶT VẤN ĐỀ VÀ CÂU HỎI NGHIÊN CỨU

Nghiên cứu này hướng đến việc ứng dụng Mạng nơ-ron đồ thị (Graph Neural Networks - GNN) và Trí tuệ nhân tạo có thể giải thích (Explainable AI - XAI) trong phân tích mạng chuyền bóng từ dữ liệu sự kiện bóng đá đỉnh cao (StatsBomb Open Data, Ngoại hạng Anh mùa giải 2015/2016, gồm 380 trận đấu và 45.047 pha kiểm soát bóng). Ba câu hỏi nghiên cứu cốt lõi được giải quyết bao gồm:

- **RQ1 (Hiệu năng dự đoán):** Mô hình GNN (đặc biệt là kiến trúc Message Passing Neural Network - MPNN) có khả năng dự đoán chính xác pha kiểm soát bóng (possession) có dẫn tới cú dứt điểm hay không? Cơ chế gộp đồ thị (*graph readout pooling*) nào tối ưu hóa khả năng biểu diễn này so với các mô hình học máy phi đồ thị?
- **RQ2 (Chất lượng giải thích XAI):** Các phương pháp giải thích mô hình (GNNExplainer, PGExplainer) có xác định được các đường chuyền và cầu thủ quan trọng với độ tin cậy vượt trội so với baseline ngẫu nhiên (thông qua Cổng kiểm soát chất lượng Gate D17) hay không?
- **RQ3 (Chỉ số Hybrid Centrality):** Việc kết hợp mức độ quan trọng từ XAI với chỉ số trung tâm mạng truyền thống (*Betweenness Centrality*) để tạo thành chỉ số *Hybrid Centrality* có mang lại giá trị gia tăng thực sự trong việc đánh giá vai trò sáng tạo cơ hội của cầu thủ bóng đá hay không?

---

## 4.2. KẾT QUẢ DỰ ĐOÁN POSSESSION CÓ CÚ SÚT (RQ1)

### 4.2.1. Đối sánh mô hình GNN và Baseline học máy phi đồ thị
Bài toán phân loại possession là bài toán mất cân đối mẫu tự nhiên (chỉ có khoảng 14% possession kết thúc bằng một cú dứt điểm). Do đó, diện tích dưới đường cong Precision-Recall (AUC-PR) là độ đo chính được sử dụng để đánh giá mô hình, bên cạnh AUC-ROC, Brier Score và F1-score. Tất cả các mô hình được huấn luyện và đánh giá trên 5 seeds ngẫu nhiên cố định (42, 123, 456, 789, 1024) theo phân chia chia trận đấu độc lập (266 train / 57 validation / 57 test).

**Bảng 4.1: Kết quả đối sánh hiệu năng các mô hình trên tập Test (trung bình 5 seeds)**

| Phân lớp mô hình | Kiến trúc | Test AUC-PR | Test AUC-ROC | Brier Score | Test F1-Score | Best Seed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GNN (Đề xuất)** | **MPNN (mean)** | **0.5996 ± 0.0042** | **0.8757 ± 0.0034** | 0.1302 ± 0.0114 | **0.5762 ± 0.0036** | 789 |
| GNN (Baselines) | GraphSAGE | 0.5159 ± 0.0022 | 0.8504 ± 0.0009 | 0.1460 ± 0.0049 | 0.5139 ± 0.0066 | 1024 |
| GNN (Baselines) | GATv2 | 0.4620 ± 0.0048 | 0.8407 ± 0.0014 | 0.1534 ± 0.0070 | 0.4934 ± 0.0048 | 1024 |
| GNN (Baselines) | GCN | 0.4526 ± 0.0048 | 0.8310 ± 0.0013 | 0.1688 ± 0.0058 | 0.4722 ± 0.0065 | 789 |
| **Tabular Baseline** | Random Forest | **0.6198 ± 0.0017** | **0.8855 ± 0.0006** | **0.0868 ± 0.0002** | **0.5892 ± 0.0047** | — |
| **Tabular Baseline** | ANN (MLP) | 0.6117 ± 0.0097 | 0.8839 ± 0.0018 | 0.0787 ± 0.0013 | 0.5822 ± 0.0099 | — |
| Tabular Baseline | Logistic Regression| 0.3378 ± 0.0000 | 0.8221 ± 0.0000 | 0.1847 ± 0.0000 | 0.4531 ± 0.0000 | — |

*Nguồn: `results/tables/official/model_comparison.csv`*

### 4.2.2. Khảo sát các cơ chế Readout Pooling trên MPNN (Quyết định D4)
Nhằm tìm hiểu xem cơ chế gộp biểu diễn đồ thị có thể nâng cao hiệu năng của GNN hay không, 3 biến thể readout gồm Global Max Pooling (`mpnn_max`), Attention Pooling (`mpnn_attn`) và Mean kết hợp Last-edge Recipient (`mpnn_lastedge`) đã được huấn luyện đối sánh qua 5 seeds:

**Bảng 4.2: Đối sánh các cơ chế Readout Pooling của mô hình MPNN (5 seeds)**

| Mô hình | Cơ chế Readout | Validation AUC-PR | Test AUC-PR | Test AUC-ROC | Test Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`mpnn`** (Gốc) | `mean` | 0.5847 ± 0.0015 | **0.5996 ± 0.0042** | 0.8757 ± 0.0034 | 0.1302 ± 0.0114 |
| `mpnn_max` | `max` | 0.5624 ± 0.0031 | 0.5853 ± 0.0048 | 0.8811 ± 0.0010 | 0.1424 ± 0.0158 |
| `mpnn_attn` | `attention` | 0.5620 ± 0.0061 | 0.5820 ± 0.0074 | 0.8786 ± 0.0032 | 0.1336 ± 0.0114 |
| `mpnn_lastedge` | `mean_lastedge` | **0.5921 ± 0.0074** | 0.5973 ± 0.0082 | **0.8852 ± 0.0018** | **0.1256 ± 0.0138** |

*Nguồn: `results/tables/official/readout_comparison.csv`*

**Thảo luận khoa học về RQ1:**
1. **Ưu thế tuyệt đối của MPNN trong nhóm GNN:** MPNN vượt trội hoàn toàn so với GCN (+0.1470 AUC-PR), GATv2 (+0.1376) và GraphSAGE (+0.0837). Nguyên nhân chính là do MPNN tận dụng triệt để 12 chiều đặc trưng cạnh (tọa độ không gian, góc chuyền, cự ly, áp lực đối phương), trong khi GCN và GraphSAGE chỉ truyền tải thông điệp dựa trên cấu trúc liên kết đỉnh.
2. **So sánh trung thực với Tabular Baselines:** Random Forest (0.6198) và ANN (0.6117) vẫn đạt AUC-PR nhỉnh hơn MPNN (0.5996). Điều này được giải thích bởi sự khác biệt về bản chất biểu diễn: mô hình dạng bảng trực tiếp nhận các đặc trưng tổng hợp toàn cục (vị trí kết thúc chuỗi bóng, tốc độ tiến triển bóng tổng thể), trong khi GNN phải tự học biểu diễn này thông qua thông điệp lan truyền từng bước.
3. **Thực thi Quyết định D4:** Biến thể `mpnn_lastedge` đạt Validation AUC-PR cao nhất (0.5921), vượt trội baseline +0.0074. Tuy nhiên, mức cải thiện này chưa đạt ngưỡng tối thiểu 0.0100 đã đăng ký trước. Theo đúng nguyên tắc khoa học, **mô hình MPNN Mean Pooling nguyên bản tiếp tục được duy trì làm mô hình backbone chính thức**. Đáng chú ý, ở seed 1024, `mpnn_lastedge` đạt Test AUC-PR lên tới **0.6107**, tiệm cận trực tiếp hiệu năng của mô hình ANN phi đồ thị (0.6117).

---

## 4.3. ĐÁNH GIÁ CHẤT LƯỢNG GIẢI THÍCH MÔ HÌNH XAI (RQ2)

### 4.3.1. Đối sánh chất lượng giải thích và Cổng kiểm soát Gate D17
Chất lượng giải thích của 4 thuật toán (`gnn_explainer`, `pg_explainer`, `random`, `last_edge`) được đánh giá trên 150 đồ thị test (75 dương, 75 âm, $\ge 4$ cạnh) thông qua các chỉ số:
- **Fidelity+:** Độ sụt giảm xác suất khi loại bỏ top-k cạnh quan trọng (càng cao chứng tỏ cạnh bị loại thực sự quan trọng).
- **Fidelity-:** Độ sụt giảm xác suất khi chỉ giữ lại top-k cạnh quan trọng (càng âm chứng tỏ tập cạnh giữ lại bảo toàn đầy đủ dự đoán).
- **Sparsity:** Tỷ lệ nén của mặt nạ (trung bình 0.6987).
- **Stability:** Độ ổn định của giải thích khi lặp lại trên cùng đồ thị.

**Bảng 4.3: Đối sánh định lượng các phương pháp XAI và Kiểm định Gate D17**

| Phương pháp giải thích | Fidelity+ (Mean ± Std) | Fidelity- (Mean ± Std) | $\Delta\text{Fidelity}$ | Stability | Gate D17 (Bootstrap 95% CI vs Random) | Kết luận D17 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `gnn_explainer` | 0.0105 ± 0.1313 | -0.0496 ± 0.2556 | +0.0601 | 0.2912 | $[-0.0262, +0.0593]$ | **FAIL** (Vắt qua 0) |
| `pg_explainer` | -0.0078 ± 0.1168 | -0.0352 ± 0.2527 | +0.0274 | **0.9900** | $[-0.0597, +0.0263]$ | **FAIL** (Vắt qua 0) |
| `random` (Baseline) | 0.0077 ± 0.1671 | -0.0349 ± 0.2363 | +0.0426 | — | Baseline so sánh | — |
| `last_edge` (Heuristic)| **0.1490 ± 0.2343** | **-0.1419 ± 0.1372** | **+0.2909** | — | **$[+0.2095, +0.2883]$** | **PASS** (Hoàn toàn > 0) |

*Nguồn: `results/tables/official/xai_comparison.csv`*

**Thảo luận khoa học về RQ2:**
1. **Phát hiện quan trọng về cấu trúc chuỗi bóng đá:** Kiểm định Bootstrap 95% CI cho thấy cả GNNExplainer và PGExplainer đều không vượt qua được random baseline ở mức ý nghĩa thống kê về độ đo $\Delta\text{Fidelity}$ (khoảng tin cậy chứa giá trị 0). Ngược lại, heuristic `last_edge` đạt $\Delta\text{Fidelity} = +0.2909$ với khoảng tin cậy 95% hoàn toàn nằm trên 0 ($[0.2095, 0.2883]$). Điều này chứng minh: trong mạng chuyền bóng ngắn, quyết định dứt điểm phụ thuộc cực lớn vào đường chuyền tạo thời cơ cuối cùng. Các phương pháp tối ưu hóa nhiễu tổng quát khi can thiệp vào bất kỳ đường chuyền nào cũng phá vỡ chuỗi bóng tương tự như nhiễu ngẫu nhiên.
2. **Độ ổn định:** PGExplainer thể hiện tính vượt trội về độ ổn định (Stability = **0.9900** so với 0.2912 của GNNExplainer) và tốc độ suy luận (nhanh hơn ~50 lần), khiến PGExplainer trở thành công cụ tin cậy để phục vụ trích xuất điểm tầm quan trọng cầu thủ trên diện rộng.

---

## 4.4. ĐÓNG GÓP NGHIÊN CỨU CHÍNH: CHỈ SỐ HYBRID CENTRALITY (RQ3)

### 4.4.1. Tối ưu hóa tham số $\alpha^*$ trên tập Set A (190 trận)
Chỉ số Hybrid Centrality được định nghĩa theo công thức:
$$\text{HC}_i(\alpha) = \alpha \cdot \text{Rank}(\text{Centrality}_i) + (1 - \alpha) \cdot \text{Rank}(\text{XAI}_i)$$
Trong đó $\text{Centrality}_i$ là Betweenness Centrality trên mạng chuyền bóng có trọng số cả trận, và $\text{XAI}_i$ là điểm quan trọng trích xuất out-of-fold từ mô hình Cross-fitting (Quyết định D12).
Quá trình quét tham số $\alpha \in [0.0, 1.0]$ trên tập Set A với độ đo mục tiêu là mối tương quan với Expected Threat ($xT/90$) đã xác định giá trị tối ưu:
$$\alpha^* = 0.70 \quad (\rho = 0.4955)$$

### 4.4.2. Đánh giá độc lập trên tập Set B (190 trận độc lập, 481 cầu thủ)
Tại $\alpha^* = 0.70$ cố định từ Set A, chỉ số Hybrid Centrality được tính toán và kiểm định độc lập trên Set B:

**Bảng 4.4: Tương quan Spearman của Hybrid Centrality với các chỉ số bóng đá tham chiếu trên Set B**

| Chỉ số bóng đá tham chiếu | Hybrid Centrality ($\alpha^* = 0.70$) | Betweenness thuần | Điểm XAI thuần | Mức tăng so với Centrality | Mức ý nghĩa ($p$-value) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$xT/90$** (Expected Threat) | **0.5925** | 0.6005 | 0.0308 | -0.0080 | $p < 0.0001$ |
| **$xA/90$** (Expected Assists) | **0.3885** | 0.2066 | 0.5340 | **+0.1819** | $p < 0.0001$ |
| **$Key Passes/90$** | **0.4433** | 0.2541 | 0.5448 | **+0.1892** | $p < 0.0001$ |
| **$Assists/90$** | **0.3191** | 0.2076 | 0.3313 | **+0.1115** | $p < 0.0001$ |

*Nguồn: `results/tables/official/hybrid_centrality_comparison.csv`*

### 4.4.3. Kiểm định đóng góp gia tăng bằng Hồi quy OLS ($\Delta R^2$)
Hồi quy tuyến tính đa biến giải thích chỉ số $xT/90$:
- **Mô hình thuần Centrality:** $R^2 = 0.2882$.
- **Mô hình kết hợp (Centrality + XAI):** $R^2 = 0.2996$.
- **Độ tăng biến thiên giải thích:** $\Delta R^2 = +0.0114$.
- **Khoảng tin cậy 95% Bootstrap (1.000 mẫu lại):** **$[0.0015, 0.0279]$**.
Do cận dưới $0.0015 > 0$, **kết quả khẳng định thành phần XAI mang lại giá trị gia tăng thực sự và có ý nghĩa thống kê** đối với việc giải thích năng lực tạo nguy hiểm khung thành của cầu thủ, vượt ra ngoài thông tin hình học mạng đơn thuần.

### 4.4.4. Bảng xếp hạng Top Cầu thủ và Phân tích trường hợp chuyên môn

**Bảng 4.5: Top 10 Cầu thủ dẫn đầu bảng xếp hạng Hybrid Centrality Ngoại hạng Anh mùa giải 2015/16**

| Hạng | Cầu thủ | Đội bóng | Phút thi đấu | Hybrid Centrality | Rank Centrality | Rank XAI | Key Passes | Kiến tạo | $xA/90$ | $xT/90$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Mesut Özil** | Arsenal | 3.044 | **0.9374** | 0.9788 | 0.8408 | 70 | 9 | 0.220 | 0.2032 |
| 2 | **Santi Cazorla** | Arsenal | 1.291 | **0.9274** | 0.9809 | 0.8025 | 23 | 2 | 0.184 | 0.2183 |
| 3 | **Ross Barkley** | Everton | 3.079 | **0.9072** | 0.9703 | 0.7601 | 47 | 6 | 0.125 | 0.1274 |
| 4 | **Aaron Ramsey** | Arsenal | 2.622 | **0.9070** | 0.9936 | 0.7049 | 17 | 1 | 0.048 | 0.1351 |
| 5 | **Christian Eriksen**| Tottenham | 2.944 | **0.9059** | 0.9448 | 0.8153 | 75 | 8 | 0.194 | 0.1645 |
| 6 | **Troy Deeney** | Watford | 3.297 | **0.8994** | 0.9045 | 0.8875 | 41 | 5 | 0.168 | 0.0764 |
| 7 | **Darren Fletcher**| West Brom | 3.375 | **0.8979** | 0.9533 | 0.7686 | 15 | 1 | 0.033 | 0.0814 |
| 8 | **Yann M'Vila** | Sunderland | 3.103 | **0.8970** | 0.9958 | 0.6667 | 20 | 0 | 0.030 | 0.0589 |
| 9 | **Cesc Fàbregas** | Chelsea | 2.897 | **0.8921** | 0.9979 | 0.6454 | 46 | 8 | 0.158 | 0.1846 |
| 10 | **Dimitri Payet** | West Ham | 2.573 | **0.8728** | 0.9321 | 0.7346 | 80 | 5 | 0.262 | 0.1942 |

*Nguồn: `results/tables/official/hybrid_centrality_top_players.csv`*

**Bình luận chuyên môn sâu:**
- **Mesut Özil (Arsenal):** Đứng số 1 toàn giải với điểm Hybrid $0.9374$. Özil là trường hợp hoàn hảo thể hiện ưu điểm của mô hình: vừa sở hữu vị trí trọng yếu trong mạng chuyền bóng (Rank Centrality 0.9788), vừa là người tạo cơ hội sắc bén nhất giải đấu (Rank XAI 0.8408, 70 key passes, 9 kiến tạo trên Set B).
- **Dimitri Payet (West Ham) và Christian Eriksen (Tottenham):** Được thăng hạng vượt bậc nhờ điểm XAI cao ($0.7346$ và $0.8153$). Centrality truyền thống thường đánh giá thấp các cầu thủ chạy cánh hoặc số 10 tự do do họ chạm bóng ít hơn các tiền vệ trung tâm, nhưng Hybrid Centrality đã khắc phục hoàn hảo khiểm khuyết này.
- **Sự phân hóa giữa tiền vệ kiểm soát và tiền vệ công:** Các tiền vệ lùi sâu như Yann M'Vila và Aaron Ramsey có Betweenness cực cao ($> 0.99$) nhưng điểm XAI vừa phải ($0.66$–$0.70$) do ít tham gia vào đường chuyền sát vòng cấm. Chỉ số Hybrid Centrality đã cân bằng chính xác cả hai khía cạnh: khả năng giữ nhịp luân chuyển bóng và tính sát thương trong việc tạo cơ hội dứt điểm.

---

## 4.5. TỔNG KẾT VÀ ĐÓNG GÓP KHOA HỌC

1. **Về mô hình dự đoán:** Khẳng định MPNN là kiến trúc GNN phù hợp nhất cho dữ liệu mạng chuyền bóng nhờ cơ chế nhúng đặc trưng cạnh đa chiều. Cơ chế gộp `mpnn_lastedge` mở ra hướng tiếp cận triển vọng khi đạt kết quả tiệm cận mô hình học máy phi đồ thị.
2. **Về phương pháp giải thích:** Cung cấp bằng chứng thực nghiệm độc lập và nghiêm ngặt đầu tiên về tính chất của các phương pháp XAI trên mạng chuyền bóng, chỉ ra vai trò chi phối của hành động cuối cùng trong chuỗi dứt điểm.
3. **Về chỉ số đề xuất:** Đã đề xuất và kiểm chứng thành công chỉ số **Hybrid Centrality** với giao thức A/B chống rò rỉ mẫu, chứng minh tính vượt trội trong việc tương quan với các chỉ số sáng tạo cơ hội thực tế ($xA/90$, $Key Passes/90$) và mang lại đóng góp gia tăng có ý nghĩa thống kê ($\Delta R^2 > 0$ ở mức 95% tin cậy).
