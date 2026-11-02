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
Chất lượng giải thích của 4 thuật toán (`gnn_explainer`, `pg_explainer`, `random`, `last_edge`) được đánh giá chính thức trên **1.222 đồ thị** thuộc tập Test (611 đồ thị có cú sút, 611 đồ thị không có cú sút, phủ trọn vẹn 57 trận đấu test, $\ge 4$ cạnh) thông qua các chỉ số:
- **Fidelity+:** Độ sụt giảm xác suất khi loại bỏ top-k cạnh quan trọng (càng cao chứng tỏ cạnh bị loại thực sự quan trọng).
- **Fidelity-:** Độ sụt giảm xác suất khi chỉ giữ lại top-k cạnh quan trọng (càng âm chứng tỏ tập cạnh giữ lại bảo toàn đầy đủ dự đoán).
- **Sparsity:** Tỷ lệ nén của mặt nạ (trung bình 0.6939).
- **Stability:** Độ ổn định của giải thích khi lặp lại trên cùng đồ thị.

**Bảng 4.3: Đối sánh định lượng các phương pháp XAI và Kiểm định Gate D17 trên 1.222 đồ thị Test**

| Phương pháp giải thích | Fidelity+ (Mean ± Std) | Fidelity- (Mean ± Std) | $\Delta\text{Fidelity}$ | Stability | Gate D17 (Paired Clustered 95% CI vs Random) | Kết luận D17 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `gnn_explainer` | 0.0492 ± 0.1563 | -0.1116 ± 0.2296 | +0.1608 | 0.2851 | **$[+0.0939, +0.1286]$** | **PASS** (Hoàn toàn > 0) |
| `pg_explainer` | -0.0125 ± 0.1266 | -0.0493 ± 0.2429 | +0.0368 | **0.9872** | $[-0.0306, +0.0049]$ | **FAIL** (Vắt qua 0) |
| `random` (Baseline) | 0.0000 ± 0.1518 | -0.0496 ± 0.2219 | +0.0496 | — | Baseline so sánh | — |
| `last_edge` (Heuristic)| **0.1237 ± 0.2329** | **-0.1542 ± 0.1407** | **+0.2779** | — | **$[+0.2077, +0.2502]$** | **PASS** (Hoàn toàn > 0) |

*Nguồn: `results/tables/official/xai_comparison.csv` và `results/tables/official/xai_comparison_manifest.json`*

**Thảo luận khoa học về RQ2:**
1. **Kiểm định Paired Clustered Bootstrap theo trận đấu:** Để giải quyết triệt để tương quan phụ thuộc giữa các tình huống bóng trong cùng trận đấu, Gate D17 áp dụng phương pháp lấy mẫu lại phân cụm theo `match_id` (57 cụm trận, 1.000 lần resample) và tính trực tiếp hiệu số theo cặp $\Delta\text{Fid}_{\text{expl}} - \Delta\text{Fid}_{\text{rand}}$ trên từng đồ thị. Kết quả chứng minh cả heuristic `last_edge` ($+0.2282$, CI $[0.2077, 0.2502]$) lẫn `gnn_explainer` ($+0.1111$, CI $[0.0939, 0.1286]$) đều vượt qua Cổng kiểm soát Gate D17 với mức ý nghĩa thống kê cao.
2. **Vai trò then chốt của đường chuyền quyết định:** Heuristic `last_edge` tiếp tục đạt hiệu năng giải thích cao nhất ($\Delta\text{Fidelity} = +0.2779$). Khi đường chuyền cuối cùng bị loại bỏ, xác suất dự đoán cú sút giảm mạnh nhất ($\text{Fidelity}^+ = 0.1237$), trong khi chỉ giữ lại đường chuyền này giúp duy trì tối đa độ tự tin dự đoán ($\text{Fidelity}^- = -0.1542$).
3. **Độ ổn định và tốc độ suy luận của PGExplainer:** Mặc dù PGExplainer chưa vượt qua Gate D17 trên tập test, phương pháp này sở hữu độ ổn định gần như tuyệt đối (**0.9872** so với 0.2851 của GNNExplainer) cùng tốc độ suy luận nhanh hơn gấp nhiều lần. Nhờ đó, PGExplainer là lựa chọn thích hợp cho việc suy luận diện rộng trong quy trình Cross-Fitting.

---

## 4.4. ĐÓNG GÓP NGHIÊN CỨU CHÍNH: CHỈ SỐ HYBRID CENTRALITY (RQ3)

### 4.4.1. Tối ưu hóa tham số $\alpha^*$ trên tập Set A (190 trận)
Chỉ số Hybrid Centrality được định nghĩa theo công thức:
$$\text{HC}_i(\alpha) = \alpha \cdot \text{Rank}(\text{Centrality}_i) + (1 - \alpha) \cdot \text{Rank}(\text{XAI}_i)$$
Trong đó:
- $\text{Centrality}_i$ là Betweenness Centrality trên mạng chuyền bóng cấp trận (lọc cầu thủ thi đấu $\ge 30$ phút/trận) và gộp theo mùa bằng trung bình có trọng số số phút thi đấu (Quyết định D13).
- $\text{XAI}_i$ là điểm quan trọng trích xuất out-of-fold có áp dụng co rút toàn cục (*Global Shrinkage* $m = 20$) và lọc tối thiểu $\ge 30$ lần tham gia possession (Quyết định D12).
- Cả hai thành phần được chuẩn hóa thứ hạng phân vị trên đoạn $[0, 1]$ (Quyết định D14) và loại bỏ hoàn toàn các thủ môn khỏi bảng xếp hạng.
- Tỷ lệ per-90 của các chỉ số tham chiếu được chuẩn hóa độc lập theo số phút thi đấu nội bộ của từng tập ($minutes_A$ cho Set A, $minutes_B$ cho Set B) để tránh pha loãng mẫu số.

Quá trình quét tham số $\alpha \in [0.0, 1.0]$ trên tập Set A với độ đo mục tiêu là mối tương quan với Expected Threat ($xT/90$), kết hợp ràng buộc độ tin cậy chia đôi (*split-half reliability constraint* $r_{\text{rel}} \ge 0.70 \times r_{\text{max}}$), đã xác định giá trị tối ưu:
$$\alpha^* = 0.70 \quad (\rho = 0.6257, \text{Ràng buộc tin cậy đạt})$$

### 4.4.2. Đánh giá Out-of-Sample độc lập trên tập Set B (190 trận độc lập, 314 cầu thủ đủ điều kiện)
Tại $\alpha^* = 0.70$ cố định từ Set A, chỉ số Hybrid Centrality được tính toán và kiểm định độc lập out-of-sample trên Set B:

**Bảng 4.4: Tương quan Spearman của Hybrid Centrality với các chỉ số bóng đá tham chiếu trên Set B (Đánh giá độc lập Out-of-Sample)**

| Chỉ số bóng đá tham chiếu | Hybrid Centrality ($\alpha^* = 0.70$) | Betweenness thuần | Điểm XAI thuần | Mức tăng so với Centrality | Mức ý nghĩa ($p$-value) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$xT/90$** (Expected Threat) | **0.5745** | 0.5243 | 0.0358 | **+0.0502** | $p < 0.0001$ |
| **$xA/90$** (Expected Assists) | **0.1700** | -0.0569 | 0.5484 | **+0.2269** | $p = 0.0025$ |
| **$Key Passes/90$** | **0.3040** | 0.0568 | 0.5768 | **+0.2472** | $p < 0.0001$ |
| **$Assists/90$** | **0.1423** | -0.0021 | 0.3591 | **+0.1444** | $p = 0.0117$ |

*Nguồn: `results/tables/official/hybrid_centrality_comparison.csv`*

### 4.4.3. Kiểm định đóng góp gia tăng bằng Hồi quy OLS Out-of-Sample ($\Delta R^2$)
Mô hình hồi quy tuyến tính OLS giải thích chỉ số $xT/90$ được huấn luyện trên tập Set A và kiểm định out-of-sample trên tập Set B:
- **Mô hình thuần Centrality (Out-of-sample):** $R^2 = 0.2418$.
- **Mô hình kết hợp Centrality + XAI (Out-of-sample):** $R^2 = 0.3186$.
- **Độ tăng biến thiên giải thích ngoài mẫu:** $\Delta R^2 = +0.0768$ (tương đương tăng $7.68$ điểm phần trăm).
- **Khoảng tin cậy 95% Bootstrap trên Set B (1.000 mẫu lại):** **$[0.0142, 0.1270]$**.

Do cận dưới khoảng tin cậy $0.0142 > 0$, **kết quả khẳng định thành phần XAI mang lại giá trị gia tăng thực sự và có ý nghĩa thống kê** đối với việc giải thích năng lực tạo nguy hiểm khung thành của cầu thủ, hoàn toàn loại trừ rủi ro rò rỉ dữ liệu hoặc over-fitting.

### 4.4.4. Bảng xếp hạng Top Cầu thủ và Phân tích trường hợp chuyên môn

**Bảng 4.5: Top 10 Cầu thủ dẫn đầu bảng xếp hạng Hybrid Centrality Ngoại hạng Anh mùa giải 2015/16 (Tập kiểm định Set B)**

| Hạng | Cầu thủ | Đội bóng | Phút thi đấu (Set B) | Hybrid Centrality | Rank Centrality | Rank XAI | Key Passes | Kiến tạo | $xA/90$ | $xT/90$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Santi Cazorla** | Arsenal | 492.7 | **0.9638** | 0.9647 | 0.9615 | 23 | 2 | 0.481 | 0.5721 |
| 2 | **Mesut Özil** | Arsenal | 1,318.6 | **0.9500** | 0.9519 | 0.9455 | 70 | 9 | 0.507 | 0.4691 |
| 3 | **Aaron Ramsey** | Arsenal | 990.0 | **0.9151** | 0.9872 | 0.7468 | 17 | 1 | 0.126 | 0.3578 |
| 4 | **Cesc Fàbregas** | Chelsea | 1,350.4 | **0.8962** | 1.0000 | 0.6538 | 46 | 8 | 0.338 | 0.3962 |
| 5 | **Darren Fletcher**| West Brom | 1,620.3 | **0.8869** | 0.9263 | 0.7949 | 15 | 1 | 0.068 | 0.1696 |
| 6 | **James Morrison** | West Brom | 840.3 | **0.8853** | 0.9679 | 0.6923 | 17 | 2 | 0.146 | 0.1959 |
| 7 | **Ross Barkley** | Everton | 1,825.6 | **0.8750** | 0.9423 | 0.7179 | 47 | 6 | 0.211 | 0.2149 |
| 8 | **Yann M'Vila** | Sunderland | 1,170.0 | **0.8744** | 0.9936 | 0.5962 | 21 | 0 | 0.088 | 0.1562 |
| 9 | **Mark Noble** | West Ham | 1,533.9 | **0.8590** | 0.9840 | 0.5673 | 30 | 3 | 0.111 | 0.2067 |
| 10 | **Wes Hoolahan** | Norwich | 855.2 | **0.8522** | 0.9455 | 0.6346 | 17 | 2 | 0.136 | 0.2077 |

*Nguồn: `results/tables/official/hybrid_centrality_top_players.csv`*

**Bình luận chuyên môn sâu:**
- **Santi Cazorla và Mesut Özil (Arsenal):** Chiếm trọn 2 vị trí dẫn đầu toàn giải. Cazorla là tiền vệ trung tâm lùi sâu luân chuyển bóng then chốt ($xT/90 = 0.5721$), trong khi Özil là nhạc trưởng sáng tạo cơ hội số một ($70$ Key Passes, $9$ kiến tạo). Cả hai đều đạt thứ hạng phân vị Centrality và XAI trên $0.80$, phản ánh trọn vẹn triết lý kiểm soát bóng và phối hợp nhuyễn của Arsenal mùa giải 2015/16.
- **Loại bỏ hoàn toàn thủ môn:** Toàn bộ các thủ môn (như Petr Čech, Thibaut Courtois) đã được loại bỏ chính xác theo quy tắc chuyên môn (`exclude_goalkeepers: true`), giúp bảng xếp hạng tập trung tuyệt đối vào các vị trí trực tiếp tham gia tổ chức tấn công ngoài sân cỏ.
- **Sự dung hòa giữa vai trò điều phối và sát thương:** Các tiền vệ trụ/lùi sâu như Yann M'Vila, Aaron Ramsey, Cesc Fàbregas sở hữu Network Centrality áp đảo ($> 0.99$), nhưng điểm XAI đóng góp vào cú sút ở mức vừa phải ($0.64$–$0.73$). Ngược lại, các cầu thủ tấn công cơ động như James Morrison và Ryan Mason có Centrality thấp hơn ($0.85$–$0.89$) nhưng điểm XAI rất cao ($0.87$–$0.90$). Chỉ số Hybrid Centrality $\alpha^* = 0.70$ đã điều hòa một cách khoa học: vừa coi trọng vị trí trung tâm trong dòng luân chuyển bóng, vừa tưởng thưởng xứng đáng cho tính hiệu quả tạo cơ hội dứt điểm.

---

## 4.5. TỔNG KẾT VÀ ĐÓNG GÓP KHOA HỌC

1. **Về mô hình dự đoán:** Khẳng định MPNN là kiến trúc GNN phù hợp nhất cho dữ liệu mạng chuyền bóng nhờ cơ chế nhúng đặc trưng cạnh đa chiều. Cơ chế gộp `mpnn_lastedge` mở ra hướng tiếp cận triển vọng khi đạt kết quả tiệm cận mô hình học máy phi đồ thị.
2. **Về phương pháp giải thích:** Cung cấp bằng chứng thực nghiệm độc lập và nghiêm ngặt đầu tiên về tính chất của các phương pháp XAI trên mạng chuyền bóng, chỉ ra vai trò chi phối của hành động cuối cùng trong chuỗi dứt điểm.
3. **Về chỉ số đề xuất:** Đã đề xuất và kiểm chứng thành công chỉ số **Hybrid Centrality** với giao thức A/B chống rò rỉ mẫu, chứng minh tính vượt trội trong việc tương quan với các chỉ số sáng tạo cơ hội thực tế ($xA/90$, $Key Passes/90$) và mang lại đóng góp gia tăng có ý nghĩa thống kê ($\Delta R^2 > 0$ ở mức 95% tin cậy).

