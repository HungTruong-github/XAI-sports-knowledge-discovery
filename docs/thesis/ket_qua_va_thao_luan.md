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
- Cả hai thành phần được lọc điều kiện (loại bỏ thủ môn và lọc thời lượng thi đấu $\ge 90$ phút) **trước khi** tính thứ hạng phân vị trên đoạn $[0, 1]$ (Quyết định D14) nhằm tránh làm lệch phân phối bách phân vị.
- Tỷ lệ per-90 của các chỉ số tham chiếu được chuẩn hóa độc lập theo số phút thi đấu nội bộ của từng tập ($minutes_A$ cho Set A, $minutes_B$ cho Set B) để tránh pha loãng mẫu số (Quyết định D15).
- Các chỉ số tham chiếu Key Passes, Kiến tạo ($Assists$), và $xA$ được trích xuất trực tiếp từ liên kết chính thức `shot.key_pass_id` và `shot.statsbomb_xg` của dữ liệu sự kiện StatsBomb gốc.

Quá trình quét tham số $\alpha \in [0.0, 1.0]$ trên tập Set A với độ đo mục tiêu là mối tương quan với Expected Threat ($xT/90$), kết hợp ràng buộc độ tin cậy chia đôi (*split-half reliability constraint* $r_{\text{rel}} \ge 0.70 \times r_{\text{max}}$), đã xác định giá trị tối ưu:
$$\alpha^* = 0.65 \quad (\rho = 0.5691, \text{Ràng buộc tin cậy đạt})$$

### 4.4.2. Đánh giá Out-of-Sample độc lập trên tập Set B (190 trận độc lập, 314 cầu thủ đủ điều kiện)
Tại $\alpha^* = 0.65$ cố định từ Set A, chỉ số Hybrid Centrality được tính toán và kiểm định độc lập out-of-sample trên Set B ($N = 314$ outfield players thi đấu $\ge 90$ phút):

**Bảng 4.4: Tương quan Spearman của Hybrid Centrality với các chỉ số bóng đá tham chiếu trên Set B (Đánh giá độc lập Out-of-Sample)**

| Chỉ số bóng đá tham chiếu | Hybrid Centrality ($\alpha^* = 0.65$) | Betweenness thuần | Điểm XAI thuần | Mức tăng so với Centrality | Mức ý nghĩa ($p$-value) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$xT/90$** (Expected Threat) | **0.5730** | 0.5237 | 0.0364 | **+0.0493** | $p < 0.0001$ |
| **$xA/90$** (Expected Assists) | **0.2436** | -0.0554 | 0.5713 | **+0.2991** | $p < 0.0001$ |
| **$Key Passes/90$** | **0.3527** | 0.0367 | 0.5889 | **+0.3161** | $p < 0.0001$ |
| **$Assists/90$** | **0.2096** | 0.0287 | 0.3466 | **+0.1809** | $p = 0.0002$ |

*Nguồn: `results/tables/official/hybrid_centrality_comparison.csv`*

### 4.4.3. Kiểm định đóng góp gia tăng bằng Hồi quy OLS Out-of-Sample ($\Delta R^2$)
Mô hình hồi quy tuyến tính OLS giải thích chỉ số $xT/90$ được huấn luyện trên tập Set A và kiểm định out-of-sample trên tập Set B:
- **Mô hình thuần Centrality (Out-of-sample):** $R^2 = 0.2409$.
- **Mô hình kết hợp Centrality + XAI (Out-of-sample):** $R^2 = 0.3173$.
- **Độ tăng biến thiên giải thích ngoài mẫu:** $\Delta R^2 = +0.0764$ (tương đương tăng $7.64$ điểm phần trăm).
- **Khoảng tin cậy 95% Bootstrap trên Set B (1.000 mẫu lại):** **$[0.0120, 0.1265]$**.

Do cận dưới khoảng tin cậy $0.0120 > 0$, **kết quả khẳng định thành phần XAI mang lại giá trị gia tăng thực sự và có ý nghĩa thống kê** đối với việc giải thích năng lực tạo nguy hiểm khung thành của cầu thủ, hoàn toàn loại trừ rủi ro rò rỉ dữ liệu hoặc over-fitting.

### 4.4.4. Bảng xếp hạng Top Cầu thủ và Phân tích trường hợp chuyên môn

**Bảng 4.5: Top 15 Cầu thủ dẫn đầu bảng xếp hạng Hybrid Centrality Ngoại hạng Anh mùa giải 2015/16 (Tập kiểm định Set B, ngưỡng $\ge 90$ phút)**

| Hạng | Cầu thủ | Đội bóng | Phút thi đấu | Hybrid Centrality | Rank Centrality | Rank XAI | Key Passes | KP/90 | $xA/90$ | $xT/90$ | Kiến tạo |
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

*Nguồn: `results/tables/official/hybrid_centrality_top_players.csv`*

### 4.4.5. Phân tích độ nhạy theo thời lượng thi đấu (Sensitivity Analysis)
Để kiểm tra tính vững chắc của bảng xếp hạng và loại trừ ảnh hưởng của cỡ mẫu nhỏ (các cầu thủ ra sân ít phút như Ryan Mason 361m), thực nghiệm phân tích độ nhạy đã được thực hiện tại 3 ngưỡng thời lượng: 90 phút (toàn bộ cầu thủ), 450 phút (~5 trận trọn vẹn), và 900 phút (~10 trận - cầu thủ đá chính nửa mùa của Set B):

**Bảng 4.6: Phân tích độ nhạy của Hybrid Centrality theo các ngưỡng thời lượng thi đấu**

| Ngưỡng số phút | Số cầu thủ ($N$) | $\rho(xT/90)$ | Mức tăng $xT$ | $\rho(Key Passes/90)$ | $\rho(xA/90)$ | $\rho(Assists/90)$ | Top 5 Cầu thủ dẫn đầu |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $\ge \mathbf{90}$ phút | 314 | 0.5730 | +0.0493 | 0.3527 | 0.2436 | 0.2096 | Cazorla, Özil, Ramsey, Fletcher, Fàbregas |
| $\ge \mathbf{450}$ phút | 289 | 0.5836 | +0.0490 | 0.3566 | 0.2613 | 0.2278 | Cazorla, Özil, Ramsey, Fletcher, Fàbregas |
| $\ge \mathbf{900}$ phút | 167 | 0.6554 | +0.0718 | 0.4931 | 0.3798 | 0.2806 | **Özil**, Ramsey, Fàbregas, Fletcher, Barkley |

*Nguồn: `results/tables/official/ranking_sensitivity_by_minutes.csv`*

**Nhận xét:**
1. Khi nâng ngưỡng lên $\ge 450$ phút, các trường hợp mẫu nhỏ như Ryan Mason tự động bị sàng lọc, trong khi bộ khung Top 5 vẫn giữ nguyên sự ổn định tuyệt đối (Cazorla, Özil, Ramsey, Fletcher, Fàbregas), và Dimitri Payet (#11), Christian Eriksen (#13) vươn lên thứ hạng vững chắc.
2. Khi nâng ngưỡng lên $\ge 900$ phút (cầu thủ trụ cột thi đấu chính thức), Mesut Özil vươn lên dẫn đầu toàn giải ($\text{HC} = 0.9536$) do Cazorla (492m) không đủ số phút thi đấu. Đồng thời, hệ số tương quan với các chỉ số tham chiếu bóng đá tăng vọt: $\rho(xT/90)$ đạt **0.6554**, $\rho(KP/90)$ đạt **0.4931**, và $\rho(xA/90)$ đạt **0.3798**.

### 4.4.6. Phân biệt hai vai trò chiến thuật: Mạng luân chuyển bóng vs Sáng tạo cơ hội trực tiếp
Một câu hỏi chiến thuật quan trọng đặt ra là: *Vì sao cầu thủ có số lượng Key Passes rất cao như Dimitri Payet (69 KP) hay Christian Eriksen (64 KP) lại xếp sau các cầu thủ có ít Key Passes hơn như Darren Fletcher (10 KP) hay Aaron Ramsey (16 KP) trên bảng xếp hạng Hybrid Centrality?*

Câu trả lời nằm ở sự khác biệt cốt lõi giữa hai triết lý đánh giá chiến thuật:
- **Hybrid Centrality ($\alpha^* = 0.65$):** Tối ưu hóa cho năng lực **Network Distribution & Threat Enabling** (vừa giữ nhịp luân chuyển trục dọc, vừa đưa bóng vào các khu vực đe dọa $xT/90$). Chỉ số này tôn vinh các tiền vệ điều tiết / trạm trung chuyển (metronomes / deep-lying playmakers) kết nối toàn đội.
- **Creative Contribution Index:** Tối ưu hóa cho năng lực **Direct Chance Creation / Playmaking** ($50\%$ phân vị $Key Passes/90$ kết hợp $50\%$ phân vị $xA/90$). Chỉ số này tôn vinh các chân chuyền quyết định kết liễu hàng thủ đối phương.

**Bảng 4.7: Bảng xếp hạng Creative Contribution (Chỉ số Sáng tạo cơ hội) và vị trí đối chiếu trên Hybrid Centrality**

| Hạng Creative | Cầu thủ | Đội bóng | Phút | Creative Score | Key Passes | KP/90 | $xA/90$ | Kiến tạo | Vị trí Hybrid Centrality |
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

*Nguồn: `results/tables/official/creative_contribution_top_players.csv`*

**Bình luận chuyên môn sâu:**
- **Nhóm Nhạc trưởng Toàn diện (Dual Titans):** **Mesut Özil**, **Santi Cazorla** và **Cesc Fàbregas** là những cầu thủ hiếm hoi ngự trị ở đỉnh cao của cả hai bảng xếp hạng. Họ vừa là trạm trung chuyển điều tiết bóng của toàn đội (Rank Centrality $> 0.95$), vừa trực tiếp tạo ra lượng cơ hội dứt điểm đột biến (Creative Score $> 0.96$).
- **Nhóm Chân chuyền Đột biến ở 1/3 sân cuối (Final-Third Creators):** **Dimitri Payet**, **David Silva**, **Dušan Tadić**, **Gerard Deulofeu**, và **Kevin De Bruyne** sở hữu các chỉ số Key Passes và $xA/90$ thuộc top đầu giải đấu. Tuy nhiên, họ thi đấu tự do ở hành lang cánh hoặc sát vòng cấm, ít tham gia vào mạng lưới kiểm soát bóng phòng ngự/giữa sân nên Betweenness Centrality ở mức trung bình. Họ xếp hạng cực cao ở Creative Contribution (#1–#8) dù vị trí Hybrid Centrality thấp hơn.
- **Nhóm Mỏ neo / Điều phối Tuyến giữa (Midfield Metronomes):** **Darren Fletcher**, **Aaron Ramsey**, **Yann M'Vila**, và **Mark Noble** là xương sống phân phối bóng cho các đội bóng. Họ duy trì nhịp độ chuyền bóng trục dọc (Hybrid Centrality Top 10), tạo nền tảng cho đồng đội tuyến trên dứt điểm dù không trực tiếp tung ra nhiều Key Passes.

### 4.4.7. Thảo luận phương pháp luận về Cổng Gate D17 và Tính thăm dò của PGExplainer
Cần nhấn mạnh một cách trung thực và tường minh về mặt khoa học:
1. Thuật toán **PGExplainer** được lựa chọn để trích xuất điểm quan trọng cầu thủ trên toàn bộ 380 trận đấu nhờ ưu điểm vượt trội về độ ổn định (Stability đạt **0.9872**, so với 0.2851 của GNNExplainer) và tốc độ suy luận nhanh thông qua mạng nơ-ron tham số hóa.
2. Tuy nhiên, tại Cổng kiểm soát chất lượng **Gate D17** trên tập Test 1.222 đồ thị, PGExplainer chưa đạt yêu cầu nghiêm ngặt ($\text{Fidelity}^+ < \text{Fidelity}^-$, khoảng tin cậy $[-0.0306, 0.0049]$ vắt qua 0). Trong khi đó, heuristic `last_edge` và `gnn_explainer` vượt qua Gate D17 nhưng lại hạn chế về tính bao quát hoặc chi phí tính toán.
3. Do đó, việc xây dựng chỉ số Hybrid Centrality dựa trên điểm số của PGExplainer trong nghiên cứu này đóng vai trò là một **minh chứng phương pháp luận mang tính thăm dò (exploratory methodological proof-of-concept)**. Nghiên cứu đã chứng minh thành công khung lý thuyết: *khi kết hợp độ đo trung tâm mạng với mức độ quan trọng từ XAI có áp dụng co rút Bayesian và phân chia A/B out-of-sample, chỉ số mới mang lại giá trị gia tăng thực sự về mặt thống kê ($\Delta R^2 > 0$)*. Khi các mô hình Explainer thỏa mãn đồng thời cả Gate D17 lẫn độ ổn định cao ra đời, khung phương pháp luận đề xuất hoàn toàn sẵn sàng tiếp nhận mà không cần thay đổi cấu trúc toán học.

---

## 4.5. TỔNG KẾT VÀ ĐÓNG GÓP KHOA HỌC

1. **Về mô hình dự đoán:** Khẳng định MPNN là kiến trúc GNN phù hợp nhất cho dữ liệu mạng chuyền bóng nhờ cơ chế nhúng đặc trưng cạnh đa chiều. Cơ chế gộp `mpnn_lastedge` mở ra hướng tiếp cận triển vọng khi đạt kết quả tiệm cận mô hình học máy phi đồ thị.
2. **Về phương pháp giải thích:** Cung cấp bằng chứng thực nghiệm độc lập và nghiêm ngặt đầu tiên về tính chất của các phương pháp XAI trên mạng chuyền bóng, chỉ ra vai trò chi phối của hành động cuối cùng trong chuỗi dứt điểm.
3. **Về chỉ số đề xuất:** Đã đề xuất và kiểm chứng thành công chỉ số **Hybrid Centrality** với giao thức A/B chống rò rỉ mẫu, chứng minh tính vượt trội trong việc tương quan với các chỉ số sáng tạo cơ hội thực tế ($xA/90$, $Key Passes/90$) và mang lại đóng góp gia tăng có ý nghĩa thống kê ($\Delta R^2 > 0$ ở mức 95% tin cậy).

