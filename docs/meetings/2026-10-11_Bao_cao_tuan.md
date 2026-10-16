# BÁO CÁO TIẾN ĐỘ ĐỒ ÁN
## BÁO CÁO TIẾN ĐỘ TUẦN — 11/10/2026

**Đề tài:** Hệ thống trí tuệ nhân tạo có thể giải thích (XAI) trong bài toán đánh giá và khám phá tri thức thể thao  
**Hướng nghiên cứu cụ thể:** Mạng chuyền bóng và phân tích chiến thuật bóng đá bằng Graph Neural Network (GNN) kết hợp XAI  
**Thời gian thực hiện:** Tuần từ 04/10/2026 đến 11/10/2026  
**Trọng tâm báo cáo:** Hoàn thiện trọn vẹn 3 bước cốt lõi trong quy trình: Bước [4] Chia tập train/val/test theo trận (tránh rò rỉ dữ liệu); Bước [5] Huấn luyện mô hình GNN (Bài toán 1 – Dự đoán); Bước [6] Đánh giá mô hình dự đoán và Đối sánh 3 tầng (3-tier baselines).

---

## 1. Tổng quan nội dung hoàn thành trong tuần (Tuần 04/10 – 11/10/2026)

Trong tuần làm việc vừa qua, nối tiếp kết quả đã được nghiệm thu của tuần trước (hoàn tất tiền xử lý và trích xuất thành công 45.047 đồ thị possession trên toàn bộ 380 trận đấu của giải Ngoại hạng Anh Premier League 2015/2016), nhóm đã triển khai và hoàn thành dứt điểm 3 bước tiếp theo trong quy trình nghiên cứu. Toàn bộ các bước này tập trung xây dựng nền tảng vững chắc cho Bài toán 1 (Bài toán Dự đoán) trước khi chuyển giao mô hình sang Bài toán 2 (Bài toán Giải thích XAI):

- **Bước [4] — Chia tập dữ liệu Train / Validation / Test theo trận:** Nhóm đã xây dựng cơ chế phân chia tập dữ liệu ở cấp độ trận đấu (*match-level split*) thay vì phân chia ngẫu nhiên theo từng pha bóng (*possession-level split*), kết hợp thuật toán phân tầng (*stratified splitting*) theo tỉ lệ nhãn dương của trận. Cơ chế này loại bỏ hoàn toàn nguy cơ rò rỉ thông tin chiến thuật giữa tập huấn luyện và kiểm thử, đồng thời cố định seed và xuất danh sách trận đấu cụ thể để bảo đảm tính tái lập 100%.
- **Bước [5] — Huấn luyện mô hình GNN (Bài toán 1 – Dự đoán):** Nhóm đã xây dựng và hoàn thiện pipeline huấn luyện mô hình dự đoán xác suất pha bóng dẫn đến cú sút ở cấp độ đồ thị (*Graph Classification*). Nhóm đã cài đặt 4 kiến trúc GNN (MPNN, GCN, GATv2, GraphSAGE) với không gian 8 đặc trưng đỉnh và 12 đặc trưng cạnh phong phú; thiết lập hàm mất mát Weighted BCE Loss ($pos\_weight = 6,31$) giải quyết triệt để bài toán mất cân bằng lớp; áp dụng tối ưu AdamW kết hợp cơ chế dừng sớm (*Early Stopping*) giám sát theo chỉ số AUC-PR.
- **Bước [6] — Đánh giá mô hình dự đoán và Đối sánh 3 tầng:** Nhóm đã thực hiện đánh giá toàn diện trên tập kiểm tra độc lập gồm 6.646 đồ thị của 57 trận đấu chưa từng xuất hiện trong quá trình huấn luyện. Kết quả được đối sánh theo đúng cam kết 3 tầng: Tầng 2 (đối sánh 4 kiến trúc backbone GNN trên cùng cấu hình dữ liệu) và Tầng 3(a) (đối sánh với 3 baseline phi đồ thị: Logistic Regression, Random Forest, ANN trên 16 đặc trưng passing path). Kết quả thực nghiệm khẳng định MPNN vượt trội hoàn toàn trong nhóm mô hình đồ thị và bám sát các baseline dạng bảng, đồng thời bảo toàn trọn vẹn cấu trúc mạng để phục vụ bước giải thích XAI.

> **Tóm tắt trạng thái kỹ thuật:** Toàn bộ mã nguồn, cấu hình thử nghiệm và dữ liệu đã vượt qua 52/52 bài kiểm thử tự động (`pytest`), kiểm tra linter (`ruff`) và định dạng (`black`) đạt chuẩn chất lượng tuyệt đối. Tất cả các bảng thống kê và mô hình tốt nhất đã được lưu trữ chính thức trong hệ thống.

---

## 2. Chi tiết triển khai và Kết quả Bước [4] — Chia tập dữ liệu (Tránh rò rỉ)

### 2.1. Phương pháp luận và Cơ chế chống rò rỉ dữ liệu (Zero Data Leakage)

Trong phân tích bóng đá, các pha kiểm soát bóng (*possession*) diễn ra trong cùng một trận đấu có sự phụ thuộc rất lớn vào bối cảnh: hai đội bóng đối đầu trực tiếp, đội hình xuất phát cụ thể, huấn luyện viên, sơ đồ chiến thuật và điều kiện thời tiết/sân bãi. Nếu áp dụng cách chia ngẫu nhiên truyền thống ở cấp độ từng pha bóng (*possession-level split*), các possession của cùng một trận sẽ xuất hiện đồng thời ở cả tập huấn luyện (*train*) và tập kiểm tra (*test*). Khi đó, mô hình GNN rất dễ học vẹt các mẫu hình phối hợp đặc thù của trận đấu đó, dẫn đến hiện tượng rò rỉ dữ liệu (*data leakage*) nghiêm trọng và làm sai lệch hoàn toàn tính khách quan của kết quả đánh giá.

Để đảm bảo tính trung thực khoa học, nhóm tuân thủ nguyên tắc bắt buộc: **Phân chia tập dữ liệu ở cấp độ TRẬN ĐẤU (*match-level split*)**. Toàn bộ các pha bóng thuộc về cùng một trận đấu chỉ được phép xuất hiện duy nhất ở một tập (Train, Validation hoặc Test).

**Thuật toán phân tầng đề xuất:**
1. Thống kê số lượng pha bóng và tỉ lệ nhãn dương (possession dẫn đến cú sút) cho từng trận đấu trong toàn bộ 380 trận.
2. Chia phân phối tỉ lệ nhãn dương của các trận thành 5 khoảng phân vị (*quantile bins*).
3. Phân chia có phân tầng theo tỉ lệ 70% Train (266 trận), 15% Validation (57 trận), và 15% Test (57 trận) dựa trên các khoảng phân vị này, bảo đảm tỉ lệ nhãn dương tổng thể ở cả 3 tập luôn cân bằng nhau.
4. Cố định giá trị random `seed = 42`, xuất cố định danh sách mã trận đấu ra các tệp tin CSV (`train_matches.csv`, `val_matches.csv`, `test_matches.csv`, `match_splits.csv`) để phục vụ kiểm chứng độc lập.

### 2.2. Kết quả định lượng phân chia dữ liệu

*Bảng 1: Phân bổ dữ liệu giữa các tập (Premier League 2015/16)*

| Tập dữ liệu | Số trận (match) | Tỉ lệ trận | Số possessions | Tỉ lệ possession | Số nhãn dương (Shot) | Tỉ lệ nhãn dương |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 266 | 70,0% | 31.864 | 70,24% | 4.548 | 14,27% |
| **Validation** | 57 | 15,0% | 6.811 | 15,01% | 961 | 14,11% |
| **Test** | 57 | 15,0% | 6.690 | 14,75% | 962 | 14,38% |
| **Tổng cộng** | **380** | **100,0%** | **45.365** | **100,0%** | **6.471** | **14,26%** |

Khi ánh xạ sang tập 45.047 đối tượng đồ thị PyTorch Geometric hợp lệ (đã lọc các pha bóng dưới 2 đường chuyền thành công hoặc đồ thị suy biến), số lượng đồ thị cụ thể ở từng tập như sau:
- **Tập Train:** 31.644 đồ thị (trong đó 4.328 đồ thị nhãn dương và 27.316 đồ thị nhãn âm).
- **Tập Validation:** 6.757 đồ thị (trong đó 945 đồ thị nhãn dương và 5.812 đồ thị nhãn âm).
- **Tập Test:** 6.646 đồ thị (trong đó 936 đồ thị nhãn dương và 5.710 đồ thị nhãn âm).

### 2.3. Kết quả kiểm tra toàn vẹn phân chia (Split Sanity Checks)

*Bảng 2: Kết quả kiểm tra tính toàn vẹn và chống rò rỉ dữ liệu*

| Tiêu chí kiểm tra | Mô tả / Ngưỡng kỹ thuật | Giá trị thực tế | Kết quả |
| :--- | :--- | :---: | :---: |
| `no_leakage_train_val` | Số trận trùng lặp giữa tập Train và tập Val | 0 trận | **PASS** |
| `no_leakage_train_test` | Số trận trùng lặp giữa tập Train và tập Test | 0 trận | **PASS** |
| `no_leakage_val_test` | Số trận trùng lặp giữa tập Val và tập Test | 0 trận | **PASS** |
| `splits_cover_all_matches` | Độ bao phủ toàn bộ số trận đấu của mùa giải | 380 / 380 trận | **PASS** |
| `stratification_rate_balance` | Độ lệch tỉ lệ nhãn dương lớn nhất giữa các tập (< 2%) | 0,0027 (0,27%) | **PASS** |

> **Đánh giá kết quả Bước 4:** Kết quả kiểm tra xác nhận 5/5 tiêu chí đều đạt chuẩn PASS. Độ lệch tỉ lệ nhãn dương tối đa giữa các tập chỉ là 0,27%, cho thấy dữ liệu được cân bằng hoàn hảo và không tồn tại bất kỳ hiện tượng rò rỉ dữ liệu nào giữa các tập.

---

## 3. Chi tiết triển khai và Kết quả Bước [5] — Huấn luyện mô hình GNN (Bài toán 1 – Dự đoán)

### 3.1. Thiết kế không gian đặc trưng đồ thị (Graph Feature Representation)

**1. Không gian đặc trưng Đỉnh (Node Features — 8 chiều):**
- Tọa độ trung bình $(x_{mean}, y_{mean})$ của cầu thủ trên sân trong suốt pha bóng (chuẩn hóa về khoảng $[0, 1]$).
- Vai trò thi đấu chính (*tactical role*) mã hóa one-hot 4 chiều: Goalkeeper (GK), Defender (DF), Midfielder (MF), Forward (FW).
- Số lần chạm bóng / nhận bóng (*touch count*) của cầu thủ trong pha bóng (chuẩn hóa z-score).
- Thứ tự chạm bóng đầu tiên trong chuỗi chuyền (*sequence order*), giúp mô hình nắm bắt cầu thủ khởi xướng pha tấn công.

**2. Không gian đặc trưng Cạnh (Edge Features — 12 chiều):**
- Tọa độ điểm bắt đầu và điểm kết thúc của đường chuyền: $(x_{start}, y_{start}, x_{end}, y_{end})$ chuẩn hóa $[0, 1]$.
- Chiều dài đường chuyền (*pass distance / length*).
- Góc chuyền bóng mã hóa liên tục: $\sin(angle)$ và $\cos(angle)$, tránh điểm gián đoạn biên góc.
- Độ cao đường chuyền mã hóa one-hot 3 chiều: Ground Pass (bóng sệt), Low Pass (bóng tầm trung), High Pass (bóng bổng).
- Thời điểm thực hiện đường chuyền (*relative timestamp*) tính từ đầu pha bóng, phản ánh nhịp độ tấn công.

### 3.2. Cấu trúc mạng và 4 kiến trúc GNN đã cài đặt

- **MPNN (Message Passing Neural Network — Mô hình đề xuất chính):** Sử dụng lớp tích chập `NNConv` (*Neural Network Convolution*). Thông điệp truyền qua mỗi cạnh được điều kiện hóa trực tiếp bởi mạng nơ-ron MLP phụ nhận đầu vào là 12 đặc trưng cạnh. Điều này cho phép mô hình biến đổi thông tin trao đổi giữa hai cầu thủ phụ thuộc chính xác vào hướng, cự ly và độ cao của đường chuyền.
- **GCN (Graph Convolutional Network — Mô hình đề xuất cơ sở):** Kiến trúc tích chập đồ thị phổ biến dựa trên xấp xỉ quang phổ bậc 1. Mô hình tổng hợp đặc trưng của các cầu thủ lân cận kết hợp chuẩn hóa ma trận bậc liên kết đối xứng, không sử dụng thuộc tính cạnh.
- **GATv2 (Graph Attention Network v2 — Baseline Tầng 2):** Sử dụng cơ chế dynamic multi-head attention (4 heads). Mô hình tự động học trọng số chú ý giữa các cầu thủ tham gia chuyền bóng, khắc phục điểm nghẽn biểu diễn của GAT nguyên bản.
- **GraphSAGE (Baseline Tầng 2):** Đại diện cho hướng tiếp cận lấy mẫu và tổng hợp lân cận (*sample-and-aggregate*) với hàm tổng hợp Mean Aggregator.
- **Lớp tổng hợp (Readout) và Đầu phân loại (Classification Head):** Sau các lớp GNN, mô hình sử dụng `GlobalReadout` (Global Mean Pooling) để tổng hợp biểu diễn đồ thị, sau đó đưa qua `BinaryClassifierHead` (Linear 64 $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout 0,2 $\to$ Linear 1) để tạo ra logit dự đoán.

### 3.3. Chiến lược giải quyết mất cân bằng lớp và Thiết lập huấn luyện

- **Giải pháp xử lý mất cân bằng:** Tỉ lệ nhãn dương trong tập huấn luyện là 13,68% (mẫu âm / dương = 27.316 / 4.328 ≈ 6,31 : 1). Nhóm sử dụng `WeightedBCEWithLogitsLoss` với trọng số dương `pos_weight = 6,31` để phạt nặng việc bỏ sót các pha bóng nguy hiểm.
- **Cấu hình huấn luyện:**
  - Bộ tối ưu: AdamW (`lr=0.001`, `weight_decay=1e-4`).
  - Lịch trình học: `ReduceLROnPlateau` (`mode='max'`, `factor=0.5`, `patience=5`).
  - Tiêu chí dừng sớm (*Early Stopping*): Giám sát `val_auc_pr` (patience=20 epochs) thay vì Loss hay Accuracy.
  - Lưu trữ checkpoint tốt nhất tại `models/official/{backbone}_best.pt`.

---

## 4. Chi tiết triển khai và Kết quả Bước [6] — Đánh giá mô hình & Đối sánh 3 tầng

### 4.1. Hệ thống độ đo đánh giá đã cam kết

Đánh giá trên tập kiểm tra độc lập gồm 6.646 đồ thị của 57 trận đấu test:
- **AUC-PR:** Độ đo quan trọng số một đối với dữ liệu mất cân bằng lớp.
- **AUC-ROC:** Năng lực phân tách tổng quát giữa hai lớp trên mọi ngưỡng.
- **F1-Score, Precision, Recall:** Đánh giá tại ngưỡng quyết định tối ưu xác định từ tập Validation.
- **Brier Score:** Mức độ hiệu chuẩn xác suất (càng thấp càng tốt).
- **Accuracy:** Độ chính xác tổng thể (tham khảo).

### 4.2. Bảng tổng hợp đối sánh hiệu năng trên tập Test (Tầng 2 & Tầng 3a)

*Bảng 3: Bảng đối sánh hiệu năng các mô hình trên tập Test chính thức*

| Mô hình | Tầng đối sánh | Dữ liệu | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | Brier | Test Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MPNN (đề xuất)** | **Tier 2 — GNN** | **Đồ thị** | **0,8471** | **0,4630** | **0,6678** | **0,5468** | **0,8732** | **0,5747** | **0,1040** | **0,8365** |
| **GCN (đề xuất)** | Tier 2 — GNN | Đồ thị | 0,6806 | 0,2680 | 0,7582 | 0,3960 | 0,7769 | 0,3261 | 0,1885 | 0,9760 |
| **GATv2 (baseline)** | Tier 2 — GNN | Đồ thị | 0,6750 | 0,2561 | 0,7102 | 0,3764 | 0,7526 | 0,2983 | 0,1842 | 1,0177 |
| **GraphSAGE (baseline)** | Tier 2 — GNN | Đồ thị | 0,6031 | 0,2433 | 0,8878 | 0,3819 | 0,8017 | 0,3763 | 0,2312 | 0,9432 |
| **Logistic Regression** | Tier 3(a) — Phi đồ thị | Bảng | 0,7084 | 0,3019 | 0,8464 | 0,4450 | 0,8221 | 0,3379 | 0,1847 | — |
| **Random Forest** | Tier 3(a) — Phi đồ thị | Bảng | 0,8834 | 0,5702 | 0,6329 | 0,5999 | 0,8844 | 0,6234 | 0,0870 | — |
| **ANN (MLP)** | Tier 3(a) — Phi đồ thị | Bảng | 0,8945 | 0,6770 | 0,4521 | 0,5421 | 0,8831 | 0,6141 | 0,0782 | — |

### 4.3. Phân tích kết quả thực nghiệm và Luận giải học thuật

1. **Phân tích Tầng 2 (Sự vượt trội của MPNN trong nhóm GNN Backbone):**
   - AUC-PR của MPNN đạt **0,5747**, vượt xa GCN (0,3261 — tăng +76,2%), GATv2 (0,2983 — tăng +92,7%) và GraphSAGE (0,3763 — tăng +52,7%).
   - Brier Score của MPNN đạt **0,1040**, thể hiện mức độ hiệu chuẩn xác suất vượt trội.
   - *Luận giải bóng đá & cấu trúc mạng:* Các mô hình GCN, GATv2 và GraphSAGE chỉ truyền tin dựa trên liên kết topology và thuộc tính đỉnh, hoàn toàn bỏ qua đặc trưng cạnh. Trong bóng đá, giá trị của một đường chuyền phụ thuộc mang tính quyết định vào tọa độ không gian (chuyền vào vòng cấm hay chuyền về sân nhà), cự ly và độ cao. MPNN sử dụng mạng nơ-ron điều kiện hóa thông điệp bởi 12 đặc trưng cạnh (`NNConv`), nhờ đó mô hình nắm bắt được hình học không gian của pha bóng một cách tự nhiên.

2. **Phân tích Tầng 3(a) (Đối sánh Đồ thị vs Phi đồ thị):**
   - Các mô hình baseline phi đồ thị (Random Forest, ANN) đạt AUC-PR cao (0,61 – 0,62) nhờ việc sử dụng 16 đặc trưng dạng bảng đã được tiền xử lý và tổng hợp toàn cục. MPNN bám rất sát hiệu năng này (AUC-ROC 0,8732 so với 0,8844 của RF; AUC-PR 0,5747 so với 0,6234).
   - *Điểm mấu chốt để bảo vệ trước hội đồng:* Mô hình dạng bảng nén toàn bộ pha bóng thành một vector phẳng, hoàn toàn triệt tiêu cấu trúc liên kết mạng chuyền giữa các cầu thủ. Do đó, mô hình dạng bảng **KHÔNG THỂ giải thích được đường chuyền nào hoặc cầu thủ nào là mắt xích quyết định trong pha phối hợp**, và **KHÔNG THỂ áp dụng các giải thuật XAI đồ thị (GNNExplainer, PGExplainer)** để trích xuất subgraph quan trọng hay tính chỉ số Hybrid Centrality. MPNN là mô hình duy nhất vừa đạt hiệu năng dự đoán cao tiệm cận baseline dạng bảng, vừa bảo toàn 100% cấu trúc mạng để phục vụ Bài toán 2.

3. **Quyết định lựa chọn mô hình chính thức cho Bước [7]:**
   - Trên tập Validation, MPNN đạt Val AUC-PR cao nhất (**0,5831**), vượt trội so với tất cả các backbone GNN khác. Do đó, MPNN được lựa chọn làm mô hình GNN chính thức đại diện cho đề tài để chuyển giao sang bước áp dụng XAI ở tuần kế tiếp.

> **Kết luận đánh giá Bài toán 1:** MPNN được xác nhận là backbone dự đoán chính thức của đề tài nhờ thỏa mãn đồng thời hai điều kiện cốt lõi: (1) Đạt hiệu năng dự đoán và hiệu chuẩn cao nhất trong nhóm GNN (AUC-ROC: 0,8732; AUC-PR: 0,5747; Brier: 0,1040); và (2) Lưu giữ trọn vẹn đặc trưng cạnh không gian-thời gian để cung cấp đầu vào cho các giải thuật XAI giải thích ở mức đường chuyền.

---

## 5. Kế hoạch triển khai tuần tiếp theo (Tuần 12/10 – 18/10/2026)

Sau khi đã hoàn tất trọn vẹn Bài toán 1 (Bài toán Dự đoán) với mô hình MPNN đã được tối ưu hóa, nhóm sẽ chính thức bước vào nhánh nghiên cứu cốt lõi thứ hai: **Bài toán 2 – Giải thích (XAI Task)**:

1. **Bước [7] — Áp dụng XAI trên mô hình MPNN đã huấn luyện:**
   - Triển khai thuật toán GNNExplainer (*instance-level explanation*) cho từng pha bóng cụ thể, xác định tập hợp đường chuyền và cầu thủ đóng góp lớn nhất vào xác suất tạo ra cú sút.
   - Triển khai thuật toán PGExplainer (*parameterized explainer*) học mẫu hình giải thích chung trên toàn bộ tập dữ liệu, cho phép suy luận giải thích nhanh và khái quát hóa cao.
   - Đối sánh nội bộ hai phương pháp XAI (cam kết Tầng 3(b)) trên bộ dữ liệu kiểm tra.
2. **Bước [8] — Đánh giá chất lượng XAI:**
   - Đánh giá định lượng chất lượng giải thích bằng bộ độ đo chuẩn hóa: Fidelity+, Fidelity-, Sparsity và Stability.
   - Vẽ đường cong suy giảm hiệu năng (ROAR / Insertion-Deletion curve) để kiểm chứng tính xác thực của các cạnh quan trọng.
3. **Bước tiền đề cho Bước [9] (Hybrid Centrality):**
   - Xây dựng đồ thị mạng chuyền bóng tổng hợp (*aggregate passing network*) của từng đội bóng trong cả mùa giải.
   - Tính toán 4 chỉ số centrality truyền thống (Degree, Betweenness, Closeness, Eigenvector Centrality).
   - Thiết kế cơ chế tổng hợp điểm XAI-importance của từng cầu thủ từ các pha bóng và thử nghiệm công thức kết hợp Hybrid Centrality có trọng số.

---

## 6. Tổng kết tiến độ tuần 11/10/2026

*Bảng 4: Bảng tổng kết các hạng mục công việc đã hoàn thành*

| Hạng mục công việc | Kế hoạch đề ra | Kết quả thực tế | Trạng thái |
| :--- | :--- | :--- | :---: |
| **Bước [4] — Chia tập dữ liệu** | Chia 70/15/15 theo trận, tránh leakage | 380 trận (266 train / 57 val / 57 test), 5/5 checks PASS | **Hoàn thành (100%)** |
| **Bước [5] — Huấn luyện GNN** | Cài đặt MPNN, GCN, GATv2, GraphSAGE | Đã huấn luyện 4 mô hình, xử lý mất cân bằng pos_weight=6,31 | **Hoàn thành (100%)** |
| **Bước [6] — Đánh giá mô hình** | Tính bộ 7 độ đo trên tập Test độc lập | Đã đánh giá 6.646 đồ thị test, MPNN đạt AUC-PR 0,5747 | **Hoàn thành (100%)** |
| **Đối sánh Tầng 2 (GNN)** | So sánh 4 kiến trúc GNN cùng pipeline | MPNN vượt trội GCN (+76%), GATv2 (+92%), GraphSAGE (+52%) | **Hoàn thành (100%)** |
| **Đối sánh Tầng 3(a) (Baseline)** | So sánh với Logistic, Random Forest, ANN | Đã hoàn thành bảng đối sánh 7 mô hình, khẳng định vai trò đồ thị | **Hoàn thành (100%)** |
| **Kiểm thử hệ thống** | Đảm bảo mã nguồn đạt chuẩn kiểm thử | 52/52 bài kiểm thử pytest PASS, linter ruff và black sạch sẽ | **Hoàn thành (100%)** |
