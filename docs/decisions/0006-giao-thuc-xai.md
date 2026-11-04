# 0006 — Giao thức thực nghiệm và đánh giá chất lượng XAI

- **Trạng thái:** Đã chốt (nhóm, 2026-11-02; giảng viên có thể xem lại)
- **Ngày tạo:** 2026-11-02
- **Chặn:** Bước 7 và Bước 8
- **Cấu hình liên quan:** `configs/xai.yaml`, `configs/hybrid_centrality.yaml`

---

## Chi tiết các quyết định thực nghiệm XAI

### 1. Phạm vi pha bóng và phân tích độ nhạy (D3)
- Phân tích chính thức sử dụng toàn bộ các play pattern trong bóng đá (bao gồm open-play, tình huống cố định, phản công...).
- Khảo sát open-play thuần túy được thực hiện dưới dạng phân tích độ nhạy (sensitivity analysis) thông qua lọc hậu kiểm trên trường `graph.play_pattern` (loại bỏ From Corner, From Free Kick, From Throw In, From Kick Off, Penalty).
- *Lý do:* Tránh việc phải dựng lại dữ liệu và huấn luyện lại mô hình, duy trì tính nhất quán xuyên suốt pipeline.

### 2. Mục tiêu giải thích (D5)
- Mục tiêu giải thích luôn được cố định là **lớp 1 (có cú sút)** cho mọi đồ thị trên cả GNNExplainer và PGExplainer (`explanation_type="phenomenon"`, `target_class=1`).
- Mọi độ đo Fidelity (Fidelity+, Fidelity-) đều được tính trên xác suất dự đoán của lớp 1.
- *Lý do:* Câu hỏi nghiên cứu của Hybrid Centrality là "cầu thủ đóng góp như thế nào cho khả năng tạo ra cú sút của pha bóng", không phải giải thích tại sao pha bóng thất bại.

### 3. Ngưỡng cắt Subgraph quan trọng (D6)
- **Chính:** Quy tắc tỉ lệ $k = \max(1, \text{round}(0,3 \times E))$.
- **Phụ:** Quét toàn bộ $k$ từ $1$ đến $E - 1$ để vẽ đường cong phân bố Sparsity–Fidelity toàn diện.

### 4. Quy mô và phương pháp lấy mẫu (D7)
- **Pilot test:** 150 đồ thị (75 dương, 75 âm) phục vụ kiểm tra đường truyền và thời gian thực thi.
- **Thực nghiệm chính thức:** 1.500 đồ thị từ tập Test (750 dương, 750 âm), áp dụng phân tầng theo 57 trận đấu và 3 nhóm kích thước đồ thị (4 cạnh, 5–6 cạnh, $\ge 7$ cạnh).
- *Lý do:* Đảm bảo tính đại diện cho tập kiểm thử, đủ năng lực thống kê cho bootstrap, khả thi khi thực thi trên CPU.

### 5. Huấn luyện PGExplainer (D8)
- Huấn luyện trên 500 đồ thị tập Train (có từ 4 cạnh trở lên), lấy mẫu phân tầng theo nhãn và kích thước, huấn luyện 30 epoch với mục tiêu cố định `target=1`.

### 6. Đánh giá tính ổn định Stability (D9)
- Lấy mẫu 100 đồ thị, thực hiện 5 lần lặp với nhiễu Gaussian ($\sigma = 0,01$) trên ma trận đặc trưng đỉnh, đo lường độ tương đồng Jaccard của tập cạnh top-k.

### 7. Chính sách đối với ROAR (D10)
- Thiết lập `roar: false` trong cấu hình và tài liệu hóa dưới dạng hạn chế nghiên cứu do yêu cầu huấn luyện lại hàng trăm lần vượt quá giới hạn tài nguyên tính toán.

### 8. Cross-fitting cấp cầu thủ cho Bước 9 (D11)
- Áp dụng kiểm thử chéo 5-fold theo trận trên toàn bộ 380 trận với seed 42.
- Mỗi trận lấy ngẫu nhiên tối đa 40 đồ thị (có từ 4 cạnh trở lên) để giải thích out-of-fold nhằm thu thập đủ điểm đóng góp cho cầu thủ thi đấu ở các trận ngoài tập test.

### 9. Cổng chất lượng XAI (D17)
- Một explainer được công nhận có giá trị thông tin nếu chênh lệch $(\text{Fidelity}^+ - \text{Fidelity}^-)$ so với RandomExplainer có khoảng tin cậy 95% bootstrap theo trận nằm hoàn toàn trên 0.
- Nếu không có mô hình nào vượt qua cổng, Hybrid Centrality vẫn được tính toán và công bố kèm nhãn "kết quả thăm dò".
