# 0005 — Lựa chọn cơ chế Readout cho mô hình MPNN

- **Trạng thái:** Đã chốt (nhóm, 2026-11-02; giảng viên có thể xem lại)
- **Ngày tạo:** 2026-11-02
- **Chặn:** Bước 5 và Bước 7
- **Cấu hình liên quan:** `configs/model/mpnn.yaml`, `configs/model/mpnn_attn.yaml`, `configs/model/mpnn_max.yaml`, `configs/model/mpnn_lastedge.yaml`

---

## Bối cảnh

Mô hình MPNN (Message Passing Neural Network) là backbone đồ thị có hiệu năng cao nhất tại Bước 5. Để khảo sát xem cơ chế tổng hợp biểu diễn đồ thị (*graph readout / pooling*) có thể cải thiện khả năng nắm bắt pha bóng dẫn tới cú sút hay không, nhóm thiết kế 3 biến thể readout đối sánh với cơ chế `mean` mặc định:
1. `attention`: Global Attention pooling với mạng gating MLP học trọng số đóng góp của từng đỉnh.
2. `max`: Global Max pooling lấy giá trị kích hoạt lớn nhất theo từng chiều đặc trưng đỉnh.
3. `mean_lastedge`: Ghép nối vector trung bình (mean pooling) với embedding của đỉnh nhận bóng ở đường chuyền cuối cùng của đồ thị (chiều đầu vào của classification head mở rộng lên $2 \times \text{hidden\_dim}$).

---

## Tiêu chí lựa chọn backbone chính thức (D4)

- **Quy tắc tiền nghiệm (Pre-registered rule):**
  - Huấn luyện cả 4 biến thể (`mean`, `attention`, `max`, `mean_lastedge`) trên 5 seeds ngẫu nhiên cố định (42, 123, 456, 789, 1024).
  - So sánh giá trị trung bình AUC-PR trên tập **Validation**.
  - Nếu biến thể tốt nhất vượt trội hơn `mean` ít nhất **0,01 AUC-PR** trên tập Validation thì chọn biến thể đó làm backbone chính thức cho phần XAI và Hybrid Centrality.
  - Ngược lại (chênh lệch $< 0,01$), tiếp tục giữ nguyên cơ chế `mean` pooling để ưu tiên tính đơn giản và tính khái quát hóa của mô hình.
