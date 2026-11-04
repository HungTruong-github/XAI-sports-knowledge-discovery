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

---

## Kết quả thực nghiệm đối sánh (5 seeds)

Bảng tổng hợp từ `results/tables/official/readout_comparison.csv`:

| Mô hình | Cơ chế Readout | Val AUC-PR | Test AUC-PR | Test AUC-ROC | Test Brier Score | Best Seed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`mpnn`** (baseline) | `mean` | 0.5847 ± 0.0015 | **0.5996 ± 0.0042** | 0.8757 ± 0.0034 | 0.1302 ± 0.0114 | 789 |
| `mpnn_max` | `max` | 0.5624 ± 0.0031 | 0.5853 ± 0.0048 | 0.8811 ± 0.0010 | 0.1424 ± 0.0158 | 42 |
| `mpnn_attn` | `attention` | 0.5620 ± 0.0061 | 0.5820 ± 0.0074 | 0.8786 ± 0.0032 | 0.1336 ± 0.0114 | 1024 |
| `mpnn_lastedge` | `mean_lastedge` | **0.5921 ± 0.0074** | 0.5973 ± 0.0082 | **0.8852 ± 0.0018** | **0.1256 ± 0.0138** | 1024 |

### Đánh giá theo quy tắc D4:
- Biến thể có Val AUC-PR cao nhất là `mpnn_lastedge`: $0.5921$.
- Baseline `mpnn` (mean pooling) có Val AUC-PR: $0.5847$.
- Chênh lệch (margin): $\Delta = 0.5921 - 0.5847 = +0.0074$.
- Do $+0.0074 < +0.0100$ (ngưỡng tiền nghiệm), điều kiện chuyển đổi backbone không được kích hoạt.

---

## Quyết định cuối cùng

1. **Backbone chính thức:** Tiếp tục chọn **`mpnn` (mean pooling)** làm mô hình backbone chính thức cho toàn bộ Pipeline XAI (Bước 8-9) và Hybrid Centrality (Bước 10).
2. **Vai trò của `mpnn_lastedge`:** Được lưu trữ và báo cáo trong tài liệu nghiên cứu / khóa luận như một biến thể kiến trúc nâng cao:
   - Đạt Val AUC-PR cao nhất qua 5 seeds ($0.5921$).
   - Đạt điểm cao nhất ở seed 1024: Val AUC-PR = $0.5997$, Test AUC-PR = $0.6107$ (tiếp cận mức của tabular ANN $0.6117$ và Random Forest $0.6198$).
   - Đạt độ bền vững cao hơn trên các đồ thị có nhiều đường chuyền ($\ge 6$ edges: AUC-PR 0.6054 so với 0.5693 của mean pooling).
3. **Artifacts đã lưu:**
   - Bảng đối sánh: `results/tables/official/readout_comparison.csv`
   - Manifest kiểm toán: `results/tables/official/readout_comparison_manifest.json`
   - Checkpoints 5 seeds: `models/official/mpnn_lastedge_best.pt`, `mpnn_max_best.pt`, `mpnn_attn_best.pt`.
