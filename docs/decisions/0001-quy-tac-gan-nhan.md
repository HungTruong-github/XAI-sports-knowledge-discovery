# 0001 — Quy tắc gán nhãn possession

- **Trạng thái:** Chưa quyết định
- **Ngày tạo:** 2026-09-04 (Cập nhật số liệu mùa giải chính thức: 2026-10-16)
- **Chặn:** Bước 3 (xây dựng đồ thị) và toàn bộ pipeline phía sau
- **Cấu hình liên quan:** `configs/data.yaml` → `labeling`

## Bối cảnh

`docs/Quy_trinh_XAI_Football.md` mục 2.3 nêu rõ đây là quyết định **ảnh hưởng dây chuyền đến toàn bộ pipeline**, phải chốt sớm và nhất quán, không được thay đổi giữa chừng sau khi đã huấn luyện mô hình (mục 12 — Rủi ro).

### Làm rõ khái niệm ngữ nghĩa (Wording Precision)
Cần phân biệt rõ hai khái niệm liên quan đến nhãn sút ở cấp độ possession, tuyệt đối không dùng thay thế cho nhau:
- `has_shot`: Pha kiểm soát bóng (possession) có chứa ít nhất một sự kiện dứt điểm (`Shot`) của đội kiểm soát bóng ở bất kỳ thời điểm nào trong chuỗi.
- `ends_with_shot`: Pha kiểm soát bóng kết thúc bằng một sự kiện `Shot` theo đúng định nghĩa sự kiện hợp lệ cuối cùng của pha bóng. Khi áp dụng cơ chế cắt chuỗi (`truncate_at_first_shot: true`), các đường chuyền sau cú sút đầu tiên bị loại bỏ để tránh rò rỉ nhãn, biến chuỗi đường chuyền thành chuỗi kết thúc ngay trước/tại cú sút đầu tiên.

---

## Các phương án

### Phương án A — Nhãn ở mức possession (`possession_level`)

Gán nhãn nhị phân cho toàn bộ pha bóng: `1` nếu possession dẫn đến một cú sút (`has_shot` / `ends_with_shot`), `0` nếu ngược lại.

- **Ưu điểm:** Khớp trực tiếp với bài toán phân loại ở cấp độ đồ thị (*graph-level classification*); một possession sạch tương ứng với đúng một đồ thị và một nhãn dự đoán.
- **Nhược điểm:** Với các possession kéo dài, một số đường chuyền ban đầu có thể cách xa thời điểm dứt điểm, tạo ra độ thưa tín hiệu khi XAI học trọng số cạnh.

### Phương án B — Nhãn ở mức đường chuyền (`pass_level`)

Gán nhãn độc lập cho từng đường chuyền: `1` nếu đường chuyền dẫn đến cú sút trong cửa sổ $N$ giây tiếp theo (tương tự nguyên lý của các mô hình *Expected Threat - xT* hoặc *VAEP*).

- **Ưu điểm:** Tín hiệu dự đoán cục bộ sắc nét, xác định trực tiếp giá trị gia tăng của từng đường chuyền.
- **Nhược điểm:** Phải chuyển đổi đơn vị đồ thị từ possession hoàn chỉnh sang cửa sổ trượt (*sliding window*), phá vỡ tính liên tục của pha bóng, làm phức tạp hóa cấu trúc đồ thị ở Bước 3 và gây khó khăn lớn cho việc tổng hợp điểm XAI của cầu thủ ở Bước 9.

---

## Câu hỏi cần xác nhận trước khi chốt chính thức

- [ ] Giảng viên hướng dẫn xác nhận chọn Phương án A (possession-level) làm đơn vị phân loại chuẩn cho Bài toán 1.
- [ ] Xác nhận lựa chọn mục tiêu là `shot` thay vì `goal` (do tỉ lệ bàn thắng quá hiếm, gây mất cân bằng cực độ).
- [ ] Giữ nguyên cơ chế cắt chuỗi `truncate_at_first_shot: true` ở Bước 3 để bảo đảm không rò rỉ diễn biến sau cú sút.
- [x] **Tỉ lệ lớp dương thực tế của mỗi phương án trên toàn mùa giải là bao nhiêu?** — Đã đo lường đầy đủ trên toàn bộ 380 trận, xem bảng dưới.

---

## Bằng chứng định lượng toàn mùa giải (Premier League 2015/16)

Nguồn dữ liệu: 380 trận đấu chính thức Premier League 2015/16, trích xuất 45.365 possession sạch và 261.205 đường chuyền thành công.  
Tệp artifact gốc: `results/tables/official/label_comparison.csv`.

| Phương án | Mục tiêu (`target`) | Cửa sổ thời gian | Tổng số đơn vị | Số nhãn dương | Tỉ lệ nhãn dương |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A — possession** | **`shot`** | — | **45.365** | **6.471** | **14,26%** |
| **A — possession** | `goal` | — | 45.365 | 662 | **1,46%** |
| **B — đường chuyền** | `shot` | 5 giây | 261.205 | 8.627 | **3,30%** |
| **B — đường chuyền** | `shot` | 10 giây | 261.205 | 16.321 | **6,25%** |
| **B — đường chuyền** | `shot` | 15 giây | 261.205 | 22.594 | **8,65%** |
| **B — đường chuyền** | `goal` | 5 giây | 261.205 | 960 | **0,37%** |
| **B — đường chuyền** | `goal` | 10 giây | 261.205 | 1.775 | **0,68%** |
| **B — đường chuyền** | `goal` | 15 giây | 261.205 | 2.432 | **0,93%** |

### Nhận định chuyên môn từ bằng chứng thực tế
1. **Target `shot` vs `goal`:** Tỉ lệ có bàn thắng ở mức possession chỉ đạt 1,46% (662 bàn trên 45.365 pha bóng, tức tỉ lệ mất cân bằng xấp xỉ 1 : 68). Mức mất cân bằng này quá khắc nghiệt đối với việc huấn luyện GNN ổn định. Trong khi đó, target `shot` đạt 14,26% (tỉ lệ 1 : 6,01), là mức mất cân bằng tiêu chuẩn hoàn toàn có thể kiểm soát và tối ưu hóa hiệu quả thông qua kỹ thuật cân bằng trọng số hàm mất mát (`WeightedBCEWithLogitsLoss` với `pos_weight ≈ 6.31`).
2. **Phương án A vs Phương án B:** Phương án B ở mọi cửa sổ thời gian (5s, 10s, 15s) đều cho tỉ lệ nhãn dương rất thấp (3,30% – 8,65%), đồng thời đòi hỏi phải định nghĩa lại cấu trúc đồ thị trượt rất phức tạp.

---

## Đề xuất kỹ thuật & Trạng thái

- **Đề xuất kỹ thuật của nhóm:** Chọn **Phương án A với `target = shot`** và giữ cơ chế cắt chuỗi tại cú sút đầu tiên.
- **Trạng thái hiện tại:** `Chưa quyết định` (Giữ nguyên trạng thái này cho đến khi nhận được văn bản xác nhận chính thức từ Giảng viên hướng dẫn).
