# 0001 — Quy tắc gán nhãn possession

- **Trạng thái:** Đã chốt (nhóm, 2026-11-02; giảng viên có thể xem lại)
- **Ngày tạo:** 2026-09-04 (Cập nhật số liệu mùa giải chính thức: 2026-10-16; Chốt quyết định: 2026-11-02)
- **Chặn:** Bước 3 (xây dựng đồ thị) và toàn bộ pipeline phía sau
- **Cấu hình liên quan:** `configs/data.yaml` → `labeling`

## Bối cảnh

`docs/Quy_trinh_XAI_Football.md` mục 2.3 nêu rõ đây là quyết định **ảnh hưởng dây chuyền đến toàn bộ pipeline**, phải chốt sớm và nhất quán, không được thay đổi giữa chừng sau khi đã huấn luyện mô hình (mục 12 — Rủi ro).

### Làm rõ khái niệm ngữ nghĩa (Wording Precision)
Cần phân biệt rõ hai khái niệm liên quan đến nhãn sút ở cấp độ possession, tuyệt đối không dùng thay thế cho nhau:
- `has_shot`: Pha kiểm soát bóng (possession) có chứa ít nhất một sự kiện dứt điểm (`Shot`) của đội kiểm soát bóng ở bất kỳ thời điểm nào trong chuỗi.
- `ends_with_shot`: Pha kiểm soát bóng kết thúc bằng một sự kiện `Shot` theo đúng định nghĩa sự kiện hợp lệ cuối cùng của pha bóng. Khi áp dụng cơ chế cắt chuỗi (`truncate_at_first_shot: true`), các đường chuyền sau cú sút đầu tiên bị loại bỏ để tránh rò rỉ nhãn, biến chuỗi đường chuyền thành chuỗi kết thúc ngay trước/tại cú sút đầu tiên.

---

## Quyết định chính thức (D1)

- **Lựa chọn:** Phương án A — Nhãn ở mức possession (`possession_level`), mục tiêu là cú sút (`shot`), kèm theo cơ chế cắt chuỗi `truncate_at_first_shot: true`.
- **Lý do:** Khớp chặt chẽ với đề cương nghiên cứu ban đầu; đồ thị đại diện cho chuỗi chuẩn bị tấn công; tỉ lệ lớp dương 14,26% bảo đảm tính phân tách và đủ mẫu để huấn luyện GNN và đánh giá mô hình.

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

---

## Xác nhận kiểm toán
- [x] Chọn Phương án A (possession-level) làm đơn vị phân loại chuẩn cho Bài toán 1.
- [x] Chọn mục tiêu `shot` thay vì `goal`.
- [x] Giữ nguyên cơ chế cắt chuỗi `truncate_at_first_shot: true` ở Bước 3.
- [x] Kiểm kê tỉ lệ lớp dương: 14,26% (possession shot).
