# 0001 — Quy tắc gán nhãn possession

- **Trạng thái:** Chưa quyết định
- **Ngày tạo:** 2026-09-04
- **Chặn:** Bước 3 (xây dựng đồ thị) và toàn bộ pipeline phía sau
- **Cấu hình liên quan:** `configs/data.yaml` → `labeling`

## Bối cảnh

`docs/Quy_trinh_XAI_Football.md` mục 2.3 nêu rõ đây là quyết định **ảnh hưởng
dây chuyền đến toàn bộ pipeline**, phải chốt sớm và nhất quán, không được thay
đổi giữa chừng sau khi đã huấn luyện mô hình (mục 12 — Rủi ro).

## Các phương án

### Phương án A — Nhãn ở mức possession (`possession_level`)

Gán `1` nếu possession kết thúc bằng một cú `Shot`, `0` nếu ngược lại.

- Ưu: đơn giản, khớp trực tiếp với bài toán graph-level classification;
  một possession = một đồ thị = một nhãn.
- Nhược: possession dài có nhiều đường chuyền không liên quan tới cú sút,
  có thể làm nhiễu tín hiệu mà XAI học được.

### Phương án B — Nhãn ở mức đường chuyền (`pass_level`)

Gán nhãn theo từng đường chuyền: `1` nếu dẫn đến sút trong `N` giây tiếp theo
(tương tự cách tiếp cận của các mô hình xThreat).

- Ưu: tín hiệu sắc nét hơn, gần với cách làm phổ biến trong sports analytics.
- Nhược: phải định nghĩa lại đơn vị đồ thị (cửa sổ trượt thay vì possession
  trọn vẹn), làm phức tạp Bước 3 và cách tổng hợp XAI ở Bước 9.2.

## Câu hỏi cần trả lời trước khi chốt

- [ ] Đơn vị đồ thị là possession trọn vẹn hay cửa sổ trượt?
- [ ] Target là `shot` hay `goal`? (Nếu `goal`, lớp dương sẽ hiếm hơn nhiều,
      ảnh hưởng trực tiếp tới chiến lược xử lý mất cân bằng lớp ở `configs/train.yaml`)
- [ ] Nếu chọn B: `N` giây bằng bao nhiêu, căn cứ từ tài liệu nào?
- [ ] Tỉ lệ lớp dương thực tế của mỗi phương án là bao nhiêu?
      (Cần chạy thống kê mô tả trên dữ liệu trước khi chốt)

## Quyết định

_Chưa chốt. Cập nhật mục này kèm ngày và lý do khi đã quyết định._

## Hệ quả

_Điền sau khi chốt: những bước nào phải điều chỉnh theo._
