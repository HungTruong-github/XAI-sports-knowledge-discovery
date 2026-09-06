# Nhật ký quyết định nghiên cứu (Decision Records)

Thư mục này ghi lại **các quyết định nghiên cứu quan trọng** cùng lý do lựa chọn.

## Vì sao cần thư mục này

Cả `docs/XAI-Football_Project_Content.md` (mục 10, 18, 21) lẫn
`docs/Quy_trinh_XAI_Football.md` (mục 12) đều nhấn mạnh: có những quyết định
**phải chốt trước khi code** và **phải trình bày rõ căn cứ** trong khóa luận,
nếu không hội đồng sẽ đặt câu hỏi về tính "thủ công, thiếu căn cứ".

Ghi lại tại đây để:

1. Không thay đổi giữa chừng sau khi đã huấn luyện mô hình.
2. Có sẵn nội dung viết vào chương Phương pháp của khóa luận.
3. Trả lời được câu hỏi "vì sao chọn cách này" khi bảo vệ.

## Quy ước

- Mỗi quyết định = một file `NNNN-ten-quyet-dinh.md`, đánh số tăng dần.
- Trạng thái: `Chưa quyết định` → `Đã chốt` → (nếu cần) `Đã thay thế bởi NNNN`.
- Khi đã chốt: cập nhật giá trị tương ứng trong `configs/` và ghi ngày chốt.

## Danh sách

| # | Quyết định | Trạng thái | Chặn bước nào |
|---|---|---|---|
| [0001](0001-quy-tac-gan-nhan.md) | Quy tắc gán nhãn possession | Chưa quyết định | Bước 3 trở đi |
| [0002](0002-cong-thuc-hybrid-centrality.md) | Công thức Hybrid Centrality | Chưa quyết định | Bước 9 |
