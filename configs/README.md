# configs/ — Cấu hình thực nghiệm

Mọi tham số nằm ở đây, **không hard-code trong mã nguồn**. Nhờ vậy có thể đổi
cấu hình mà không sửa code, và mỗi lần chạy đều lưu lại được bản chụp cấu hình
để tái lập kết quả.

## Các file

| File | Bước | Nội dung |
|---|---|---|
| `data.yaml` | 1-2 | Nguồn dữ liệu, lọc event, làm sạch, **quy tắc gán nhãn** |
| `graph.yaml` | 3 | Định nghĩa node/edge và đặc trưng của đồ thị possession |
| `split.yaml` | 4 | Chia train/val/test theo trận, tỉ lệ, seed |
| `train.yaml` | 5 | Optimizer, loss, early stopping — **dùng chung cho mọi backbone** |
| `model/*.yaml` | 5 | Kiến trúc từng backbone và baseline |
| `xai.yaml` | 7-8 | Cấu hình explainer và bộ độ đo XAI |
| `hybrid_centrality.yaml` | 9 | Thành phần, chuẩn hóa, hệ số alpha, cách kiểm chứng |

## Nguyên tắc quan trọng

**Mọi backbone ở Tầng 2 phải dùng chung `train.yaml` và `split.yaml`.**
Đây là điều kiện để việc đối sánh cô lập đúng biến số kiến trúc — đúng như
nhóm đã cam kết với giảng viên ở mục 1.3 Tầng 2. Nếu mỗi backbone dùng một
cấu hình huấn luyện khác nhau thì không thể kết luận chênh lệch hiệu năng
đến từ kiến trúc.

## Các giá trị `null` với ghi chú `TODO`

Những chỗ này là **quyết định nghiên cứu chưa chốt**, cố ý để trống thay vì
điền bừa. Xem `docs/decisions/` để biết cần trả lời câu hỏi gì trước khi điền:

- `data.yaml → labeling` → [quyết định 0001](../docs/decisions/0001-quy-tac-gan-nhan.md)
- `hybrid_centrality.yaml → combination` → [quyết định 0002](../docs/decisions/0002-cong-thuc-hybrid-centrality.md)

## Cấu hình cục bộ

Nếu cần ghi đè tham số trên máy riêng (đường dẫn dữ liệu, số worker...),
tạo `configs/local.yaml` — file này đã được `.gitignore` bỏ qua.
