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

## Trạng thái các Quyết định Nghiên cứu (Decision Records)

Toàn bộ các quyết định nghiên cứu then chốt đã được chốt và phản ánh đầy đủ trong cấu hình (xem `docs/decisions/`):

- `data.yaml → labeling`: Đã chốt theo [Quyết định 0001](../docs/decisions/0001-quy-tac-gan-nhan.md) (bài toán phân loại nhị phân cấp possession, nhãn cú sút `shot`, tỷ lệ dương 14,26%).
- `hybrid_centrality.yaml → combination`: Đã chốt theo [Quyết định 0002](../docs/decisions/0002-cong-thuc-hybrid-centrality.md) (chuẩn hóa Percentile Rank, tham số tối ưu $\alpha^* = 0.65$ từ Set A có ràng buộc độ tin cậy chia đôi).
- `configs/model/mpnn.yaml`: Duy trì MPNN Mean Pooling nguyên bản làm backbone chính thức theo [Quyết định 0005](../docs/decisions/0005-readout-mpnn.md) (quy tắc D4).
- `configs/xai.yaml`: Chốt cấu hình XAI, Gate D17 và Cross-fitting theo [Quyết định 0006](../docs/decisions/0006-giao-thuc-xai.md).

## Cấu hình cục bộ

Nếu cần ghi đè tham số trên máy riêng (đường dẫn dữ liệu, số worker...),
tạo `configs/local.yaml` — file này đã được `.gitignore` bỏ qua.
