# docs/ — Tài liệu nghiên cứu

## Tài liệu nền

| File | Nội dung |
|---|---|
| [`XAI-Football_Project_Content.md`](XAI-Football_Project_Content.md) | Bối cảnh đề tài: hướng nghiên cứu, khoảng trống, kiến trúc hệ thống, thuật ngữ, và **những điều KHÔNG được giả định** (mục 18) |
| [`Quy_trinh_XAI_Football.md`](Quy_trinh_XAI_Football.md) | Quy trình triển khai chi tiết 12 bước, từ thu thập dữ liệu đến viết báo cáo |

Hai file này là **baseline hiện tại của dự án**. Chỉ thay thế khi có báo cáo
tiến độ mới hơn ghi rõ thay đổi hướng nghiên cứu.

## Khảo sát chuyên đề

| File | Nội dung | Bản .docx |
|---|---|---|
| [`Nguon_du_lieu_XAI_Football.md`](Nguon_du_lieu_XAI_Football.md) | Khảo sát và đề xuất nguồn dữ liệu (04/09/2026). **Phát hiện quan trọng:** kế hoạch dữ liệu trong báo cáo tiến độ chỉ cho ~52 trận. Đề xuất thay bằng StatsBomb Big-5 2015/16 + Wyscout/Pappalardo để kiểm chứng chéo. | [2026-09-04](meetings/2026-09-04_Khao_sat_nguon_du_lieu.docx) |
| [`Rui_ro_va_phuong_an_XAI_Football.md`](Rui_ro_va_phuong_an_XAI_Football.md) | Phân tích **38 rủi ro** kỹ thuật và dữ liệu (05/09/2026), kèm phương án xử lý, lộ trình 5 giai đoạn và bảng kiểm trước khi code. Có mục riêng cho **rủi ro âm thầm** — loại vẫn cho ra số liệu nhưng số liệu sai. | [2026-09-05](meetings/2026-09-05_Phan_tich_rui_ro.docx) |

**Khi sửa nội dung, sửa ở bản `.md` trước** rồi mới dựng lại bản `.docx`, để hai
bản không lệch nhau.

## Các thư mục con

### `decisions/`
Nhật ký các quyết định nghiên cứu quan trọng và lý do lựa chọn.
Hiện có **2 quyết định chưa chốt** đang chặn việc triển khai — xem
[`decisions/README.md`](decisions/README.md).

### `meetings/`
Báo cáo tiến độ theo tuần và nhận xét của giảng viên.
Đặt tên theo định dạng `YYYY-MM-DD_ten-file` để sắp xếp theo thời gian.

| Ngày | File | Nội dung chính |
|---|---|---|
| 2026-09-01 | `2026-09-01_Phan_hoi_gop_y_GV.docx` | Phản hồi 3 góp ý của GV buổi 25/08: xác nhận định nghĩa bài toán, chốt bộ độ đo, chốt chiến lược baseline 3 tầng |
| 2026-09-04 | `2026-09-04_Khao_sat_nguon_du_lieu.docx` | Báo cáo khảo sát nguồn dữ liệu: kế hoạch cũ chỉ có ~52 trận, đề xuất 2 bộ dữ liệu mới và thiết kế kiểm chứng chéo |
| 2026-09-05 | `2026-09-05_Phan_tich_rui_ro.docx` | Phân tích 38 rủi ro kỹ thuật và dữ liệu, phương án xử lý từng rủi ro, lộ trình 5 giai đoạn |

### `figures/`
Hình vẽ dùng trong tài liệu (sơ đồ pipeline, kiến trúc hệ thống).
Khác với `results/figures/` là hình sinh ra từ kết quả thực nghiệm.

### `thesis/`
Các chương khóa luận khi bắt đầu viết.
