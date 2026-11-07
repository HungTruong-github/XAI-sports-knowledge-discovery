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
Hiện toàn bộ **6/6 quyết định nghiên cứu then chốt** (Decision 0001 đến 0006) đã được chốt chính thức — xem
[`decisions/README.md`](decisions/README.md).

### `meetings/`
Báo cáo tiến độ theo tuần và nhận xét của giảng viên.
Đặt tên theo định dạng `YYYY-MM-DD_ten-file` để sắp xếp theo thời gian.

| Ngày | File | Nội dung chính |
|---|---|---|
| 2026-09-01 | `2026-09-01_Phan_hoi_gop_y_GV.docx` | Phản hồi 3 góp ý của GV buổi 25/08: xác nhận định nghĩa bài toán, chốt bộ độ đo, chốt chiến lược baseline 3 tầng |
| 2026-09-04 | `2026-09-04_Khao_sat_nguon_du_lieu.docx` | Báo cáo khảo sát nguồn dữ liệu: kế hoạch cũ chỉ có ~52 trận, đề xuất dataset Premier League 2015/16 |
| 2026-09-05 | `2026-09-05_Phan_tich_rui_ro.docx` | Phân tích 38 rủi ro kỹ thuật và dữ liệu, phương án xử lý từng rủi ro, lộ trình 5 giai đoạn |
| 2026-09-19 | `2026-09-19_Bao_cao_tuan.md` | Báo cáo tiến độ tuần: hoàn thiện pipeline dữ liệu và dựng đồ thị possession |
| 2026-10-11 | `2026-10-11_Bao_cao_tuan.md` | Báo cáo tiến độ tuần: huấn luyện GNN multi-seed và baselines phi đồ thị |
| 2026-10-22 | `2026-10-22_Phan_hoi_nhan_xet_GV.md` | Phản hồi nhận xét GV: phân tích feature ablation và khảo sát readout pooling |
| 2026-10-23 | `report_23_10.docx` | Báo cáo tiến độ đối sánh XAI và kiểm định Cổng Gate D17 |

### `reproducibility/`
Báo cáo tái lập độc lập 5 giai đoạn (`PHASE_1_REPORT.md` đến `PHASE_5_REPORT.md`) cùng báo cáo kiểm toán toàn diện (`COMPLETION_AUDIT.md`).

### `figures/`
Hình vẽ dùng trong tài liệu (sơ đồ pipeline, kiến trúc hệ thống).
Khác với `results/figures/` là hình sinh ra từ kết quả thực nghiệm.

### `thesis/`
Các chương khóa luận hoàn chỉnh (`docs/thesis/ket_qua_va_thao_luan.md`).
