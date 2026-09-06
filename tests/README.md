# tests/ — Kiểm thử

Chạy:

```bash
pytest
```

## Vì sao cần test trong một dự án nghiên cứu

Lỗi thầm lặng trong pipeline dữ liệu nguy hiểm hơn lỗi làm chương trình dừng:
nó vẫn cho ra số liệu, chỉ là số liệu **sai**, và bạn sẽ đưa số liệu đó vào
khóa luận mà không biết.

## Những thứ nên test trước tiên

| Ưu tiên | Test | Vì sao |
|---|---|---|
| Cao | Không rò rỉ dữ liệu giữa train/val/test | Kiểm tra tập `match_id` của 3 tập rời nhau hoàn toàn — đây là rủi ro số 1 được nêu ở mục 12 Quy trình |
| Cao | Quy tắc gán nhãn | Nhãn sai làm hỏng toàn bộ kết quả phía sau |
| Cao | Dựng đồ thị đúng | `edge_index` khớp với chuỗi chuyền, số node bằng số cầu thủ tham gia |
| Vừa | Chuẩn hóa tọa độ | Mọi trận tấn công cùng một hướng, tọa độ nằm trong 120x80 |
| Vừa | Công thức độ đo | Đối chiếu với `sklearn` trên ví dụ nhỏ tự tính tay được |
| Vừa | Chuẩn hóa trong Hybrid Centrality | Hai thành phần thực sự cùng thang đo trước khi kết hợp |

## Cấu trúc

Đặt file test theo đúng cấu trúc của `src/xai_football/`:

```
tests/
├── data/test_splits.py
├── graphs/test_build.py
├── evaluation/test_metrics.py
└── centrality/test_hybrid.py
```
