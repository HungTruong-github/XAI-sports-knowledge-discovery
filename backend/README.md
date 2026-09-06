# backend/ — API cho ứng dụng demo (Giai đoạn 2)

> **Trạng thái:** chưa triển khai. Đây là phần **demo hệ thống**, không phải
> phần nghiên cứu cốt lõi.

## Vai trò

Trọng tâm của đồ án là nghiên cứu (`src/xai_football/` + `experiments/` +
`results/`). Thư mục này chỉ phục vụ phần "hệ thống" trong tên đề tài — một
API mỏng gọi lại pipeline đã có để trình diễn trước hội đồng.

**Không đưa logic nghiên cứu vào đây.** Backend chỉ gọi `xai_football`.

## Cấu trúc đề xuất khi triển khai

```
backend/
├── app/
│   ├── main.py          # khởi tạo ứng dụng
│   ├── api/             # các endpoint
│   ├── schemas/         # định nghĩa request/response
│   └── services/        # lớp mỏng gọi sang xai_football
└── requirements.txt     # phụ thuộc riêng của backend
```

## Endpoint dự kiến

| Endpoint | Trả về |
|---|---|
| `POST /predict` | Xác suất possession dẫn đến sút |
| `POST /explain` | Subgraph quan trọng do XAI xác định |
| `GET /centrality/{match_id}` | Bảng xếp hạng Hybrid Centrality |

## Lưu ý

Nếu khóa luận không yêu cầu demo chạy được, có thể **bỏ hẳn** `backend/` và
`frontend/`, chỉ trình bày kết quả bằng notebook và hình trong `results/`.
Nên hỏi ý kiến giảng viên hướng dẫn về việc này trước khi đầu tư thời gian.
