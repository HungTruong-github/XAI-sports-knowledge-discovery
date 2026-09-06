# experiments/ — Đầu ra thô của các lần chạy

Nơi chứa log, metric theo epoch, và sản phẩm tạm của **từng lần chạy thực
nghiệm**. Nội dung không commit lên git (xem `.gitignore`).

## Phân biệt ba thư mục dễ nhầm

| Thư mục | Chứa gì | Commit? |
|---|---|---|
| `experiments/` | Log thô, metric theo epoch, output từng lần chạy | Không |
| `models/` | Trọng số model đã huấn luyện | Không |
| `results/` | Bảng và hình **cuối cùng** đưa vào khóa luận | **Có** |

## Cấu trúc đề xuất cho mỗi lần chạy

```
experiments/{YYYYMMDD_HHMMSS}_{backbone}_{seed}/
├── config_snapshot.yaml    # bản chụp toàn bộ cấu hình đã dùng
├── train_log.csv           # loss/metric theo epoch
├── metrics.json            # kết quả cuối trên val và test
└── notes.md                # ghi chú tay nếu có
```

Nếu dùng `mlflow`, thư mục `mlruns/` cũng đã được `.gitignore` bỏ qua.

## Nguyên tắc

- Mỗi lần chạy phải ghi lại **seed** và **bản chụp cấu hình** — không có hai
  thứ này thì kết quả không tái lập được.
- Theo kế hoạch đã báo cáo với giảng viên, mỗi cấu hình chạy `n_runs = 5` lần
  để báo cáo trung bình và độ lệch chuẩn.
- Chỉ đánh giá trên **tập test một lần duy nhất**, sau khi đã chốt xong model
  và siêu tham số trên tập validation (Bước 6).
