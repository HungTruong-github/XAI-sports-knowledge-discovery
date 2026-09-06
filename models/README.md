# models/ — Model đã huấn luyện

Nơi lưu **trọng số model đã huấn luyện** (file `.pt` / `.pkl`).
Nội dung không commit lên git vì kích thước lớn (xem `.gitignore`).

> **Phân biệt với `src/xai_football/modeling/`:**
> - `src/xai_football/modeling/` = **mã nguồn định nghĩa kiến trúc** (có commit)
> - `models/` (thư mục này) = **kết quả huấn luyện dạng nhị phân** (không commit)

## Quy ước đặt tên

```
models/{backbone}/{run_id}/best.pt
models/{backbone}/{run_id}/config_snapshot.yaml
```

Mỗi checkpoint phải đi kèm bản chụp cấu hình đã dùng để huấn luyện, nếu không
sẽ không biết model được tạo ra từ siêu tham số nào.

## Model chính thức

Model được chọn để đưa sang bước XAI phải là model tốt nhất trên **tập
validation**, không phải tập test (Bước 6). Ghi rõ model nào được chọn và
vì sao tại đây khi đã chốt.
