# data/ — Dữ liệu

**Toàn bộ nội dung dữ liệu KHÔNG được commit lên git** (xem `.gitignore`).
Chỉ cấu trúc thư mục, file README và `data/splits/` được commit.

Dữ liệu chảy một chiều: `raw → interim → processed`. Không bao giờ sửa trực
tiếp `raw/` — mọi biến đổi phải đi qua code trong `src/xai_football/data/`
để đảm bảo tái lập được (reproducibility).

## Các thư mục

### `raw/`
Dữ liệu thô tải từ StatsBomb Open Data, **giữ nguyên không chỉnh sửa**.

```
raw/{competition_id}/{season_id}/{match_id}.json
```

Lưu nguyên JSON gốc để truy vết lại và không phải gọi lại API nhiều lần
(Bước 1.4). Kèm theo log số trận, số event mỗi trận, tỉ lệ trận bị loại —
phục vụ phần mô tả dataset trong khóa luận.

### `interim/`
Dữ liệu trung gian đã qua tiền xử lý: possession sequence đã làm sạch, đã
chuẩn hóa tọa độ, đã gán nhãn (đầu ra Bước 2).

### `processed/`
Đồ thị possession đã đóng gói sẵn cho mô hình — danh sách đối tượng
`torch_geometric.data.Data` lưu dạng `.pt`, tách theo giải/mùa (Bước 3.4).
Có file này thì không phải dựng lại đồ thị mỗi lần huấn luyện.

### `splits/`  ← **CÓ commit lên git**
Danh sách `match_id` thuộc từng tập train/val/test, kèm seed.

Đây là **điều kiện bắt buộc để tái lập kết quả** và để đối sánh công bằng
giữa các backbone ở Tầng 2 (Bước 4). File nhỏ, dạng JSON, nên được commit.

### `external/`
Dữ liệu tham chiếu từ nguồn ngoài dùng để kiểm chứng Hybrid Centrality ở
Bước 9.4: key passes, assists, xA, xT...

## Lưu ý về nguồn dữ liệu

StatsBomb Open Data có điều khoản sử dụng riêng — cần trích dẫn đúng
StatsBomb trong khóa luận và bài báo.
