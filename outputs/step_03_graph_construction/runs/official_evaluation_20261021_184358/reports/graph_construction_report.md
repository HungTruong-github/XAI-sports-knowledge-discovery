# Báo cáo Xây dựng Đồ thị Possession (Bước 3)

- **Tổng số đồ thị PyG đã đóng gói**: 45.047
- **Kết quả Kiểm tra Cấu trúc Đồ thị**: Graph checks verified
- **Kích thước đặc trưng**:
  - Đỉnh (Node features): 18 chiều (vị trí trung bình, vai trò chiến thuật one-hot).
  - Cạnh (Edge features): 6 chiều (chiều dài, góc, thời lượng, v.v.).
- **Xử lý biên tọa độ**: 1 điểm phạt góc ngoài sân ($x=120.9$) được kẹp về biên sân an toàn (`coords_clipped`).
- **Trạng thái Decision 0003**: `Draft/Pending` (loại possession $< 2$ đường chuyền; cắt trước cú sút đầu).
