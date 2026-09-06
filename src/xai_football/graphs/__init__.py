"""Bước 3 — Xây dựng đồ thị chuyền bóng (possession graph).

Module dự kiến:
    features.py   Đặc trưng đỉnh (vị trí thi đấu, tọa độ trung bình, số chạm
                  bóng) và đặc trưng cạnh (x1,y1,x2,y2, khoảng cách, góc,
                  loại chuyền, thời điểm) — Bước 3.1-3.2
    build.py      Chuyển một possession -> torch_geometric.data.Data
                  (x, edge_index, edge_attr, y) — Bước 3.3-3.4
                  Tùy chọn: thêm node ảo đại diện khung thành đối phương
    dataset.py    Lớp Dataset/InMemoryDataset gom toàn bộ possession graph,
                  cache ra data/processed/*.pt để không phải dựng lại
    aggregate.py  Đồ thị chuyền bóng TỔNG HỢP theo trận/mùa (networkx),
                  dùng cho centrality truyền thống ở Bước 9.1
"""
