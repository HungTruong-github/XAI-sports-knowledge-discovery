"""Bước 3 — Xây dựng đồ thị chuyền bóng (possession graph).

Module:
    features.py   Đặc trưng đỉnh (vai trò GK/DF/MF/FW, tọa độ trung bình, số
                  lần chuyền/nhận) và đặc trưng cạnh (x1,y1,x2,y2, độ dài, góc
                  sin/cos, độ cao đường chuyền, thứ tự/thời điểm) — Bước 3.1-3.2
    build.py      Chuyển một possession -> torch_geometric.data.Data
                  (x, edge_index, edge_attr, y + metadata truy ngược XAI);
                  cắt possession tại cú sút đầu tiên — Bước 3.3-3.4
    dataset.py    Dựng toàn bộ tập đồ thị, đếm mọi possession bị loại, cache
                  ra data/processed/<run_name>/graphs.pt + graph_index.parquet
    aggregate.py  [Planned / Not Implemented] Đồ thị chuyền bóng TỔNG HỢP theo
                  trận/mùa (networkx), dùng cho centrality truyền thống ở Bước 9.1
                  Tùy chọn chưa cài đặt: node ảo đại diện khung thành đối phương
"""
