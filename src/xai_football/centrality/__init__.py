"""Bước 9-10 — Hybrid Centrality: đóng góp nghiên cứu chính.

Module dự kiến:
    traditional.py    Degree (in/out), betweenness, closeness, eigenvector
                      trên đồ thị chuyền bóng tổng hợp (networkx) — Bước 9.1
    xai_importance.py Tổng hợp điểm XAI của một cầu thủ qua TẤT CẢ possession
                      cầu thủ đó tham gia — Bước 9.2
    hybrid.py         Chuẩn hóa + kết hợp hai thành phần theo hệ số alpha
                      XEM docs/decisions/0002-cong-thuc-hybrid-centrality.md
                      — CÔNG THỨC VÀ ALPHA CHƯA CHỐT
    validation.py     Kiểm chứng: so với centrality thuần túy (baseline Tầng 3b)
                      và đối chiếu key passes / assists / xA / xT — Bước 9.4

CẢNH BÁO: alpha không được chọn tùy ý. Phải trình bày rõ cách xác định
(dò trên tập validation + tiêu chí lựa chọn) — mục 12 trong Quy trình.
"""
