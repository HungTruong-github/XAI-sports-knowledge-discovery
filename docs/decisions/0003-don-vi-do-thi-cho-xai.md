# 0003 — Đơn vị đồ thị và ngưỡng lọc độ dài possession cho phần XAI

- **Trạng thái:** Đã chốt (nhóm, 2026-11-02; giảng viên có thể xem lại)
- **Ngày tạo:** 2026-10-16 (Chốt quyết định: 2026-11-02)
- **Chặn:** Bước 7 (áp dụng XAI) và Bước 8 (đánh giá XAI)
- **Cấu hình liên quan:** `configs/graph.yaml`, `configs/xai.yaml`

---

## Bối cảnh

Trong mạng chuyền bóng bóng đá, nhiều pha kiểm soát bóng chỉ gồm 2 hoặc 3 đường chuyền trước khi dứt điểm hoặc mất bóng. Với đồ thị chỉ có 2–3 cạnh, việc thuật toán XAI gán mặt nạ trọng số cạnh trở nên tầm thường (*trivial*). Hơn nữa, khi xóa 1 cạnh trong đồ thị 2 cạnh để tính Fidelity, cấu trúc liên thông bị đứt gãy hoàn toàn khiến kết quả đánh giá XAI bị biến dạng topo thay vì đo lường đúng đóng góp chiến thuật.

---

## Quyết định chính thức (D2)

- **Lựa chọn:** Phương án B — Huấn luyện mô hình GNN trên toàn bộ tập đồ thị hợp lệ ($\ge 2$ đường chuyền), nhưng đánh giá XAI chuyên sâu chỉ thực hiện trên tập đồ thị có từ **4 cạnh trở lên** ($E \ge 4$).
- **Lý do:**
  1. Bảo toàn toàn bộ 45.047 đồ thị cho quá trình huấn luyện mô hình dự đoán mà không cần dựng lại dữ liệu hay làm giảm kích thước mẫu học của GNN.
  2. Tập đồ thị có $\ge 4$ cạnh chiếm hơn 66,7% tổng số pha bóng sạch mùa giải 2015/16, bảo đảm kích thước mẫu dồi dào và đa dạng về cấu trúc topo.
  3. Tránh việc Fidelity bị chi phối vô nghĩa bởi các chuỗi chuyền bóng quá ngắn.
