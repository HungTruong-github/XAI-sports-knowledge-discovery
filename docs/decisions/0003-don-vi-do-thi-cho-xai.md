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

---

## Quy trình chọn mẫu đánh giá XAI (Sample Selection Flow)

Để minh bạch hóa số lượng mẫu thực tế $N=1.222$ đồ thị đối sánh giữa các thuật toán giải thích (thay vì con số trần cấu hình tối đa 1.500 đồ thị):

| Giai đoạn chọn mẫu | Số lượng đồ thị | Tỉ lệ / Tiêu chí loại trừ |
| :--- | :---: | :--- |
| **Tổng số possession sạch toàn mùa (380 trận)** | 45.365 | Dữ liệu sau làm sạch sự kiện và lọc hiệp chính |
| **Đồ thị hợp lệ đưa vào huấn luyện GNN** | 45.047 | Lọc possession có $\ge 2$ đường chuyền |
| **Đồ thị thuộc tập Test độc lập (57 trận)** | 6.646 | Chiếm 14,75% tổng số đồ thị, chia theo trận đấu |
| **Đồ thị tập Test có $\ge 4$ cạnh ($E \ge 4$)** | 4.432 | Điều kiện ngưỡng để việc giải thích XAI có ý nghĩa |
| **— Trong đó: Lớp Positive ($y=1$, có sút)** | **611** | Toàn bộ các pha bóng có sút thỏa $E \ge 4$ trong Test set |
| **— Trong đó: Lớp Negative ($y=0$, không sút)** | 3.821 | Các pha bóng không có sút thỏa $E \ge 4$ trong Test set |
| **Mẫu XAI chính thức (Lấy mẫu cân bằng 1:1)** | **1.222** | **611 Positive + 611 Negative** lấy mẫu tất định (seed 42) |

> [!NOTE]
> Mặc dù cấu hình `configs/xai.yaml` đặt ngưỡng trần tối đa là 1.500 mẫu (`n_samples: 1500`, kỳ vọng $750$ dương + $750$ âm), toàn bộ tập Test (57 trận) chỉ có tối đa 611 đồ thị lớp dương thỏa điều kiện $E \ge 4$. Để duy trì tính cân bằng nghiêm ngặt $1:1$ (tránh việc lệch lớp làm sai lệch đánh giá Fidelity và đường cong Insertion/Deletion), quy trình đã chọn toàn bộ 611 đồ thị dương và ghép đôi với 611 đồ thị âm được lấy mẫu ngẫu nhiên tất định, tạo thành tập mẫu chuẩn xác gồm **1.222 đồ thị**.
