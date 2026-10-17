# 0003 — Đơn vị đồ thị và ngưỡng lọc độ dài possession cho phần XAI

- **Trạng thái:** Chưa quyết định
- **Ngày tạo:** 2026-10-16
- **Chặn:** Bước 7 (áp dụng XAI) và Bước 8 (đánh giá XAI)
- **Cấu hình liên quan:** `configs/graph.yaml`, `configs/xai.yaml` (dự kiến)

---

## Bối cảnh

Trong đề cương nghiên cứu và tài liệu rủi ro (`docs/Rui_ro_va_phuong_an_XAI_Football.md` mục 2.2), nhóm đã chỉ ra một thách thức cố hữu khi áp dụng giải thuật XAI dạng đồ thị (GNNExplainer, PGExplainer) lên mạng chuyền bóng bóng đá:

1. **Vấn đề đồ thị thưa / quá ngắn:** Nhiều pha kiểm soát bóng chỉ gồm 2 hoặc 3 đường chuyền trước khi dứt điểm hoặc mất bóng. Với đồ thị chỉ có 2–3 cạnh, việc thuật toán XAI gán mặt nạ trọng số cạnh (edge importance mask) trở nên tầm thường (*trivial*). Hơn nữa, khi đánh giá chất lượng XAI bằng độ đo Fidelity (xóa cạnh quan trọng nhất để đo mức suy giảm dự đoán), việc xóa 1 cạnh trong đồ thị 2 cạnh thường làm đứt gãy hoàn toàn cấu trúc liên thông, khiến kết quả đánh giá XAI bị nhiễu do biến dạng topo thay vì đo lường đúng tầm quan trọng chiến thuật.
2. **Yêu cầu đối sánh chất lượng giải thích (Bước 8):** Để các chỉ số như Fidelity+, Fidelity-, Sparsity, Stability và đường cong ROAR phản ánh đúng năng lực giải thích của mô hình, đồ thị cần có đủ độ dày liên kết tối thiểu.

---

## Các phương án đề xuất

### Phương án A — Đánh giá XAI trên toàn bộ tập đồ thị hợp lệ ($\ge 2$ đường chuyền)
- Áp dụng GNNExplainer và PGExplainer lên toàn bộ các đồ thị được sinh ra từ Bước 3 (không lọc thêm).
- **Ưu điểm:** Tính đồng nhất tuyệt đối giữa tập dự đoán (Bước 5–6) và tập giải thích (Bước 7–8).
- **Nhược điểm:** Phân phối điểm Fidelity bị chi phối bởi các đồ thị siêu ngắn; khó quan sát các cấu trúc chuỗi phối hợp phức tạp (tam giác chuyền bóng, phối hợp đập-nhả trung lộ).

### Phương án B — Huấn luyện trên toàn bộ ($\ge 2$), nhưng lọc tập đánh giá XAI chuyên sâu ($\ge 4$ đường chuyền)
- Mô hình GNN dự đoán vẫn được huấn luyện trên toàn bộ 45.047 đồ thị ($\ge 2$ đường chuyền) để tối đa hóa kích thước mẫu học.
- Đối với phân tích XAI (Bước 7) và đánh giá độ đo giải thích (Bước 8), chỉ chọn lọc các possession có tối thiểu 4 (hoặc 5) đường chuyền thành công.
- **Ưu điểm:** Đồ thị đủ phong phú về mặt cấu trúc để GNNExplainer học được các subgraph chiến thuật có ý nghĩa; đánh giá Fidelity/Sparsity có tính phân hóa cao.
- **Bằng chứng định lượng:** Theo phân bố kích thước đồ thị mùa giải 2015/16, số pha bóng có $\ge 4$ đường chuyền vẫn chiếm hơn 66,7% tổng số pha bóng sạch, hoàn toàn bảo đảm độ lớn mẫu cho phân tích thống kê.

### Phương án C — Làm giàu đồ thị bằng các hành động bổ sung (Carry, Dribble, Virtual Goal Node)
- Bổ sung hành động dắt bóng (`Carry`), lừa bóng (`Dribble`) thành các cạnh có hướng, hoặc thêm một đỉnh ảo đại diện khung thành đối phương (tương tự kiến trúc TacticAI).
- **Ưu điểm:** Đồ thị dày dặn hơn, tăng số lượng liên kết.
- **Nhược điểm:** Làm thay đổi định nghĩa toán học cốt lõi của "Mạng chuyền bóng" (*passing network*) thành mạng đa hành động (*action network*); làm phức tạp hóa đáng kể cấu trúc đầu vào và việc giải thích ý nghĩa của các cạnh không phải đường chuyền.

---

## Câu hỏi cần xác nhận với Giảng viên hướng dẫn

- [ ] Giảng viên duyệt lựa chọn Phương án B (áp dụng ngưỡng $\ge 4$ đường chuyền cho phần thực nghiệm XAI chuyên sâu) hay muốn giữ nguyên Phương án A?
- [ ] Xác định danh sách các case study trực quan hóa: chọn các pha bóng tấn công tiêu biểu (possession dài dẫn đến bàn thắng) hay lấy mẫu ngẫu nhiên có phân tầng?
- [ ] Có cần bổ sung node ảo khung thành đối phương cho bài toán open-play hay giữ nguyên đồ thị thuần túy giữa các cầu thủ?

---

## Trạng thái

- **Trạng thái:** `Chưa quyết định` (Dự kiến báo cáo và xin ý kiến giảng viên tại buổi họp tiếp theo trước khi triển khai Bước 7).
