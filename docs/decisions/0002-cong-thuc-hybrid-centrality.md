# 0002 — Công thức Hybrid Centrality

- **Trạng thái:** Đã chốt (nhóm, 2026-11-02; giảng viên có thể xem lại)
- **Ngày tạo:** 2026-09-04 (Cập nhật tiến độ Bước 7–8: 2026-10-17; Chốt quyết định: 2026-11-02)
- **Chặn:** Bước 9 (đóng góp nghiên cứu chính)
- **Cấu hình liên quan:** `configs/hybrid_centrality.yaml`

---

## Bối cảnh

Hybrid Centrality là **đóng góp chính** của đề tài. Mục tiêu là kết hợp cấu trúc liên kết mạng truyền thống với trọng số đóng góp chiến thuật do mô hình XAI trích xuất, hỗ trợ đánh giá vai trò cầu thủ trong quá trình phối hợp.

---

## Công thức và Thiết kế Phương pháp luận (D12 đến D17)

### 1. Dạng công thức kết hợp (D14)
$$\text{HybridCentrality}(v) = \alpha \cdot \text{Centrality}_{\text{norm}}(v) + (1 - \alpha) \cdot \text{XAIImportance}_{\text{norm}}(v)$$

- **Chuẩn hóa (D14):** Sử dụng Percentile Rank trên tập cầu thủ đủ điều kiện thi đấu (loại thủ môn do đặc thù vị trí không tham gia chuỗi chuyền bóng thông thường). Thử nghiệm z-score làm phân tích độ nhạy.
- **Hệ số $\alpha$ (D16):** Dò lưới $\alpha \in [0.0, 1.0]$ với bước nhảy 0.05 trên tập A. Tiêu chí chọn $\alpha$: tối đa hóa hệ số tương quan hạng Spearman giữa Hybrid Centrality và chỉ số đe dọa kỳ vọng ($xT$) trên 90 phút, đi kèm ràng buộc độ tin cậy chia đôi (split-half reliability giữa các trận chẵn/lẻ theo ngày thi đấu) tối thiểu bằng $0.7 \times \text{max\_reliability}$ trên lưới.

### 2. Thành phần 1: Centrality truyền thống (D13)
- Tính toán trên mạng chuyền bóng của toàn đội theo từng trận đấu (bao gồm mọi đường chuyền thành công, trọng số cạnh bằng số lần chuyền bóng, khoảng cách = $1 / \text{trọng số}$).
- Chỉ số chính: **Betweenness Centrality**.
- Chỉ số phân tích độ nhạy: In-Degree, Out-Degree, Closeness, Eigenvector.
- Tổng hợp theo mùa giải bằng trung bình có trọng số theo số phút thi đấu, chỉ tính các trận cầu thủ thi đấu tối thiểu 30 phút.

### 3. Thành phần 2: Điểm quan trọng từ XAI (D12)
- Điểm XAI của mỗi cầu thủ được tính bằng trung bình importance của các cạnh liên thuộc (cả chuyền đi và nhận bóng) trong mỗi lần tham gia pha bóng được giải thích.
- Áp dụng co về trung bình toàn cục (empirical Bayes shrinkage) với tham số $m = 20$ lần tham gia:
  $$\text{XAIImportance}_{\text{shrunk}}(v) = \frac{n_v \cdot \bar{I}_v + m \cdot \bar{I}_{\text{global}}}{n_v + m}$$
- Ngưỡng lọc cỡ mẫu tối thiểu: $n_v \ge 30$ pha bóng được giải thích.
- Bổ sung biến thể triệt tiêu loại bỏ cạnh cuối cùng (last-edge ablation) để kiểm tra tính độc lập với đường chuyền dọn cỗ.

### 4. Thiết kế thực nghiệm và Chia tập A/B (D15)
- Chia 380 trận thành 2 nửa rời nhau: Tập A (190 trận) và Tập B (190 trận) bằng seed ngẫu nhiên 42, bảo đảm mỗi câu lạc bộ đều xuất hiện cân bằng ở cả hai nửa.
- Tập A: chỉ dùng để huấn luyện mô hình $xT$ và tìm kiếm siêu tham số $\alpha$.
- Tập B: dùng độc lập hoàn toàn để kiểm chứng, tính $\Delta R^2$ và đánh giá đối sánh.

### 5. Cổng chất lượng giải thích (D17)
- Một thuật toán XAI chỉ được coi là đạt yêu cầu nguồn cho Hybrid Centrality nếu chênh lệch $(\text{Fidelity}^+ - \text{Fidelity}^-)$ so với explainer ngẫu nhiên (RandomExplainer) có khoảng tin cậy 95% bootstrap theo trận nằm hoàn toàn trên 0.
- Nếu không có explainer nào vượt qua cổng, Hybrid Centrality vẫn được tính toán nhưng gắn nhãn "kết quả thăm dò" (provisional/exploratory).
