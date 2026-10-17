# 0002 — Công thức Hybrid Centrality

- **Trạng thái:** Chưa quyết định (Pending)
- **Ngày tạo:** 2026-09-04 (Cập nhật tiến độ Bước 7–8: 2026-10-17)
- **Chặn:** Bước 9 (đóng góp nghiên cứu chính)
- **Cấu hình liên quan:** `configs/hybrid_centrality.yaml`

---

## Bối cảnh

Hybrid Centrality là **đóng góp chính** của đề tài, nhưng
`docs/XAI-Football_Project_Content.md` mục 10 ghi rõ: công thức toán học
chưa được đặc tả đầy đủ trong báo cáo tiến độ và **phải được định nghĩa
chính thức** trước khi triển khai.

`docs/Quy_trinh_XAI_Football.md` mục 12 cảnh báo thêm: hệ số `alpha` không
được chọn tùy ý, phải trình bày rõ cách xác định để hội đồng không đặt câu
hỏi về tính "thủ công, thiếu căn cứ".

---

## Dạng công thức đang đề xuất

```text
HybridCentrality = alpha * Centrality_norm + (1 - alpha) * XAIImportance_norm
```

---

## Dữ liệu và Hạ tầng đã sẵn sàng từ Bước 7–8

1. **Điểm quan trọng XAI cấp cầu thủ (Player-level XAI importance):**
   - Đã triển khai module `src/xai_football/xai/subgraph.py:aggregate_player_importance()`.
   - Sinh ra artifact `player_importance_by_possession.csv` với các trường:
     `player_id`, `outgoing_importance`, `incoming_importance`, `total_importance`, `pass_sent_count`, `pass_received_count`.
2. **Kiểm định chất lượng giải thích (Bước 8):**
   - Đã triển khai bộ độ đo `Fidelity+`, `Fidelity-`, `Sparsity`, `Stability`, `Insertion/Deletion curves` tại `src/xai_football/xai/metrics.py`.
   - Xác nhận MPNN và GNNExplainer/PGExplainer cung cấp tín hiệu giải thích có kiểm chứng toán học.
3. **Cơ chế phòng hộ cấu hình (Safeguards):**
   - Đã cài đặt `validate_hybrid_centrality_config()` trong `src/xai_football/centrality/interface.py` nhằm từ chối mọi lệnh chạy chính thức khi các tham số cốt lõi trong `configs/hybrid_centrality.yaml` còn null.
   - Các thử nghiệm kỹ thuật buộc phải truyền cờ `allow_provisional=True` và gắn nhãn cột `provisional_hybrid_centrality`.

---

## Những điểm bắt buộc phải xác nhận với Giảng viên hướng dẫn

- [ ] **Lựa chọn độ đo Centrality truyền thống:**
  - Degree (in/out), Betweenness, Closeness, Eigenvector — dùng độ đo nào làm đại diện chính?
  - Nếu gộp nhiều độ đo: Dùng trung bình hình học, PCA hay tổng chuẩn hóa?
- [ ] **Định nghĩa tổng hợp điểm XAI của cầu thủ qua mùa giải:**
  - Tổng điểm qua mọi possession hay trung bình mỗi 90 phút?
  - Trọng số theo số đường chuyền hay theo xác suất thành công của pha bóng?
- [ ] **Phương pháp chuẩn hóa:**
  - Min-Max normalization $[0, 1]$ hay Z-score Standardization? (Min-max trực quan hơn cho biểu thức lồi $\alpha C + (1-\alpha) X$).
- [ ] **Ngưỡng lọc cỡ mẫu tối thiểu:**
  - Thiết lập `min_possessions` (ví dụ $\ge 50$ pha bóng hoặc $\ge 270$ phút thi đấu) để loại bỏ các cầu thủ thi đấu quá ít gây nhiễu thống kê.
- [ ] **Tiêu chí tối ưu hóa $\alpha$ trên tập Validation:**
  - Tìm $\alpha \in [0, 1]$ tối đa hóa tương quan hạng Spearman với các chỉ số kiến tạo kỳ vọng ($xT$, $xA$, $Key Passes$) trên tập Validation.
- [ ] **Chọn Explainer nguồn:**
  - Chốt giữa GNNExplainer và PGExplainer dựa trên kết quả đối sánh tại Bước 8.

---

## Cách kiểm chứng đã dự kiến (mục 9.4)

1. So với centrality truyền thống thuần túy — baseline Tầng 3(b).
2. Đối chiếu định lượng với key passes, assists, xA, xT (face validity).
3. Đối chiếu định tính với vai trò thực tế của một số cầu thủ nổi bật (tiền vệ kiến thiết vs tiền đạo mục tiêu).

---

## Quyết định hiện tại

**Trạng thái: Chưa quyết định (Pending).**  
*Không được giả định công thức này đã hợp lệ trước khi có buổi làm việc và phê duyệt chính thức từ Giảng viên hướng dẫn.*
