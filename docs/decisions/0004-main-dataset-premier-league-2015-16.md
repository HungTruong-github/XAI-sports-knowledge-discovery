# Decision 0004 — Chốt Dataset chính: Premier League 2015/2016 (StatsBomb Open Data)

- **Trạng thái**: Accepted
- **Ngày chốt**: 2026-09-26
- **Người đề xuất**: Nhóm nghiên cứu XAI-Football
- **Phạm vi tác động**: Toàn bộ Data Pipeline, Graph Construction, Model Training, và XAI Evaluation

---

## 1. Bối cảnh & Thay đổi so với kế hoạch ban đầu

Ban đầu, đề tài dự kiến sử dụng **Bundesliga 2023/24** kết hợp với **UEFA Champions League** (xem [Decision 0002](0002-chuyen-huong-sang-statsbomb-open-data.md) và `docs/Nguon_du_lieu_XAI_Football.md`).

Tuy nhiên, qua kiểm tra kho StatsBomb Open Data thực tế:
- StatsBomb Open Data **không cung cấp trọn vẹn mùa giải Bundesliga** (chỉ có các trận đấu đơn lẻ / sự kiện đặc biệt).
- Việc nghiên cứu mạng lưới chuyền bóng và đánh giá độ quan trọng của cầu thủ (Centrality / XAI) đòi hỏi **tính đồng bộ toàn giải đấu** (full league season), theo dõi xuyên suốt qua 380 trận đấu của 20 đội bóng.

---

## 2. Quyết định chính thức

Chuyển dataset chính thức duy nhất của đồ án sang:

- **Nguồn cung cấp**: StatsBomb Open Data
- **Giải đấu**: Premier League
- **Mùa giải**: 2015/2016
- **competition_id**: `2`
- **season_id**: `27`
- **Số trận đấu kỳ vọng**: `380` trận (20 đội, vòng tròn 2 lượt)

---

## 3. Lý do lựa chọn Premier League 2015/16

1. **Tính đầy đủ (Full Season Coverage)**: Đúng 380/380 trận đấu có sẵn dữ liệu event thô và đội hình (lineups).
2. **Đầy đủ trường thuộc tính cho Graph & XAI**:
   - Tọa độ chi tiết (x, y, pass end_x, pass end_y).
   - Định danh cầu thủ (`player_id`) và người nhận bóng (`recipient_id`).
   - Kết quả đường chuyền (`outcome`: Complete / Incomplete / Out / ...).
   - Kết quả cú sút (`shot.outcome`, `statsbomb_xg`).
3. **Phù hợp trực tiếp với bài toán**: Đủ quy mô để huấn luyện và đánh giá Graph Neural Networks, đồng thời tính toán các chỉ số centrality truyền thống và giải thích mô hình (GNNExplainer / PGExplainer / SHAP).

---

## 4. Lưu ý về các nguồn dữ liệu khác

- **Bundesliga & UCL**: Là kế hoạch ban đầu, được giữ lại trong tài liệu lịch sử, không xóa vết.
- **Wyscout Data**: Là nguồn kiểm chứng ngoài (optional external validation) nếu còn thời gian ở các giai đoạn sau; KHÔNG thuộc phạm vi bắt buộc của Data Stage hiện tại.
