# Phản hồi nhận xét của Giảng viên — Báo cáo tuần 11/10/2026, mục 4.2

- **Ngày:** 22/10/2026
- **Nhận xét (19/10/2026):** *"Nhóm phi đồ thị lại cho kết quả tốt hơn?"* — tại Bảng 3, mục 4.2.
- **Phạm vi:** chỉ Bước 5–6 (huấn luyện & đánh giá). Không thay đổi Bước 7–10.

---

## 1. Trả lời ngắn

1. **Bảng 3 trong báo cáo 11/10 là số liệu cũ (1 seed).** Sau khi chạy lại 4 GNN × 5 seed và kiểm định thống kê (22/10), **MPNN không còn thua có ý nghĩa thống kê so với Random Forest và ANN** (AUC-PR: 0,600 so với 0,620 và 0,612; p sau hiệu chỉnh Holm = 0,27). Ba GNN còn lại (GCN, GATv2, GraphSAGE) vẫn thua.
2. **Lợi thế của nhóm phi đồ thị đến gần như hoàn toàn từ một đặc trưng: tọa độ điểm rơi của đường chuyền cuối cùng (`final_x`, `final_y`).**
   - Bỏ hai đặc trưng này, AUC-PR của RF giảm từ 0,620 xuống 0,268, thấp hơn **mọi** GNN.
   - Chỉ giữ hai đặc trưng này, ANN vẫn đạt AUC-PR 0,623, ngang bộ 10 đặc trưng.
3. **Nguyên nhân phía GNN:** đường chuyền cuối là thông tin rất mạnh, nhưng readout của GNN hiện tại là *mean pooling trên các cầu thủ*. Cách gộp này làm loãng thông tin "đường chuyền nào là cuối cùng". GCN, GATv2 và GraphSAGE còn không dùng đặc trưng cạnh, nên không thấy tọa độ đường chuyền.

Nói cách khác, nhóm phi đồ thị không thắng nhờ mô hình tốt hơn. Chúng thắng vì được đưa sẵn đúng một đặc trưng tổng hợp mạnh, đặc trưng mà kiến trúc GNN hiện tại không trích ra được.

---

## 2. Số liệu cập nhật (thay cho Bảng 3 cũ)

Tập test độc lập: 6.646 possession của 57 trận. Mỗi mô hình là trung bình ± độ lệch chuẩn của 5 seed (42, 123, 456, 789, 1024). Khoảng tin cậy 95% lấy từ bootstrap theo trận, 10.000 vòng. Nguồn: `results/tables/official/model_comparison.csv`, `model_metric_ci.csv`.

*Bảng 3 (cập nhật 22/10/2026): Hiệu năng trên tập Test, 5 seed*

| Mô hình | Tầng | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR (KTC 95%) | Brier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MPNN** | Tier 2 — GNN | 0,878 ± 0,004 | 0,557 ± 0,020 | 0,599 ± 0,025 | 0,576 ± 0,004 | 0,876 ± 0,003 | **0,600** (0,566–0,632) | 0,130 ± 0,011 |
| GraphSAGE | Tier 2 — GNN | 0,845 ± 0,012 | 0,455 ± 0,025 | 0,594 ± 0,038 | 0,514 ± 0,007 | 0,850 ± 0,001 | 0,516 (0,481–0,551) | 0,146 ± 0,005 |
| GATv2 | Tier 2 — GNN | 0,824 ± 0,012 | 0,411 ± 0,017 | 0,620 ± 0,040 | 0,493 ± 0,005 | 0,841 ± 0,001 | 0,462 (0,427–0,497) | 0,153 ± 0,007 |
| GCN | Tier 2 — GNN | 0,802 ± 0,014 | 0,375 ± 0,017 | 0,641 ± 0,033 | 0,472 ± 0,007 | 0,831 ± 0,001 | 0,453 (0,417–0,489) | 0,169 ± 0,006 |
| Logistic Regression | Tier 3(a) — Phi đồ thị | 0,789 | 0,353 | 0,632 | 0,453 | 0,822 | 0,338 (0,313–0,366) | 0,185 |
| Random Forest | Tier 3(a) — Phi đồ thị | 0,866 ± 0,006 | 0,513 ± 0,016 | 0,693 ± 0,016 | 0,589 ± 0,005 | 0,886 ± 0,001 | **0,620** (0,583–0,655) | 0,087 ± 0,000 |
| ANN (MLP) | Tier 3(a) — Phi đồ thị | 0,870 ± 0,010 | 0,527 ± 0,032 | 0,654 ± 0,030 | 0,582 ± 0,010 | 0,884 ± 0,002 | 0,612 (0,578–0,645) | 0,079 ± 0,001 |

Logistic Regression cho cùng một kết quả ở mọi seed (mô hình lồi, nghiệm duy nhất), nên độ lệch chuẩn bằng 0.

**Kiểm định cặp MPNN với nhóm phi đồ thị** (paired cluster bootstrap theo trận, hiệu chỉnh Holm; nguồn: `significance_tests.csv`):

| So sánh | Độ đo | Chênh lệch (MPNN − đối thủ) | KTC 95% | p (Holm) | Kết luận |
| :--- | :---: | :---: | :---: | :---: | :--- |
| MPNN vs Random Forest | AUC-PR | −0,020 | (−0,042; +0,002) | 0,27 | Không khác biệt có ý nghĩa |
| MPNN vs ANN | AUC-PR | −0,012 | (−0,026; +0,002) | 0,27 | Không khác biệt có ý nghĩa |
| MPNN vs Random Forest | AUC-ROC | −0,010 | (−0,018; −0,002) | 0,05 | Không khác biệt có ý nghĩa (sát ngưỡng) |
| MPNN vs ANN | AUC-ROC | −0,008 | (−0,015; −0,002) | 0,05 | Không khác biệt có ý nghĩa (sát ngưỡng) |
| MPNN vs Logistic Regression | AUC-PR | +0,262 | (+0,236; +0,285) | 0,004 | MPNN tốt hơn |
| MPNN vs RF / ANN | Brier | +0,043 / +0,052 | — | 0,004 | RF / ANN hiệu chuẩn tốt hơn |

**Sửa thông tin sai trong báo cáo cũ:** báo cáo 11/10 ghi baseline dùng "16 đặc trưng passing path". Thực tế chỉ có **10 đặc trưng** (`configs/model/baseline_tabular.yaml`): `n_passes, n_players_involved, sequence_length, mean_pass_distance, mean_pass_angle, duration, start_x, start_y, final_x, final_y`.

---

## 3. Thí nghiệm kiểm chứng: Ablation đặc trưng của baseline phi đồ thị

**Cách làm:** huấn luyện lại LR, RF, ANN với cùng split theo trận, cùng 5 seed, cùng siêu tham số và cùng quy tắc chọn ngưỡng trên validation. Chỉ thay đổi tập đặc trưng đầu vào.
Script: `scripts/06c_baseline_feature_ablation.py`. Kết quả: `results/tables/official/baseline_feature_ablation.csv`.
Cấu hình A cho ra **đúng** số của lần chạy chính thức, xác nhận thí nghiệm khớp pipeline gốc.

*Bảng 5: AUC-PR trên tập Test (trung bình 5 seed) theo tập đặc trưng*

| Tập đặc trưng | Số đặc trưng | LR | RF | ANN |
| :--- | :---: | :---: | :---: | :---: |
| A — Đầy đủ (như báo cáo) | 10 | 0,338 | 0,620 | 0,612 |
| B — Bỏ `final_x`, `final_y` | 8 | 0,190 | **0,268** | **0,271** |
| C — Bỏ cả tọa độ đầu và cuối | 6 | 0,166 | 0,168 | 0,186 |
| D — **Chỉ** `final_x`, `final_y` | 2 | 0,332 | 0,534 | **0,623** |
| *Tham chiếu: GNN thấp nhất (GCN) / MPNN* | — | | *0,453 / 0,600* | |

*Bảng 6: AUC-ROC trên tập Test tương ứng*

| Tập đặc trưng | LR | RF | ANN |
| :--- | :---: | :---: | :---: |
| A — Đầy đủ | 0,822 | 0,886 | 0,884 |
| B — Bỏ `final_x`, `final_y` | 0,640 | 0,702 | 0,709 |
| C — Bỏ tọa độ đầu và cuối | 0,573 | 0,581 | 0,605 |
| D — Chỉ `final_x`, `final_y` | 0,821 | 0,854 | 0,888 |

**Độ quan trọng hoán vị (permutation importance) của Random Forest** (đo trên validation, mức giảm AUC-PR khi xáo trộn từng đặc trưng, 10 lần lặp; nguồn: `rf_permutation_importance.csv`):

| Đặc trưng | Mức giảm AUC-PR |
| :--- | :---: |
| `final_x` | **0,380** |
| `final_y` | **0,319** |
| `duration` | 0,019 |
| `sequence_length` | 0,019 |
| `mean_pass_distance` | 0,014 |
| `start_x` | 0,011 |
| 4 đặc trưng còn lại | < 0,010 |

**Vì sao `final_x/final_y` mạnh như vậy** (thống kê trên tập train, nguồn: `final_x_by_label.csv`):

| Nhãn possession | Số mẫu | `final_x` trung vị (m) | Tỉ lệ đường chuyền cuối rơi vào vòng cấm |
| :--- | :---: | :---: | :---: |
| Không có cú sút (0) | 27.316 | 73,7 | 4,6% |
| Có cú sút (1) | 4.328 | 101,2 | **44,6%** |

Do đồ thị được cắt tại cú sút đầu tiên (`truncate_at_first_shot: true`), đường chuyền cuối của một possession có sút chính là đường chuyền ngay trước cú sút. Vị trí của nó gần như là vị trí dứt điểm. Đây **không phải rò rỉ nhãn** theo nghĩa kỹ thuật, vì cả GNN (qua `end_x`, `end_y`, `order_in_possession` trong đặc trưng cạnh) lẫn baseline đều có thông tin này. Tuy vậy, đây là tín hiệu rất sát nhãn, và baseline dạng bảng được đưa sẵn nó dưới dạng một cột riêng.

---

## 4. Luận giải: vì sao GNN không tận dụng được thông tin này

1. **Readout làm loãng tín hiệu.** GNN gộp embedding các *cầu thủ* bằng mean pooling (`configs/model/*.yaml → readout: mean`). Đường chuyền cuối chỉ là một trong nhiều cạnh, và nó chỉ ảnh hưởng tới đỉnh người nhận sau 2–3 lớp truyền tin. Khi lấy trung bình trên khoảng 5 cầu thủ (trung bình 4,9 đỉnh mỗi đồ thị), tín hiệu "pha bóng kết thúc ở đâu" bị pha loãng. Baseline dạng bảng thì nhận trực tiếp `final_x/final_y`.
2. **Ba GNN không dùng đặc trưng cạnh.** GCN, GATv2 và GraphSAGE chỉ dùng đặc trưng đỉnh (vai trò, tọa độ trung bình của cầu thủ, số lần chạm bóng). Chúng không thấy tọa độ từng đường chuyền. Đó là lý do ba mô hình này thua cả RF/ANN, còn MPNN, mô hình duy nhất dùng đặc trưng cạnh, thì ngang RF/ANN.
3. **Khi baseline mất `final_x/final_y`, mọi GNN đều thắng rõ.** RF/ANN chỉ còn AUC-PR khoảng 0,27, trong khi GNN đạt 0,45–0,60. Tức là phần thông tin GNN học được từ *cấu trúc pha bóng* có giá trị thật, vượt xa các thống kê tổng hợp khác của chuỗi chuyền.

---

## 5. Hướng xử lý đề xuất (chờ GV góp ý)

| # | Đề xuất | Phạm vi | Trạng thái |
| :--- | :--- | :--- | :--- |
| 1 | Thay Bảng 3 cũ bằng bảng 5 seed và kiểm định ở mục 2, kèm Bảng 5–6 (ablation) trong báo cáo/luận văn | Bước 6 | **Đã làm** (tài liệu này) |
| 2 | Báo cáo hai cấu hình baseline: (a) đầy đủ 10 đặc trưng, (b) bỏ `final_x/final_y`. Như vậy hội đồng thấy rõ phần đóng góp của đặc trưng điểm cuối | Bước 6 | **Đã có số liệu** (Bảng 5–6) |
| 3 | Bổ sung cho GNN một readout biết đường chuyền cuối: ghép embedding của đỉnh nhận đường chuyền cuối, hoặc attention pooling (`readout: attention` đã có sẵn trong code), hoặc thêm vector đặc trưng toàn cục vào head. Mục tiêu: so sánh công bằng với cùng lượng thông tin | Bước 5 — **thay đổi mô hình, cần huấn luyện lại 5 seed** | Chờ GV đồng ý |
| 4 | Nhấn mạnh trong luận văn rằng mục tiêu chính là **giải thích** (Bài toán 2). MPNN đạt hiệu năng tương đương baseline mạnh nhất (không khác biệt thống kê), đồng thời cho phép XAI ở mức đường chuyền/cầu thủ, điều mà mô hình dạng bảng không làm được | Văn bản | Đã nêu ở mục 1 |

**Câu hỏi xin ý kiến thầy:**
- [ ] Thầy có đồng ý để nhóm thử đề xuất #3 (readout biết đường chuyền cuối) và huấn luyện lại MPNN không? Nếu có, kết quả XAI ở Bước 7–9 sẽ phải chạy lại trên mô hình mới.
- [ ] Trong luận văn, thầy muốn nhóm lấy cấu hình baseline nào làm chính: đầy đủ 10 đặc trưng, hay bỏ `final_x/final_y`?

---

## 6. Tái lập

```bash
python scripts/06c_baseline_feature_ablation.py --config configs/data.yaml
```

Thời gian chạy khoảng vài phút trên CPU. Đầu ra nằm trong `results/tables/official/`: `baseline_feature_ablation.csv`, `baseline_feature_ablation_all_seeds.csv`, `rf_permutation_importance.csv`, `final_x_by_label.csv`.
