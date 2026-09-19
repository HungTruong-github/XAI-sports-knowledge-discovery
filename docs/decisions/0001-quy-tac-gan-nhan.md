# 0001 — Quy tắc gán nhãn possession

- **Trạng thái:** Chưa quyết định
- **Ngày tạo:** 2026-09-04
- **Chặn:** Bước 3 (xây dựng đồ thị) và toàn bộ pipeline phía sau
- **Cấu hình liên quan:** `configs/data.yaml` → `labeling`

## Bối cảnh

`docs/Quy_trinh_XAI_Football.md` mục 2.3 nêu rõ đây là quyết định **ảnh hưởng
dây chuyền đến toàn bộ pipeline**, phải chốt sớm và nhất quán, không được thay
đổi giữa chừng sau khi đã huấn luyện mô hình (mục 12 — Rủi ro).

## Các phương án

### Phương án A — Nhãn ở mức possession (`possession_level`)

Gán `1` nếu possession kết thúc bằng một cú `Shot`, `0` nếu ngược lại.

- Ưu: đơn giản, khớp trực tiếp với bài toán graph-level classification;
  một possession = một đồ thị = một nhãn.
- Nhược: possession dài có nhiều đường chuyền không liên quan tới cú sút,
  có thể làm nhiễu tín hiệu mà XAI học được.

### Phương án B — Nhãn ở mức đường chuyền (`pass_level`)

Gán nhãn theo từng đường chuyền: `1` nếu dẫn đến sút trong `N` giây tiếp theo
(tương tự cách tiếp cận của các mô hình xThreat).

- Ưu: tín hiệu sắc nét hơn, gần với cách làm phổ biến trong sports analytics.
- Nhược: phải định nghĩa lại đơn vị đồ thị (cửa sổ trượt thay vì possession
  trọn vẹn), làm phức tạp Bước 3 và cách tổng hợp XAI ở Bước 9.2.

## Câu hỏi cần trả lời trước khi chốt

- [ ] Đơn vị đồ thị là possession trọn vẹn hay cửa sổ trượt?
- [ ] Target là `shot` hay `goal`? (Nếu `goal`, lớp dương sẽ hiếm hơn nhiều,
      ảnh hưởng trực tiếp tới chiến lược xử lý mất cân bằng lớp ở `configs/train.yaml`)
- [ ] Nếu chọn B: `N` giây bằng bao nhiêu, căn cứ từ tài liệu nào?
- [x] **Tỉ lệ lớp dương thực tế của mỗi phương án là bao nhiêu?** — đã đo, xem dưới.

## Bằng chứng định lượng (đo ngày 19/09/2026)

Nguồn: Premier League 2015/16, 5 trận chạy thử, 836 possession thô → **585
possession** sau làm sạch, 4.234 đường chuyền.
Sinh lại bằng: `python scripts/02_descriptive_stats.py`
Bảng đầy đủ: `results/tables/label_comparison.csv`

| Phương án | Target | Cửa sổ | Số đơn vị | Số dương | Tỉ lệ dương |
|---|---|---|---|---|---|
| A — possession | `shot` | — | 585 | 106 | **18,12%** |
| A — possession | `goal` | — | 585 | 9 | **1,54%** |
| B — đường chuyền | `shot` | 5 giây | 4.234 | 138 | 3,26% |
| B — đường chuyền | `shot` | 10 giây | 4.234 | 254 | 6,00% |
| B — đường chuyền | `shot` | 15 giây | 4.234 | 382 | 9,02% |
| B — đường chuyền | `goal` | 10 giây | 4.234 | 33 | 0,78% |

**Nhận định:** Phương án A với `target = shot` cho mức mất cân bằng còn xử lý
được bằng weighted BCE. Chuyển sang `goal` làm lớp dương hiếm hơn gần **12 lần**
(18,12% → 1,54%), đúng như lo ngại ban đầu.

**Lưu ý về cỡ mẫu:** đây là số đo trên 5 trận, phải chạy lại trên toàn bộ phạm vi
dữ liệu sau khi chốt nguồn (xem báo cáo `docs/meetings/2026-09-19_Bao_cao_tuan.md`
mục 2) trước khi coi là kết luận cuối cùng.

## Quyết định

_Chưa chốt. Đề xuất mang ra buổi 19/09: **Phương án A, `target = shot`**.
Cập nhật mục này kèm ngày và ý kiến giảng viên khi đã quyết định._

## Hệ quả

_Điền sau khi chốt: những bước nào phải điều chỉnh theo._
