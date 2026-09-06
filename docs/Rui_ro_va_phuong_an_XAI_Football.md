# Phân tích rủi ro và phương án xử lý

**Đề tài:** Hệ thống XAI trong đánh giá và khám phá tri thức thể thao
**Hướng nghiên cứu:** Mạng chuyền bóng và phân tích chiến thuật bóng đá bằng GNN kết hợp XAI

| | |
|---|---|
| **Ngày lập** | 05/09/2026 |
| **Trạng thái** | Đề xuất — chờ trao đổi với giảng viên hướng dẫn |
| **Phạm vi** | Toàn bộ pipeline, từ thu thập dữ liệu đến chỉ số chiến thuật cấp đội |
| **Tài liệu liên quan** | `Quy_trinh_XAI_Football.md`, `Nguon_du_lieu_XAI_Football.md`, `decisions/` |

---

## 1. Tóm tắt

Tài liệu này liệt kê **38 rủi ro** đã xác định, chia thành 6 nhóm, kèm phương án xử lý
cụ thể cho từng cái.

Điểm quan trọng nhất cần nắm: **rủi ro nguy hiểm nhất của đề tài này không phải là
rủi ro làm chương trình chạy sai.** Chương trình chạy sai thì biết ngay mà sửa. Nguy
hiểm hơn nhiều là **rủi ro âm thầm** — những lỗi vẫn cho ra số liệu đẹp, vẫn vẽ được
biểu đồ, nhưng con số đó **sai**. Loại này sẽ đi thẳng vào khóa luận mà không ai phát
hiện, cho tới khi hội đồng hỏi đúng câu.

Tài liệu tách riêng nhóm rủi ro âm thầm ở mục 9.

### Phân bố rủi ro

| Nhóm | Số lượng | Rất cao | Cao |
|---|---|---|---|
| A — Dữ liệu | 11 | 0 | 1 |
| B — Tiền xử lý và xây dựng đồ thị | 5 | 1 | 2 |
| C — XAI | 5 | 0 | 2 |
| D — Hybrid Centrality | 5 | 1 | 3 |
| E — Quy trình chiến thuật cấp đội | 6 | 0 | 2 |
| F — Tiến độ và quản lý | 6 | 0 | 1 |
| **Tổng** | **38** | **2** | **11** |

---

## 2. Cách đọc tài liệu

### Mức nghiêm trọng

| Mức | Nghĩa là |
|---|---|
| **Rất cao** | Nếu không xử lý, một phần lớn đề tài không hoàn thành được |
| **Cao** | Làm sai lệch kết quả hoặc chặn một bước quan trọng |
| **Trung bình** | Gây tốn thời gian hoặc thu hẹp kết luận |
| **Thấp** | Cần lưu ý nhưng không ảnh hưởng kết quả nghiên cứu |

### Loại rủi ro

| Ký hiệu | Loại | Đặc điểm |
|---|---|---|
| 🚧 | **Chặn** | Không xử lý thì không đi tiếp được. Dễ phát hiện. |
| 🔇 | **Âm thầm** | Vẫn cho ra số liệu, nhưng số liệu sai. **Nguy hiểm nhất.** |
| 📏 | **Ràng buộc phạm vi** | Không sai, nhưng giới hạn điều được phép kết luận |

---

## 3. Năm rủi ro cần xử lý trước tiên

### 🥇 K1 — Đồ thị possession quá nhỏ để XAI có ý nghĩa

**Mức: Rất cao 🚧** — đây là rủi ro nguy hiểm nhất và chưa từng được ghi nhận trong
các tài liệu trước.

Một possession bóng sống điển hình chỉ có 3–6 đường chuyền, tạo ra đồ thị khoảng
**4–5 đỉnh và 4–5 cạnh**.

Hệ quả trên đồ thị nhỏ như vậy:

- Câu hỏi "cạnh nào quan trọng" chỉ có 16 tập con khả dĩ — gần như tầm thường
- **Sparsity** mất ý nghĩa: bỏ 1 trong 4 cạnh đã là 25%, không có độ phân giải
- **Fidelity** trở nên rời rạc và nhiễu

Nếu không xử lý, **toàn bộ Bài toán 2 — tức nửa sau của đề tài — có nguy cơ cho ra kết
quả không nói lên điều gì**, dù mọi thứ vẫn chạy trơn tru.

**Phương án — làm cả ba, không chọn một:**

1. **Lọc possession dài hơn cho phần XAI.** Chỉ giữ possession có ≥ 4–5 đường chuyền.
   Đánh đổi cỡ mẫu lấy tính diễn giải. Với ~150.000 possession, giữ 20% dài nhất vẫn
   còn ~30.000 — dư dùng.
2. **Làm giàu đồ thị.** Thêm `Carry` và `Dribble` làm cạnh chứ không chỉ đường chuyền.
   Bật `add_goal_node` trong `configs/graph.yaml` (tiền lệ: TacticAI).
3. **Dựa vào tổng hợp.** Hybrid Centrality vốn gộp qua nhiều possession, nên nhiễu ở
   từng đồ thị sẽ triệt tiêu bớt. Không dùng giải thích của một possession đơn lẻ làm
   bằng chứng.

**Cần làm:** tạo quyết định `0003-don-vi-do-thi-cho-xai.md`, chốt **trước** khi code Bước 3.

---

### 🥈 H1–H3 — Hybrid Centrality: công thức, hệ số alpha, và bẫy vòng lặp logic

**Mức: Rất cao 🚧** — đây là **đóng góp chính** của đề tài.

Ba vấn đề chồng lên nhau:

**H1 — Chưa có công thức.** Báo cáo tiến độ chỉ nêu ý tưởng kết hợp, chưa có định nghĩa
toán học. Chưa chốt thì không code được.

**H2 — Hệ số alpha không được chọn tùy ý.** Phải dò trên tập validation với tiêu chí
được trình bày rõ ràng.

**H3 — Bẫy vòng lặp logic.** 🔇 Đây là cái tinh vi nhất và dễ mắc nhất:

```
Dò alpha sao cho TỐI ĐA tương quan với xA
        ↓
Rồi dùng chính xA để KIỂM CHỨNG Hybrid Centrality
        ↓
Kết luận: "chỉ số của em tương quan cao với xA"  ← hiển nhiên, vì đã tối ưu cho nó
```

Đây là lập luận vòng tròn. Hội đồng có kinh nghiệm sẽ nhận ra ngay, và nó phá hỏng
toàn bộ phần kiểm chứng.

**Phương án:**

- **Tách bạch tiêu chí dò và tiêu chí kiểm chứng.** Nếu dò alpha bằng xA thì phải kiểm
  chứng bằng chỉ số khác (key passes, assists), và ngược lại.
- Hoặc dò alpha bằng **tiêu chí nội tại** không liên quan tới chỉ số kiểm chứng — ví dụ
  độ ổn định split-half của bảng xếp hạng.
- Hoặc chia **tập kiểm chứng riêng** ở cấp giải đấu: dò alpha trên Premier League, kiểm
  chứng trên La Liga.

---

### 🥉 D4 — Wyscout không có trường `possession`

**Mức: Cao 🚧**

Đã xác minh từ nguồn gốc: mục Data Records của bài báo Pappalardo et al. liệt kê đủ
12 trường của một event (`eventId`, `eventName`, `subEventId`, `subEventName`, `tags`,
`eventSec`, `id`, `matchId`, `matchPeriod`, `playerId`, `positions`, `teamId`) — **không
có trường nào nhóm event thành possession**. Xác nhận độc lập lần hai từ
`socceraction/data/wyscout/schema.py`.

Nghĩa là toàn bộ phương án kiểm chứng chéo phụ thuộc vào việc **tự tái tạo possession**.

**Phương án:**

- Dùng `socceraction` chuyển **cả hai** nguồn về định dạng trung gian **SPADL**
- Quy tắc tái tạo possession phải **giống hệt nhau cho cả hai nguồn**. Nếu StatsBomb
  dùng trường có sẵn còn Wyscout dùng quy tắc tự chế, so sánh sẽ không công bằng và
  toàn bộ kết luận kiểm chứng chéo mất giá trị.
- **Khuyến nghị mạnh:** áp dụng cùng một quy tắc tái tạo cho **cả StatsBomb**, bỏ qua
  trường `possession` có sẵn, để hai nguồn hoàn toàn đồng nhất về cách chia.

**Phép thử quyết định:** chuyển thử 1 trận mỗi nguồn sang SPADL, so sánh tập đặc trưng
còn lại sau khi lấy phần giao. **Làm sớm nhất**, trước khi đầu tư vào bước khác.

---

### 4️⃣ X2 + X4 — Gộp điểm importance sai cách

**Mức: Cao 🔇** — cả hai đều là rủi ro âm thầm.

**X2 — Mask không cùng thang giữa các possession.** GNNExplainer sinh mask riêng cho
từng đồ thị, giá trị trong (0,1). "0,8" trên đồ thị 4 cạnh và "0,8" trên đồ thị 12 cạnh
**không cùng ý nghĩa**. Gộp thẳng giống như cộng phần trăm của hai tổng số khác nhau.

**X4 — Nhầm lẫn giữa tổng đóng góp và hiệu suất.** Nếu **cộng** importance, cầu thủ tham
gia nhiều possession tự động điểm cao. Nếu lấy **trung bình**, cầu thủ chỉ tham gia 3
possession được đặt ngang hàng với người tham gia 200 — cực kỳ nhiễu.

**Phương án:**

| Vấn đề | Cách xử lý |
|---|---|
| X2 | Chuẩn hóa min-max **trong từng possession trước**, cho tổng mask mỗi đồ thị bằng 1. Khi gộp dùng **median** thay vì mean — bền hơn với đồ thị chênh lệch số cạnh |
| X4 | Đặt ngưỡng `min_possessions`. Báo cáo **cả hai** biến thể: tổng (đóng góp toàn cục) và trung bình (hiệu suất mỗi lần tham gia) — chúng đo hai thứ khác nhau, đừng gộp làm một |
| Bổ sung | Nhân trọng số importance với **xác suất dự đoán của chính possession đó**, để "quan trọng trong pha nguy hiểm" khác với "quan trọng trong pha vô hại" |

---

### 5️⃣ K3 — Rò rỉ dữ liệu giữa train và test

**Mức: Cao 🔇**

Nếu chia theo possession thay vì theo trận, các possession cùng một trận (cùng đội hình,
cùng chiến thuật, cùng đối thủ) sẽ nằm ở cả hai tập. Mô hình sẽ cho kết quả **rất đẹp
nhưng vô giá trị**.

Đây là rủi ro âm thầm điển hình: không có thông báo lỗi, chỉ số chỉ đơn giản là cao bất
thường — và cao bất thường thì dễ bị nhầm là thành công.

**Phương án:**

- Chia theo `match_id`, tuyệt đối không chia theo possession
- Viết **test tự động** kiểm tra ba tập `match_id` rời nhau hoàn toàn — đây là test ưu
  tiên số 1 trong `tests/`
- Commit `data/splits/` lên git để cố định và tái lập được
- Cảnh giác: nếu AUC-PR cao bất thường, **nghi ngờ rò rỉ trước khi ăn mừng**

---

## 4. Nhóm A — Rủi ro dữ liệu

| Mã | Rủi ro | Mức | Phương án xử lý |
|---|---|---|---|
| **A1** | Kế hoạch gốc chỉ ~52 trận (Bundesliga 2023/24 = 34 trận Leverkusen; UCL = 18 trận chung kết) | Cao 🚧 | Chuyển sang StatsBomb Big-5 mùa 2015/16 (~1.517 trận). Đã có phương án đầy đủ trong `Nguon_du_lieu_XAI_Football.md` |
| **A2** | Bundesliga không có full season trong StatsBomb | Trung bình 📏 | Wyscout có Bundesliga 2017/18 đủ 306 trận. Dùng nguồn này nếu bắt buộc giữ Bundesliga |
| **A3** | Tập huấn luyện **không có dữ liệu 360/tracking** | Trung bình 📏 | Tuyên bố phạm vi rõ ràng ở đầu khóa luận. Dùng Leverkusen 2023/24 (34 trận, **có** 360) làm case study bổ sung |
| **A4** | Wyscout không có trường `possession` | Cao 🚧 | Xem mục 3, rủi ro D4 |
| **A5** | Hệ tọa độ khác nhau: 120×80 tuyệt đối và 0–100 phần trăm | Trung bình 🔇 | Chuẩn hóa về một hệ **trước** khi trích đặc trưng cạnh. Kiểm tra thủ công hướng tấn công đã lật đúng chưa |
| **A6** | Bộ phân loại event khác nhau giữa hai nhà cung cấp | Trung bình 📏 | SPADL làm lớp trung gian. Chấp nhận mất `under_pressure`, `technique`, `body_part` — đặc trưng cạnh phải lấy **phần giao** |
| **A7** | Chênh lệch mùa giải (2015/16 và 2017/18) gây nhiễu biến | Thấp 📏 | Nêu rõ trong phần hạn chế. Khi kết quả tốt đừng nhận công cho cả hai yếu tố; khi xấu phải nêu cả hai khả năng |
| **A8** | Đội hình hai nguồn không trùng nhau | Thấp 📏 | **Cấm dùng player ID làm node feature.** Chỉ dùng vai trò/vị trí và thống kê trong possession |
| **A9** | Dung lượng: 1 file event ≈ 3,19 MB, tổng ~4,9 GB | Thấp | Dùng `statsbombpy` tải chọn lọc. **Không `git clone` cả kho 7,4 GB.** Cache lại `data/raw/` |
| **A10** | Giấy phép StatsBomb bắt buộc ghi nguồn **và chèn logo** | Thấp 🚧 | Chèn logo StatsBomb trong khóa luận, bài báo và slide bảo vệ. Lấy từ Media Pack |
| **A11** | Một số con số về cỡ mẫu là **ước lượng** từ dung lượng file, chưa đếm trực tiếp | Thấp | Chạy kiểm chứng ở mục 9 của `Nguon_du_lieu_XAI_Football.md` trước khi đưa số vào khóa luận |

---

## 5. Nhóm B — Tiền xử lý và xây dựng đồ thị

| Mã | Rủi ro | Mức | Phương án xử lý |
|---|---|---|---|
| **B1** | **Đồ thị possession quá nhỏ cho XAI** | **Rất cao** 🚧 | Xem mục 3, rủi ro K1 |
| **B2** | Quy tắc gán nhãn chưa chốt (quyết định 0001) | Cao 🚧 | Chốt **trước** khi code Bước 3. Chạy thống kê mô tả trước để biết tỉ lệ lớp thực tế. **Khuyến nghị target là `shot`, không phải `goal`** — nhãn `goal` chỉ khoảng 1–2%, quá hiếm để huấn luyện ổn định |
| **B3** | Rò rỉ dữ liệu giữa train và test | Cao 🔇 | Xem mục 3, rủi ro K3 |
| **B4** | Mất cân bằng lớp | Trung bình | Weighted BCE hoặc Focal Loss. **AUC-PR là độ đo chính**, accuracy chỉ tham khảo. Early stopping theo AUC-PR |
| **B5** | Over-smoothing khi GNN quá sâu so với đồ thị nhỏ | Trung bình | Đồ thị 4–5 đỉnh thì **2 lớp là đủ**. Đưa số lớp vào tinh chỉnh siêu tham số, đừng cố định 3 lớp |

---

## 6. Nhóm C — XAI

| Mã | Rủi ro | Mức | Phương án xử lý |
|---|---|---|---|
| **C1** | Mask không so sánh được giữa các possession | Cao 🔇 | Chuẩn hóa min-max trong từng đồ thị trước; gộp bằng **median** |
| **C2** | Nhầm lẫn tổng đóng góp và hiệu suất khi gộp | Cao 🔇 | Đặt `min_possessions`; báo cáo cả hai biến thể riêng biệt |
| **C3** | GNNExplainer quá chậm cho quy trình áp dụng hàng loạt | Trung bình | **PGExplainer** cho quy trình chuẩn (huấn luyện một lần, giải thích nhanh). GNNExplainer chỉ dùng cho phân tích sâu trên mẫu nhỏ. Đây là lý do thực tế để dùng cả hai, không chỉ để đối sánh |
| **C4** | Diễn giải điểm XAI thành quan hệ nhân quả | Trung bình 🔇 | Kiểm tra **Stability** trước khi diễn giải bất kỳ subgraph nào. Ngôn ngữ báo cáo phải viết "mô hình dựa vào", không viết "cầu thủ này tạo ra" |
| **C5** | Importance không tính đến độ nguy hiểm của possession | Trung bình | Nhân trọng số với xác suất dự đoán của possession đó |

---

## 7. Nhóm D — Hybrid Centrality

| Mã | Rủi ro | Mức | Phương án xử lý |
|---|---|---|---|
| **D1** | Chưa có công thức toán học chính thức | **Rất cao** 🚧 | Chốt quyết định 0002 trước khi code Bước 9 |
| **D2** | Hệ số alpha bị chọn tùy ý | Cao | Dò trên tập validation, trình bày rõ dải dò và tiêu chí chọn |
| **D3** | **Bẫy vòng lặp logic khi chọn alpha** | Cao 🔇 | Tách tiêu chí dò khỏi tiêu chí kiểm chứng. Xem mục 3 |
| **D4** | Xung đột chuẩn hóa nội bộ đội và giữa các đội | Cao 🔇 | Cần **hai thang chuẩn hóa**: toàn cục để so sánh giữa đội, nội bộ đội để xếp hạng trong đội. Không dùng lẫn |
| **D5** | Chọn z-score sẽ làm Gini/HHI không tính được (giá trị âm) | Trung bình 🚧 | Nếu Bước 4 cần Gini/HHI thì **bắt buộc dùng min-max**. Ràng buộc này phải ghi vào quyết định 0002 |

---

## 8. Nhóm E — Quy trình so sánh chiến thuật cấp đội

Nhóm này liên quan tới quy trình 5 bước đã đề xuất để đáp ứng yêu cầu "định lượng và
lặp lại được" của giảng viên.

| Mã | Rủi ro | Mức | Phương án xử lý |
|---|---|---|---|
| **E1** | **Jaccard đang đo tính ổn định đội hình, không phải ổn định chiến thuật** | Cao 🔇 | Định nghĩa subgraph theo **vai trò và vùng sân**, không theo danh tính cầu thủ. Khi đó so sánh được qua các trận dù đội hình đổi, và so sánh được giữa các đội. Nhất quán với ràng buộc A8 |
| **E2** | So sánh **thứ hạng** giữa hai đội là vô nghĩa | Trung bình 🔇 | Hạng 1 của đội A trừ hạng 1 của đội B luôn bằng 0. Phải so sánh **giá trị** (thang toàn cục) hoặc **dạng phân bố** |
| **E3** | Thiếu mô hình null và kiểm định thống kê | Cao 🔇 | Chỉ số 0,42 và 0,38 khác nhau thật hay chỉ là nhiễu? Cần **permutation test** (xáo trộn ngẫu nhiên importance rồi tính lại) và **khoảng tin cậy bootstrap** cho mọi chỉ số |
| **E4** | Chưa kiểm tra độ tin cậy của chính chỉ số | Trung bình | **Split-half reliability**: chia số trận của một đội làm đôi, tính chỉ số trên từng nửa. Hai nửa lệch nhau nghĩa là chỉ số không đo được gì ổn định |
| **E5** | Gini trên 11–14 cầu thủ là thống kê cỡ mẫu quá nhỏ | Trung bình | Luôn kèm khoảng tin cậy bootstrap. Cân nhắc tổng hợp theo mùa thay vì theo từng trận |
| **E6** | Tự chế chỉ số tập trung trong khi đã có chuẩn sẵn | Thấp | Dùng **Freeman centralization** — đã có tiền lệ trong tài liệu passing network (Clemente và cộng sự), diễn giải rõ ràng: 0 = mọi cầu thủ tương tác đều nhau, gần 1 = có cầu thủ chủ chốt. Tính trên **cả** mạng truyền thống (baseline) **và** Hybrid Centrality (đề xuất) — chênh lệch chính là đóng góp. Giữ Gini/HHI làm chỉ số bổ trợ |

---

## 9. Nhóm F — Tiến độ và quản lý

| Mã | Rủi ro | Mức | Phương án xử lý |
|---|---|---|---|
| **F1** | **Quá tải phạm vi** — Hybrid Centrality cộng thêm 3 chỉ số cấp đội, mỗi cái đều cần kiểm chứng riêng | Cao | Chọn **một** chỉ số cấp đội làm thật chặt (khuyến nghị: chỉ số tập trung), hai cái còn lại trình bày ở mức minh họa và ghi rõ là hướng phát triển |
| **F2** | Chi phí tính toán: 4 backbone × 5 seed = 20 lần chạy | Trung bình | Đồ thị nhỏ nên **CPU chạy được**, không bắt buộc GPU. Colab bản miễn phí là đủ. Nếu thiếu thời gian, giảm `n_runs` xuống 3 và ghi rõ trong báo cáo |
| **F3** | Đổi phạm vi dữ liệu cần giảng viên đồng ý | Trung bình | Chuẩn bị lập luận 3 ý đã soạn trong `Nguon_du_lieu_XAI_Football.md` mục 11 |
| **F4** | Tên đề tài hứa nhiều hơn dữ liệu cho phép | Trung bình 📏 | Thêm chữ "tấn công" vào hướng nghiên cứu cụ thể. Quan trọng hơn: thêm mục **Phạm vi và Giới hạn** ngay đầu khóa luận |
| **F5** | Nguồn thứ cấp và công cụ tự động có thể sai | Thấp 🔇 | Đã gặp thực tế: một lần đọc file bị cắt ngắn cho kết quả sai; một danh mục dataset ghi sai số trận Bundesliga. **Chỉ trích dẫn số liệu tự kiểm chứng từ nguồn gốc** |
| **F6** | Thay đổi quyết định giữa chừng sau khi đã huấn luyện | Trung bình 🚧 | Chốt các quyết định trong `docs/decisions/` **trước** khi code. Đã ghi rồi thì không đổi, trừ khi làm lại từ đầu |

---

## 10. Danh sách rủi ro âm thầm

Đây là nhóm cần cảnh giác nhất. Chúng **không báo lỗi**, vẫn cho ra số liệu và biểu đồ
bình thường, nhưng con số **sai**.

| Mã | Rủi ro | Biểu hiện dễ nhầm là thành công |
|---|---|---|
| **B3** | Rò rỉ dữ liệu train/test | AUC-PR cao bất thường → dễ tưởng mô hình tốt |
| **C1** | Mask không cùng thang | Bảng xếp hạng cầu thủ trông hợp lý nhưng thứ tự sai |
| **C2** | Nhầm tổng đóng góp với hiệu suất | Cầu thủ chơi nhiều phút luôn đứng đầu → trông "đúng trực giác" nên không ai nghi ngờ |
| **C4** | Diễn giải XAI thành nhân quả | Câu chuyện kể ra rất thuyết phục nhưng không có căn cứ |
| **D3** | Vòng lặp logic khi chọn alpha | Tương quan với xA rất cao → trông như kiểm chứng thành công |
| **D4** | Xung đột chuẩn hóa | So sánh giữa các đội vẫn ra số, nhưng số đó vô nghĩa |
| **E1** | Jaccard đo nhầm đối tượng | Ra chỉ số ổn định chiến thuật, thực chất là ổn định đội hình |
| **E2** | So sánh thứ hạng | Luôn ra 0 hoặc số vô nghĩa mà không ai để ý |
| **E3** | Thiếu kiểm định | Mọi chênh lệch đều được diễn giải như thể có thật |
| **A5** | Tọa độ chưa chuẩn hóa đồng bộ | Mô hình vẫn học được, chỉ là học sai không gian |
| **F5** | Số liệu từ nguồn thứ cấp | Trích dẫn trông chuyên nghiệp nhưng sai sự thật |

**Nguyên tắc phòng vệ:** với mỗi rủi ro trong bảng này, phải có **một phép kiểm tra tự
động hoặc một bước xác nhận thủ công bắt buộc** — không dựa vào việc "nhìn thấy kết quả
hợp lý".

---

## 11. Lộ trình xử lý

### Giai đoạn 0 — Trước khi viết dòng code đầu tiên

| Việc | Rủi ro xử lý |
|---|---|
| Chốt quyết định **0001** — quy tắc gán nhãn | B2 |
| Chốt quyết định **0002** — công thức Hybrid Centrality, hệ số alpha, cách tránh vòng lặp logic, chuẩn hóa | D1, D2, D3, D4, D5 |
| Tạo và chốt quyết định **0003** — đơn vị đồ thị đủ lớn cho XAI | B1 |
| Chạy kiểm chứng dữ liệu 5 bước | A11 |
| **Phép thử SPADL** trên 1 trận mỗi nguồn | A4, A6 |
| Trao đổi với giảng viên về phạm vi dữ liệu và tên đề tài | F3, F4 |

> Phép thử SPADL là **cửa quyết định**. Nếu thất bại, chuyển sang phương án dự phòng:
> dùng Wyscout chỉ để kiểm chứng ở mức xếp hạng cầu thủ, không chuyển cả mô hình sang.

### Giai đoạn 1 — Dữ liệu và đồ thị

| Việc | Rủi ro xử lý |
|---|---|
| Viết test chống rò rỉ **trước** khi viết code chia tập | B3 |
| Chuẩn hóa tọa độ, kiểm tra thủ công hướng tấn công | A5 |
| Dựng đồ thị theo quyết định 0003, kiểm tra phân bố kích thước đồ thị | B1 |
| Thống kê mô tả tỉ lệ lớp, đặt `pos_weight` | B4 |

### Giai đoạn 2 — Huấn luyện và đánh giá

| Việc | Rủi ro xử lý |
|---|---|
| Đưa số lớp GNN vào tinh chỉnh siêu tham số | B5 |
| Cố định seed, lưu bản chụp cấu hình mỗi lần chạy | F6 |
| **Nếu AUC-PR cao bất thường: dừng lại, kiểm tra rò rỉ trước** | B3 |

### Giai đoạn 3 — XAI

| Việc | Rủi ro xử lý |
|---|---|
| Chuẩn hóa mask trong từng đồ thị trước khi gộp | C1 |
| Dùng PGExplainer cho quy trình chuẩn | C3 |
| Kiểm tra Stability trước khi diễn giải bất kỳ subgraph nào | C4 |

### Giai đoạn 4 — Hybrid Centrality và chỉ số chiến thuật

| Việc | Rủi ro xử lý |
|---|---|
| Áp dụng hai thang chuẩn hóa cho hai mục đích | D4 |
| Định nghĩa subgraph theo vai trò/vùng sân | E1 |
| Bổ sung permutation test và bootstrap CI cho mọi chỉ số | E3, E5 |
| Kiểm tra split-half reliability | E4 |
| Tính Freeman centralization trên cả mạng truyền thống và Hybrid | E6 |

---

## 12. Bảng kiểm trước khi bắt đầu code

- [ ] Đã chốt quyết định 0001 — quy tắc gán nhãn, target là `shot`
- [ ] Đã chốt quyết định 0002 — công thức Hybrid Centrality, cách chọn alpha **không vòng lặp logic**, chuẩn hóa min-max nếu cần Gini
- [ ] Đã tạo và chốt quyết định 0003 — ngưỡng độ dài possession cho phần XAI
- [ ] Đã chạy thống kê mô tả và biết **tỉ lệ possession dẫn đến sút thực tế**
- [ ] Đã chạy phép thử SPADL thành công trên cả hai nguồn
- [ ] Đã xác nhận số trận thực tế bằng `statsbombpy`, không dựa vào ước lượng
- [ ] Đã viết test chống rò rỉ dữ liệu
- [ ] Đã thống nhất với giảng viên về phạm vi dữ liệu mới
- [ ] Đã bổ sung mục Phạm vi và Giới hạn vào đề cương khóa luận
- [ ] Đã xác nhận node feature **không chứa player ID**

---

## 13. Nguồn tham khảo

**Đã kiểm chứng trực tiếp**

- StatsBomb Open Data — https://github.com/statsbomb/open-data
- Pappalardo et al. (2019), *Scientific Data* 6:236 — https://pmc.ncbi.nlm.nih.gov/articles/PMC6817871/
- socceraction / SPADL — https://github.com/ML-KULeuven/socceraction
- kloppy — https://github.com/PySport/kloppy

**Cơ sở lý thuyết cho chỉ số cấp đội**

- Buldú et al., *Using Network Science to Analyse Football Passing Networks* — https://pmc.ncbi.nlm.nih.gov/articles/PMC6186964/
- *Network Characteristics of Successful Performance in Association Football* (UEFA Champions League) — https://pmc.ncbi.nlm.nih.gov/articles/PMC5507993/

**Tài liệu nội bộ liên quan**

- `docs/Nguon_du_lieu_XAI_Football.md` — khảo sát và lựa chọn nguồn dữ liệu
- `docs/Quy_trinh_XAI_Football.md` — quy trình triển khai 12 bước
- `docs/decisions/` — nhật ký quyết định nghiên cứu
