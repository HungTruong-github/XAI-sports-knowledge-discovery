# Khảo sát và lựa chọn nguồn dữ liệu

**Đề tài:** Mạng chuyền bóng và phân tích chiến thuật bóng đá bằng GNN kết hợp XAI

| | |
|---|---|
| **Ngày lập** | 04/09/2026 |
| **Trạng thái** | Đề xuất — chờ trao đổi với giảng viên hướng dẫn |
| **Liên quan** | `Quy_trinh_XAI_Football.md` Bước 1; `XAI-Football_Project_Content.md` mục 6, 19 |
| **Ảnh hưởng tới** | `configs/data.yaml`, quyết định [0001](decisions/0001-quy-tac-gan-nhan.md), thiết kế node feature ở Bước 3.1 |

---

## Tóm tắt

Khi kiểm chứng trực tiếp nguồn dữ liệu ghi trong báo cáo tiến độ 01/09/2026, phát hiện
kế hoạch dữ liệu hiện tại **chỉ cho khoảng 50 trận đấu**, không phải toàn mùa giải như
giả định. Cỡ mẫu này không đủ để thực hiện chiến lược đối sánh 3 tầng đã cam kết.

Tài liệu này đề xuất thay bằng **hai bộ dữ liệu**:

| | Bộ dữ liệu | Vai trò | Cỡ mẫu |
|---|---|---|---|
| **①** | StatsBomb Open Data — Big-5 mùa 2015/16 | Huấn luyện và đánh giá chính | ~1.517 trận |
| **②** | Wyscout / Pappalardo et al. 2019 (Scientific Data) | Kiểm chứng chéo nhà cung cấp | 1.941 trận |

Cặp này không chỉ giải quyết vấn đề cỡ mẫu mà còn **bổ sung một trục đóng góp khoa học
mới** mà kế hoạch cũ không có: kiểm chứng khả năng khái quát hóa của Hybrid Centrality
qua hai nhà cung cấp dữ liệu độc lập.

---

## 1. Vấn đề phát hiện ở kế hoạch dữ liệu hiện tại

### 1.1. Giả định trong báo cáo so với thực tế

Báo cáo tiến độ và `XAI-Football_Project_Content.md` mục 19 đều ghi nguồn dữ liệu là
**StatsBomb Open Data — Bundesliga 2023/24 và UEFA Champions League**, trong đó
`Quy_trinh_XAI_Football.md` mục 1.1 mô tả rõ "Bundesliga 2023/24 (toàn bộ mùa giải)".

Kiểm chứng trực tiếp trên kho `statsbomb/open-data`:

| Nguồn ghi trong báo cáo | Thực tế | Bằng chứng |
|---|---|---|
| Bundesliga 2023/24, toàn bộ mùa giải (306 trận) | **34 trận**, chỉ gồm các trận có Bayer Leverkusen thi đấu | `data/matches/9/281.json` — 63.079 byte; đội `Bayer Leverkusen` (team_id 904) xuất hiện ở **mọi** trận |
| UEFA Champions League, các mùa có sẵn | **Chỉ trận chung kết của mỗi mùa** | `data/matches/16/4.json` chứa đúng **1 trận**: Tottenham Hotspur – Liverpool, 01/06/2019, `competition_stage = "Final"` |

Tổng cộng: **34 + ~15 ≈ 50 trận**.

### 1.2. Hệ quả về mặt thống kê

Áp dụng chia 70/15/15 theo trận như `configs/split.yaml`:

```
50 trận  →  train ~35 trận  |  val ~7 trận  |  test ~7 trận
```

Với 7 trận test, và với bài toán mất cân bằng lớp (possession dẫn đến sút là thiểu số),
khoảng tin cậy của AUC-PR sẽ rộng đến mức **không phân biệt được backbone nào tốt hơn**.
Điều này làm hỏng trực tiếp mục đích của Tầng 2 trong chiến lược đối sánh — vốn là
"cô lập biến số kiến trúc để khẳng định đóng góp không đến từ việc chọn đúng mô hình mạnh".

Đây không phải vấn đề có thể khắc phục bằng kỹ thuật; nó là giới hạn cứng của cỡ mẫu.

### 1.3. Bundesliga là ngoại lệ trong StatsBomb Open Data

Điểm đáng chú ý: **Bundesliga là giải Big-5 duy nhất StatsBomb không mở đầy đủ một mùa
giải nào.** Xác định bằng cách đối chiếu dung lượng file danh sách trận — tỉ lệ
byte/trận rất ổn định (~1.814 byte/trận, hiệu chuẩn từ file Leverkusen 34 trận):

| Giải, mùa 2015/16 (`season_id = 27`) | File | Dung lượng | Số trận suy ra | Kết luận |
|---|---|---|---|---|
| La Liga | `11/27.json` | 702.673 B | ~380 | Toàn giải ✅ |
| Premier League | `2/27.json` | 698.179 B | ~380 | Toàn giải ✅ |
| Serie A | `12/27.json` | 686.900 B | ~380 | Toàn giải ✅ |
| Ligue 1 | `7/27.json` | 667.360 B | ~377 | Toàn giải ✅ |
| **Bundesliga** | `9/27.json` | **61.666 B** | **34** | Chỉ Leverkusen ❌ |

Kiểm chứng chéo phương pháp: `9/27.json` (61.666 B) khi đọc trực tiếp cho đúng 34 trận,
tất cả đều có Leverkusen — khớp với ước lượng từ dung lượng. Tương tự với FA Women's
Super League: `37/90.json` (243.008 B) → ~134 trận, khớp con số 131 trận của mùa 2020/21.

> **Lưu ý về nguồn thứ cấp:** danh mục dataset của Jan Van Haaren ghi "German Bundesliga
> 2015/16 — 306 matches". Con số này **mâu thuẫn với file gốc** (61.666 B, 34 trận). Khi
> viết khóa luận, chỉ trích dẫn số liệu đã tự kiểm chứng từ kho dữ liệu gốc, không dẫn
> lại số liệu từ trang tổng hợp.

---

## 2. Tiêu chí lựa chọn

Xác định tiêu chí **trước** khi khảo sát, để việc lựa chọn không bị dẫn dắt bởi bộ dữ
liệu tình cờ tìm thấy:

| # | Tiêu chí | Vì sao bắt buộc |
|---|---|---|
| 1 | Là **event data** có định danh cầu thủ | Đỉnh đồ thị là cầu thủ (Bước 3.1) |
| 2 | Đường chuyền có **người nhận** xác định được | Cạnh có hướng A→B; không có trường này thì không dựng được đồ thị |
| 3 | Có **tọa độ** điểm chuyền đi và điểm nhận | Đặc trưng cạnh (Bước 3.2) |
| 4 | Nhóm được thành **possession sequence** | Đơn vị đồ thị của bài toán |
| 5 | Có **kết cục cú sút** để gán nhãn | Nhãn ở mức đồ thị (Bước 2.3) |
| 6 | Đủ **cỡ mẫu** cho chia theo trận | Xem mục 1.2 |
| 7 | **Truy cập tự do**, không cần xin quyền | Rủi ro tiến độ đồ án |
| 8 | Có **tài liệu và kiểm chứng chất lượng** công khai | Yêu cầu "kiểm chứng và xác thực đầy đủ"; cần trích dẫn được khi bảo vệ |
| 9 | Giấy phép cho phép **công bố nghiên cứu** | Đề tài hướng tới bài báo khoa học |

---

## 3. Dataset ① — StatsBomb Open Data

### 3.1. Định danh

| | |
|---|---|
| **Nhà cung cấp** | Hudl StatsBomb |
| **Kho** | `https://github.com/statsbomb/open-data` |
| **Thư viện** | `statsbombpy` |
| **Quy mô kho** | 81 cặp giải–mùa (`data/competitions.json`) |
| **Định dạng** | JSON |

### 3.2. Phạm vi đề xuất sử dụng

**Tập chính — Big-5 mùa 2015/16 (trừ Bundesliga):**

| Giải | `competition_id` | `season_id` | Số trận |
|---|---|---|---|
| Premier League | 2 | 27 | ~380 |
| La Liga | 11 | 27 | ~380 |
| Serie A | 12 | 27 | ~380 |
| Ligue 1 | 7 | 27 | ~377 |
| | | **Tổng** | **~1.517** |

**Tập mở rộng tùy chọn** (nếu cần tăng cỡ mẫu hoặc kiểm tra tính ổn định qua thể thức
giải đấu khác):

| Giải | Số trận | Ghi chú |
|---|---|---|
| FIFA World Cup 2018 | 64 | Giải đấu quốc tế |
| FIFA World Cup 2022 | 64 | Có kèm dữ liệu 360 |
| UEFA Euro 2020 / 2024 | 51 / 51 | Có kèm dữ liệu 360 |
| FA Women's Super League 2018/19–2023/24 | ~460 | **Toàn giải, nhiều mùa liên tiếp** |
| Indian Super League 2020/21 | ~115 | Toàn giải |

> Riêng FA Women's Super League là trường hợp đáng cân nhắc: đây là một trong số ít
> giải được StatsBomb mở **đầy đủ nhiều mùa liên tiếp**. Nếu muốn phân tích diễn biến
> Hybrid Centrality của cùng một cầu thủ qua nhiều mùa, đây là lựa chọn tốt nhất trong
> toàn bộ kho.

### 3.3. Cấu trúc dữ liệu — đã kiểm chứng trực tiếp

Kiểm tra trên `data/events/3754217.json` (Chelsea – Arsenal, Premier League 2015/16):

**Khóa cấp cao nhất của một event:**

```
id, index, period, timestamp, minute, second, type, team, duration,
possession, possession_team, play_pattern, location, player
```

✅ Ba trường quyết định đều có mặt trong dữ liệu mùa 2015/16:

| Trường | Dùng cho |
|---|---|
| `possession` | Nhóm event thành possession sequence — **Bước 2.1** |
| `possession_team` | Xác định đội đang kiểm soát bóng |
| `play_pattern` | Lọc bóng sống / tình huống cố định; hữu ích khi giới hạn phạm vi open-play |

**Đối tượng `pass` của một đường chuyền:**

| Trường | Dùng cho |
|---|---|
| `recipient` | **Đỉnh đích của cạnh** — trường quan trọng nhất, không có thì không dựng được đồ thị |
| `end_location` | Tọa độ điểm nhận `(x2, y2)` — Bước 3.2 |
| `length` | Khoảng cách chuyền |
| `angle` | Góc chuyền (radian) |
| `height` | Ground / Low / High Pass |
| `outcome` | Rỗng nghĩa là chuyền thành công; có giá trị nghĩa là hỏng (`Incomplete`, `Out`...) |
| `body_part`, `technique`, `cross`, `under_pressure` | Đặc trưng bổ sung tùy chọn |

Toàn bộ đặc trưng cạnh liệt kê ở `Quy_trinh_XAI_Football.md` mục 3.2 đều lấy được
trực tiếp, **không cần suy diễn**.

### 3.4. Giấy phép và nghĩa vụ trích dẫn

Terms of use hướng nghiên cứu. Trích nguyên văn từ README của kho:

> "If you publish, share or distribute any research, analysis or insights based on this
> data, please state the data source as StatsBomb and use our logo"

**Nghĩa vụ cụ thể:** ghi rõ nguồn StatsBomb **và chèn logo StatsBomb** trong khóa luận,
bài báo, và slide bảo vệ. Logo lấy từ Media Pack của StatsBomb.

### 3.5. Đánh giá

| Ưu điểm | Hạn chế |
|---|---|
| Có sẵn `possession` — không phải tự tái tạo | Không có Bundesliga full season |
| Đặc trưng đường chuyền chi tiết nhất trong các nguồn mở | Mùa 2015/16 đã cũ so với 2023/24 |
| Giữ đúng nhà cung cấp đã cam kết với giảng viên | Tài liệu kỹ thuật là spec PDF, không phải bài báo bình duyệt |
| Cỡ mẫu ~1.517 trận, gấp ~30 lần kế hoạch cũ | Nghĩa vụ chèn logo khi công bố |
| Thư viện `statsbombpy` chính chủ, ổn định | |

---

## 4. Dataset ② — Wyscout / Pappalardo et al. 2019

### 4.1. Trích dẫn đầy đủ

> Pappalardo, L., Cintia, P., Rossi, A., Massucco, E., Ferragina, P., Pedreschi, D.,
> & Giannotti, F. (2019). **A public data set of spatio-temporal match events in soccer
> competitions.** *Scientific Data*, 6, 236.

| | |
|---|---|
| **DOI dữ liệu** | `10.6084/m9.figshare.c.4415000.v2` |
| **Giấy phép** | **CC BY 4.0** |
| **Định dạng** | JSON |
| **Nhà cung cấp gốc** | Wyscout |

### 4.2. Phạm vi

| Giải | Mùa | Số trận |
|---|---|---|
| Premier League | 2017/18 | 380 |
| La Liga | 2017/18 | 380 |
| Serie A | 2017/18 | 380 |
| Ligue 1 | 2017/18 | 380 |
| **Bundesliga** | **2017/18** | **306** |
| FIFA World Cup | 2018 | 64 |
| UEFA Euro | 2016 | 51 |
| **Tổng** | | **1.941 trận** |

Tổng thể: **1.941 trận — 3.251.294 event — 4.299 cầu thủ**.

> Bộ này **có Bundesliga đầy đủ 306 trận** — đúng giải đấu báo cáo tiến độ nhắm tới,
> thứ mà StatsBomb Open Data không có. Nếu bắt buộc phải giữ Bundesliga trong đề tài,
> đây là nguồn duy nhất khả dụng.

### 4.3. Vì sao đây là bộ "có kiểm chứng và xác thực đầy đủ" đúng nghĩa

Đây là điểm mạnh quyết định của bộ dữ liệu này và là lý do chính để đưa vào đề tài.
Bộ dữ liệu **đã qua bình duyệt khoa học** tại *Nature Scientific Data*, với một mục
**Technical Validation** riêng mô tả quy trình xác thực bốn tầng:

| Tầng | Nội dung |
|---|---|
| **1. Thu thập có kiểm soát** | Mỗi trận được gán nhãn bởi **3 chuyên viên**: một người phụ trách mỗi đội, một người giám sát toàn trận |
| **2. Kiểm soát chất lượng tự động** | Thuật toán đối chiếu chéo event giữa các operator, phát hiện và loại các chuỗi sự kiện bất khả thi |
| **3. Kiểm soát chất lượng thủ công** | Quality controller rà lại toàn bộ các trận được chọn theo lấy mẫu thống kê, sau khi trận đấu kết thúc |
| **4. Kiểm chứng phân phối** | Xác nhận tính toàn vẹn qua phân tích thống kê: trung bình **1.682 ± 101 event/trận**, ~**50% là đường chuyền**, phân phối thời điểm ghi bàn khớp kỳ vọng |

Trong các bộ dữ liệu bóng đá mở hiện có, **không bộ nào khác công bố quy trình xác thực
ở mức chi tiết và đã qua bình duyệt như vậy**. Đây là căn cứ trực tiếp trả lời câu hỏi
của hội đồng về độ tin cậy của dữ liệu đầu vào.

### 4.4. Cấu trúc pass event

| Trường | Ghi chú |
|---|---|
| `positions` | Tọa độ điểm đi và điểm đến, thang **0–100** (phần trăm chiều sân), **không phải** đơn vị tuyệt đối |
| `playerId` | Định danh cầu thủ thực hiện |
| `teamId` | Đội |
| `eventSec` | Thời điểm trong hiệp, tính bằng giây |
| `matchPeriod` | Hiệp |
| Tag `accurate` / `not accurate` | Đường chuyền thành công hay hỏng |
| Tag `key pass`, `assist`, `opportunity`, `goal` | Nhãn ngữ nghĩa bổ sung |

> **Giá trị bổ sung quan trọng:** tag `key pass` và `assist` có sẵn ngay trong bộ dữ liệu.
> Đây chính là hai trong bốn chỉ số đối chiếu mà `Quy_trinh_XAI_Football.md` mục 9.4 yêu
> cầu để kiểm tra face validity của Hybrid Centrality. Dùng được trực tiếp, không phải
> đi tìm nguồn dữ liệu thứ ba.

### 4.5. Cách truy cập

| Phương án | Đường dẫn | Ghi chú |
|---|---|---|
| Gốc | figshare collection `4415000` | JSON nguyên bản của bài báo |
| **Khuyến nghị** | `github.com/koenvo/wyscout-soccer-match-event-dataset` | Cùng dữ liệu, đóng gói lại theo định dạng Wyscout chuẩn, **nạp trực tiếp bằng `kloppy`** |

### 4.6. Đánh giá

| Ưu điểm | Hạn chế |
|---|---|
| Quy trình xác thực đã bình duyệt, trích dẫn được | **Không có trường `possession`** — phải tự tái tạo (xem rủi ro R1) |
| CC BY 4.0 — giấy phép thoáng nhất | Tọa độ theo phần trăm, kém chính xác hơn tọa độ tuyệt đối |
| 1.941 trận, lớn hơn dataset ① | Đặc trưng đường chuyền ít chi tiết hơn StatsBomb |
| Có Bundesliga đầy đủ 306 trận | Chỉ một mùa giải (2017/18) cho các giải quốc nội |
| Có sẵn tag `key pass` / `assist` cho Bước 9.4 | |
| Có DOI, trích dẫn chuẩn học thuật | |

---

## 5. Vì sao chọn đúng cặp này

### 5.1. Bốn trục khác biệt độc lập

| Trục | Dataset ① | Dataset ② |
|---|---|---|
| Nhà cung cấp | StatsBomb | Wyscout |
| Mùa giải | 2015/16 | 2017/18 |
| Quy trình gán nhãn | Nội bộ StatsBomb | 3 operator + QC, đã bình duyệt |
| Hệ tọa độ | 120 × 80 (tuyệt đối) | 0–100 (phần trăm) |

**Bốn giải trùng nhau** — Premier League, La Liga, Serie A, Ligue 1 — nhưng khác mùa và
khác nhà cung cấp. Đây chính là điều kiện lý tưởng cho một phép thử khái quát hóa: cùng
bối cảnh bóng đá, khác hoàn toàn về nguồn và quy trình sinh dữ liệu.

### 5.2. Thiết kế thực nghiệm

```
Dataset ① (StatsBomb 2015/16, ~1.517 trận)
        │
        ├── train / val / test  ──►  Bước 5-6: huấn luyện + đối sánh Tầng 2, Tầng 3(a)
        │                            Bước 7-8: XAI + đối sánh Tầng 3(b)
        │                            Bước 9:   tính Hybrid Centrality
        │
        └──────────────────────────────────┐
                                           │  mô hình đã chốt (không huấn luyện lại)
                                           ▼
Dataset ② (Wyscout 2017/18, 1.941 trận) ──► KIỂM CHỨNG NGOÀI
                                           │
                                           ├── Hiệu năng dự đoán có giữ được không?
                                           ├── Subgraph quan trọng có cùng dạng mẫu không?
                                           └── Thứ hạng Hybrid Centrality có ổn định không?
```

Nguyên tắc bắt buộc: **dataset ② không được dùng ở bất kỳ bước nào của quá trình chọn
mô hình hay dò siêu tham số.** Nếu vi phạm, nó không còn là kiểm chứng ngoài nữa.

### 5.3. Liên hệ với câu hỏi nghiên cứu đã đặt

Thiết kế này trả lời trực tiếp ba câu hỏi trong `XAI-Football_Project_Content.md` mục 17
mà kế hoạch dữ liệu cũ **không thể** trả lời:

| Câu hỏi | Cách trả lời |
|---|---|
| **RQ5** — Thành phần XAI quan trọng có nhất quán giữa các possession không? | Mở rộng thành: có nhất quán qua **hai nhà cung cấp** không |
| **RQ7** — Phương pháp có phân biệt được khác biệt chiến thuật giữa đội/trận không? | Kiểm tra trên hai tập giải đấu độc lập |
| **RQ8** — Giải thích có chuyển thành tri thức **tái lập được** không? | Tái lập trên nguồn dữ liệu khác chính là định nghĩa chặt nhất của "tái lập được" |

Nếu Hybrid Centrality giữ được thứ hạng cầu thủ ổn định qua hai nhà cung cấp khác nhau,
đó là bằng chứng mạnh hơn hẳn kết quả chỉ chạy trên một nguồn — và là câu trả lời sẵn
cho câu hỏi "kết quả này có khái quát được không" mà hội đồng gần như chắc chắn sẽ hỏi.

---

## 6. Rủi ro kỹ thuật

| # | Rủi ro | Mức | Hướng xử lý |
|---|---|---|---|
| **R1** | **Wyscout không có trường `possession`** — phải tự tái tạo possession sequence | **Cao** | Dùng `socceraction`: chuyển **cả hai** nguồn về định dạng trung gian **SPADL**; module xT của thư viện này đã có sẵn logic chia trận thành possession. Quy tắc tái tạo phải ghi vào quyết định 0001 và áp dụng **giống hệt nhau cho cả hai nguồn** — nếu không, so sánh sẽ không công bằng và toàn bộ kết luận kiểm chứng chéo mất giá trị. |
| **R2** | Hệ tọa độ khác nhau (120×80 tuyệt đối vs 0–100 phần trăm) | Vừa | Chuẩn hóa về một hệ duy nhất **trước** khi trích đặc trưng cạnh. `kloppy` hỗ trợ; vẫn phải kiểm tra thủ công hướng tấn công đã được lật đúng chưa (Bước 2.4). |
| **R3** | Bộ phân loại event khác nhau giữa hai nhà cung cấp | Vừa | SPADL là lớp trung gian được thiết kế đúng cho vấn đề này. Đánh đổi: **đặc trưng cạnh phải lấy phần giao của hai nguồn**, chấp nhận bỏ các trường chỉ StatsBomb mới có (`under_pressure`, `technique`, `body_part`). |
| **R4** | Chênh lệch mùa giải (2015/16 vs 2017/18) | Thấp | Là chủ ý thiết kế, không phải lỗi. Nhưng **phải nêu rõ trong phần hạn chế**: chênh lệch hiệu năng khi chuyển nguồn có thể đến từ khác biệt mùa giải chứ không chỉ từ khác biệt nhà cung cấp. Không được diễn giải quá mức. |
| **R5** | Đội hình cầu thủ hai nguồn không trùng nhau | Thấp | **Ràng buộc ngược lên thiết kế:** node feature chỉ được dùng **vai trò/vị trí và thống kê trong possession**, tuyệt đối **không dùng player ID** làm đặc trưng. Nếu vi phạm, mô hình không chạy được trên nguồn ②. |

**R1 và R5 phải được xử lý trước khi bắt đầu code**, vì chúng ràng buộc thiết kế ở
Bước 2.1 và Bước 3.1.

---

## 7. Các bộ dữ liệu đã xét và loại

| Bộ dữ liệu | Quy mô | Vì sao loại |
|---|---|---|
| **PFF FC — FIFA World Cup 2022** | 64 trận, event + broadcast tracking, có grade từng pha | Phải **gửi yêu cầu xin quyền truy cập**, không tải tự do được → vi phạm tiêu chí 7, rủi ro tiến độ. Cỡ mẫu 64 trận cũng chưa đủ. *Giữ làm dự phòng nếu sau này muốn mở rộng sang tracking data.* |
| **Bassek et al. 2025** — *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data, CC BY 4.0 | **7 trận** Bundesliga 1 và 2 | Chất lượng rất cao: dữ liệu vị trí **chính thức** đầu tiên của một giải đỉnh cao, đồng bộ event + position, đã bình duyệt. Nhưng 7 trận là quá nhỏ để huấn luyện. *Có thể dùng cho case study định tính ở Bước 10, hoặc kiểm tra trực quan xem subgraph XAI có khớp với vị trí thực tế của cầu thủ không.* |
| **SkillCorner Open Data** | 10 trận A-League 2024/25 | Tracking từ sóng truyền hình, không phải event data đầy đủ. Không có người nhận đường chuyền → vi phạm tiêu chí 2. |
| **Metrica Sports Sample Data** | 3 trận | Đã **ẩn danh cầu thủ** → không phân tích được vai trò cầu thủ, vi phạm tiêu chí 1. |
| **StatsBomb — Bundesliga 2023/24 + UCL** | ~50 trận | Kế hoạch cũ. Xem mục 1. |
| **football.db / openfootball** | — | Chỉ có kết quả trận và metadata, **không có event data** → vi phạm tiêu chí 1, 2, 3. |

---

## 8. Ảnh hưởng ngược lên thiết kế pipeline

Việc chọn hai nguồn kéo theo một số ràng buộc phải áp dụng ngay từ đầu, không thể sửa
sau:

| Mục trong `Quy_trinh_XAI_Football.md` | Thay đổi cần thiết |
|---|---|
| **1.1** Nguồn dữ liệu | Thay Bundesliga 2023/24 + UCL bằng Big-5 2015/16; bổ sung nguồn kiểm chứng Wyscout |
| **1.4** Lưu trữ dữ liệu thô | Tách `data/raw/` thành hai nhánh theo nhà cung cấp |
| **2.1** Trích possession | Phải viết được **hai đường vào**: dùng `possession` sẵn có với StatsBomb, tái tạo với Wyscout — cùng cho ra một cấu trúc chung |
| **3.1** Node feature | **Cấm dùng player ID.** Chỉ vai trò/vị trí + thống kê trong possession (rủi ro R5) |
| **3.2** Edge feature | Chỉ dùng phần giao của hai nguồn (rủi ro R3) |
| **4** Chia tập | Chia theo trận **chỉ trong dataset ①**; dataset ② là tập kiểm chứng ngoài, không tham gia chia |
| **9.4** Kiểm chứng Hybrid Centrality | Tag `key pass` / `assist` lấy trực tiếp từ dataset ②, không cần nguồn thứ ba |
| **10** Kết quả đầu ra | Bổ sung một bảng mới: so sánh hiệu năng và thứ hạng Hybrid Centrality giữa hai nguồn |

---

## 9. Kế hoạch kiểm chứng trước khi triển khai

Các con số trong tài liệu này được suy ra từ dung lượng file và từ đọc mẫu. **Phải xác
nhận lại bằng dữ liệu tải thật** trước khi đưa vào khóa luận. Thực hiện trong notebook
thăm dò, chưa cần viết pipeline:

| # | Việc | Tiêu chí đạt |
|---|---|---|
| 1 | Gọi `statsbombpy.sb.matches()` cho từng cặp `competition_id`/`season_id` ở mục 3.2 | Số trận khớp bảng, sai lệch không quá vài trận |
| 2 | Lấy ngẫu nhiên 5 trận mỗi giải, kiểm tra sự tồn tại của `possession`, `possession_team`, `pass.recipient` | Có mặt ở **100%** trận kiểm tra, không chỉ ở trận đã kiểm |
| 3 | Thống kê mô tả: số possession/trận, phân phối độ dài chuỗi chuyền, **tỉ lệ possession dẫn đến sút** | Có con số cụ thể — đây là **đầu vào bắt buộc** để chốt quyết định 0001 và đặt `pos_weight` trong `configs/train.yaml` |
| 4 | Tải thử dataset ② bằng `kloppy` từ repo `koenvo/...` | Đọc được, đếm đúng 1.941 trận |
| 5 | **Chuyển thử 1 trận StatsBomb + 1 trận Wyscout sang SPADL** bằng `socceraction`, so sánh tập đặc trưng còn lại sau khi lấy phần giao | Đặc trưng cạnh còn lại đủ dùng cho Bước 3.2 |

> **Bước 5 là phép thử quyết định tính khả thi của toàn bộ phương án kiểm chứng chéo.**
> Nên làm sớm nhất, trước khi đầu tư công sức vào các bước khác.
>
> **Phương án dự phòng nếu bước 5 thất bại:** dùng dataset ② chỉ để kiểm chứng Hybrid
> Centrality ở mức **xếp hạng cầu thủ** — tính centrality truyền thống và các chỉ số đối
> chiếu trên nguồn ②, so sánh dạng phân phối và tương quan — thay vì chuyển cả mô hình
> dự đoán sang. Cách này vẫn giữ được giá trị kiểm chứng chéo nhưng ở mức yếu hơn.

---

## 10. Việc cần làm tiếp

- [ ] Chạy kiểm chứng 5 bước ở mục 9
- [ ] Cập nhật `configs/data.yaml`: thay khối `competitions`, bổ sung khối nguồn kiểm chứng
- [ ] Tạo `docs/decisions/0003-lua-chon-nguon-du-lieu.md` ghi lại quyết định này và lý do
- [ ] Cập nhật `docs/decisions/0001-quy-tac-gan-nhan.md`: quy tắc gán nhãn giờ phải áp
      dụng được cho **cả hai nguồn**, trong đó Wyscout không có `possession` sẵn — ràng
      buộc này thu hẹp không gian lựa chọn của quyết định 0001
- [ ] Cập nhật `data/README.md`: hai nguồn, hai giấy phép, cấu trúc `raw/` theo nhà cung cấp
- [ ] Cập nhật `README.md` mục nguồn dữ liệu
- [ ] Bổ sung `socceraction` và `kloppy` vào `requirements.txt`
- [ ] Trao đổi với giảng viên hướng dẫn (xem mục 11)

---

## 11. Nội dung cần trao đổi với giảng viên

Đề xuất trình bày theo ba ý, kèm bằng chứng cụ thể:

**1. Vì sao phải đổi.** Trình bày bảng ở mục 1.1 với đường dẫn file và dung lượng cụ
thể. Nhấn mạnh đây là giới hạn của kho dữ liệu, không phải lựa chọn của nhóm, và phát
hiện được nhờ kiểm chứng trực tiếp trước khi bắt tay vào code — nếu phát hiện muộn hơn
sẽ phải làm lại từ đầu.

**2. Đổi ít nhất có thể.** Vẫn giữ nguyên nhà cung cấp StatsBomb đã báo cáo, chỉ đổi
giải đấu và mùa. Toàn bộ phương pháp, bộ độ đo và chiến lược đối sánh 3 tầng ở mục 1.2
và 1.3 của báo cáo trước **giữ nguyên không thay đổi**.

**3. Việc đổi làm mạnh thêm đề tài.** Kế hoạch cũ chỉ có một nguồn dữ liệu. Kế hoạch mới
bổ sung một trục đóng góp mà trước đó không có: kiểm chứng khả năng khái quát hóa qua
hai nhà cung cấp độc lập, trong đó nguồn thứ hai là bộ dữ liệu **đã qua bình duyệt tại
Nature Scientific Data**. Đây là câu trả lời sẵn cho câu hỏi về tính khái quát mà hội
đồng nhiều khả năng sẽ đặt ra.

Nếu thầy muốn giữ Bundesliga: nêu phương án dùng Wyscout Bundesliga 2017/18 (306 trận
đầy đủ) làm nguồn chính thay vì StatsBomb — đổi mùa giải thay vì đổi giải đấu.

---

## 12. Nhật ký kiểm chứng

| Ngày | Nội dung kiểm chứng | Nguồn | Kết quả |
|---|---|---|---|
| 04/09/2026 | Danh sách giải–mùa trong StatsBomb Open Data | `data/competitions.json` | 81 cặp giải–mùa |
| 04/09/2026 | Số trận Bundesliga 2023/24 | `data/matches/9/281.json` | 34 trận, chỉ Bayer Leverkusen |
| 04/09/2026 | Số trận UEFA Champions League 2018/19 | `data/matches/16/4.json` | 1 trận, `competition_stage = "Final"` |
| 04/09/2026 | Dung lượng file matches của Big-5 mùa 2015/16 | GitHub Contents API | Bundesliga 61.666 B; bốn giải còn lại 667–702 KB |
| 04/09/2026 | Sự tồn tại của `possession` trong dữ liệu 2015/16 | `data/events/3754217.json` | Có `possession`, `possession_team`, `play_pattern`, `pass.recipient` |
| 04/09/2026 | Quy mô và quy trình xác thực bộ Wyscout | Pappalardo et al. 2019, PMC6817871 | 1.941 trận / 3.251.294 event / 4.299 cầu thủ; xác thực 4 tầng |
| 04/09/2026 | Phân bố số trận theo giải của bộ Wyscout | `koenvo/wyscout-soccer-match-event-dataset` | 380×4 + 306 + 64 + 51 = 1.941 ✓ |

---

## Nguồn tham khảo

**Bộ dữ liệu**

- StatsBomb Open Data — https://github.com/statsbomb/open-data
- Danh sách giải–mùa — https://github.com/statsbomb/open-data/blob/master/data/competitions.json
- Pappalardo et al. (2019), *Scientific Data* 6:236 — https://pmc.ncbi.nlm.nih.gov/articles/PMC6817871/
- Wyscout dataset trên figshare — https://figshare.com/collections/Soccer_match_event_dataset/4415000
- Bản chuyển đổi cho kloppy — https://github.com/koenvo/wyscout-soccer-match-event-dataset

**Công cụ**

- `statsbombpy` — https://github.com/statsbomb/statsbombpy
- `socceraction` / SPADL — https://github.com/ML-KULeuven/socceraction
- `kloppy` — https://github.com/PySport/kloppy

**Đã xét và loại**

- Bassek et al. (2025), *Scientific Data* — https://www.nature.com/articles/s41597-025-04505-y
- PFF FC World Cup 2022 — https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data
- SkillCorner Open Data — https://github.com/SkillCorner/opendata
- Metrica Sports Sample Data — https://github.com/metrica-sports/sample-data

**Tham khảo tổng hợp**

- Danh mục dataset của Jan Van Haaren — https://www.janvanhaaren.be/resources.html
  *(lưu ý: mục Bundesliga 2015/16 của trang này mâu thuẫn với file gốc — xem mục 1.3)*
