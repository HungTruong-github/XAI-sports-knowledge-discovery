# BÁO CÁO TIẾN ĐỘ ĐỒ ÁN — KẾ HOẠCH TUẦN 19/09/2026

**Đề tài:** Hệ thống trí tuệ nhân tạo có thể giải thích (XAI) trong bài toán đánh giá và khám phá tri thức thể thao

**Hướng nghiên cứu cụ thể:** Mạng chuyền bóng và phân tích chiến thuật bóng đá bằng Graph Neural Network (GNN) kết hợp XAI

---

## 1. Phản hồi các góp ý của giảng viên (18/09/2026)

Thầy để lại ba ý kiến trên báo cáo ngày 12/09. Nhóm xin phản hồi từng ý theo đúng thứ tự.

| # | Vị trí góp ý | Nội dung | Phản hồi tại mục |
|---|---|---|---|
| 1 | Mục 1.2(b) — dòng Fidelity | *"đồ thị bị đứt gãy khi loại bỏ subgraph thì sao?"* | 1.1 |
| 2 | Mục 1.3 — Tổng kết baseline 3 tầng | *"nhất trí"* | 1.2 |
| 3 | Mục 2 — Nội dung thực hiện trong tuần | *"nhóm hãy sớm chuyển qua giai đoạn thực nghiệm"* | 1.3 |

---

### 1.1. Trả lời: đồ thị bị đứt gãy khi loại bỏ subgraph

Nhóm xác nhận đây là một vấn đề **có thật và nghiêm trọng**, không phải chi tiết kỹ thuật nhỏ. Sau khi rà soát lại, nhóm nhận thấy cách định nghĩa Fidelity trong báo cáo 12/09 là chưa đủ chặt chẽ. Dưới đây là phân tích và phương án xử lý.

#### a) Vấn đề cụ thể

Độ đo Fidelity+ được định nghĩa là `Fidelity+ = f(G) − f(G∖S)`, trong đó `G` là đồ thị possession, `S` là subgraph được XAI xác định là quan trọng, và `f` là xác suất do mô hình dự đoán. Phép tính này đòi hỏi phải **xóa** `S` khỏi `G`.

Vấn đề nằm ở **kích thước đồ thị possession**. Nhóm đã đo trực tiếp trên dữ liệu thật (Premier League 2015/16, 5 trận, 585 possession sau làm sạch — chi tiết ở mục 1.3):

| Chỉ số | p25 | **Trung vị** | p75 | Trung bình |
|---|---|---|---|---|
| Số đỉnh (cầu thủ) mỗi đồ thị | 4,0 | **5,0** | 7,0 | 5,60 |
| Số cạnh (đường chuyền) mỗi đồ thị | 3,0 | **5,0** | 8,0 | 6,58 |

Đồ thị possession trung vị chỉ có **5 đỉnh và 5 cạnh**, về cấu trúc gần như một chuỗi đường nối tiếp (path graph). Trên đồ thị nhỏ và thưa như vậy, việc xóa đi vài cạnh gần như **chắc chắn** làm đồ thị vỡ thành nhiều thành phần liên thông và sinh ra đỉnh cô lập. Điều này gây ra ba hệ quả.

**Hệ quả 1 — Lệch phân phối (out-of-distribution).** Sau khi xóa, `G∖S` không còn là một possession hợp lệ: nó là một đồ thị rời rạc mà mô hình **chưa từng gặp trong quá trình huấn luyện**. Giá trị `f(G∖S)` vì thế là một phép ngoại suy, không đáng tin cậy. Độ sụt xác suất đo được là **hỗn hợp của hai nguyên nhân không tách rời được**: (i) mất thông tin thực sự quan trọng, và (ii) mô hình gặp đầu vào bất thường. Đây chính là phê phán mà Hooker và cộng sự (2019) nêu ra khi đề xuất phương pháp ROAR, dành cho toàn bộ họ độ đo dựa trên phép che/xóa (occlusion-based).

**Hệ quả 2 — Cơ chế truyền tin sụp đổ.** GNN hoạt động bằng cách truyền thông điệp dọc theo cạnh. Khi cắt một cạnh ở giữa chuỗi chuyền, các đỉnh phía sau không còn nhận được thông điệp từ phía trước, biểu diễn của chúng suy biến về đúng đặc trưng riêng của bản thân. Do lớp readout dùng mean-pooling trên toàn bộ đỉnh, sự xuất hiện của đỉnh cô lập làm thay đổi hẳn vector đại diện đồ thị — **kể cả khi cạnh bị xóa vốn không quan trọng**.

**Hệ quả 3 — Fidelity mất khả năng phân biệt.** Vì trên đồ thị nhỏ mọi phép xóa đều gây đứt gãy, Fidelity+ của subgraph do XAI chọn và Fidelity+ của một subgraph **ngẫu nhiên cùng kích thước** sẽ đều cao như nhau. Khi đó độ đo không còn chứng minh được rằng XAI đã chọn đúng — nó chỉ chứng minh rằng phá vỡ đồ thị thì mô hình dự đoán sai. Nhóm đã ghi nhận vấn đề này trong hồ sơ rủi ro nội bộ với mã **K1**, xếp mức "Rất cao".

Số liệu đo được ở trên cho thấy mức nghiêm trọng cụ thể: với đồ thị trung vị 5 cạnh, xóa 1 cạnh đã là **20%** cấu trúc, và chỉ tồn tại 31 tập con cạnh khác rỗng. Độ đo Sparsity gần như không có độ phân giải ở quy mô này.

#### b) Phương án xử lý — năm lớp, triển khai đồng thời

Nhóm không chọn một phương án mà áp dụng cả năm, xếp theo chi phí tăng dần.

**(1) Che mềm thay vì xóa cứng — soft-mask thay cho hard-delete.**
Thay vì loại bỏ cạnh khỏi `edge_index`, nhóm **giữ nguyên cấu trúc đồ thị** và chỉ triệt tiêu thông tin đi qua cạnh đó: nhân trọng số thông điệp của cạnh với `(1 − m_e)`, hoặc thay `edge_attr` của cạnh bằng một vector tham chiếu (vector 0 hoặc giá trị trung bình trên tập huấn luyện). Số đỉnh và số cạnh không đổi, đồ thị không đứt gãy, phân phối cấu trúc đầu vào không đổi — chỉ có nội dung thông tin bị xóa. Thư viện `torch_geometric.explain` hỗ trợ sẵn cơ chế này. **Đây là câu trả lời trực tiếp nhất cho câu hỏi của thầy, và nhóm đề xuất lấy làm giao thức chuẩn chính.**

**(2) Đối chứng ngẫu nhiên cùng kích thước.**
Với mỗi possession, ngoài subgraph do XAI chọn, nhóm lấy thêm `k` subgraph ngẫu nhiên **cùng số cạnh** và **cùng mức đứt gãy** (sinh ra cùng số thành phần liên thông). Kết quả báo cáo là hiệu số:

```
ΔFidelity = Fidelity(subgraph XAI) − Fidelity(subgraph ngẫu nhiên)
```

Phần độ sụt do đứt gãy gây ra xuất hiện ở cả hai vế nên bị triệt tiêu; phần dư lại chính là **đóng góp thực sự của giải thích**. Đây là phép đối chứng rẻ nhưng có sức thuyết phục cao trước hội đồng, vì nó cô lập đúng biến số đang tranh cãi.

**(3) Chẩn đoán tính toàn vẹn đồ thị.**
Mỗi lần đo Fidelity, nhóm ghi log kèm bốn chỉ số: số thành phần liên thông trước và sau khi xóa, số đỉnh cô lập sinh ra, và chuỗi chuyền có còn nối từ đỉnh đầu tới đỉnh cuối hay không. Kết quả Fidelity sau đó được **phân tầng** theo hai nhóm "có đứt gãy" và "không đứt gãy". Cách này biến chính điểm yếu thầy chỉ ra thành **một bảng kết quả có giá trị khoa học** trong khóa luận, thay vì né tránh nó.

**(4) ROAR — huấn luyện lại mô hình.**
Xóa dần các thành phần theo thứ tự importance, sau đó **huấn luyện lại mô hình từ đầu** trên chính tập dữ liệu đã bị xóa. Vì mô hình mới được học trên đúng phân phối đã đứt gãy, vấn đề lệch phân phối ở Hệ quả 1 **biến mất hoàn toàn**. Báo cáo 12/09 chỉ liệt kê ROAR như một độ đo bổ sung; nhóm xin **nâng ROAR lên thành lời giải chính thức cho vấn đề lệch phân phối**, đúng theo hướng câu hỏi của thầy. Chi phí tính toán cao (phải huấn luyện lại nhiều lần) nên chỉ chạy trên tập con và chỉ áp dụng cho mô hình cuối cùng đã được chọn.

**(5) Chặn đứt gãy ngay từ gốc.**
Hai biện pháp ở tầng dữ liệu và tầng thuật toán:

- **Ràng buộc subgraph liên thông.** Buộc explainer chỉ trả về subgraph liên thông, và ưu tiên đọc `Fidelity−` (chỉ **giữ lại** subgraph quan trọng, bỏ phần còn lại) — vì phần giữ lại khi đó là một đồ thị hợp lệ, không rời rạc.
- **Làm giàu đồ thị.** Bổ sung `Carry` và `Dribble` làm cạnh chứ không chỉ đường chuyền; bật node ảo đại diện khung thành đối phương (`add_goal_node`, tiền lệ TacticAI); lọc riêng các possession có ≥ 4 đường chuyền cho phần XAI. Đồ thị dày hơn thì việc xóa một cạnh không còn làm vỡ đồ thị. Nội dung này được đưa thành quyết định **0003** ở mục 3.

#### c) Cam kết về cách báo cáo số liệu

Mọi con số Fidelity trong khóa luận sẽ được báo cáo theo **cặp** — giao thức xóa cứng đặt cạnh giao thức che mềm — kèm `ΔFidelity` so với đối chứng ngẫu nhiên và bảng phân tầng theo mức đứt gãy. Mục đích là chứng minh kết luận **không phụ thuộc vào lựa chọn giao thức đo**, tránh đúng tình huống thầy vừa chỉ ra.

---

### 1.2. Ghi nhận ý kiến "nhất trí" về chiến lược baseline ba tầng

Nhóm ghi nhận thầy đã nhất trí với chiến lược đối sánh ba tầng trình bày ở mục 1.3 báo cáo 12/09. Nhóm xin **đóng băng** nội dung này, coi như đã chốt và không sửa đổi trong các báo cáo sau:

- **Tầng 1** — đã rà soát, không tồn tại công trình giải quyết đồng thời cả ba thành phần (dự đoán outcome từ đồ thị chuyền bóng open-play, trích subgraph bằng graph-XAI, hợp nhất XAI-importance với centrality mạng).
- **Tầng 2** — đối sánh backbone: GCN, MPNN (đề xuất) so với GAT/GATv2, GraphSAGE (baseline), trên cùng dữ liệu, cùng cách chia tập, cùng bộ độ đo.
- **Tầng 3(a)** — baseline phi đồ thị: Logistic Regression, Random Forest, ANN trên đặc trưng passing path.
- **Tầng 3(b)** — baseline giải thích: GNNExplainer so với PGExplainer; centrality truyền thống thuần túy làm baseline "không dùng XAI"; SHAP trên mô hình phi đồ thị làm baseline giải thích ở mức đặc trưng.

Mọi thay đổi ở mục 2 dưới đây (phạm vi dữ liệu) **không làm thay đổi một chữ nào** trong chiến lược này.

---

### 1.3. Tiến độ chuyển sang giai đoạn thực nghiệm

Nhóm tiếp thu chỉ đạo của thầy và xin báo cáo **đúng mức độ đã đạt được, không nói quá**:

> **Nhóm đã hoàn thành Bước 1–2 (thu thập và tiền xử lý dữ liệu). Nhóm CHƯA huấn luyện mô hình, nên chưa có kết quả dự đoán để báo cáo.** Bước 3–6 sẽ thực hiện trong tuần 19/09 → 26/09.

Trạng thái trên toàn bộ quy trình 10 bước của đề tài:

| Bước | Nội dung | Trạng thái |
|---|---|---|
| 1–2 | Thu thập, tiền xử lý, gán nhãn | ✅ Xong, đã chạy thử trên 5 trận |
| 3–4 | Dựng đồ thị possession, chia tập theo trận | ⏳ Tuần 19/09 → 26/09 |
| 5–6 | Huấn luyện GNN, đánh giá và đối sánh baseline | ⏳ Tuần 19/09 → 26/09 |
| 7–8 | Áp dụng XAI, đo Fidelity / Sparsity | ⏳ Sau khi có mô hình |
| 9–10 | Hybrid Centrality, kiểm chứng | ⏳ Giai đoạn sau |

Dù mới ở Bước 1–2, phần việc này **không phải chuẩn bị thuần túy**: chính số liệu đo được từ đây là căn cứ để trả lời câu hỏi của thầy ở mục 1.1 (đồ thị possession trung vị chỉ 5 đỉnh và 5 cạnh), và để chốt quyết định 0001 bằng số thật thay vì chọn theo cảm tính. Chi tiết dưới đây.

#### a) Đã hoàn thành — đường ống Bước 1–2 chạy thông

Nhóm đã viết và chạy được toàn bộ mã cho Bước 1 (thu thập) và Bước 2 (tiền xử lý, gán nhãn):

| Thành phần | Tệp | Chức năng |
|---|---|---|
| Thu thập | `src/xai_football/data/collect.py` | Tải qua `statsbombpy`, lưu JSON thô theo giải–mùa–trận, ghi log số trận và tỉ lệ trận lỗi |
| Trích possession | `src/xai_football/data/possession.py` | Nhóm sự kiện theo trường `possession`, dựng chuỗi đường chuyền thành công |
| Làm sạch | `src/xai_football/data/clean.py` | Loại possession quá ngắn, hiệp phụ, thiếu tọa độ; chẩn đoán hướng tấn công |
| Gán nhãn | `src/xai_football/data/labeling.py` | Cài đặt **cả hai** phương án A và B để so tỉ lệ lớp dương |
| Script chạy | `scripts/01_download_data.py`, `scripts/02_descriptive_stats.py` | Giao diện dòng lệnh, xuất bảng ra `results/tables/` |

#### b) Kết quả chạy thử — Premier League 2015/16, 5 trận

Quy mô chạy thử: 5 trận, 18.245 sự kiện, 836 possession thô → **585 possession** sau làm sạch (loại 251 pha dưới 2 đường chuyền), tổng 4.234 đường chuyền.

> **Giới hạn của mẫu — nhóm xin nêu trước.** Đây là 5 trận trên tổng 380 trận của mùa giải, và do lấy 5 mã trận đầu sau khi sắp xếp nên **4/5 trận có Arsenal thi đấu**. Số liệu dưới đây vì vậy dùng để **xác nhận đường ống xử lý chạy đúng**, chưa phải kết luận khoa học. Nhóm sẽ chạy lại trên toàn bộ phạm vi ngay sau khi thầy chốt nguồn dữ liệu ở mục 2, và báo cáo lại nếu các con số thay đổi đáng kể.

**Bảng 1 — Tỉ lệ lớp dương của từng cách gán nhãn** (căn cứ chốt quyết định 0001):

| Phương án | Mục tiêu | Cửa sổ | Số đơn vị | Số dương | **Tỉ lệ dương** |
|---|---|---|---|---|---|
| A — possession | `shot` | — | 585 | 106 | **18,12%** |
| A — possession | `goal` | — | 585 | 9 | **1,54%** |
| B — đường chuyền | `shot` | 5 giây | 4.234 | 138 | 3,26% |
| B — đường chuyền | `shot` | 10 giây | 4.234 | 254 | 6,00% |
| B — đường chuyền | `shot` | 15 giây | 4.234 | 382 | 9,02% |
| B — đường chuyền | `goal` | 10 giây | 4.234 | 33 | 0,78% |

**Nhận định sơ bộ:** con số này **ủng hộ đề xuất của nhóm ở mục 3.1**. Phương án A với `target = shot` cho tỉ lệ lớp dương 18,12% — mức mất cân bằng còn xử lý được bằng weighted BCE. Nếu chuyển sang `goal`, tỉ lệ tụt xuống 1,54%, tức lớp dương hiếm hơn gần **12 lần**, đúng như lo ngại đã nêu.

**Bảng 2 — Phân bố độ dài possession** (căn cứ chốt quyết định 0003):

| Chỉ số | Giá trị |
|---|---|
| Số đường chuyền mỗi possession (p25 / trung vị / p75) | 3 / 5 / 9 |
| Tỉ lệ possession có ≥ 4 đường chuyền | **66,7%** |
| Tỉ lệ possession có ≥ 5 đường chuyền | **57,1%** |

Con số này **tốt hơn nhiều so với dự đoán ban đầu** của nhóm. Hồ sơ rủi ro nội bộ ước tính việc lọc possession dài sẽ mất khoảng 80% cỡ mẫu; đo thực tế cho thấy lọc ngưỡng ≥ 4 chỉ mất **33,3%**. Nghĩa là quyết định 0003 rẻ hơn nhiều so với lo ngại.

#### c) Bốn phép kiểm tra chống "rủi ro âm thầm" — đều đạt

Nhóm xác định trước một nhóm rủi ro nguy hiểm: loại lỗi **vẫn cho ra số liệu đẹp nhưng số đó sai**, sẽ đi thẳng vào khóa luận mà không ai phát hiện. Với mỗi rủi ro nhóm cài một phép kiểm tra tự động:

| Phép kiểm tra | Giá trị đo | Kết quả |
|---|---|---|
| Số possession mỗi trận trong khoảng [90, 130] | 117,0 | ✅ Đạt |
| Tỉ lệ lớp dương (possession / `shot`) trong [10%, 25%] | 18,12% | ✅ Đạt |
| Số possession dẫn đến sút mỗi trận trong [15, 35] | 21,2 | ✅ Đạt |
| Hướng tấn công: pha dẫn đến sút phải có `x` trung bình cao hơn (rủi ro A5) | chênh +15,22 | ✅ Đạt |

Phép kiểm tra thứ ba là phép **neo vào thực tế bóng đá**. Nhóm đếm trực tiếp trong 5 trận này được **26,8 cú sút mỗi trận** (cả hai đội cộng lại) — đúng mức thường thấy ở Premier League. Con số 21,2 ở bảng trên thấp hơn là hợp lý, vì nó đếm **số pha bóng** dẫn đến sút chứ không phải số cú sút: một pha có thể có nhiều cú sút liên tiếp, và các pha dưới 2 đường chuyền đã bị loại ở bước làm sạch.

Ý nghĩa của phép kiểm tra này: nếu chỉ nhìn tỉ lệ phần trăm thì không biết con số đúng hay sai; quy đổi về đơn vị bóng đá mà người làm chuyên môn kiểm tra được thì phát hiện sai sót ngay.

#### d) Một lỗi đã phát hiện và sửa nhờ chính các phép kiểm tra này

Bản mã đầu tiên có logic **tự động lật tọa độ** những possession có cú sút nhưng đường chuyền lại dừng ở nửa sân nhà, vì cho rằng đó là lỗi ghi ngược hướng tấn công. Chạy thử phát hiện 2 possession bị lật. Nhóm kiểm tra thủ công từng pha và xác định đó **không phải lỗi tọa độ**: đó là các pha cầu thủ **dẫn bóng (carry) từ giữa sân rồi sút**, nên đường chuyền cuối cùng dừng ở khoảng `x ≈ 51` là hoàn toàn đúng.

Logic tự lật đã bị gỡ bỏ, thay bằng chế độ chỉ chẩn đoán và cảnh báo. Đây đúng là loại rủi ro âm thầm nguy hiểm nhất: nếu để nguyên, mô hình vẫn huấn luyện bình thường, vẫn ra số liệu, nhưng đã học sai không gian sân trên một phần dữ liệu.

#### e) Việc còn lại

Phần chưa làm: Bước 3 (dựng đồ thị PyTorch Geometric), Bước 4 (chia tập theo trận), Bước 5–6 (huấn luyện và đánh giá). Mốc bàn giao cụ thể ở mục 4.

Lý do chưa chạy trên toàn bộ dữ liệu: **phạm vi dữ liệu chưa được chốt** (mục 2). Mã đã được tham số hóa hoàn toàn qua `configs/data.yaml`, nên ngay khi thầy quyết định, nhóm chỉ cần đổi cấu hình và chạy lại, không phải sửa dòng mã nào.

---

## 2. Vấn đề chặn: quy mô thực tế của nguồn dữ liệu

### 2.1. Phát hiện

Báo cáo 12/09 ghi nguồn dữ liệu là **StatsBomb Open Data — Bundesliga 2023/24 (toàn bộ mùa giải) và UEFA Champions League**. Nhóm đã kiểm chứng và phát hiện giả định này **không đúng với thực tế**.

Toàn bộ số liệu dưới đây được **đếm trực tiếp bằng thư viện `statsbombpy`** trong ngày 19/09, không phải ước lượng gián tiếp:

| Nguồn ghi trong báo cáo | Thực tế đếm được | Bằng chứng |
|---|---|---|
| Bundesliga 2023/24, toàn mùa (306 trận) | **34 trận** | `sb.matches(competition_id=9, season_id=281)` trả về 34 dòng; Bayer Leverkusen xuất hiện ở **cả 34 trận**, mọi đội khác chỉ 2 trận |
| UEFA Champions League, các mùa có sẵn | **1 trận mỗi mùa** | `sb.matches(competition_id=16, season_id=4)` trả về đúng 1 trận: Tottenham – Liverpool (chung kết 2018/19) |
| | **Tổng cộng ≈ 50 trận** | |

Nhóm cũng xác định thêm: **Bundesliga là giải Big-5 duy nhất mà StatsBomb không mở đầy đủ bất kỳ mùa nào** — mùa 2015/16 cũng chỉ có đúng 34 trận, và cũng chỉ gồm các trận của Bayer Leverkusen.

> **Lưu ý về phương pháp.** Một danh mục dataset thứ cấp phổ biến ghi "German Bundesliga 2015/16 — 306 matches", mâu thuẫn với con số đếm trực tiếp ở trên. Trong khóa luận nhóm chỉ trích dẫn số liệu tự kiểm chứng từ kho gốc, không dẫn lại từ trang tổng hợp.

### 2.2. Vì sao đây là vấn đề chặn

Áp dụng tỉ lệ chia 70/15/15 theo trận như đã thống nhất:

```
50 trận  →  train ~35 trận  |  validation ~7 trận  |  test ~7 trận
```

Với 7 trận trong tập test, và với bài toán mất cân bằng lớp (possession dẫn đến sút là thiểu số), khoảng tin cậy của AUC-PR sẽ rộng đến mức **không phân biệt được backbone nào tốt hơn backbone nào**. Điều này phá hỏng trực tiếp mục đích của **Tầng 2** — tầng mà thầy vừa nhất trí — vốn được lập ra để cô lập biến số kiến trúc và chứng minh đóng góp của nhóm không đến từ việc chọn đúng mô hình mạnh.

Đây là **giới hạn cứng của cỡ mẫu**, không khắc phục được bằng bất kỳ kỹ thuật nào.

### 2.3. Phương án đề xuất

**Phương án A (đề xuất chính) — đổi ít nhất có thể.**
Giữ nguyên nhà cung cấp StatsBomb đã báo cáo với thầy, chỉ đổi giải đấu và mùa:

| Giải | `competition_id` | `season_id` | Số trận (đếm bằng `statsbombpy`) |
|---|---|---|---|
| Premier League | 2 | 27 | 380 |
| La Liga | 11 | 27 | 380 |
| Serie A | 12 | 27 | 380 |
| Ligue 1 | 7 | 27 | 377 |
| | | **Tổng** | **1.517** |

Cỡ mẫu gấp khoảng **30 lần** kế hoạch cũ. Với 1.517 trận, tập test theo tỉ lệ 15% sẽ có khoảng **228 trận** thay vì 7 trận — đủ để khoảng tin cậy của AUC-PR đủ hẹp mà phân biệt được các backbone ở Tầng 2.

Quan trọng hơn, nhóm **đã chạy thật trên nguồn này** (mục 1.3): tải được dữ liệu, trích được possession, gán được nhãn, và cả bốn phép kiểm tra chất lượng đều đạt. Dữ liệu mùa 2015/16 có đủ ba trường quyết định `possession`, `possession_team`, `play_pattern`, và đối tượng `pass` có đủ `recipient`, `end_location`, `length`, `angle`, `height`, `outcome`. Toàn bộ đặc trưng cạnh liệt kê trong quy trình đều lấy được trực tiếp, không cần suy diễn.

Đánh đổi: mùa 2015/16 cũ hơn 2023/24.

**Phương án B (tùy chọn, mạnh hơn) — bổ sung nguồn thứ hai.**
Như Phương án A, cộng thêm bộ dữ liệu Wyscout (Pappalardo và cộng sự, 2019, công bố trên *Nature Scientific Data*, 1.941 trận, 3.251.294 sự kiện, đã qua quy trình xác thực bốn tầng). Việc này mở thêm một trục đóng góp mà kế hoạch cũ không có: **kiểm chứng khả năng khái quát hóa qua hai nhà cung cấp dữ liệu độc lập** — câu trả lời sẵn cho câu hỏi về tính khái quát mà hội đồng nhiều khả năng sẽ đặt ra.

Đánh đổi: Wyscout **không có trường `possession` sẵn**, phải tái tạo possession sequence qua thư viện `socceraction`/SPADL. Đây là rủi ro kỹ thuật thật, nhóm đã ghi nhận với mã **D4**, và cần một phép thử SPADL trên 1 trận mỗi nguồn trước khi cam kết.

**Phương án C (dự phòng nếu thầy muốn giữ Bundesliga).**
Dùng Wyscout Bundesliga 2017/18 (306 trận đầy đủ) làm nguồn chính — tức đổi mùa giải và nhà cung cấp thay vì đổi giải đấu.

**Nhóm xin nhấn mạnh hai điểm:**

1. Đây là giới hạn của kho dữ liệu công khai, **không phải lựa chọn của nhóm**.
2. Việc phát hiện được **trước khi viết dòng mã đầu tiên** đã tiết kiệm toàn bộ công sức thực nghiệm; nếu phát hiện muộn hơn, nhóm sẽ phải làm lại từ đầu.

---

## 3. Ba quyết định xin thầy chốt ngay tại buổi họp

Ba quyết định dưới đây đang **chặn trực tiếp** việc viết mã. Nhóm xin trình bày đề xuất kèm lý do để thầy cho ý kiến ngay trong buổi hôm nay.

### 3.1. Quyết định 0001 — Quy tắc gán nhãn possession

**Đề xuất:** Phương án A — nhãn ở mức possession, gán `1` nếu possession kết thúc bằng một cú sút; mục tiêu dự đoán là **`shot`**, không mở rộng tới `goal`.

**Lý do:**
- Một possession ứng với đúng một đồ thị và một nhãn, khớp trực tiếp với bài toán phân loại ở mức đồ thị đã xác nhận ở mục 1.1 báo cáo 12/09.
- Phương án B (nhãn ở mức đường chuyền) buộc phải chuyển đơn vị đồ thị sang cửa sổ trượt, làm phức tạp Bước 3 và cách tổng hợp điểm XAI ở Bước 9.
- Chọn `shot` thay vì `goal` vì lớp dương của `goal` hiếm hơn khoảng một bậc độ lớn, khiến AUC-PR trở nên rất nhạy với cỡ mẫu.

**Số liệu thực tế đã đo được** (Bảng 1, mục 1.3) ủng hộ lựa chọn này: Phương án A với `target = shot` cho tỉ lệ lớp dương **18,12%**, còn `target = goal` chỉ **1,54%** — hiếm hơn gần 12 lần.

**Cam kết kèm theo:** nhóm đã cài đặt **cả hai** phương án trong mã và sẽ chạy lại bảng so sánh này trên toàn bộ dữ liệu sau khi thầy chốt phạm vi. Nếu số liệu ở quy mô lớn bác bỏ lựa chọn này, nhóm xin đổi **trước khi huấn luyện bất kỳ mô hình nào**.

### 3.2. Quyết định 0003 — Đơn vị đồ thị cho phần XAI *(tạo mới)*

**Đề xuất:** với riêng phần XAI, lọc các possession có **≥ 4 đường chuyền**; bổ sung `Carry` và `Dribble` làm cạnh; bật node ảo đại diện khung thành đối phương.

**Lý do:** đây chính là phần (5) trong câu trả lời ở mục 1.1. Đồ thị trung vị đo được chỉ **5 đỉnh và 5 cạnh** — quá nhỏ để Sparsity và Fidelity có độ phân giải. Làm dày đồ thị là cách xử lý tận gốc vấn đề đứt gãy.

**Đánh đổi — nhỏ hơn nhiều so với lo ngại ban đầu.** Nhóm từng ước tính việc lọc này sẽ mất khoảng 80% cỡ mẫu. Đo thực tế (Bảng 2, mục 1.3) cho thấy **66,7% possession đã có ≥ 4 đường chuyền**, tức chỉ mất 33,3%. Với 1.517 trận của Phương án A, ước tính còn khoảng **118.000 possession** cho phần XAI — dư dùng.

### 3.3. Quyết định 0004 — Giao thức đo Fidelity *(tạo mới)*

**Đề xuất:** chính thức hóa năm lớp phương án ở mục 1.1(b) thành giao thức bắt buộc, với che mềm (soft-mask) làm chuẩn chính, kèm đối chứng ngẫu nhiên, chẩn đoán toàn vẹn đồ thị, và ROAR trên tập con.

**Lý do:** ghi thành văn bản để phần XAI ở Bước 7–8 chỉ việc triển khai theo, không phải tranh luận lại giữa chừng.

### 3.4. Quyết định 0002 — Công thức Hybrid Centrality *(chưa xin chốt)*

Nhóm **chưa xin chốt** quyết định này trong buổi hôm nay, vì nó phụ thuộc vào kết quả đối sánh GNNExplainer và PGExplainer ở Bước 8 — chưa có dữ liệu để quyết. Nhóm chỉ xin ý kiến sơ bộ của thầy về vấn đề nêu ở câu hỏi số 5, mục 5.

---

## 4. Kế hoạch thực nghiệm 19/09 → 26/09

Mục tiêu tuần: chạy đường ống trên **toàn bộ** dữ liệu và huấn luyện được mô hình đầu tiên.

| Nội dung công việc | Sản phẩm cụ thể | Trạng thái |
|---|---|---|
| Thu thập dữ liệu (Bước 1) | `src/xai_football/data/collect.py` + `scripts/01_download_data.py` | ✅ Xong, đã chạy thử 5 trận |
| Trích xuất possession (Bước 2.1) | `src/xai_football/data/possession.py` | ✅ Xong |
| Làm sạch, chuẩn hóa (Bước 2.2, 2.4) | `src/xai_football/data/clean.py` | ✅ Xong |
| Gán nhãn hai phương án (Bước 2.3) | `src/xai_football/data/labeling.py` | ✅ Xong |
| Bảng thống kê mô tả | `scripts/02_descriptive_stats.py` → 5 bảng CSV | ✅ Xong, đã chạy thử |
| Chạy lại trên **toàn bộ** phạm vi đã chốt | 5 bảng CSV ở quy mô đầy đủ | ⏳ Chờ thầy quyết mục 2 |
| Dựng đồ thị possession (Bước 3) | `src/xai_football/graphs/` → đối tượng PyTorch Geometric | ⏳ Tuần này |
| Chia tập theo trận (Bước 4) | `data/splits/` + test tự động chống rò rỉ | ⏳ Tuần này |
| Huấn luyện backbone đầu tiên (Bước 5) | GCN chạy thông, có đường cong loss và bộ độ đo | ⏳ Tuần này |

### 4.1. Năm bảng kết quả đã sinh ra được

Toàn bộ xuất ra `results/tables/`, phục vụ ba mục đích cùng lúc: chốt quyết định 0001, chốt quyết định 0003, và làm phần mô tả dữ liệu cho khóa luận.

| Tệp | Nội dung |
|---|---|
| `collection_report.csv` | Số trận, số sự kiện, tỉ lệ trận tải lỗi theo từng giải |
| `descriptive_stats.csv` | Số possession, phân bố số đường chuyền, phân bố số đỉnh và số cạnh mỗi đồ thị |
| `label_comparison.csv` | Tỉ lệ lớp dương của **tất cả** cách gán nhãn (2 phương án × 2 mục tiêu × 3 cửa sổ) |
| `graph_size_distribution.csv` | Bảng tần suất theo cặp (số đỉnh, số cạnh) |
| `sanity_checks.csv` | Kết quả bốn phép kiểm tra chống rủi ro âm thầm |

### 4.2. Các phép kiểm tra bắt buộc trước khi tin số liệu

Nhóm xác định trước một nhóm rủi ro "âm thầm" — loại lỗi vẫn cho ra số liệu đẹp nhưng số đó sai, sẽ đi thẳng vào khóa luận mà không ai phát hiện cho tới khi hội đồng hỏi đúng câu. Nguyên tắc phòng vệ: **mỗi rủi ro phải có một phép kiểm tra tự động**, không dựa vào việc "nhìn thấy kết quả có vẻ hợp lý".

Bốn phép kiểm tra đã được cài vào `scripts/02_descriptive_stats.py` và chạy tự động mỗi lần sinh bảng — kết quả lần chạy thử ở mục 1.3(c), cả bốn đều đạt. Ngoài ra còn hai việc kiểm tra thủ công bắt buộc:

- Vẽ heatmap điểm chuyền của cả hai đội trong một trận và xác nhận bằng mắt rằng cả hai tấn công cùng một hướng.
- Số trận luôn đếm bằng `statsbombpy`, không dùng lại con số ước lượng từ nguồn thứ cấp.

Khi bước sang Bước 4, phép kiểm tra ưu tiên số một là **test tự động xác nhận ba tập `match_id` rời nhau hoàn toàn** — chống rò rỉ dữ liệu giữa train và test. Đây là rủi ro âm thầm điển hình: không báo lỗi, chỉ đơn giản cho AUC-PR cao bất thường, mà cao bất thường thì rất dễ bị nhầm là thành công.

---

## 5. Nội dung xin ý kiến giảng viên

**Câu hỏi 1 (chặn).** Thầy duyệt cho nhóm đổi phạm vi dữ liệu sang Phương án A — Big-5 mùa 2015/16 trừ Bundesliga, khoảng 1.517 trận — chứ ạ? Và thầy có muốn nhóm bổ sung Wyscout làm nguồn thứ hai (Phương án B) để có thêm trục kiểm chứng khái quát hóa, hay giữ một nguồn duy nhất cho gọn phạm vi đồ án?

**Câu hỏi 2 (chặn).** Giao thức đo Fidelity mà nhóm đề xuất ở mục 1.1(b) — che mềm làm chuẩn chính, kèm đối chứng ngẫu nhiên cùng kích thước và ROAR huấn luyện lại trên tập con — đã đủ chặt chẽ theo ý thầy chưa, hay thầy muốn nhóm bổ sung hướng xử lý nào khác?

**Câu hỏi 3 (chặn).** Thầy duyệt quy tắc gán nhãn Phương án A với `target = shot` để nhóm bắt đầu viết mã ngay trong tuần này chứ ạ? Nhóm cam kết báo cáo lại tỉ lệ lớp dương thực tế của cả hai phương án ở buổi 26/09.

**Câu hỏi 4.** Việc lọc riêng possession có ≥ 4 đường chuyền cho phần XAI sẽ làm hẹp phạm vi kết luận (chỉ kết luận được trên các pha tấn công dài). Mức thu hẹp này có được thầy chấp nhận không, hay thầy muốn nhóm giữ toàn bộ possession và chấp nhận Fidelity nhiễu hơn?

**Câu hỏi 5.** Về hệ số `alpha` trong công thức Hybrid Centrality: nếu nhóm dò `alpha` sao cho tối đa hóa tương quan với xA, rồi lại dùng chính xA để kiểm chứng chỉ số, thì đó là lập luận vòng tròn — kết luận "chỉ số tương quan cao với xA" trở thành hiển nhiên. Thầy gợi ý tiêu chí dò `alpha` nào **tách rời** khỏi tiêu chí dùng để kiểm chứng?

**Câu hỏi 6.** Nội dung "so sánh chiến thuật cấp đội" (giai đoạn 7 trong đặc tả hệ thống) có nằm trong phạm vi khóa luận lần này không, hay nhóm nên cắt để tập trung nguồn lực vào Hybrid Centrality — vốn là đóng góp nghiên cứu chính?

---

## 6. Tổng kết trạng thái công việc

**Điểm cần nắm trước tiên:** nhóm đang ở Bước 1–2 của quy trình 10 bước. **Chưa có kết quả mô hình dự đoán** — đó là mục tiêu của tuần 19/09 → 26/09.

| Nội dung | Trạng thái |
|---|---|
| Định nghĩa bài toán (mục 1.1 báo cáo 12/09) | ✅ Đã chốt |
| Bộ độ đo cho Bài toán 1 (mục 1.2a) | ✅ Đã chốt |
| Bộ độ đo cho Bài toán 2 (mục 1.2b) | ⚠️ Chốt lại giao thức Fidelity — quyết định 0004 |
| Chiến lược baseline ba tầng (mục 1.3) | ✅ Thầy đã nhất trí — đóng băng |
| Phạm vi dữ liệu | 🚧 **Chặn — xin thầy quyết trong buổi này** |
| Quy tắc gán nhãn (0001) | 🚧 Chặn — xin thầy duyệt Phương án A |
| Đơn vị đồ thị cho XAI (0003) | 🚧 Chặn — xin thầy duyệt |
| Giao thức đo Fidelity (0004) | 🚧 Chặn — xin thầy duyệt |
| Công thức Hybrid Centrality (0002) | ⏸ Hoãn — chờ kết quả Bước 8 |
| Mã nguồn Bước 1–2 (thu thập, tiền xử lý, gán nhãn) | ✅ Xong, đã chạy thử 5 trận, 4/4 phép kiểm tra đạt |
| Mã nguồn Bước 3–4 (đồ thị, chia tập) | 🔨 Làm trong tuần 19/09 → 26/09 |
| Mã nguồn Bước 5–6 (huấn luyện, đánh giá) | 🔨 Làm trong tuần 19/09 → 26/09 |
