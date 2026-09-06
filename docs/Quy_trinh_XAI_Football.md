# Quy trình thực hiện đề tài "Mạng chuyền bóng và phân tích chiến thuật bằng GNN kết hợp XAI"

## 0. Sơ đồ tổng thể pipeline

```
[1] Thu thập dữ liệu (StatsBomb Open Data)
        │
[2] Tiền xử lý & gán nhãn (possession → label sút/bàn thắng)
        │
[3] Xây dựng đồ thị chuyền bóng (possession graph)
        │
[4] Chia tập train/val/test (theo trận, tránh leakage)
        │
[5] Huấn luyện mô hình GNN (Bài toán 1 – Dự đoán)
        │
[6] Đánh giá mô hình dự đoán (metrics + baseline 3 tầng)
        │
[7] Áp dụng XAI trên mô hình đã huấn luyện (Bài toán 2 – Giải thích)
        │
[8] Đánh giá chất lượng XAI (Fidelity, Sparsity, Stability...)
        │
[9] Tính chỉ số Hybrid Centrality (centrality truyền thống + XAI-importance)
        │
[10] Kiểm chứng Hybrid Centrality & trực quan hóa
        │
[11] Tổng hợp kết quả, viết báo cáo/paper
```

Pipeline gồm hai nhánh chính chạy nối tiếp: **nhánh dự đoán** (bước 1–6) tạo ra mô hình GNN đáng tin cậy, và **nhánh giải thích** (bước 7–10) khai thác mô hình đó để tạo ra sản phẩm nghiên cứu chính là chỉ số Hybrid Centrality.

---

## 1. Thu thập dữ liệu (Data Collection)

### 1.1. Nguồn dữ liệu
- **StatsBomb Open Data** — nguồn duy nhất (đã chốt ở phần trao đổi trước), gồm các subset:
  - Bundesliga 2023/24 (toàn bộ mùa giải, event data chi tiết)
  - UEFA Champions League (các mùa có sẵn trong Open Data)
  - Có thể mở rộng thêm La Liga / World Cup nếu cần tăng cỡ mẫu, miễn giữ cùng một nhà cung cấp.

### 1.2. Công cụ lấy dữ liệu
- Thư viện `statsbombpy` (Python) hoặc gọi trực tiếp REST/JSON từ repo GitHub `statsbomb/open-data`.
- Ba loại endpoint cần lấy:
  1. `competitions` — danh sách giải đấu, mùa giải có sẵn.
  2. `matches` — danh sách trận đấu theo `competition_id` + `season_id`.
  3. `events` — toàn bộ sự kiện trong từng trận (theo `match_id`), bao gồm pass, shot, carry, dribble, duel...
- Lấy kèm `lineups` (đội hình, vị trí xuất phát) để bổ sung node feature (vị trí thi đấu của cầu thủ).

### 1.3. Phạm vi trích xuất
- Chỉ giữ lại các loại event cần thiết cho việc dựng possession graph: `Pass`, `Shot`, `Carry`, `Dribble`, `Ball Receipt`, `Dispossessed`.
- Loại bỏ ngay từ bước này các trận thiếu dữ liệu (không có tọa độ, thiếu event 360 nếu dùng, hoặc log lỗi từ nhà cung cấp).

### 1.4. Lưu trữ dữ liệu thô
- Lưu nguyên JSON gốc theo cấu trúc `raw/{competition_id}/{season_id}/{match_id}.json` để đảm bảo có thể truy vết lại (reproducibility) và không phải gọi lại API nhiều lần.
- Ghi log số lượng trận, số event mỗi trận, tỉ lệ trận bị loại — phục vụ phần mô tả dataset trong báo cáo/paper.

---

## 2. Tiền xử lý dữ liệu (Data Preprocessing)

### 2.1. Trích xuất possession sequences
- Dùng trường `possession` có sẵn trong StatsBomb event data để nhóm các event thuộc cùng một pha kiểm soát bóng của một đội.
- Với mỗi possession, chỉ giữ lại chuỗi các event `Pass` thành công liên tiếp (loại các event không liên quan trực tiếp đến mạng chuyền như duel, pressure...).

### 2.2. Làm sạch dữ liệu
- Loại possession quá ngắn (ví dụ < 2 đường chuyền) vì không đủ để tạo đồ thị có ý nghĩa.
- Loại các possession xảy ra trong hiệp phụ/luân lưu (nếu có) để tránh lệch phân phối so với thi đấu chính thức.
- Xử lý các trường hợp thiếu tọa độ (x, y) hoặc timestamp lỗi — loại bỏ event lỗi, không suy diễn (impute) tọa độ để tránh đưa nhiễu vào đồ thị.

### 2.3. Gán nhãn (labeling)
- Nhãn nhị phân cho mỗi possession: `1` nếu possession kết thúc bằng — hoặc trong một cửa sổ thời gian ngắn sau đường chuyền cuối dẫn đến — một `Shot`; `0` nếu ngược lại.
- Cần quyết định rõ và ghi lại trong báo cáo: nhãn tính theo **toàn bộ possession** hay theo **từng đường chuyền** (dẫn đến sút trong N giây tiếp theo, tương tự cách tiếp cận của các mô hình xThreat). Đây là quyết định ảnh hưởng trực tiếp đến cách xây dựng đồ thị ở bước 3, nên cần chốt trước khi code.
- Kiểm tra và báo cáo tỉ lệ mất cân bằng lớp (class imbalance) — thường possession dẫn đến sút là thiểu số — để chọn chiến lược xử lý phù hợp (class weight, focal loss, oversampling...).

### 2.4. Chuẩn hóa không gian sân
- Chuẩn hóa tọa độ về hệ quy chiếu thống nhất (chuẩn StatsBomb: sân 120×80), đảm bảo mọi trận đều tấn công theo cùng một hướng (lật tọa độ nếu cần) để mô hình học đúng ngữ nghĩa không gian (gần khung thành đối phương).

### 2.5. Đầu ra của bước 2
- Một tập các possession đã làm sạch, mỗi possession gồm: danh sách cầu thủ tham gia, chuỗi đường chuyền theo thứ tự thời gian, tọa độ, loại chuyền, và nhãn outcome.

---

## 3. Xây dựng đồ thị chuyền bóng (Graph Construction)

### 3.1. Định nghĩa đỉnh (node)
- Mỗi đỉnh là một cầu thủ **tham gia possession đó** (không phải toàn bộ 11 cầu thủ, để đồ thị không quá thưa).
- Node feature đề xuất:
  - Vị trí thi đấu (role: DF/MF/FW, hoặc chi tiết hơn theo `position` của StatsBomb).
  - Vị trí trung bình trên sân trong possession (x, y trung bình).
  - Số lần chạm bóng trong possession.
  - (Tuỳ chọn) chỉ số phong độ/thống kê mùa giải nếu muốn làm giàu đặc trưng.

### 3.2. Định nghĩa cạnh (edge)
- Mỗi cạnh có hướng biểu diễn một đường chuyền thành công từ cầu thủ A → cầu thủ B.
- Edge feature đề xuất:
  - Tọa độ điểm chuyền đi và điểm nhận bóng (x1, y1, x2, y2).
  - Khoảng cách và góc chuyền.
  - Loại đường chuyền (ground pass, high pass, through ball...).
  - Thời điểm chuyền (thời gian trong possession, dùng để giữ thứ tự chuỗi nếu cần mô hình hóa yếu tố thời gian).
  - Trọng số cạnh nếu có nhiều hơn một đường chuyền giữa cùng cặp cầu thủ trong possession (gộp hoặc giữ multi-edge tuỳ kiến trúc).

### 3.3. Thuộc tính đồ thị
- Đồ thị có hướng (directed), có trọng số/đặc trưng cạnh, một nhãn duy nhất ở mức đồ thị (graph-level label) = outcome của possession.
- Optional: thêm 1 node ảo đại diện khung thành đối phương (goal node), giúp mô hình học trực tiếp mối quan hệ "gần khung thành" — cách tiếp cận này đã được dùng trong các nghiên cứu liên quan (ví dụ TacticAI dùng node đại diện cho khung thành trong đồ thị tình huống phạt góc).

### 3.4. Định dạng lưu trữ
- Chuyển từng possession thành một đối tượng `torch_geometric.data.Data` (nếu dùng PyTorch Geometric) gồm `x` (node features), `edge_index`, `edge_attr`, `y` (nhãn).
- Lưu toàn bộ tập đồ thị dưới dạng danh sách `Data` object (`.pt`) theo từng giải/mùa để tiện load lại khi huấn luyện, tránh phải dựng lại đồ thị mỗi lần chạy.

---

## 4. Chia tập dữ liệu (Train / Validation / Test Split)

- **Chia theo trận đấu** (match-level split), không chia ngẫu nhiên theo từng possession — vì các possession trong cùng một trận có tương quan (cùng đội hình, cùng chiến thuật), chia theo possession sẽ gây rò rỉ dữ liệu (data leakage) giữa train và test.
- Tỉ lệ đề xuất: 70% trận cho train, 15% cho validation, 15% cho test.
- Dùng **stratified split** theo tỉ lệ nhãn dương (possession dẫn đến sút) ở cấp độ trận, để tránh một tập bị lệch hẳn về một loại kết quả.
- Ghi rõ danh sách `match_id` thuộc từng tập và cố định (fix seed) để đảm bảo có thể tái lập kết quả giữa các lần chạy và giữa các backbone khi đối sánh (điều kiện bắt buộc cho phần baseline ở mục 1.3 báo cáo trước).

---

## 5. Xây dựng mô hình dự đoán (Bài toán 1 – GNN Prediction Model)

### 5.1. Kiến trúc chính
- **GCN** (Graph Convolutional Network) và **MPNN** (Message Passing Neural Network) — hai backbone chính đã đề xuất trong báo cáo tiến độ.

### 5.2. Backbone đối sánh (Tầng 2 – baseline)
- **GAT/GATv2** (Graph Attention Network) và **GraphSAGE**, huấn luyện trên cùng dữ liệu, cùng cấu hình chia tập, để đối sánh công bằng.

### 5.3. Kiến trúc chi tiết đề xuất
- 2–3 lớp message-passing (tăng số lớp có thể gây over-smoothing với đồ thị possession vốn nhỏ, nên không cần quá sâu).
- Sau các lớp GNN, dùng **readout/pooling** (mean pooling hoặc attention pooling) để tổng hợp biểu diễn từng node thành một vector đại diện cho toàn đồ thị (possession).
- Vector này đi qua MLP + sigmoid để cho ra xác suất possession dẫn đến sút/bàn thắng.

### 5.4. Hàm mất mát (loss function)
- Weighted Binary Cross-Entropy hoặc Focal Loss để xử lý mất cân bằng lớp đã xác định ở bước 2.3.

### 5.5. Huấn luyện
- Optimizer: Adam/AdamW, learning rate scheduler (ví dụ giảm dần khi validation loss không cải thiện).
- Early stopping theo AUC-PR trên tập validation (ưu tiên AUC-PR hơn accuracy vì dữ liệu mất cân bằng, như đã thống nhất ở mục độ đo).
- Cố định seed và log đầy đủ (loss, metrics theo epoch) để phục vụ so sánh giữa các backbone.

### 5.6. Tinh chỉnh siêu tham số (hyperparameter tuning)
- Grid search hoặc random search trên: số lớp GNN, kích thước hidden dimension, dropout, learning rate — thực hiện trên tập validation, không đụng vào tập test.

---

## 6. Đánh giá mô hình dự đoán (Evaluation – Bài toán 1)

- Áp dụng bộ độ đo đã thống nhất: **Accuracy, Precision, Recall, F1-score, AUC-ROC, AUC-PR, Brier score (calibration)**.
- Đối sánh theo đúng chiến lược 3 tầng đã xác định:
  - Tầng 2: GCN/MPNN vs GAT/GATv2 vs GraphSAGE (cùng pipeline, khác backbone).
  - Tầng 3(a): so với baseline phi đồ thị (Logistic Regression, Random Forest, ANN trên đặc trưng passing-path).
- Chọn mô hình có hiệu năng tốt nhất trên tập validation (không phải test) làm mô hình chính thức để đưa sang bước XAI — tránh việc "nhìn" kết quả test trước khi hoàn tất lựa chọn mô hình.
- Chỉ đánh giá lần cuối trên tập test sau khi đã chốt mô hình và siêu tham số.

---

## 7. Áp dụng XAI (Bài toán 2 – Explainability)

### 7.1. Phương pháp
- **GNNExplainer**: học một mask riêng cho từng possession (instance-level), xác định tập cạnh (đường chuyền) và node (cầu thủ) quan trọng nhất đối với dự đoán của mô hình cho possession đó.
- **PGExplainer**: học một mạng giải thích chung (parameterized) trên toàn bộ tập possession, cho phép giải thích nhanh hơn và có tính khái quát (generalizable) hơn khi áp dụng cho possession mới.
- Nhóm dùng cả hai để vừa có giải thích chi tiết theo từng possession (GNNExplainer), vừa có giải thích tổng quát theo mẫu hình chung của mô hình (PGExplainer), đồng thời đối sánh hai phương pháp với nhau (Tầng 3(b) baseline).

### 7.2. Đầu ra của bước XAI
- Với mỗi possession, thu được một **subgraph quan trọng**: tập con cạnh (đường chuyền) có trọng số importance cao nhất, kèm điểm importance cho từng cầu thủ tham gia (tổng hợp từ trọng số các cạnh liên quan đến cầu thủ đó).

### 7.3. Trực quan hóa
- Vẽ lại possession gốc trên sân, tô đậm các đường chuyền/cầu thủ thuộc subgraph quan trọng — phục vụ minh họa trực quan trong báo cáo và kiểm tra định tính (sanity check) xem subgraph có hợp lý về mặt bóng đá hay không.

---

## 8. Đánh giá chất lượng XAI

- **Fidelity+/Fidelity−**: đo mức thay đổi của xác suất dự đoán khi chỉ giữ lại (hoặc chỉ loại bỏ) subgraph được xác định là quan trọng.
- **Sparsity**: subgraph giải thích càng gọn (ít cạnh/node) mà vẫn giữ fidelity cao thì càng tốt.
- **Stability/Robustness**: chạy lại XAI nhiều lần hoặc thêm nhiễu nhẹ vào input, kiểm tra subgraph quan trọng có ổn định không.
- **ROAR / Insertion-Deletion curve**: xóa hoặc thêm dần các cạnh theo thứ tự importance, quan sát đường cong suy giảm/tăng hiệu năng dự đoán để đánh giá gián tiếp chất lượng giải thích.
- So sánh GNNExplainer vs PGExplainer trên chính bộ độ đo này để chọn phương pháp (hoặc kết hợp cả hai) đưa vào bước tính Hybrid Centrality.

---

## 9. Tính chỉ số Hybrid Centrality

### 9.1. Thành phần 1 — Centrality truyền thống
- Tính trên đồ thị chuyền bóng tổng hợp (aggregate passing network) của mỗi cầu thủ theo trận hoặc theo mùa giải:
  - Degree centrality (in/out).
  - Betweenness centrality.
  - Closeness centrality.
  - Eigenvector centrality.

### 9.2. Thành phần 2 — XAI-importance score
- Với mỗi cầu thủ, tổng hợp (trung bình hoặc trung bình có trọng số) điểm importance mà XAI gán cho cầu thủ đó qua **tất cả các possession** cầu thủ tham gia, thay vì chỉ dựa vào một possession đơn lẻ — đảm bảo chỉ số phản ánh mức độ quan trọng mang tính hệ thống, không bị nhiễu bởi một tình huống cá biệt.

### 9.3. Công thức kết hợp
- Chuẩn hóa (min-max hoặc z-score) cả hai thành phần về cùng thang đo trước khi kết hợp, tránh trường hợp một thành phần có phương sai lớn lấn át thành phần còn lại.
- Kết hợp theo trọng số: `HybridCentrality = α × Centrality_norm + (1 − α) × XAIImportance_norm`, với `α` là siêu tham số cần được xác định thực nghiệm (ví dụ dò trên tập validation dựa vào mức độ tương quan với một tiêu chí đối chiếu ở bước 9.4) — đây là phần cần làm rõ và trình bày công thức chính thức trong báo cáo/paper.

### 9.4. Kiểm chứng Hybrid Centrality
- So sánh **Hybrid Centrality** với **centrality truyền thống thuần túy** (baseline Tầng 3(b)) để chứng minh giá trị gia tăng.
- Đối chiếu định lượng với các chỉ số thực tế đã được công nhận trong bóng đá (key passes, assists, xA, xT/expected threat) để kiểm tra tính hợp lý (face validity) của chỉ số mới.
- Có thể bổ sung đánh giá định tính: đối chiếu bảng xếp hạng Hybrid Centrality với nhận định thực tế về vai trò của một số cầu thủ nổi bật trong dữ liệu, để tăng tính thuyết phục khi trình bày trước hội đồng.

---

## 10. Trực quan hóa & Kết quả đầu ra

- **Bảng xếp hạng cầu thủ** theo Hybrid Centrality (theo trận / theo mùa / theo đội).
- **Passing network diagram** có tô trọng số theo Hybrid Centrality (kích thước node theo chỉ số).
- **Overlay subgraph quan trọng** lên sơ đồ possession cụ thể (dùng để minh họa case study trong báo cáo).
- **Bảng so sánh hiệu năng** giữa các backbone (Tầng 2) và giữa mô hình đề xuất với các baseline phi đồ thị (Tầng 3(a)).
- **Bảng so sánh chất lượng XAI** giữa GNNExplainer và PGExplainer, và giữa Hybrid Centrality với centrality truyền thống (Tầng 3(b)).
- Tổng hợp toàn bộ thành báo cáo khóa luận / bài báo khoa học, với phần Methodology bám theo đúng thứ tự các bước 1–9 ở trên.

---

## 11. Đề xuất công cụ & thư viện

| Giai đoạn | Công cụ đề xuất |
|---|---|
| Thu thập dữ liệu | `statsbombpy`, `requests` |
| Xử lý dữ liệu | `pandas`, `numpy` |
| Xây dựng đồ thị | `networkx` (đồ thị centrality truyền thống), `torch_geometric` hoặc `dgl` (đồ thị cho GNN) |
| Huấn luyện GNN | `PyTorch` + `PyTorch Geometric` (hoặc `DGL`) |
| XAI | `torch_geometric.explain` (đã tích hợp GNNExplainer, PGExplainer) |
| Centrality truyền thống | `networkx` |
| Trực quan hóa | `matplotlib`, `mplsoccer` (vẽ sân bóng đá), `plotly` (dashboard tương tác nếu cần) |
| Quản lý thực nghiệm | `mlflow` hoặc ghi log thủ công bằng CSV/JSON có versioning |

---

## 12. Rủi ro & lưu ý khi triển khai

- **Định nghĩa nhãn (2.3)** là quyết định ảnh hưởng dây chuyền đến toàn bộ pipeline — cần chốt sớm và nhất quán, không thay đổi giữa chừng sau khi đã huấn luyện mô hình.
- **Leakage giữa train/test** dễ xảy ra nếu chia theo possession thay vì theo trận — luôn kiểm tra lại danh sách `match_id` trước khi huấn luyện.
- **Mất cân bằng lớp** cần được xử lý nhất quán ở cả bước huấn luyện (loss weighting) lẫn bước đánh giá (ưu tiên AUC-PR, F1 hơn Accuracy).
- **Overfitting của XAI lên nhiễu dữ liệu**: cần luôn kiểm tra Stability (mục 8) trước khi tin tưởng subgraph quan trọng, tránh diễn giải quá mức (over-interpretation) những gì có thể chỉ là nhiễu.
- **Hệ số α trong Hybrid Centrality (9.3)** không nên chọn tùy ý — cần trình bày rõ cách xác định (dò tham số + tiêu chí lựa chọn) để hội đồng không đặt câu hỏi về tính "thủ công, thiếu căn cứ" của công thức.
