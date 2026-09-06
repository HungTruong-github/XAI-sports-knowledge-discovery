# XAI-sports-knowledge-discovery

**Hệ thống trí tuệ nhân tạo có thể giải thích (XAI) trong bài toán đánh giá và
khám phá tri thức thể thao**

Hướng nghiên cứu cụ thể: *Mạng chuyền bóng và phân tích chiến thuật bóng đá
bằng Graph Neural Network (GNN) kết hợp XAI.*

---

## Bài toán

Đề tài gồm **hai bài toán con nối tiếp nhau**, đã được xác nhận với giảng viên
hướng dẫn (xem `docs/meetings/`):

**Bài toán 1 — Dự đoán.** Biểu diễn một chuỗi tấn công (possession) thành đồ
thị chuyền bóng có hướng (đỉnh = cầu thủ, cạnh = đường chuyền), mô hình GNN dự
đoán xác suất chuỗi đó dẫn đến cú sút. Đây là bài toán phân loại nhị phân ở
**mức đồ thị**.

**Bài toán 2 — Giải thích.** Trên mô hình đã huấn luyện, áp dụng XAI cho đồ thị
để (a) xác định subgraph quan trọng, và (b) tổng hợp điểm quan trọng qua nhiều
possession thành chỉ số **Hybrid Centrality** cho từng cầu thủ.

Bài toán 2 phụ thuộc kết quả Bài toán 1: phải có mô hình dự đoán đáng tin cậy
thì giải thích của XAI mới có ý nghĩa.

---

## Cấu trúc dự án

```
.
├── configs/              # Toàn bộ tham số thực nghiệm (YAML)
│   ├── data.yaml             Bước 1-2: thu thập, tiền xử lý, gán nhãn
│   ├── graph.yaml            Bước 3:   định nghĩa đồ thị possession
│   ├── split.yaml            Bước 4:   chia train/val/test theo trận
│   ├── train.yaml            Bước 5:   cấu hình huấn luyện dùng chung
│   ├── xai.yaml              Bước 7-8: explainer và độ đo XAI
│   ├── hybrid_centrality.yaml Bước 9:  công thức Hybrid Centrality
│   └── model/                Kiến trúc từng backbone và baseline
│
├── data/                 # Dữ liệu (KHÔNG commit, trừ splits/)
│   ├── raw/                  JSON thô từ StatsBomb, giữ nguyên
│   ├── interim/              Possession đã làm sạch, đã gán nhãn
│   ├── processed/            Đồ thị PyG đã đóng gói (.pt)
│   ├── splits/               Danh sách match_id từng tập  ← CÓ commit
│   └── external/             Chỉ số tham chiếu (xA, xT...) để kiểm chứng
│
├── src/xai_football/     # Mã nguồn chính (package cài bằng pip -e .)
│   ├── data/                 Bước 1-2, 4
│   ├── graphs/               Bước 3
│   ├── modeling/             Bước 5  (gnn/ + baselines/)
│   ├── training/             Bước 5.4-5.6
│   ├── evaluation/           Bước 6
│   ├── xai/                  Bước 7-8
│   ├── centrality/           Bước 9-10
│   ├── visualization/        Bước 10
│   └── utils/                seed, đường dẫn, I/O, log
│
├── scripts/              # Điểm vào chạy từng bước pipeline
├── notebooks/            # Thăm dò dữ liệu, trực quan hóa
├── tests/                # Kiểm thử (ưu tiên: chống rò rỉ dữ liệu)
│
├── experiments/          # Log thô từng lần chạy      (KHÔNG commit)
├── models/               # Trọng số đã huấn luyện     (KHÔNG commit)
├── results/              # Bảng + hình cho khóa luận  (CÓ commit)
│
├── docs/                 # Tài liệu nghiên cứu
│   ├── XAI-Football_Project_Content.md   Bối cảnh, khoảng trống, thuật ngữ
│   ├── Quy_trinh_XAI_Football.md         Quy trình 12 bước chi tiết
│   ├── decisions/                        Quyết định nghiên cứu + lý do
│   ├── meetings/                         Báo cáo tuần, nhận xét GV
│   ├── figures/                          Sơ đồ cho tài liệu
│   └── thesis/                           Các chương khóa luận
│
├── backend/              # API demo   (Giai đoạn 2, chưa triển khai)
└── frontend/             # Giao diện  (Giai đoạn 2, chưa triển khai)
```

Ba thư mục dễ nhầm: **`experiments/`** chứa log thô từng lần chạy,
**`models/`** chứa trọng số, **`results/`** chứa bảng và hình cuối cùng đưa
vào khóa luận. Chỉ `results/` được commit.

---

## Pipeline

```
[1] Thu thập dữ liệu (StatsBomb Open Data)
[2] Tiền xử lý & gán nhãn (possession → sút/bàn thắng)
[3] Xây dựng đồ thị chuyền bóng
[4] Chia train/val/test theo TRẬN (chống rò rỉ dữ liệu)
[5] Huấn luyện GNN                        ← Bài toán 1
[6] Đánh giá dự đoán + baseline 3 tầng
[7] Áp dụng XAI trên mô hình đã huấn luyện ← Bài toán 2
[8] Đánh giá chất lượng XAI
[9] Tính Hybrid Centrality
[10] Kiểm chứng & trực quan hóa
```

Chi tiết đầy đủ: [`docs/Quy_trinh_XAI_Football.md`](docs/Quy_trinh_XAI_Football.md)

### Chiến lược đối sánh 3 tầng

Theo đúng hướng giảng viên gợi ý (buổi 25/08/2026):

- **Tầng 1** — Rà soát: chưa có công trình nào giải đồng thời cả ba thành phần
  (dự đoán từ đồ thị chuyền bóng open-play + subgraph XAI + centrality hợp
  nhất). Gần nhất là TacticAI nhưng chỉ áp dụng cho tình huống phạt góc.
- **Tầng 2** — So sánh backbone: GCN / MPNN (đề xuất) vs GATv2 / GraphSAGE,
  cùng dữ liệu, cùng cách chia tập, cùng cấu hình huấn luyện.
- **Tầng 3** — Tách hai nhóm baseline:
  - *(a) Dự đoán:* thêm baseline phi đồ thị (Logistic Regression, Random
    Forest, ANN trên đặc trưng passing path)
  - *(b) XAI:* GNNExplainer vs PGExplainer, cộng baseline "không dùng XAI"
    là centrality truyền thống thuần túy, và SHAP ở mức đặc trưng

---

## Cài đặt

```bash
# 1. Tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Cài PyTorch trước, theo đúng phiên bản CUDA của máy
#    https://pytorch.org/get-started/locally/

# 3. Cài phần còn lại
pip install -r requirements.txt

# 4. Cài package ở chế độ dev — chạy MỘT LẦN, sau đó mọi notebook và
#    script đều import được `xai_football` mà không cần sửa sys.path
pip install -e .
```

---

## Trạng thái hiện tại

Dự án đang ở **giai đoạn thiết kế**. Cấu trúc thư mục và cấu hình đã dựng
xong, **chưa triển khai mã nguồn**.

### Hai quyết định đang chặn việc bắt đầu code

| # | Quyết định | Vì sao phải chốt trước |
|---|---|---|
| [0001](docs/decisions/0001-quy-tac-gan-nhan.md) | Quy tắc gán nhãn possession | Ảnh hưởng dây chuyền tới cách dựng đồ thị ở Bước 3 và toàn bộ pipeline phía sau; không được đổi sau khi đã huấn luyện |
| [0002](docs/decisions/0002-cong-thuc-hybrid-centrality.md) | Công thức Hybrid Centrality | Là đóng góp chính của đề tài nhưng chưa có công thức toán học chính thức; hệ số alpha phải có căn cứ rõ ràng |

Các giá trị `null` kèm ghi chú `TODO` trong `configs/` tương ứng với hai
quyết định này — cố ý để trống thay vì điền bừa.

---

## Nguyên tắc nghiên cứu

Trích từ [`docs/XAI-Football_Project_Content.md`](docs/XAI-Football_Project_Content.md)
mục 18 — **không được mặc định** rằng:

- GNN sẽ tốt hơn mô hình phi đồ thị
- GCN hoặc MPNN là mô hình cuối cùng
- GNNExplainer tốt hơn PGExplainer
- Hybrid Centrality là hợp lệ trước khi được kiểm chứng
- Một đặc trưng có điểm XAI cao là **nguyên nhân** gây ra kết quả

Mọi khẳng định phải có bằng chứng thực nghiệm hoặc tài liệu đáng tin cậy.

Ngoài ra: điểm importance của XAI phản ánh **hành vi của mô hình**, không phải
quan hệ nhân quả trong bóng đá. Cần kiểm tra Stability trước khi diễn giải bất
kỳ subgraph nào.

---

## Nguồn dữ liệu

[StatsBomb Open Data](https://github.com/statsbomb/open-data) — Bundesliga
2023/24 và UEFA Champions League. Cần trích dẫn đúng StatsBomb trong khóa luận
và bài báo theo điều khoản sử dụng của họ.

## Giấy phép

Xem [LICENSE](LICENSE).
