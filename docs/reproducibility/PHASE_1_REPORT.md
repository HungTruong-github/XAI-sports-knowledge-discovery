# BÁO CÁO TÁI LẬP GIAI ĐOẠN 1: HUẤN LUYỆN VÀ ĐỐI SÁNH READOUT POOLING CHO MPNN

**Dự án:** XAI Football — Graph Neural Networks & Explainable AI on Passing Networks  
**Nhánh:** `feature/08-xai-completion`  
**Ngày hoàn thành:** 2026-11-02  
**Mục tiêu:** Huấn luyện 3 biến thể Readout Pooling của MPNN (`max`, `attention`, `mean_lastedge`) qua 5 seeds ngẫu nhiên cố định (42, 123, 456, 789, 1024), so sánh với baseline `mean` pooling, và thực thi Quyết định D4 để chọn backbone chính thức cho XAI và Hybrid Centrality.

---

## 1. Thiết kế thực nghiệm và Tiêu chí Quyết định D4

Căn cứ theo quyết định tiền nghiệm tại `docs/decisions/0005-readout-mpnn.md`:
- Mô hình đồ thị tốt nhất tại Bước 5 là MPNN với cơ chế `mean` pooling.
- Để kiểm tra giả thuyết liệu cơ chế gộp đồ thị có thể cải thiện hiệu năng dự đoán possession dẫn đến cú sút, 3 biến thể được xây dựng:
  1. `mpnn_max`: Global Max Pooling trên các đỉnh đồ thị.
  2. `mpnn_attn`: Global Attention Pooling học trọng số đóng góp của từng đỉnh.
  3. `mpnn_lastedge`: Ghép nối Mean Pooling với biểu diễn của đỉnh nhận bóng ở đường chuyền cuối cùng (last-edge recipient node).
- **Quy tắc D4:** Nếu biến thể tốt nhất vượt trội hơn `mean` pooling $\ge 0.0100$ AUC-PR trên tập Validation (trung bình 5 seeds), biến thể đó được chọn làm backbone chính thức. Ngược lại, giữ nguyên `mpnn` (mean pooling).

---

## 2. Kết quả đối sánh chi tiết (Bảng số liệu 5 seeds)

Nguồn dữ liệu: `results/tables/official/readout_comparison.csv`

| Mô hình | Readout Pooling | Val AUC-PR (mean ± std) | Test AUC-PR (mean ± std) | Test AUC-ROC (mean ± std) | Test Brier Score | Best Seed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`mpnn`** (baseline) | `mean` | 0.5847 ± 0.0015 | **0.5996 ± 0.0042** | 0.8757 ± 0.0034 | 0.1302 ± 0.0114 | 789 |
| `mpnn_max` | `max` | 0.5624 ± 0.0031 | 0.5853 ± 0.0048 | 0.8811 ± 0.0010 | 0.1424 ± 0.0158 | 42 |
| `mpnn_attn` | `attention` | 0.5620 ± 0.0061 | 0.5820 ± 0.0074 | 0.8786 ± 0.0032 | 0.1336 ± 0.0114 | 1024 |
| `mpnn_lastedge` | `mean_lastedge` | **0.5921 ± 0.0074** | 0.5973 ± 0.0082 | **0.8852 ± 0.0018** | **0.1256 ± 0.0138** | 1024 |

### Đánh giá Robustness Subsets trên tập Test (Seed tốt nhất của mỗi mô hình)

| Mô hình | AUC-PR (2–3 chuyền) | AUC-PR (4–5 chuyền) | AUC-PR ($\ge 6$ chuyền) | AUC-PR (Open-play) |
| :--- | :---: | :---: | :---: | :---: |
| `mpnn` | 0.5689 | 0.6037 | 0.5693 | 0.5843 |
| `mpnn_max` | 0.5916 | 0.6081 | 0.5804 | 0.6028 |
| `mpnn_attn` | 0.5983 | 0.6136 | 0.5818 | 0.6098 |
| `mpnn_lastedge` | **0.6074** | **0.6464** | **0.6054** | **0.6202** |

---

## 3. Thực thi Quyết định D4

- Giá trị trung bình Validation AUC-PR của `mpnn_lastedge`: $0.5921$.
- Giá trị trung bình Validation AUC-PR của `mpnn` baseline: $0.5847$.
- Chênh lệch cải thiện: $\Delta = 0.5921 - 0.5847 = +0.0074$.
- So với ngưỡng kiểm định $+0.0100$: Chênh lệch $+0.0074 < +0.0100$.
- **Kết luận Quyết định D4:**
  - Không thỏa mãn điều kiện vượt trội tối thiểu $\ge 0.0100$ trên Validation set.
  - **Giữ nguyên `mpnn` (mean pooling) làm backbone chính thức** cho các bước giải thích mô hình (Bước 8-9) và tính toán chỉ số Hybrid Centrality (Bước 10).
  - Ghi nhận `mpnn_lastedge` là biến thể kiến trúc đồ thị có AUC-ROC cao nhất ($0.8852$) và Brier score tốt nhất ($0.1256$), đồng thời ở seed 1024 đạt Test AUC-PR $0.6107$, mở ra hướng phân tích sâu về vai trò của đường chuyền kết thúc chuỗi bóng.

---

## 4. Danh mục Artifacts và Checkpoints đã tạo

1. **Bảng kết quả & Manifest:**
   - `results/tables/official/readout_comparison.csv`
   - `results/tables/official/readout_comparison_manifest.json`
2. **Cấu hình mô hình:**
   - `configs/model/mpnn_attn.yaml`
   - `configs/model/mpnn_max.yaml`
   - `configs/model/mpnn_lastedge.yaml`
3. **Mã nguồn bổ sung:**
   - `src/xai_football/modeling/gnn/readout.py` (hỗ trợ `MeanLastEdgeReadout`)
   - `src/xai_football/modeling/gnn/mpnn.py` (tích hợp pooling dynamic)
   - `scripts/evaluate_readouts.py` (đối sánh và kiểm toán tự động)
4. **Trọng số mô hình chính thức (5 seeds per variant):**
   - `models/official/mpnn_best.pt`
   - `models/official/mpnn_lastedge_best.pt`
   - `models/official/mpnn_max_best.pt`
   - `models/official/mpnn_attn_best.pt`
   - Các file checkpoint từng seed trong `experiments/official/mpnn_*/seed_*/best_model.pt`

---

## 5. Lệnh tái lập (Reproducibility Commands)

```bash
# Huấn luyện 5 seeds cho từng biến thể
python scripts/06b_train_gnn_multi_seeds.py --model mpnn_max --epochs 25
python scripts/06b_train_gnn_multi_seeds.py --model mpnn_attn --epochs 25
python scripts/06b_train_gnn_multi_seeds.py --model mpnn_lastedge --epochs 25

# Đánh giá và sinh bảng đối sánh, manifest
python scripts/evaluate_readouts.py
```

