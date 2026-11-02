# BÁO CÁO TÁI LẬP GIAI ĐOẠN 5: TRỰC QUAN HÓA, TÀI LIỆU KHÓA LUẬN VÀ KIỂM TOÁN TỔNG THỂ

**Dự án:** XAI Football — Graph Neural Networks & Explainable AI on Passing Networks  
**Nhánh:** `feature/08-xai-completion`  
**Ngày hoàn thành:** 2026-11-02  
**Mục tiêu:** Sinh toàn bộ hình ảnh biểu đồ học thuật chất lượng cao, hoàn thiện bản thảo chương Kết quả và Thảo luận của Khóa luận (`docs/thesis/ket_qua_va_thao_luan.md`), cập nhật `README.md` và kiểm toán tính toàn vẹn tái lập của toàn bộ Pipeline từ Bước 0 đến Bước 10.

---

## 1. Danh mục Biểu đồ và Hình ảnh Trực quan hóa Chính thức (PNG & SVG)

Đã tạo thành công 7 bộ biểu đồ chính thức được lưu trữ kép (PNG độ phân giải cao 300 DPI và vector SVG) tại thư mục `results/figures/official/`:

1. `07_readout_comparison.png` / `.svg`: Biểu đồ cột có thanh sai số thể hiện Validation AUC-PR và Test AUC-PR qua 5 seeds của 4 biến thể Readout Pooling MPNN (`mean`, `max`, `attention`, `mean_lastedge`).
2. `08_xai_fidelity_comparison.png` / `.svg`: Đối sánh Fidelity+, Fidelity- và kiểm định Cổng chất lượng Gate D17 giữa 4 thuật toán giải thích (`gnn_explainer`, `pg_explainer`, `random`, `last_edge`).
3. `08_xai_sparsity_fidelity_curves.png` / `.svg`: Đường cong quan hệ giữa độ thưa mặt nạ (Sparsity) và độ trung thực (Fidelity+ / Fidelity-).
4. `09_hybrid_alpha_optimization.png` / `.svg`: Đường cong quét tham số $\alpha \in [0.0, 1.0]$ trên tập Set A với điểm cực trị $\alpha^* = 0.70$ tối đa hóa tương quan với Expected Threat ($xT/90$).
5. `09_hybrid_vs_centrality_scatter.png` / `.svg`: Biểu đồ phân tán trên 481 cầu thủ tập Set B giữa Betweenness Centrality và Hybrid Centrality, tô màu theo $xT/90$ và kích thước điểm theo số lượng đường chuyền quyết định (Key Passes), có gắn nhãn các ngôi sao hàng đầu.
6. `10_top_players_ranking_shift.png` / `.svg`: Biểu đồ Dumbbell thể hiện sự chuyển dịch thứ bậc Percentile Rank giữa Centrality truyền thống và Hybrid Centrality của Top 12 cầu thủ.
7. `10_case_study_pitch_plots.png` / `.svg`: Trực quan hóa sân bóng 2D ($120 \times 80$) cho 6 trường hợp chuỗi chuyền bóng thực tế dẫn tới dứt điểm của Mesut Özil, Dimitri Payet, Christian Eriksen, Santi Cazorla, Cesc Fàbregas, và Troy Deeney, với màu sắc và độ dày mũi tên thể hiện điểm quan trọng XAI.

---

## 2. Hoàn thiện Tài liệu Khóa luận

Đã xây dựng hoàn chỉnh văn bản học thuật `docs/thesis/ket_qua_va_thao_luan.md` với dung lượng đầy đủ, văn phong học thuật chuẩn mực, trả lời rốt ráo 3 câu hỏi nghiên cứu cốt lõi:
- **Phần 4.1:** Đặt vấn đề và 3 câu hỏi nghiên cứu (RQ1, RQ2, RQ3).
- **Phần 4.2:** Kết quả dự đoán cơ hội dứt điểm (RQ1), đối sánh GNN với Baseline phi đồ thị, phân tích nguyên nhân và thực thi Quyết định D4.
- **Phần 4.3:** Đánh giá chất lượng giải thích XAI (RQ2), kiểm định Gate D17 với khoảng tin cậy 95% Bootstrap, phân tích bản chất chi phối của đường chuyền kết thúc chuỗi bóng.
- **Phần 4.4:** Đóng góp nghiên cứu chính: Chỉ số Hybrid Centrality (RQ3), tối ưu $\alpha^* = 0.70$, đánh giá mù trên Set B, kiểm định hồi quy OLS $\Delta R^2$ và phân tích chuyên môn chi tiết các cầu thủ dẫn đầu.
- **Phần 4.5:** Tổng kết các đóng góp lý thuyết và thực tiễn của đề tài.

---

## 3. Cập nhật Bảng Tiến độ Pipeline trong README.md

Bảng tiến độ tại `README.md` đã được cập nhật chính thức với 100% các bước từ Bước 0 đến Bước 10 đạt trạng thái **Đã hoàn thành**, cùng các Decision Records D1 đến D6 được chốt minh bạch.

---

## 4. Lệnh tái lập (Reproducibility Commands)

```bash
# Sinh toàn bộ 7 bộ hình ảnh trực quan hóa học thuật
python scripts/12_generate_all_figures.py
```
