# results/ — Kết quả thực nghiệm, Bảng số liệu & Xuất xứ (Provenance)

Thư mục `results/` lưu trữ toàn bộ các bảng số liệu, biểu đồ trực quan hóa và siêu dữ liệu xuất xứ (provenance metadata) phục vụ cho báo cáo khóa luận và công bố khoa học. Khác với `experiments/` (chứa log thô, bị `.gitignore`), thư mục này được theo dõi phiên bản bằng Git để đảm bảo tính minh bạch và khả năng tái lập.

---

## 1. Cấu trúc thư mục

```
results/
├── figures/official/             # Biểu đồ định dạng PNG (300 DPI) và SVG
├── metadata/                     # Siêu dữ liệu hệ thống, môi trường và cấu hình
│   ├── RUN_REGISTRY.json         # Danh bạ đăng ký trung tâm toàn bộ các runs thực nghiệm
│   └── official/                 # Hash cấu hình, môi trường và commit của run chính thức
├── tables/                       # Bảng số liệu tổng hợp
│   ├── official/                 # Bảng kết quả chính thức đã được xác thực (CANONICAL)
│   │   ├── model_comparison.csv  # So sánh 4 GNN + 3 Baselines (Test set, multi-seed)
│   │   ├── significance_tests.csv# Kiểm định ý nghĩa thống kê (Bootstrap, Holm-Bonferroni)
│   │   └── xai_comparison.csv    # Bảng đánh giá XAI chính thức (được bảo vệ)
│   └── pilot/                    # Bảng kết quả chạy thử nghiệm / pilot
├── xai/                          # Đồ thị con giải thích và đánh giá chi tiết XAI
│   └── official/mpnn/
│       └── 20261017_091540/      # Artifact lịch sử (5 đồ thị kiểm tra trên branch REPRODUCIBILITY)
└── reproducibility/              # Tài liệu đối soát tính tái lập
```

---

## 2. Nguyên tắc Xuất xứ & Chống nhầm lẫn số liệu (Provenance Integrity)

### 2.1. Đăng ký Run trung tâm (`RUN_REGISTRY.json`)
Mọi lần chạy thực nghiệm (GNN, Baselines, XAI) đều được ghi nhận vào `results/metadata/RUN_REGISTRY.json` với các trường bắt buộc:
- `run_id`: Định danh duy nhất theo timestamp (ví dụ: `20261017_091540` hoặc commit-based `official_gcn_mpnn_gatv2_graphsage_20261021`).
- `run_type`: Phân loại rõ ràng:
  - `official`: Chạy đầy đủ 5 seeds, toàn bộ tập Test, dùng `configs/data.yaml` chính thức.
  - `pilot`: Chạy trên tập dữ liệu rút gọn `configs/data_pilot.yaml`.
  - `smoke`: Chạy kiểm tra nhanh 1 epoch, subset nhỏ.
  - `historical_provisional`: Run lịch sử đã thực hiện trong quá khứ, giữ lại làm bằng chứng đối soát.
- `status`: `complete`, `partial`, hoặc `historical_provisional`.
- `git_commit`: Mã SHA của commit tại thời điểm thực nghiệm.
- `config_hashes`: SHA-256 của các file cấu hình tương ứng.
- `checkpoint_sha256`: Hash của file trọng số mô hình đã nạp.
- `n_graphs_explained`: Số lượng đồ thị thực tế được đánh giá.

### 2.2. Cơ chế bảo vệ bảng số liệu chính thức
1. **Không tự động chọn run mới nhất theo tên thư mục**:
   - `scripts/09_evaluate_xai.py` yêu cầu bắt buộc truyền rõ `--run-id` hoặc `--xai-dir`.
   - Nếu không truyền tham số, script sẽ liệt kê bảng danh sách các run hợp lệ kèm trạng thái và dừng lại với mã lỗi (exit code 1).
2. **Ngăn chặn Pilot/Smoke/Partial ghi đè Official**:
   - Bất kỳ run nào có nhãn `pilot`, `smoke`, hoặc `is_debug_run: true` đều bị từ chối tổng hợp vào bảng `official/xai_comparison.csv` hay `official/model_comparison.csv`.
   - Trong `scripts/07_evaluate.py`, nếu sử dụng cờ `--allow-partial` để đánh giá mô hình chưa hoàn thành đủ 5 seeds, kết quả bắt buộc phải ghi ra file riêng: `results/tables/official/partial_model_comparison.csv`, tuyệt đối không ghi đè lên `model_comparison.csv`.
3. **Quản lý Artifact lịch sử an toàn**:
   - Run XAI `results/xai/official/mpnn/20261017_091540/` được tạo trên branch `REPRODUCIBILITY` (commit `9fae6c51`) với 5 đồ thị mẫu. Run này được bảo toàn nguyên vẹn làm minh chứng thực nghiệm lịch sử.
   - Để xuất bảng tổng hợp cho run này, người dùng phải truyền cờ `--allow-historical`. Khi đó, script sẽ xuất ra file riêng `results/tables/official/xai_comparison_historical_20261017_091540.csv` kèm cột `status = HISTORICAL_PROVISIONAL`, không làm sai lệch bảng chính thức.
4. **Ghi file an toàn nguyên tử (Atomic Writes)**:
   - Tất cả các script tổng hợp số liệu sử dụng hàm `save_atomic_csv` và `save_atomic_json` (trong `xai_football.utils.io`).
   - Dữ liệu được ghi ra file tạm (`.tmp`) rồi mới thay thế nguyên tử bằng `os.replace`. Nếu xảy ra crash hoặc ngắt tiến trình giữa chừng, file kết quả chính thức cũ không bao giờ bị cắt cụt hay hỏng hóc.

---

## 3. Danh mục các bảng kết quả hiện có

| File | Phân loại | Run ID / Nguồn gốc | Số mẫu / Seeds | Ghi chú |
|---|---|---|---|---|
| `tables/official/model_comparison.csv` | Official | `fc18a929` (branch `main`) | 5 seeds (42, 123, 456, 789, 1024) | 4 GNN + 3 Baselines, Test set (6.646 possessions). MPNN đạt AUC-PR 0.5996 ± 0.0042. |
| `tables/official/significance_tests.csv` | Official | `fc18a929` | 5 seeds | Kiểm định ghép cặp bootstrap theo trận, điều chỉnh Holm-Bonferroni. |
| `tables/official/xai_comparison_historical_20261017_091540.csv` | Historical Provisional | `20261017_091540` (`9fae6c51`, `REPRODUCIBILITY`) | 5 test graphs | Thử nghiệm sơ bộ GNNExplainer vs RandomExplainer trên MPNN. |
| `tables/pilot/model_comparison.csv` | Pilot | `configs/data_pilot.yaml` | 1 seed (42) | Kết quả kiểm thử nhanh pipeline trên tập con 2 trận. |
