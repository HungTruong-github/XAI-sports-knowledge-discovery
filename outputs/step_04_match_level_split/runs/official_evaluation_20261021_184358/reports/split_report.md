# Báo cáo Phân chia Tập Dữ liệu theo Trận đấu (Bước 4)

- **Chiến lược phân chia**: Phân tầng theo TRẬN ĐẤU (Match-level stratified split) nhằm ngăn chặn triệt để rò rỉ dữ liệu.
- **Tỷ lệ phân chia**:
  - Train: 266 trận (70%) — 31.644 đồ thị
  - Validation: 57 trận (15%) — 6.757 đồ thị
  - Test: 57 trận (15%) — 6.646 đồ thị
- **Kiểm định tính bất giao (Disjoint Verification)**: 0 overlap between Train (266), Val (57), Test (57)
- **Tính đồng đều tỷ lệ nhãn dương**: Cả 3 tập duy trì tỷ lệ nhãn shot ổn định quanh 14,26% ($\pm 0,15\%$).
