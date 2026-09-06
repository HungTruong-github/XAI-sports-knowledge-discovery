"""Bước 6 — Đánh giá Bài toán 1 (dự đoán).

Bộ độ đo đã thống nhất với giảng viên (mục 1.2a — phản hồi GV 25/08):
    Accuracy (chỉ tham khảo), Precision, Recall, F1,
    AUC-ROC, AUC-PR (độ đo chính do mất cân bằng lớp),
    Brier score (calibration).

Module dự kiến:
    metrics.py      Tính toàn bộ bộ độ đo trên
    calibration.py  Brier score + reliability diagram
    compare.py      Bảng đối sánh Tầng 2 (backbone) và Tầng 3a (phi đồ thị)
"""
