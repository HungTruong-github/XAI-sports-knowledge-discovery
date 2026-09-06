"""Tiện ích dùng chung.

Module dự kiến:
    paths.py    Đường dẫn chuẩn tới data/, models/, results/... (một nguồn duy nhất)
    seed.py     Cố định seed cho random/numpy/torch — điều kiện bắt buộc để
                tái lập kết quả khi đối sánh giữa các backbone (Bước 4)
    io.py       Đọc/ghi JSON, YAML, checkpoint
    logging.py  Cấu hình log thống nhất
"""
