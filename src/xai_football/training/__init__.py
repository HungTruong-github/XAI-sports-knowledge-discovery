"""Bước 5.4-5.6 — Huấn luyện mô hình.

Module dự kiến:
    losses.py     Weighted BCE / Focal Loss (xử lý mất cân bằng lớp — Bước 2.3)
    trainer.py    Vòng huấn luyện, Adam/AdamW, LR scheduler,
                  early stopping theo AUC-PR trên tập validation
    tuning.py     Grid/random search siêu tham số — CHỈ chạy trên tập
                  validation, tuyệt đối không đụng vào tập test
    tracking.py   Ghi log thực nghiệm (mlflow hoặc CSV/JSON có versioning)
"""
