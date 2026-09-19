"""Cố định seed ngẫu nhiên.

Điều kiện bắt buộc để tái lập kết quả và để việc đối sánh giữa các backbone
ở Tầng 2 là công bằng (mục 2.3 báo cáo 01/09 — danh sách mã trận của ba tập
phải cố định kèm seed).
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Cố định seed cho random, numpy và torch (nếu đã cài).

    Trả lại chính seed để ghi vào log thực nghiệm.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
