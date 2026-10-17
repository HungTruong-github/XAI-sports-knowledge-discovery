"""Bộ khởi tạo Optimizer và Learning Rate Scheduler (Factory Pattern).

Tham chiếu: configs/train.yaml -> optimizer, scheduler.
Đảm bảo mọi cấu hình trong YAML được phản ánh chính xác hoặc bị từ chối tường minh.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

SUPPORTED_OPTIMIZERS = {"adam", "adamw", "sgd"}
SUPPORTED_SCHEDULERS = {"reduce_on_plateau", "reducelronplateau", "cosine", "step", "none", None}


def build_optimizer(
    model: nn.Module, opt_cfg: dict[str, Any] | None = None
) -> torch.optim.Optimizer:
    """Tạo optimizer từ cấu hình train.yaml.

    Args:
        model: Mô hình PyTorch cần tối ưu.
        opt_cfg: Cấu hình optimizer (name, lr, weight_decay, momentum, ...).

    Returns:
        torch.optim.Optimizer instance.

    Raises:
        ValueError: Nếu tên optimizer không được hỗ trợ.
    """
    cfg = opt_cfg or {}
    name = str(cfg.get("name", "adamw")).strip().lower()
    lr = float(cfg.get("lr", 0.001))
    weight_decay = float(cfg.get("weight_decay", 0.0001))

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    else:
        raise ValueError(
            f"Optimizer '{name}' không được hỗ trợ. Các thuật toán được hỗ trợ: "
            f"{sorted(SUPPORTED_OPTIMIZERS)}"
        )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_cfg: dict[str, Any] | None = None,
    epochs: int = 200,
) -> Any:
    """Tạo Learning Rate Scheduler từ cấu hình train.yaml.

    Args:
        optimizer: Optimizer liên kết với scheduler.
        sched_cfg: Cấu hình scheduler (name, monitor, factor, patience, step_size, gamma).
        epochs: Tổng số epochs (cần thiết cho CosineAnnealingLR).

    Returns:
        LR scheduler instance hoặc None nếu name là 'none' hoặc không cấu hình.

    Raises:
        ValueError: Nếu tên scheduler không được hỗ trợ.
    """
    if not sched_cfg:
        return None

    name_raw = sched_cfg.get("name")
    if name_raw is None:
        return None

    name = str(name_raw).strip().lower()
    if name in ("none", "null", ""):
        return None

    if name in ("reduce_on_plateau", "reducelronplateau"):
        monitor = str(sched_cfg.get("monitor", "val_auc_pr"))
        mode = "min" if "loss" in monitor else "max"
        factor = float(sched_cfg.get("factor", 0.5))
        patience = int(sched_cfg.get("patience", 10))
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    elif name in ("cosine", "cosine_annealing"):
        t_max = int(sched_cfg.get("t_max", epochs))
        eta_min = float(sched_cfg.get("eta_min", 1e-6))
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif name in ("step", "step_lr"):
        step_size = int(sched_cfg.get("step_size", 30))
        gamma = float(sched_cfg.get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(
            f"Scheduler '{name}' không được hỗ trợ. Các scheduler được hỗ trợ: "
            f"['reduce_on_plateau', 'cosine', 'step', 'none']"
        )
