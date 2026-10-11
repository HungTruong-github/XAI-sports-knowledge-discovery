"""Bước 5.4 — Hàm mất mát (Loss functions) xử lý mất cân bằng lớp.

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 5.4.
Cấu hình:   configs/train.yaml -> loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss xử lý mất cân bằng lớp trầm trọng:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
        logits: Tensor dự đoán raw logits [batch_size] hoặc [batch_size, 1].
        targets: Tensor nhãn nhị phân [batch_size] hoặc [batch_size, 1] (0 hoặc 1).
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_loss = alpha_t * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_loss_fn(
    loss_cfg: dict | None = None,
    pos_weight: float | None = None,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Tạo loss function phù hợp dựa trên config và tỉ lệ mất cân bằng lớp.

    Args:
        loss_cfg: Cấu hình loss từ configs/train.yaml.
        pos_weight: Trọng số lớp dương (n_negative / n_positive).
        device: Thiết bị chạy torch (cpu / cuda).

    Returns:
        nn.Module hàm mất mát.
    """
    cfg = loss_cfg or {}
    name = (cfg.get("name") or "weighted_bce").lower()

    # Nếu config chỉ định pos_weight rõ ràng thì ưu tiên dùng
    cfg_pos_weight = cfg.get("pos_weight")
    if cfg_pos_weight is not None:
        weight_val = float(cfg_pos_weight)
    elif pos_weight is not None:
        weight_val = float(pos_weight)
    else:
        weight_val = 1.0

    if name in ("focal", "focal_loss"):
        gamma = float(cfg.get("focal_gamma", 2.0))
        # alpha có thể suy từ pos_weight hoặc để mặc định
        alpha = float(cfg.get("focal_alpha", 0.75))
        return BinaryFocalLoss(gamma=gamma, alpha=alpha)

    # Mặc định: Weighted BCE With Logits Loss
    pw_tensor = torch.tensor([weight_val], device=device, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
