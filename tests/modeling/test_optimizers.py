"""Unit tests cho factory khởi tạo Optimizer và Scheduler."""

import pytest
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from xai_football.training.optimizers import build_optimizer, build_scheduler


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)


def test_build_optimizer_supported():
    model = DummyModel()

    # AdamW
    opt_adamw = build_optimizer(model, {"name": "adamw", "lr": 0.005, "weight_decay": 0.01})
    assert isinstance(opt_adamw, AdamW)
    assert opt_adamw.param_groups[0]["lr"] == 0.005
    assert opt_adamw.param_groups[0]["weight_decay"] == 0.01

    # Adam
    opt_adam = build_optimizer(model, {"name": "adam", "lr": 0.001})
    assert isinstance(opt_adam, Adam)

    # SGD
    opt_sgd = build_optimizer(model, {"name": "sgd", "lr": 0.01, "momentum": 0.95})
    assert isinstance(opt_sgd, SGD)
    assert opt_sgd.param_groups[0]["momentum"] == 0.95


def test_build_optimizer_unsupported():
    model = DummyModel()
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        build_optimizer(model, {"name": "rmsprop_invalid"})


def test_build_scheduler_supported():
    model = DummyModel()
    opt = AdamW(model.parameters(), lr=0.001)

    # ReduceLROnPlateau
    sched_plateau = build_scheduler(
        opt, {"name": "reduce_on_plateau", "factor": 0.2, "patience": 5, "monitor": "val_auc_pr"}
    )
    assert isinstance(sched_plateau, ReduceLROnPlateau)
    assert sched_plateau.factor == 0.2
    assert sched_plateau.patience == 5
    assert sched_plateau.mode == "max"

    # ReduceLROnPlateau với loss monitor -> mode min
    sched_loss = build_scheduler(opt, {"name": "reduce_on_plateau", "monitor": "val_loss"})
    assert isinstance(sched_loss, ReduceLROnPlateau)
    assert sched_loss.mode == "min"

    # Cosine
    sched_cosine = build_scheduler(opt, {"name": "cosine", "t_max": 50})
    assert isinstance(sched_cosine, CosineAnnealingLR)

    # Step
    sched_step = build_scheduler(opt, {"name": "step", "step_size": 15, "gamma": 0.5})
    assert isinstance(sched_step, StepLR)

    # None / null
    assert build_scheduler(opt, {"name": "none"}) is None
    assert build_scheduler(opt, None) is None


def test_build_scheduler_unsupported():
    model = DummyModel()
    opt = AdamW(model.parameters(), lr=0.001)
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        build_scheduler(opt, {"name": "cyclic_lr_invalid"})
