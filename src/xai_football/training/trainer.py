"""Bước 5.5 — Khung huấn luyện mô hình GNN (GNN Trainer).

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 5.5 và configs/train.yaml.

Tính năng:
    - Huấn luyện theo mini-batch bằng torch_geometric.loader.DataLoader
    - Tối ưu hóa bằng AdamW / Adam, hỗ trợ gradient clipping
    - Lập lịch learning rate: ReduceLROnPlateau giám sát val_auc_pr
    - Early stopping theo AUC-PR trên tập validation (chống overfitting)
    - Tự động lưu checkpoint mô hình tốt nhất (best_model.pt) và lịch sử huấn luyện
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from xai_football.evaluation.metrics import compute_binary_metrics
from xai_football.utils.logging import get_logger

logger = get_logger("trainer")


class GNNTrainer:
    """Quản lý toàn bộ chu trình huấn luyện, đánh giá và early stopping cho mô hình GNN."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
        scheduler: Any = None,
        early_stopping_patience: int = 20,
        early_stopping_monitor: str = "val_auc_pr",
        early_stopping_mode: str = "max",
        device: torch.device | str = "cpu",
        output_dir: Path | str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler

        self.patience = early_stopping_patience
        self.monitor = early_stopping_monitor
        self.mode = early_stopping_mode
        self.output_dir = Path(output_dir) if output_dir else Path("experiments/run")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric = -float("inf") if self.mode == "max" else float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.history: list[dict[str, Any]] = []

    def train_epoch(self) -> dict[str, float]:
        """Huấn luyện 1 epoch trên tập train."""
        self.model.train()
        total_loss = 0.0
        all_probs: list[float] = []
        all_targets: list[int] = []

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=getattr(batch, "edge_attr", None),
                batch=batch.batch,
            )
            targets = batch.y.float().view(-1)
            loss = self.loss_fn(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(targets.detach().cpu().numpy().astype(int).tolist())

        avg_loss = total_loss / len(all_targets) if all_targets else 0.0
        metrics = compute_binary_metrics(all_targets, all_probs)
        metrics["loss"] = round(avg_loss, 4)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Đánh giá mô hình trên một loader (validation hoặc test)."""
        self.model.eval()
        total_loss = 0.0
        all_probs: list[float] = []
        all_targets: list[int] = []

        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=getattr(batch, "edge_attr", None),
                batch=batch.batch,
            )
            targets = batch.y.float().view(-1)
            loss = self.loss_fn(logits, targets)

            total_loss += loss.item() * batch.num_graphs
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(targets.cpu().numpy().astype(int).tolist())

        avg_loss = total_loss / len(all_targets) if all_targets else 0.0
        metrics = compute_binary_metrics(all_targets, all_probs)
        metrics["loss"] = round(avg_loss, 4)
        return metrics

    def is_improvement(self, current_val: float) -> bool:
        if self.mode == "max":
            return current_val > self.best_metric + 1e-5
        return current_val < self.best_metric - 1e-5

    def fit(self, epochs: int = 200, eval_every: int = 1) -> dict[str, Any]:
        """Thực thi vòng lặp huấn luyện đầy đủ kèm early stopping."""
        logger.info(
            "Bắt đầu huấn luyện: %d epochs (early stopping patience=%d, monitor=%s)",
            epochs,
            self.patience,
            self.monitor,
        )
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            t_train = self.train_epoch()

            if epoch % eval_every == 0:
                t_val = self.evaluate(self.val_loader)
                current_val_metric = t_val.get(
                    self.monitor.replace("val_", ""), t_val.get("auc_pr", 0.0)
                )

                # Bước điều chỉnh LR scheduler nếu có
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_val_metric)
                    else:
                        self.scheduler.step()

                # Kiểm tra cải thiện
                improved = self.is_improvement(current_val_metric)
                if improved:
                    self.best_metric = current_val_metric
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    # Lưu checkpoint mô hình tốt nhất
                    self.save_checkpoint("best_model.pt", epoch, t_val)
                else:
                    self.patience_counter += 1

                log_entry = {
                    "epoch": epoch,
                    "train_loss": t_train["loss"],
                    "train_auc_pr": t_train["auc_pr"],
                    "train_auc_roc": t_train["auc_roc"],
                    "val_loss": t_val["loss"],
                    "val_auc_pr": t_val["auc_pr"],
                    "val_auc_roc": t_val["auc_roc"],
                    "val_f1": t_val["f1"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                self.history.append(log_entry)

                if epoch % 5 == 0 or improved or epoch == 1:
                    logger.info(
                        "Ep %3d/%3d | L: %.4f, PR: %.4f | Val L: %.4f, PR: %.4f (%s) | LR: %.6f",
                        epoch,
                        epochs,
                        t_train["loss"],
                        t_train["auc_pr"],
                        t_val["loss"],
                        t_val["auc_pr"],
                        "BEST" if improved else f"p={self.patience_counter}",
                        self.optimizer.param_groups[0]["lr"],
                    )

                if self.patience_counter >= self.patience:
                    logger.info(
                        "Early stopping kích hoạt tại epoch %d (best epoch: %d với %s=%.4f)",
                        epoch,
                        self.best_epoch,
                        self.monitor,
                        self.best_metric,
                    )
                    break

        elapsed = time.time() - start_time
        logger.info("Hoàn tất huấn luyện sau %.1f giây. Best Epoch: %d", elapsed, self.best_epoch)

        # Lưu lịch sử ra CSV
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.output_dir / "training_history.csv", index=False)

        # Đánh giá lần cuối trên tập test bằng best model (nếu có test_loader)
        test_metrics: dict[str, float] = {}
        if self.test_loader is not None and (self.output_dir / "best_model.pt").exists():
            self.load_checkpoint("best_model.pt")
            test_metrics = self.evaluate(self.test_loader)
            logger.info("Kết quả trên tập TEST (best checkpoint): %s", test_metrics)
            test_df = pd.DataFrame([test_metrics])
            test_df.to_csv(self.output_dir / "test_metrics.csv", index=False)

        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "training_time": elapsed,
            "test_metrics": test_metrics,
        }

    def save_checkpoint(self, filename: str, epoch: int, metrics: dict[str, float]) -> None:
        ckpt_path = self.output_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            ckpt_path,
        )

    def load_checkpoint(self, filename: str) -> None:
        ckpt_path = self.output_dir / filename
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
