"""Bước 5.5 — Khung huấn luyện mô hình GNN (GNN Trainer).

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 5.5 và configs/train.yaml.

Tính năng:
    - Huấn luyện theo mini-batch bằng torch_geometric.loader.DataLoader
    - Tối ưu hóa bằng AdamW / Adam, hỗ trợ gradient clipping
    - Lập lịch learning rate: ReduceLROnPlateau giám sát val_auc_pr
    - Early stopping theo AUC-PR trên tập validation (chống overfitting)
    - Tách biệt hoàn toàn Validation và Test: Threshold chọn trên Validation, áp dụng cố định sang Test
    - Tự động lưu checkpoint mô hình tốt nhất (best_model.pt), lịch sử huấn luyện (training_history.csv),
      validation_metrics.csv, test_metrics.csv và run_manifest.json
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from xai_football.evaluation.metrics import compute_binary_metrics, find_best_threshold
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
        scheduler_monitor: str | None = None,
        early_stopping_patience: int = 20,
        early_stopping_monitor: str = "val_auc_pr",
        early_stopping_mode: str = "max",
        threshold_strategy: str = "val_optimal",
        device: torch.device | str = "cpu",
        output_dir: Path | str | None = None,
        seed: int = 42,
        model_name: str = "model",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.scheduler_monitor = scheduler_monitor or early_stopping_monitor

        self.patience = early_stopping_patience
        self.monitor = early_stopping_monitor
        self.mode = early_stopping_mode
        self.threshold_strategy = threshold_strategy
        self.seed = seed
        self.model_name = model_name

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
        metrics = compute_binary_metrics(all_targets, all_probs, threshold=0.5)
        metrics["loss"] = round(avg_loss, 4)
        return metrics

    @torch.no_grad()
    def predict_proba(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, float]:
        """Dự đoán xác suất trên toàn bộ DataLoader.

        Returns:
            targets: mảng nhãn nhị phân thực tế.
            probs: mảng xác suất dự đoán liên tục.
            avg_loss: mất mát trung bình.
        """
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
        return np.array(all_targets, dtype=int), np.array(all_probs, dtype=float), avg_loss

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, threshold: float = 0.5) -> dict[str, float]:
        """Đánh giá mô hình trên một loader (validation hoặc test)."""
        targets, probs, avg_loss = self.predict_proba(loader)
        metrics = compute_binary_metrics(targets, probs, threshold=threshold)
        metrics["loss"] = round(avg_loss, 4)
        return metrics

    def is_improvement(self, current_val: float) -> bool:
        if self.mode == "max":
            return current_val > self.best_metric + 1e-5
        return current_val < self.best_metric - 1e-5

    def fit(self, epochs: int = 200, eval_every: int = 1) -> dict[str, Any]:
        """Thực thi vòng lặp huấn luyện đầy đủ kèm early stopping và giao thức Validation/Test nghiêm ngặt."""
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
                t_val = self.evaluate(self.val_loader, threshold=0.5)
                current_val_metric = t_val.get(
                    self.monitor.replace("val_", ""), t_val.get("auc_pr", 0.0)
                )

                # Bước điều chỉnh LR scheduler nếu có
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        sched_metric = t_val.get(
                            self.scheduler_monitor.replace("val_", ""), current_val_metric
                        )
                        self.scheduler.step(sched_metric)
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

        # 1. Lưu lịch sử huấn luyện
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.output_dir / "training_history.csv", index=False)

        # 2. Tải lại best model để đánh giá cuối cùng
        if (self.output_dir / "best_model.pt").exists():
            self.load_checkpoint("best_model.pt")

        # 3. Đánh giá trên Validation và chọn threshold tối ưu
        val_targets, val_probs, val_loss = self.predict_proba(self.val_loader)
        if self.threshold_strategy == "val_optimal":
            best_thresh = find_best_threshold(val_targets, val_probs, metric="f1")
        else:
            best_thresh = 0.5

        val_metrics = compute_binary_metrics(val_targets, val_probs, threshold=best_thresh)
        val_metrics["loss"] = round(val_loss, 4)
        val_row = {
            "model": self.model_name,
            "seed": self.seed,
            "split": "validation",
            "best_epoch": self.best_epoch,
        }
        val_row.update(val_metrics)
        pd.DataFrame([val_row]).to_csv(self.output_dir / "validation_metrics.csv", index=False)
        logger.info("Validation metrics (best checkpoint, th=%.4f): %s", best_thresh, val_metrics)

        # 4. Đánh giá trên Test bằng đúng threshold đã chốt từ validation
        test_metrics: dict[str, float] = {}
        if self.test_loader is not None:
            test_targets, test_probs, test_loss = self.predict_proba(self.test_loader)
            test_metrics = compute_binary_metrics(test_targets, test_probs, threshold=best_thresh)
            test_metrics["loss"] = round(test_loss, 4)

            test_row = {
                "model": self.model_name,
                "seed": self.seed,
                "split": "test",
                "best_epoch": self.best_epoch,
            }
            test_row.update(test_metrics)
            pd.DataFrame([test_row]).to_csv(self.output_dir / "test_metrics.csv", index=False)
            logger.info("Test metrics (best checkpoint, th=%.4f): %s", best_thresh, test_metrics)

            # Lưu prediction scores để vẽ ROC/PR curves trong notebook
            pred_df = pd.DataFrame({"y_true": test_targets, "y_prob": test_probs})
            pred_df.to_parquet(self.output_dir / "test_predictions.parquet", index=False)

        # 5. Lưu run manifest
        manifest = {
            "model_name": self.model_name,
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "best_val_metric": self.best_metric,
            "threshold_strategy": self.threshold_strategy,
            "threshold": best_thresh,
            "training_time_seconds": round(elapsed, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        with open(self.output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "threshold": best_thresh,
            "training_time": elapsed,
            "val_metrics": val_metrics,
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
