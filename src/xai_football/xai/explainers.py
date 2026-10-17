"""Bước 7 — Giải thích dự đoán của mô hình GNN (GNNExplainer & PGExplainer).

Tham chiếu: configs/xai.yaml và docs/Quy_trinh_XAI_Football.md, mục 7.
Mục tiêu:
    - Giải thích phân loại ở cấp độ đồ thị (Graph-level Binary Classification: dự đoán cú sút).
    - Tạo mặt nạ trọng số cạnh (edge importance mask) và đỉnh (node importance mask).
    - Tương thích với API torch_geometric.explain (PyG 2.x).
    - Xử lý hoàn hảo đặc trưng cạnh (edge_attr) của backbone MPNN.
    - Xử lý biên an toàn: đồ thị 0 cạnh, đồ thị 1 cạnh, NaN/Inf, chuẩn hóa mask [0, 1].

NGUYÊN TẮC KHOA HỌC:
    - Điểm importance phản ánh hành vi nội tại của mô hình GNN, không suy diễn quan hệ nhân quả tuyệt đối.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.explain import (
    Explainer,
    ModelConfig,
)
from torch_geometric.explain import (
    GNNExplainer as PyGGNNExplainer,
)
from torch_geometric.explain import (
    PGExplainer as PyGPGExplainer,
)

from xai_football.utils.logging import get_logger

logger = get_logger("explainers")


@dataclass
class ExplanationResult:
    """Đối tượng chứa kết quả giải thích cho một đồ thị possession."""

    match_id: int
    possession_id: int
    graph_index: int
    target: int
    pred_prob: float
    edge_mask: np.ndarray  # Shape [num_edges], giá trị [0, 1]
    node_mask: np.ndarray | None  # Shape [num_nodes, num_node_features] hoặc None
    explainer_name: str
    num_nodes: int
    num_edges: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "possession_id": self.possession_id,
            "graph_index": self.graph_index,
            "target": self.target,
            "pred_prob": round(float(self.pred_prob), 4),
            "explainer_name": self.explainer_name,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "mean_edge_importance": (
                round(float(self.edge_mask.mean()), 4) if len(self.edge_mask) > 0 else 0.0
            ),
            "max_edge_importance": (
                round(float(self.edge_mask.max()), 4) if len(self.edge_mask) > 0 else 0.0
            ),
        }


def _normalize_mask(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    """Chuẩn hóa mask về miền [0, 1] một cách ổn định số học."""
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().float().numpy()
    else:
        m = np.asarray(mask, dtype=float)

    if m.size == 0:
        return m

    # Xử lý NaN hoặc Inf
    m = np.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0)

    # Đưa về miền [0, 1]
    m_min, m_max = m.min(), m.max()
    if m_max > m_min + 1e-8:
        m_norm = (m - m_min) / (m_max - m_min)
    else:
        m_norm = np.clip(m, 0.0, 1.0)
    return m_norm


class BaseGraphExplainer:
    """Interface cơ sở cho các thuật toán giải thích đồ thị bóng đá."""

    def __init__(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    def explain_graph(
        self, data: Data, graph_index: int = 0, target_class: int = 1
    ) -> ExplanationResult:
        raise NotImplementedError


class FootballGNNExplainer(BaseGraphExplainer):
    """Bộ giải thích GNNExplainer cho mô hình phân loại đồ thị possession."""

    def __init__(
        self,
        model: nn.Module,
        epochs: int = 100,
        lr: float = 0.01,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(model, device=device)
        self.epochs = epochs
        self.lr = lr

        self.explainer = Explainer(
            model=self.model,
            algorithm=PyGGNNExplainer(epochs=epochs, lr=lr),
            explanation_type="model",
            model_config=ModelConfig(
                mode="binary_classification",
                task_level="graph",
                return_type="raw",
            ),
            node_mask_type="attributes",
            edge_mask_type="object",
        )

    def explain_graph(
        self, data: Data, graph_index: int = 0, target_class: int = 1
    ) -> ExplanationResult:
        data = data.to(self.device)
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index.shape[1] if data.edge_index is not None else 0

        # Dự đoán ban đầu của mô hình
        with torch.no_grad():
            logit = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=getattr(data, "edge_attr", None),
                batch=torch.zeros(num_nodes, dtype=torch.long, device=self.device),
            )
            prob = float(torch.sigmoid(logit.view(-1))[0].item())

        # Xử lý trường hợp biên: đồ thị không có cạnh nào
        if num_edges == 0:
            return ExplanationResult(
                match_id=int(getattr(data, "match_id", 0)),
                possession_id=int(getattr(data, "possession_id", 0)),
                graph_index=graph_index,
                target=int(getattr(data, "y", torch.tensor([0])).item()),
                pred_prob=prob,
                edge_mask=np.zeros(0, dtype=float),
                node_mask=np.zeros((num_nodes, data.x.shape[1]), dtype=float),
                explainer_name="GNNExplainer",
                num_nodes=num_nodes,
                num_edges=0,
            )

        kwargs: dict[str, Any] = {}
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            kwargs["edge_attr"] = data.edge_attr

        # Chạy Explainer của PyTorch Geometric
        explanation = self.explainer(
            x=data.x,
            edge_index=data.edge_index,
            **kwargs,
        )

        edge_mask = _normalize_mask(explanation.edge_mask)
        node_mask = (
            _normalize_mask(explanation.node_mask)
            if hasattr(explanation, "node_mask") and explanation.node_mask is not None
            else None
        )

        return ExplanationResult(
            match_id=int(getattr(data, "match_id", 0)),
            possession_id=int(getattr(data, "possession_id", 0)),
            graph_index=graph_index,
            target=int(getattr(data, "y", torch.tensor([0])).item()),
            pred_prob=prob,
            edge_mask=edge_mask,
            node_mask=node_mask,
            explainer_name="GNNExplainer",
            num_nodes=num_nodes,
            num_edges=num_edges,
        )


class FootballPGExplainer(BaseGraphExplainer):
    """Bộ giải thích tham số PGExplainer với mạng nơ-ron học giải thích chung."""

    def __init__(
        self,
        model: nn.Module,
        epochs: int = 30,
        lr: float = 0.003,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(model, device=device)
        self.epochs = epochs
        self.lr = lr
        self.is_trained = False

        self.algo = PyGPGExplainer(epochs=epochs, lr=lr)
        self.explainer = Explainer(
            model=self.model,
            algorithm=self.algo,
            explanation_type="phenomenon",
            model_config=ModelConfig(
                mode="binary_classification",
                task_level="graph",
                return_type="raw",
            ),
            edge_mask_type="object",
        )

    def train_on_graphs(self, train_graphs: list[Data], max_graphs: int = 200) -> None:
        """Huấn luyện mạng giải thích PGExplainer trên một tập đồ thị mẫu."""
        logger.info(
            "Bắt đầu huấn luyện PGExplainer trên %d graphs (%d epochs)",
            min(len(train_graphs), max_graphs),
            self.epochs,
        )
        subset = [g for g in train_graphs[:max_graphs] if g.edge_index.shape[1] > 0]
        if not subset:
            logger.warning("Không có đồ thị hợp lệ (>0 cạnh) để huấn luyện PGExplainer.")
            return

        for epoch in range(self.epochs):
            total_loss = 0.0
            count = 0
            for g in subset:
                g = g.to(self.device)
                kwargs: dict[str, Any] = {}
                if hasattr(g, "edge_attr") and g.edge_attr is not None:
                    kwargs["edge_attr"] = g.edge_attr

                target = g.y.view(-1) if hasattr(g, "y") else torch.tensor([1], device=self.device)
                try:
                    loss = self.explainer.algorithm.train(
                        epoch,
                        self.model,
                        g.x,
                        g.edge_index,
                        target=target,
                        **kwargs,
                    )
                    total_loss += float(loss)
                    count += 1
                except Exception:
                    pass
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_l = total_loss / max(count, 1)
                logger.info("PGExplainer Epoch %2d/%2d | Loss: %.4f", epoch + 1, self.epochs, avg_l)
        self.is_trained = True

    def explain_graph(
        self, data: Data, graph_index: int = 0, target_class: int = 1
    ) -> ExplanationResult:
        data = data.to(self.device)
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index.shape[1] if data.edge_index is not None else 0

        with torch.no_grad():
            logit = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=getattr(data, "edge_attr", None),
                batch=torch.zeros(num_nodes, dtype=torch.long, device=self.device),
            )
            prob = float(torch.sigmoid(logit.view(-1))[0].item())

        if num_edges == 0:
            return ExplanationResult(
                match_id=int(getattr(data, "match_id", 0)),
                possession_id=int(getattr(data, "possession_id", 0)),
                graph_index=graph_index,
                target=int(getattr(data, "y", torch.tensor([0])).item()),
                pred_prob=prob,
                edge_mask=np.zeros(0, dtype=float),
                node_mask=None,
                explainer_name="PGExplainer",
                num_nodes=num_nodes,
                num_edges=0,
            )

        kwargs: dict[str, Any] = {}
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            kwargs["edge_attr"] = data.edge_attr

        explanation = self.explainer(
            x=data.x,
            edge_index=data.edge_index,
            target=torch.tensor([target_class], device=self.device),
            **kwargs,
        )

        edge_mask = _normalize_mask(explanation.edge_mask)
        return ExplanationResult(
            match_id=int(getattr(data, "match_id", 0)),
            possession_id=int(getattr(data, "possession_id", 0)),
            graph_index=graph_index,
            target=int(getattr(data, "y", torch.tensor([0])).item()),
            pred_prob=prob,
            edge_mask=edge_mask,
            node_mask=None,
            explainer_name="PGExplainer",
            num_nodes=num_nodes,
            num_edges=num_edges,
        )


def build_explainer(
    name: str,
    model: nn.Module,
    xai_cfg: dict[str, Any] | None = None,
    device: torch.device | str = "cpu",
) -> BaseGraphExplainer:
    """Factory khởi tạo Explainer từ tên thuật toán và cấu hình xai.yaml."""
    cfg = xai_cfg or {}
    explainers_cfg = cfg.get("explainers", {})
    name_clean = name.strip().lower()

    if name_clean in ("gnn_explainer", "gnnexplainer"):
        g_cfg = explainers_cfg.get("gnn_explainer", {})
        epochs = int(g_cfg.get("epochs", 100))
        lr = float(g_cfg.get("lr", 0.01))
        return FootballGNNExplainer(model=model, epochs=epochs, lr=lr, device=device)
    elif name_clean in ("pg_explainer", "pgexplainer"):
        p_cfg = explainers_cfg.get("pg_explainer", {})
        epochs = int(p_cfg.get("epochs", 30))
        lr = float(p_cfg.get("lr", 0.003))
        return FootballPGExplainer(model=model, epochs=epochs, lr=lr, device=device)
    else:
        raise ValueError(
            f"Thuật toán Explainer '{name}' không được hỗ trợ. "
            f"Các lựa chọn hợp lệ: ['gnn_explainer', 'pg_explainer']"
        )
