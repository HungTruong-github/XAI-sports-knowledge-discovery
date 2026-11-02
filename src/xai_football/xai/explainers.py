"""Bước 7 — Giải thích dự đoán của mô hình GNN (GNNExplainer, PGExplainer, Baselines).

Tham chiếu: configs/xai.yaml và docs/decisions/0006-giao-thuc-xai.md (Quyết định D5, D8, D17).
Mục tiêu:
    - Giải thích phân loại ở cấp độ đồ thị (Graph-level Binary Classification: dự đoán cú sút).
    - Tạo mặt nạ trọng số cạnh (edge importance mask) và đỉnh (node importance mask).
    - Hướng mục tiêu tường minh vào lớp 1 (có cú sút) theo D5.
    - Cung cấp đầy đủ các baseline đối chứng: RandomExplainer, LastEdgeExplainer.
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

        # D5: Dùng explanation_type="phenomenon" để giải thích hướng đích vào target_class
        self.explainer = Explainer(
            model=self.model,
            algorithm=PyGGNNExplainer(epochs=epochs, lr=lr),
            explanation_type="phenomenon",
            model_config=ModelConfig(
                mode="binary_classification",
                task_level="graph",
                return_type="raw",
            ),
            node_mask_type="object",
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

        # Chạy Explainer với target_class truyền tường minh theo D5
        explanation = self.explainer(
            x=data.x,
            edge_index=data.edge_index,
            target=torch.tensor([target_class], device=self.device),
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

    def train_on_graphs(self, train_graphs: list[Data], max_graphs: int = 500) -> None:
        """Huấn luyện mạng giải thích PGExplainer theo quyết định D8."""
        subset = [g for g in train_graphs[:max_graphs] if g.edge_index is not None and g.edge_index.shape[1] > 0]
        if not subset:
            logger.warning("Không có đồ thị hợp lệ (>0 cạnh) để huấn luyện PGExplainer.")
            return

        logger.info(
            "Bắt đầu huấn luyện PGExplainer trên %d graphs (%d epochs, target=1)",
            len(subset),
            self.epochs,
        )

        successful_steps = 0
        failed_steps = 0

        for epoch in range(self.epochs):
            total_loss = 0.0
            count = 0
            for g in subset:
                g = g.to(self.device)
                kwargs: dict[str, Any] = {}
                if hasattr(g, "edge_attr") and g.edge_attr is not None:
                    kwargs["edge_attr"] = g.edge_attr

                # D8: Luôn huấn luyện với target = 1
                target = torch.tensor([1], device=self.device)
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
                    successful_steps += 1
                except Exception as exc:
                    failed_steps += 1
                    logger.debug("Lỗi huấn luyện PGExplainer trên graph: %s", exc)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_l = total_loss / max(count, 1)
                logger.info(
                    "PGExplainer Epoch %2d/%2d | Loss: %.4f (Thành công: %d, Lỗi: %d)",
                    epoch + 1,
                    self.epochs,
                    avg_l,
                    count,
                    failed_steps,
                )

        if successful_steps == 0:
            raise RuntimeError("PGExplainer không thực hiện được bất kỳ bước huấn luyện nào thành công.")

        self.is_trained = True
        logger.info("PGExplainer đã huấn luyện thành công (%d bước tối ưu)", successful_steps)

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


class RandomExplainer(BaseGraphExplainer):
    """Explainer ngẫu nhiên đối chứng (Random Baseline theo D17).

    Rút ngẫu nhiên trọng số [0, 1] cho mỗi cạnh n_runs lần (mặc định 5),
    lấy trung bình qua các lần rút rồi chuẩn hóa về [0, 1].
    """

    def __init__(
        self,
        model: nn.Module,
        seed: int = 42,
        n_runs: int = 5,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(model, device=device)
        self.seed = seed
        self.n_runs = n_runs

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
                explainer_name="RandomExplainer",
                num_nodes=num_nodes,
                num_edges=0,
            )

        # Trung bình n_runs lần rút ngẫu nhiên
        masks = []
        for r in range(self.n_runs):
            rng = np.random.RandomState(self.seed + graph_index * 1000 + r)
            masks.append(rng.uniform(0.0, 1.0, size=num_edges))
        avg_mask = np.mean(masks, axis=0)
        norm_mask = _normalize_mask(avg_mask)

        return ExplanationResult(
            match_id=int(getattr(data, "match_id", 0)),
            possession_id=int(getattr(data, "possession_id", 0)),
            graph_index=graph_index,
            target=int(getattr(data, "y", torch.tensor([0])).item()),
            pred_prob=prob,
            edge_mask=norm_mask,
            node_mask=None,
            explainer_name="RandomExplainer",
            num_nodes=num_nodes,
            num_edges=num_edges,
        )


class LastEdgeExplainer(BaseGraphExplainer):
    """Explainer quy tắc ngây thơ (Naive Rule-based Baseline).

    Gán điểm importance = 1.0 cho đường chuyền cuối cùng (chỉ số cạnh lớn nhất),
    và 0.0 cho mọi đường chuyền khác.
    """

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
                explainer_name="LastEdgeExplainer",
                num_nodes=num_nodes,
                num_edges=0,
            )

        edge_mask = np.zeros(num_edges, dtype=float)
        edge_mask[-1] = 1.0

        return ExplanationResult(
            match_id=int(getattr(data, "match_id", 0)),
            possession_id=int(getattr(data, "possession_id", 0)),
            graph_index=graph_index,
            target=int(getattr(data, "y", torch.tensor([0])).item()),
            pred_prob=prob,
            edge_mask=edge_mask,
            node_mask=None,
            explainer_name="LastEdgeExplainer",
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
    elif name_clean in ("random_explainer", "randomexplainer", "random"):
        r_cfg = explainers_cfg.get("random_explainer", {})
        n_runs = int(r_cfg.get("n_runs", 5))
        seed = int(cfg.get("sampling", {}).get("seed", 42))
        return RandomExplainer(model=model, seed=seed, n_runs=n_runs, device=device)
    elif name_clean in ("last_edge_explainer", "lastedgeexplainer", "last_edge", "lastedge"):
        return LastEdgeExplainer(model=model, device=device)
    else:
        raise ValueError(
            f"Thuật toán Explainer '{name}' không được hỗ trợ. "
            f"Các lựa chọn hợp lệ: ['gnn_explainer', 'pg_explainer', 'random_explainer', 'last_edge_explainer']"
        )
