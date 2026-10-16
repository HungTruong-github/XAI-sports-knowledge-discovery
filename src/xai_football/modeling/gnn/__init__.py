"""Backbone GNN — Tầng 2 trong chiến lược đối sánh 3 tầng.

Backbones:
    - GCNModel       (gcn.py)       — Backbone đề xuất
    - MPNNModel      (mpnn.py)      — Backbone đề xuất (kết hợp đặc trưng cạnh)
    - GATv2Model     (gat.py)       — Baseline đối sánh
    - GraphSAGEModel (graphsage.py) — Baseline đối sánh
"""

from __future__ import annotations

import torch.nn as nn

from xai_football.modeling.gnn.gat import GATv2Model
from xai_football.modeling.gnn.gcn import GCNModel
from xai_football.modeling.gnn.graphsage import GraphSAGEModel
from xai_football.modeling.gnn.heads import BinaryClassifierHead
from xai_football.modeling.gnn.mpnn import MPNNModel
from xai_football.modeling.gnn.readout import GlobalReadout

__all__ = [
    "GCNModel",
    "MPNNModel",
    "GATv2Model",
    "GraphSAGEModel",
    "GlobalReadout",
    "BinaryClassifierHead",
    "build_gnn_model",
]


def build_gnn_model(
    model_cfg: dict,
    node_in_dim: int,
    edge_in_dim: int = 12,
) -> nn.Module:
    """Khởi tạo mô hình GNN theo cấu hình YAML.

    Args:
        model_cfg: Cấu hình mô hình (configs/model/*.yaml).
        node_in_dim: Số chiều đặc trưng đỉnh đầu vào (thường là 8).
        edge_in_dim: Số chiều đặc trưng cạnh đầu vào (thường là 12).

    Returns:
        nn.Module đã khởi tạo.
    """
    name = (model_cfg.get("name") or "gcn").lower()
    kwargs = {k: v for k, v in model_cfg.items() if k not in ("name", "type")}

    if name == "gcn":
        return GCNModel(node_in_dim=node_in_dim, **kwargs)
    elif name == "mpnn":
        return MPNNModel(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, **kwargs)
    elif name in ("gat", "gatv2"):
        return GATv2Model(node_in_dim=node_in_dim, **kwargs)
    elif name in ("graphsage", "sage"):
        return GraphSAGEModel(node_in_dim=node_in_dim, **kwargs)
    else:
        raise ValueError(
            f"Không hỗ trợ kiến trúc GNN: '{name}'. Hỗ trợ: gcn, mpnn, gatv2, graphsage."
        )
