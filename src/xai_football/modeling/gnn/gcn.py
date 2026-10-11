"""Backbone GCN (Graph Convolutional Network) — Backbone ĐỀ XUẤT (Tầng 2).

Tham chiếu: configs/model/gcn.yaml và docs/Quy_trinh_XAI_Football.md, mục 5.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from xai_football.modeling.gnn.heads import BinaryClassifierHead
from xai_football.modeling.gnn.readout import GlobalReadout


class GCNModel(nn.Module):
    """Mô hình phân loại possession bằng GCN."""

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        readout: str = "mean",
        head_hidden_dim: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout_p = dropout

        self.convs = nn.ModuleList()
        # Lớp đầu tiên chuyển từ đặc trưng đầu vào sang hidden_dim
        self.convs.append(GCNConv(node_in_dim, hidden_dim))

        # Các lớp ẩn tiếp theo
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.readout = GlobalReadout(readout, in_dim=hidden_dim)
        self.head = BinaryClassifierHead(
            in_dim=hidden_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Trích xuất node embeddings sau các lớp message passing (phục vụ XAI)."""
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = self.act(h)
                h = self.dropout(h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Lan truyền xuôi tính raw logit phân loại nhị phân ở mức đồ thị.

        Args:
            x: Đặc trưng đỉnh [num_nodes, node_in_dim].
            edge_index: Danh sách cạnh có hướng [2, num_edges].
            edge_attr: Đặc trưng cạnh (tùy chọn).
            batch: Vector chỉ mục đồ thị [num_nodes].

        Returns:
            logits: [num_graphs]
        """
        h = self.get_node_embeddings(x, edge_index, edge_attr)
        graph_emb = self.readout(h, batch)
        return self.head(graph_emb)
