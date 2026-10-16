"""Backbone GATv2 (Graph Attention Network v2) — Baseline ĐỐI SÁNH Tầng 2.

Tham chiếu: configs/model/gatv2.yaml và docs/Quy_trinh_XAI_Football.md, mục 5.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from xai_football.modeling.gnn.heads import BinaryClassifierHead
from xai_football.modeling.gnn.readout import GlobalReadout


class GATv2Model(nn.Module):
    """Mô hình phân loại possession bằng GATv2 với multi-head attention."""

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        readout: str = "mean",
        head_hidden_dim: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout_p = dropout

        # Mỗi head có số chiều = hidden_dim // heads
        head_dim = max(1, hidden_dim // heads)
        out_dim = head_dim * heads

        self.convs = nn.ModuleList()
        # Lớp đầu tiên
        self.convs.append(GATv2Conv(node_in_dim, head_dim, heads=heads, dropout=dropout))

        # Các lớp tiếp theo
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(out_dim, head_dim, heads=heads, dropout=dropout))

        self.act = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

        self.readout = GlobalReadout(readout, in_dim=out_dim)
        self.head = BinaryClassifierHead(
            in_dim=out_dim,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        h = self.get_node_embeddings(x, edge_index, edge_attr)
        graph_emb = self.readout(h, batch)
        return self.head(graph_emb)
