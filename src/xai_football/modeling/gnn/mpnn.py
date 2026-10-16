"""Backbone MPNN (Message Passing Neural Network) — Backbone ĐỀ XUẤT (Tầng 2).

Tham chiếu: configs/model/mpnn.yaml và docs/Quy_trinh_XAI_Football.md, mục 5.
Điểm cốt lõi: Tận dụng đặc trưng cạnh (edge_attr: tọa độ, góc, độ dài, loại chuyền)
để điều hòa thông điệp truyền giữa các đỉnh (cầu thủ).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from xai_football.modeling.gnn.heads import BinaryClassifierHead
from xai_football.modeling.gnn.readout import GlobalReadout


class MPNNConv(MessagePassing):
    """Một lớp truyền thông điệp điều kiện hóa theo đặc trưng cạnh (Edge-conditioned)."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int) -> None:
        super().__init__(aggr="mean")
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.node_proj = nn.Linear(in_dim, out_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # x_j: đặc trưng đỉnh nguồn; edge_attr: đặc trưng đường chuyền
        edge_weight = self.edge_mlp(edge_attr)
        return self.node_proj(x_j) * edge_weight

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class MPNNModel(nn.Module):
    """Mô hình MPNN phân loại đồ thị possession có sử dụng đặc trưng đường chuyền."""

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        readout: str = "mean",
        head_hidden_dim: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.edge_in_dim = edge_in_dim
        self.dropout_p = dropout

        self.convs = nn.ModuleList()
        # Lớp đầu tiên
        self.convs.append(MPNNConv(node_in_dim, hidden_dim, edge_in_dim))

        # Các lớp tiếp theo
        for _ in range(num_layers - 1):
            self.convs.append(MPNNConv(hidden_dim, hidden_dim, edge_in_dim))

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
        if edge_attr is None:
            # Fallback nếu không có edge_attr: tạo zero tensor
            edge_attr = torch.zeros(
                (edge_index.size(1), self.edge_in_dim),
                dtype=x.dtype,
                device=x.device,
            )

        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_attr)
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
