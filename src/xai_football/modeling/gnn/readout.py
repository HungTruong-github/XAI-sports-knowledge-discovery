"""Bước 5.3 — Readout / Pooling layer gộp node embeddings -> graph embedding.

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 5.3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GlobalAttention, global_add_pool, global_max_pool, global_mean_pool


class GlobalReadout(nn.Module):
    """Lớp pooling gộp biểu diễn các đỉnh trong cùng đồ thị thành một vector đại diện.

    Hỗ trợ:
        - "mean": global mean pooling (mặc định)
        - "max": global max pooling
        - "sum" / "add": global add pooling
        - "attention": GlobalAttention pooling với gating MLP
    """

    def __init__(self, readout_type: str = "mean", in_dim: int | None = None) -> None:
        super().__init__()
        self.readout_type = readout_type.lower()
        if self.readout_type == "attention":
            if in_dim is None:
                raise ValueError("Readout attention yêu cầu tham số in_dim.")
            gate_nn = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2 or 1),
                nn.ReLU(),
                nn.Linear(in_dim // 2 or 1, 1),
            )
            self.pool_fn = GlobalAttention(gate_nn=gate_nn)
        else:
            self.pool_fn = None

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        """Args:
            x: Node embeddings [num_nodes, in_dim].
            batch: Tensor gán đỉnh vào đồ thị [num_nodes]. Nếu None, giả định tất cả thuộc 1 đồ thị.

        Returns:
            Graph embeddings [num_graphs, in_dim].
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.readout_type == "mean":
            return global_mean_pool(x, batch)
        elif self.readout_type == "max":
            return global_max_pool(x, batch)
        elif self.readout_type in ("add", "sum"):
            return global_add_pool(x, batch)
        elif self.readout_type == "attention":
            return self.pool_fn(x, batch)
        else:
            raise ValueError(f"Không hỗ trợ readout type: {self.readout_type}")
