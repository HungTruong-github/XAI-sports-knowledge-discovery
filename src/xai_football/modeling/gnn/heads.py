"""Bước 5.3 — Head phân loại nhị phân ở mức đồ thị (Graph-level classification head).

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 5.3.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BinaryClassifierHead(nn.Module):
    """MLP Head nhận vector biểu diễn toàn đồ thị và trả về raw logit cho phân loại nhị phân."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        """Args:
            graph_emb: [num_graphs, in_dim]

        Returns:
            logits: [num_graphs]
        """
        return self.mlp(graph_emb).squeeze(-1)
