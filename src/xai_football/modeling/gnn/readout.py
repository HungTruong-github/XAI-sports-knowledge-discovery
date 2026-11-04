"""Bước 5.3 — Readout / Pooling layer gộp node embeddings -> graph embedding.

Tham chiếu: docs/Quy_trinh_XAI_Football.md, mục 5.3 và docs/decisions/0005-readout-mpnn.md.
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
        - "mean_lastedge": nối mean pooling với embedding của đỉnh nhận ở đường chuyền cuối cùng
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

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            x: Node embeddings [num_nodes, in_dim].
            batch: Tensor gán đỉnh vào đồ thị [num_nodes]. Nếu None, giả định tất cả thuộc 1 đồ thị.
            edge_index: Tensor cạnh [2, num_edges]. Bắt buộc khi readout_type == "mean_lastedge".
            edge_attr: Tensor đặc trưng cạnh [num_edges, edge_dim].

        Returns:
            Graph embeddings [num_graphs, out_dim].
            (out_dim = in_dim nếu mean/max/add/attention; out_dim = 2 * in_dim nếu mean_lastedge).
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
        elif self.readout_type == "mean_lastedge":
            mean_pool = global_mean_pool(x, batch)
            if edge_index is None or edge_index.size(1) == 0:
                return torch.cat([mean_pool, mean_pool], dim=-1)

            num_graphs = mean_pool.size(0)
            edge_batch = batch[edge_index[0]]

            # Assert kiểm tra thứ tự cạnh tăng dần theo thời gian nếu có đặc trưng thời gian
            if edge_attr is not None and edge_attr.size(1) >= 11 and edge_index.size(1) > 1:
                same_graph = edge_batch[1:] == edge_batch[:-1]
                if same_graph.any():
                    diffs = edge_attr[1:, 10] - edge_attr[:-1, 10]
                    assert torch.all(
                        diffs[same_graph] >= -1e-5
                    ), "Thứ tự cạnh trong cùng đồ thị không tăng dần theo thời gian."

            # Tìm chỉ số cạnh cuối cùng (chỉ số cạnh lớn nhất) theo từng đồ thị trong batch
            edge_ids = torch.arange(edge_index.size(1), device=edge_index.device)
            last_edge_indices = torch.full(
                (num_graphs,), -1, dtype=torch.long, device=edge_index.device
            )
            last_edge_indices.scatter_reduce_(
                0, edge_batch, edge_ids, reduce="amax", include_self=False
            )

            # Lấy embedding của đỉnh nhận ở đường chuyền cuối cùng (receiver: edge_index[1])
            valid_mask = last_edge_indices >= 0
            last_receivers = torch.zeros_like(mean_pool)
            if valid_mask.any():
                recv_nodes = edge_index[1, last_edge_indices[valid_mask]]
                last_receivers[valid_mask] = x[recv_nodes]
            if (~valid_mask).any():
                last_receivers[~valid_mask] = mean_pool[~valid_mask]

            return torch.cat([mean_pool, last_receivers], dim=-1)
        else:
            raise ValueError(f"Không hỗ trợ readout type: {self.readout_type}")
