"""
Data-driven MeshGraphNet implementation for learning PDE solutions.

This module implements a standard data-driven MeshGraphNet that learns from ground truth
FEM data, as described in the MeshGraphNets paper. It includes:
- Message passing layers with proper normalization
- Data normalization utilities
- A simplified architecture compatible with the graph structures from GraphCreator
"""

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
from torch_scatter import scatter
from typing import List, Tuple, Optional
import numpy as np


def mlp_2layer(in_dim: int, out_dim: int) -> nn.Sequential:
    """2-layer MLP with ReLU and LayerNorm(out_dim) as in the paper."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
        nn.LayerNorm(out_dim),
    )


class MeshGraphNetLayer(MessagePassing):
    """
    Message passing layer for data-driven MeshGraphNet.
    Simplified version that works with standard graph structures from GraphCreator.
    """

    def __init__(self, hidden_dim: int, aggr: str = "sum"):
        super(MeshGraphNetLayer, self).__init__()
        self.aggr = aggr  # sum, mean, max

        # Edge MLP: [x_sender, x_receiver, x_edge, g] -> delta_edge (128)
        self.edge_mlp = mlp_2layer(in_dim=hidden_dim * 4, out_dim=hidden_dim)

        # Node MLP: [x_node, aggr_msgs, g] -> delta_node (128)
        self.node_mlp = mlp_2layer(in_dim=hidden_dim * 3, out_dim=hidden_dim)

        # Global MLP: [sum_nodes, sum_edges, g] -> delta_global (128)
        self.global_mlp = mlp_2layer(in_dim=hidden_dim * 3, out_dim=hidden_dim)

    def forward(
        self,
        x_v: torch.Tensor,  # [N, 128]
        x_e: torch.Tensor,  # [E, 128]
        g: torch.Tensor,  # [B, 128]
        edge_index: torch.Tensor,  # [2, E]
        batch: Optional[torch.Tensor],  # [N]
    ):
        # ---- Edge update ----
        senders, receivers = edge_index[0], edge_index[1]
        # Use receiver's graph id to index the corresponding global
        if batch is None:
            # Single-graph fallback
            edge_batch = x_v.new_zeros((x_e.size(0),), dtype=torch.long)
            node_batch = x_v.new_zeros((x_v.size(0),), dtype=torch.long)
        else:
            node_batch = batch
            edge_batch = node_batch[receivers]  # edges belong to the receiver's graph

        xs = x_v[senders]  # [E, 128]
        xr = x_v[receivers]  # [E, 128]
        g_e = g[edge_batch]  # [E, 128]

        edge_input = torch.cat([xs, xr, x_e, g_e], dim=-1)  # [E, 512]
        delta_e = self.edge_mlp(edge_input)  # [E, 128]
        x_e = x_e + delta_e  # residual

        # ---- Node aggregation (messages = updated edges) ----
        msgs = x_e  # [E, 128]
        aggr_msgs = scatter(
            msgs, receivers, dim=0, dim_size=x_v.size(0), reduce=self.aggr
        )  # [N, 128]
        g_v = g[node_batch]  # [N, 128]

        node_input = torch.cat([x_v, aggr_msgs, g_v], dim=-1)  # [N, 384]
        delta_v = self.node_mlp(node_input)  # [N, 128]
        x_v = x_v + delta_v  # residual

        # ---- Global update (sum over nodes/edges per graph) ----
        B = g.size(0)
        sum_nodes = scatter(x_v, node_batch, dim=0, dim_size=B, reduce="sum")  # [B,128]
        sum_edges = scatter(x_e, edge_batch, dim=0, dim_size=B, reduce="sum")  # [B,128]

        g_input = torch.cat([sum_nodes, sum_edges, g], dim=-1)  # [B, 384]
        delta_g = self.global_mlp(g_input)  # [B, 128]
        g = g + delta_g  # residual

        return x_v, x_e, g


class MeshGraphNet(nn.Module):
    """
    Data-driven MeshGraphNet for learning PDE solutions from ground truth data.

    This is a simplified implementation that works with the graph structures created
    by GraphCreator and trains with MSE loss against FEM ground truth.
    """

    def __init__(
        self,
        input_dim_node: int,
        input_dim_edge: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        input_dim_global: int = 0,
        num_layers: int = 12,
    ):
        super(MeshGraphNet, self).__init__()

        self.time_window = output_dim

        self.input_dim_node = input_dim_node
        self.input_dim_edge = input_dim_edge
        self.hidden_dim = hidden_dim # latent dimension
        self.output_dim = output_dim # predict multiple time steps

        # Number of message passing layers
        self.num_layers = num_layers

        self.node_encoder = mlp_2layer(input_dim_node, hidden_dim)
        self.edge_encoder = mlp_2layer(input_dim_edge, hidden_dim)
        self.has_global_in = input_dim_global > 0
        self.global_encoder = (
            mlp_2layer(input_dim_global, hidden_dim) if self.has_global_in else None
        )

        # Message passing layers
        self.gnn_layers = nn.ModuleList(
            [MeshGraphNetLayer(hidden_dim) for _ in range(self.num_layers)]
        )
        self._build_output_head()

    def _build_output_head(self):
        """
        For time_window=20, match the paper exactly:
          Conv1d(1->8, k=15, s=4) -> ReLU -> Conv1d(8->1, k=10, s=1) to get length 20.
        For 5 or 10, reuse (k1=15, s1=4), compute k2 = L1 - T + 1 with L1 = floor((128-k1)/s1)+1 = 29.
        Otherwise, fall back to a linear projection.
        """
        T = self.time_window
        if T == 1:
            self.use_cnn = False
            self.output_head = nn.Linear(self.hidden_dim, 1)
            return

        if T == 20:
            self.use_cnn = True
            self.output_head = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=15, stride=4),
                nn.ReLU(),
                nn.Conv1d(8, 1, kernel_size=10, stride=1),
            )
            return

        if T in (5, 10):
            # L1 = 29 with k1=15, s1=4; choose k2 to hit the exact length
            k1, s1 = 15, 4
            L1 = (128 - k1) // s1 + 1  # 29
            k2 = L1 - T + 1  # 25 for T=5; 20 for T=10
            self.use_cnn = True
            self.output_head = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=k1, stride=s1),
                nn.ReLU(),
                nn.Conv1d(8, 1, kernel_size=k2, stride=1),
            )
            return

        # fallback: simple linear projection 128 -> T
        self.use_cnn = False
        self.output_head = nn.Linear(self.hidden_dim, T)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data.x:          [N, input_dim_node]
            data.edge_attr:  [E, input_dim_edge]
            data.edge_index: [2, E]
            data.batch:      [N] (optional)
            data.u or data.global_attr/globals/g: [B, input_dim_global] (optional)
        Returns:
            [N, time_window] predictions for each node
        """
        x: torch.Tensor = self.node_encoder(data.x)  # [N, 128]
        e: torch.Tensor = self.edge_encoder(data.edge_attr)  # [E, 128]

        batch = getattr(data, "batch", None)
        if batch is None:
            B = 1
        else:
            B = int(batch.max().item()) + 1

        # Prepare global inputs
        g_in = None
        if self.has_global_in:
            g_in = getattr(data, "global_attr", None)
            if g_in is None:
                # If not provided, use zeros to preserve interface
                g_in = x.new_zeros((B, self.global_encoder[0].in_features))
            g = self.global_encoder(g_in)  # [B, 128]
        else:
            g = x.new_zeros(
                (B, self.hidden_dim)
            )

        # Processor stack
        for layer in self.gnn_layers:
            x, e, g = layer(x, e, g, data.edge_index, batch)

        # Temporal bundling / decoding
        if self.use_cnn:
            out = self.output_head(x.unsqueeze(1)).squeeze(1)  # [N, T]
        else:
            out = self.output_head(x)  # [N, T]

        return out

    def __repr__(self):
        return f"MeshGraphNet(layers={self.num_layers}, hidden_dim={self.hidden_dim})"
