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
from torch_geometric.nn import InstanceNorm
from torch_geometric.data import Data
from typing import List, Tuple, Optional
import numpy as np

class MeshGraphNet_Layer(MessagePassing):
    """
    Message passing layer for data-driven MeshGraphNet.
    Simplified version that works with standard graph structures from GraphCreator.
    """

    def __init__(self, input_dim_node: int, input_dim_edge: int, hidden_dim: int):
        super(MeshGraphNet_Layer, self).__init__(aggr="mean")
        self.input_dim_node = input_dim_node
        self.input_dim_edge = input_dim_edge
        self.hidden_dim = hidden_dim

        # Message network (processes concatenated node features and edge features)
        self.message_net = nn.Sequential(
            nn.Linear(2 * input_dim_node + input_dim_edge, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Update network (processes node features and aggregated messages)
        self.update_net = nn.Sequential(
            nn.Linear(input_dim_node + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim_node),
            nn.ReLU(),
        )

        # Layer normalization
        self.norm = InstanceNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ):
        """Forward pass through message passing layer."""
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if batch is not None:
            out = self.norm(out, batch)
        return out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor):
        """Create messages by concatenating sender, receiver, and edge features."""
        message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_net(message_input)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor):
        """Update node features using aggregated messages and current node features."""
        update_input = torch.cat([x, aggr_out], dim=-1)
        update = self.update_net(update_input)
        # Residual connection
        return x + update


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
        args=None,
    ):
        super(MeshGraphNet, self).__init__()

        self.time_window = getattr(args, "time_window", 20) if args else 1

        self.input_dim_node = input_dim_node
        self.input_dim_edge = input_dim_edge
        self.hidden_dim = hidden_dim
        self.output_dim = self.time_window  # Predict multiple time steps

        # Number of message passing layers
        self.num_layers = getattr(args, "num_layers", 12) if args else 12

        # Encoder: Map input node features to hidden dimension
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim_node, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Edge encoder: Map input edge features to hidden dimension
        self.edge_encoder = nn.Sequential(
            nn.Linear(input_dim_edge, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Message passing layers
        self.gnn_layers = nn.ModuleList(
            [
                MeshGraphNet_Layer(hidden_dim, hidden_dim, hidden_dim)
                for _ in range(self.num_layers)
            ]
        )

        # Decoder: Map hidden features to output
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        if self.time_window == 20:
            self.output_mlp = nn.Sequential(
                nn.Conv1d(1, 8, 15, stride=4), nn.ReLU(), 
                nn.Conv1d(8, 1, 10, stride=1)
            )
        elif self.time_window == 5:
            self.output_mlp = nn.Sequential(
                nn.Conv1d(1, 8, 3, stride=1, padding=1), nn.ReLU(), 
                nn.Conv1d(8, 1, 3, stride=1, padding=1)
            )
        elif self.time_window == 10:
            self.output_mlp = nn.Sequential(
                nn.Conv1d(1, 8, 5, stride=1, padding=2), nn.ReLU(), 
                nn.Conv1d(8, 1, 5, stride=1, padding=2)
            )
        elif self.time_window == 1:
            # Special case for single time step prediction (no temporal bundling)
            self.output_mlp = None  # Skip temporal bundling
        else:
            # Default case for other time windows
            self.output_mlp = nn.Sequential(
                nn.Conv1d(1, 8, min(self.time_window//2, 7), stride=1, padding='same'), nn.ReLU(), 
                nn.Conv1d(8, 1, min(self.time_window//2, 7), stride=1, padding='same')
            )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of MeshGraphNet.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Predicted node values (normalized)
        """
        # Move normalization tensors to same device as data

        x = data.x
        edge_attr = data.edge_attr

        # Encode node and edge features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Message passing
        batch = getattr(data, "batch", None)
        for layer in self.gnn_layers:
            x = layer(x, data.edge_index, edge_attr, batch)

        # Decode to output
        # output = self.node_decoder(x)

        # Temporal bundling
        output = self.output_mlp(x[:, None]).squeeze(1)

        return output

    def __repr__(self):
        return f"MeshGraphNet(layers={self.num_layers}, hidden_dim={self.hidden_dim})"
