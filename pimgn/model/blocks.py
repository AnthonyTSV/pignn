"""
Message passing blocks for PI-MGN according to the methodology in the paper.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter


def build_mlp(in_size, hidden_size, out_size, layers=2, lay_norm=True):
    """Build MLP as described in the paper with LayerNorm."""
    modules = []
    
    # Input layer
    modules.append(nn.Linear(in_size, hidden_size))
    modules.append(nn.ReLU())
    
    # Hidden layers
    for _ in range(layers - 2):
        modules.append(nn.Linear(hidden_size, hidden_size))
        modules.append(nn.ReLU())
    
    # Output layer
    modules.append(nn.Linear(hidden_size, out_size))
    
    mlp = nn.Sequential(*modules)
    
    if lay_norm:
        return nn.Sequential(mlp, nn.LayerNorm(normalized_shape=out_size))
    return mlp


class EdgeBlock(nn.Module):
    """
    Edge update function: e' = f_e(h_v, h_u, e, g)
    Based on equation in paper methodology section.
    """
    
    def __init__(self, hidden_size=128):
        super(EdgeBlock, self).__init__()
        self.hidden_size = hidden_size
        
        # f_e MLP: takes [h_v, h_u, e, g] -> e'
        edge_input_dim = 4 * hidden_size  # h_v + h_u + e + g
        self.edge_mlp = build_mlp(edge_input_dim, hidden_size, hidden_size)
    
    def forward(self, x, edge_index, edge_attr, global_attr, batch=None):
        """
        Args:
            x: Node features (N, hidden_size)
            edge_index: Edge connectivity (2, E)
            edge_attr: Edge features (E, hidden_size)
            global_attr: Global features (1, hidden_size) or (B, hidden_size)
            batch: Batch information for batched graphs
        """
        num_edges = edge_attr.size(0)
        
        # Get source and target node features
        source_idx, target_idx = edge_index[0], edge_index[1]
        x_source = x[source_idx]  # Features of source nodes
        x_target = x[target_idx]  # Features of target nodes
        
        # Expand global features to match edge dimensions
        if global_attr.size(0) == 1:
            global_features = global_attr.expand(num_edges, -1)
        else:
            # For batched graphs
            global_features = global_attr.expand(num_edges, -1)
        
        # Concatenate [h_target, h_source, e, g] as per paper
        edge_input = torch.cat([x_target, x_source, edge_attr, global_features], dim=-1)
        
        # Apply edge MLP
        updated_edges = self.edge_mlp(edge_input)
        
        return updated_edges


class NodeBlock(nn.Module):
    """
    Node update function: h' = f_v(h_v, agg_e, g)
    Based on equation in paper methodology section.
    """
    
    def __init__(self, hidden_size=128):
        super(NodeBlock, self).__init__()
        self.hidden_size = hidden_size
        
        # f_v MLP: takes [h_v, agg_e, g] -> h'
        node_input_dim = 3 * hidden_size  # h_v + agg_e + g
        self.node_mlp = build_mlp(node_input_dim, hidden_size, hidden_size)
    
    def forward(self, x, edge_index, edge_attr_updated, global_attr, batch=None):
        """
        Args:
            x: Node features (N, hidden_size)
            edge_index: Edge connectivity (2, E)
            edge_attr_updated: Updated edge features from EdgeBlock (E, hidden_size)
            global_attr: Global features (1, hidden_size) or (B, hidden_size)
            batch: Batch information for batched graphs
        """
        num_nodes = x.size(0)
        
        # Aggregate edge messages to nodes (receivers)
        # edge_index[1] contains target nodes (receivers)
        aggregated_edges = scatter(edge_attr_updated, edge_index[1], 
                                 dim=0, dim_size=num_nodes, reduce='sum')
        
        # Expand global features to match node dimensions
        if global_attr.size(0) == 1:
            global_features = global_attr.expand(num_nodes, -1)
        else:
            # For batched graphs
            global_features = global_attr.expand(num_nodes, -1)
        
        # Concatenate [h_v, agg_e, g] as per paper
        node_input = torch.cat([x, aggregated_edges, global_features], dim=-1)
        return self.node_mlp(node_input)


class GlobalBlock(nn.Module):
    """
    Global update function: g' = f_g(sum_nodes, sum_edges, g)
    Based on equation in paper methodology section.
    """
    
    def __init__(self, hidden_size=128):
        super(GlobalBlock, self).__init__()
        self.hidden_size = hidden_size
        
        # f_g MLP: takes [sum_nodes, sum_edges, g] -> g'
        global_input_dim = 3 * hidden_size  # sum_nodes + sum_edges + g
        self.global_mlp = build_mlp(global_input_dim, hidden_size, hidden_size)
    
    def forward(self, x, edge_attr_updated, global_attr, batch=None):
        """
        Args:
            x: Node features (N, hidden_size)
            edge_attr_updated: Updated edge features (E, hidden_size)
            global_attr: Global features (1, hidden_size) or (B, hidden_size)
            batch: Batch information for batched graphs
        """
        # Sum over all nodes and edges (permutation-invariant aggregation)
        if batch is not None:
            # For batched graphs, sum within each graph
            sum_nodes = scatter(x, batch, dim=0, reduce='sum')
            # For edges, we need to map edges to their corresponding batch
            # This requires additional batch information for edges
            sum_edges = torch.sum(edge_attr_updated, dim=0, keepdim=True)
        else:
            # Single graph case
            sum_nodes = torch.sum(x, dim=0, keepdim=True)
            sum_edges = torch.sum(edge_attr_updated, dim=0, keepdim=True)
        
        # Concatenate [sum_nodes, sum_edges, g] as per paper
        global_input = torch.cat([sum_nodes, sum_edges, global_attr], dim=-1)
        return self.global_mlp(global_input)
