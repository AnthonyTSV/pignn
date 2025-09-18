"""
Physics-Informed MeshGraphNet (PI-MGN) implementation.
Based on the methodology described in the paper.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from .blocks import EdgeBlock, NodeBlock, GlobalBlock, build_mlp


class PIMGN(nn.Module):
    """
    Physics-Informed MeshGraphNet for solving heat equation with strict Dirichlet BC enforcement.
    
    Architecture:
    - Encoder: Maps input features to hidden representations
    - Processor: L message passing steps with EdgeBlock, NodeBlock, GlobalBlock
    - Decoder: MLP to output temporal bundle predictions for FREE DOFs ONLY
    
    Key difference from standard MGN: Only predicts values for free degrees of freedom,
    following the strict BC enforcement methodology from the paper.
    """
    
    def __init__(self, 
                 node_input_size=6,     # [x, y, T_prev, node_type_one_hot(3)]
                 edge_input_size=4,     # [dx, dy, dist, dT]
                 global_input_size=2,   # [alpha, dt]
                 hidden_size=128,
                 num_layers=12,         # L=12 message passing steps as in paper
                 output_size=1,         # Single timestep output (temporal bundling handled separately)
                 device='cpu',
                 predict_free_dofs_only=True):  # New parameter for strict BC enforcement
        super(PIMGN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.predict_free_dofs_only = predict_free_dofs_only
        self.output_size = output_size
        
        # Encoders for each feature type
        self.node_encoder = build_mlp(node_input_size, hidden_size, hidden_size)
        self.edge_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.global_encoder = build_mlp(global_input_size, hidden_size, hidden_size)
        
        # Message passing blocks
        self.edge_blocks = nn.ModuleList([
            EdgeBlock(hidden_size) for _ in range(num_layers)
        ])
        self.node_blocks = nn.ModuleList([
            NodeBlock(hidden_size) for _ in range(num_layers)
        ])
        self.global_blocks = nn.ModuleList([
            GlobalBlock(hidden_size) for _ in range(num_layers)
        ])
        
        # Decoder for free DOFs only (when predict_free_dofs_only=True)
        # For strict BC enforcement, we only decode for free nodes
        if predict_free_dofs_only:
            # Simple MLP decoder for single timestep prediction
            # IMPORTANT: No LayerNorm for final decoder to allow proper temperature scaling
            self.decoder = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)
        else:
            # CNN decoder for temporal bundling (legacy)
            self.decoder = nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(hidden_size, output_size, kernel_size=1)
            )
    
    def forward(self, data):
        """
        Forward pass through the PI-MGN with strict BC enforcement.
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features for FREE DOFs only (N_free, node_input_size)
                - edge_index: Edge connectivity (2, E)
                - edge_attr: Edge features (E, edge_input_size)
                - global_attr: Global features (1, global_input_size)
                - batch: Batch information (optional)
                - free_node_indices: Indices of free nodes in original mesh (N_free,)
        
        Returns:
            Tensor: Predictions for FREE DOFs only (N_free, output_size)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        global_attr = data.global_attr
        batch = getattr(data, 'batch', None)
        
        # Encode features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        global_attr = self.global_encoder(global_attr)
        
        # Message passing for L layers
        for i in range(self.num_layers):
            # Edge update
            edge_attr_new = self.edge_blocks[i](x, edge_index, edge_attr, 
                                              global_attr, batch)
            
            # Node update
            x_new = self.node_blocks[i](x, edge_index, edge_attr_new, 
                                      global_attr, batch)
            
            # Global update
            global_attr_new = self.global_blocks[i](x_new, edge_attr_new, 
                                                   global_attr, batch)
            
            # Residual connections as in the original MGN
            x = x + x_new
            edge_attr = edge_attr + edge_attr_new
            global_attr = global_attr_new
        
        # Decode to get final predictions for free DOFs only
        if self.predict_free_dofs_only:
            # Simple MLP output for free DOFs
            predictions = self.decoder(x)
        else:
            # Legacy CNN output for all nodes
            x_reshaped = x.unsqueeze(2)
            predictions = self.decoder(x_reshaped).squeeze(2)
        
        return predictions


def build_features(graph_data, T_current, alpha_value, dt, device='cpu', fem_solver=None, free_dofs_only=True):
    """
    Build input features for PI-MGN with strict BC enforcement.
    
    Args:
        graph_data: Dictionary containing mesh information
        T_current: Current temperature values (N_total,) for ALL DOFs
        alpha_value: Thermal diffusivity value
        dt: Time step size
        device: PyTorch device
        fem_solver: FEMSolver instance for free DOF information
        free_dofs_only: If True, build graph for free DOFs only (strict BC enforcement)
    
    Returns:
        Data: PyTorch Geometric Data object with features for free DOFs only
    """
    pos = torch.tensor(graph_data['pos'], dtype=torch.float32, device=device)
    edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long, device=device)
    boundary_mask = torch.tensor(graph_data['boundary_mask'], dtype=torch.bool, device=device)
    
    T_current_tensor = torch.tensor(T_current, dtype=torch.float32, device=device)
    if T_current_tensor.dim() == 0:
        T_current_tensor = T_current_tensor.unsqueeze(0)
    
    N_total = pos.size(0)
    E_total = edge_index.size(1)
    
    if free_dofs_only and fem_solver is not None:
        # Build graph for FREE DOFs only (strict BC enforcement)
        free_indices = fem_solver.free_indices
        N_free = len(free_indices)
        
        # Create mapping from original node indices to free node indices
        free_index_map = {int(orig_idx): free_idx for free_idx, orig_idx in enumerate(free_indices)}
        
        # Filter positions and temperature for free nodes only
        pos_free = pos[free_indices]
        T_free = T_current_tensor[free_indices]
        
        # Node features for free nodes: [x, y, T_prev, node_type_one_hot(3)]
        # For free nodes, node_type = [0, 0, 1] (inner nodes)
        node_type_free = torch.zeros(N_free, 3, dtype=torch.float32, device=device)
        node_type_free[:, 2] = 1.0  # All free nodes are inner nodes
        
        node_features = torch.cat([
            pos_free,  # x, y coordinates
            T_free.unsqueeze(1),  # T_prev
            node_type_free  # one-hot node type
        ], dim=1)
        
        # Filter edges to include only free-to-free connections
        valid_edges = []
        edge_features_list = []
        
        for e in range(E_total):
            u_orig = int(edge_index[0, e])
            v_orig = int(edge_index[1, e])
            
            # Only keep edges between free nodes
            if u_orig in free_index_map and v_orig in free_index_map:
                u_free = free_index_map[u_orig]
                v_free = free_index_map[v_orig]
                valid_edges.append([u_free, v_free])
                
                # Compute edge features for this edge
                pos_u = pos_free[u_free]
                pos_v = pos_free[v_free]
                dx_dy = pos_v - pos_u
                dist = torch.norm(dx_dy, keepdim=True)
                
                T_u = T_free[u_free]
                T_v = T_free[v_free]
                dT = (T_v - T_u).unsqueeze(0)
                
                edge_feature = torch.cat([dx_dy, dist, dT])
                edge_features_list.append(edge_feature)
        
        if len(valid_edges) == 0:
            # Handle case with no edges (shouldn't happen in practice)
            edge_index_free = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_features = torch.zeros((0, 4), dtype=torch.float32, device=device)
        else:
            edge_index_free = torch.tensor(valid_edges, dtype=torch.long, device=device).t()
            edge_features = torch.stack(edge_features_list)
        
        # Store original indices for reconstruction
        free_node_indices = torch.tensor(free_indices, dtype=torch.long, device=device)
        
    else:
        # Build graph for ALL nodes (legacy mode)
        node_type = torch.zeros(N_total, 3, dtype=torch.float32, device=device)
        node_type[boundary_mask, 0] = 1.0  # Dirichlet boundary nodes
        node_type[~boundary_mask, 2] = 1.0  # Inner nodes
        
        node_features = torch.cat([
            pos,  # x, y coordinates
            T_current_tensor.unsqueeze(1),  # T_prev
            node_type  # one-hot node type
        ], dim=1)
        
        # Edge features: [dx, dy, dist, dT]
        u_idx, v_idx = edge_index[0], edge_index[1]
        pos_u = pos[u_idx]
        pos_v = pos[v_idx]
        
        dx_dy = pos_v - pos_u
        dist = torch.norm(dx_dy, dim=1, keepdim=True)
        
        T_u = T_current_tensor[u_idx]
        T_v = T_current_tensor[v_idx]
        dT = (T_v - T_u).unsqueeze(1)
        
        edge_features = torch.cat([dx_dy, dist, dT], dim=1)
        edge_index_free = edge_index
        free_node_indices = torch.arange(N_total, dtype=torch.long, device=device)
    
    # Global features: [alpha, dt]
    global_features = torch.tensor([[alpha_value, dt]], dtype=torch.float32, device=device)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=node_features,
        edge_index=edge_index_free,
        edge_attr=edge_features,
        global_attr=global_features,
        free_node_indices=free_node_indices,  # Store for reconstruction
        pos=pos_free if free_dofs_only and fem_solver is not None else pos
    )
    
    return data
