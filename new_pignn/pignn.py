"""
Physics-Informed Graph Neural Network (PIGNN)
A simplified implementation of a physics-informed graph neural network for solving PDEs.
Uses message passing with temporal bundling for efficient time series prediction.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter


class Swish(nn.Module):
    """Swish activation function"""
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class PIGNN_Layer(MessagePassing):
    """
    Message passing layer for Physics-Informed Graph Neural Network
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: int, edge_features: int):
        super(PIGNN_Layer, self).__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        # Message networks - 2*node_features + edge_features
        message_input_size = 2 * in_features + edge_features
        self.message_net_1 = nn.Sequential(
            nn.Linear(message_input_size, hidden_features),
            nn.ReLU()
        )
        self.message_net_2 = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU()
        )

        # Update networks
        self.update_net_1 = nn.Sequential(
            nn.Linear(in_features + hidden_features, hidden_features),
            nn.ReLU()
        )
        self.update_net_2 = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        """Propagate messages along edges"""
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """Message update"""
        message = self.message_net_1(torch.cat((x_i, x_j, edge_attr), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x):
        """Node update with residual connection"""
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update  # Residual connection
        else:
            return update


class PIGNN(nn.Module):
    """
    Physics-Informed Graph Neural Network
    
    A simplified implementation that uses message passing layers to solve PDEs.
    The model processes node features, edge attributes, and global information
    to predict temporal evolution through a 1D CNN decoder.
    """
    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        global_input_size: int = 1,
        hidden_features: int = 128,
        message_passing_steps: int = 12,
        time_window: int = 20,
        device: str = "cpu",
    ):
        super(PIGNN, self).__init__()
        
        self.hidden_features = hidden_features
        self.time_window = time_window
        self.edge_input_size = edge_input_size
        self.device = torch.device(device)
        
        # Input embedding MLP - combines all input features
        total_input_size = node_input_size + 2 + global_input_size  # node + pos + global
        self.embedding_mlp = nn.Sequential(
            nn.Linear(total_input_size, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU()
        )
        
        # Message passing layers
        self.gnn_layers = nn.ModuleList([
            PIGNN_Layer(
                in_features=hidden_features,
                hidden_features=hidden_features,
                out_features=hidden_features,
                edge_features=self.edge_input_size
            ) for _ in range(message_passing_steps)
        ])
        
        # 1D CNN decoder for temporal bundling
        if time_window == 20:
            self.output_mlp = nn.Sequential(
                nn.Conv1d(1, 8, 15, stride=4),
                nn.ReLU(),
                nn.Conv1d(8, 1, 10, stride=1)
            )
        elif time_window == 25:
            self.output_mlp = nn.Sequential(
                nn.Conv1d(1, 8, 16, stride=3),
                nn.ReLU(),
                nn.Conv1d(8, 1, 14, stride=1)
            )
        elif time_window == 50:
            self.output_mlp = nn.Sequential(
                nn.Conv1d(1, 8, 12, stride=2),
                nn.ReLU(),
                nn.Conv1d(8, 1, 10, stride=1)
            )
        elif time_window == 1:
            self.output_mlp = nn.Linear(hidden_features, 1)
        else:
            # Adaptive version for other time windows
            self.output_mlp = nn.Sequential(
                nn.Conv1d(1, 8, 15, stride=4),
                nn.ReLU(),
                nn.Conv1d(8, 1, 10, stride=1),
                nn.AdaptiveMaxPool1d(time_window)
            )
    
    def to(self, device):
        """Override to update internal device tracking"""
        self.device = torch.device(device)
        return super().to(device)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of the PIGNN model
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: node features [N, node_input_size]
                - edge_index: edge connectivity [2, E]
                - edge_attr: edge features [E, edge_input_size]
                - pos: node positions [N, 2] (optional)
                - global_attr: global features [B, global_input_size] (optional)
                - batch: batch assignment [N] (optional)
        
        Returns:
            torch.Tensor: predicted temporal evolution [N, time_window]
        """
        # Extract input data and move to device
        if isinstance(data, tuple):
            data, aux = data
            
        x = data.x.to(self.device) if hasattr(data, 'x') else data['x'].to(self.device)
        edge_index = data.edge_index.to(self.device) if hasattr(data, 'edge_index') else data['edge_index'].to(self.device)
        edge_attr = data.edge_attr.to(self.device) if hasattr(data, 'edge_attr') else data['edge_attr'].to(self.device)
        
        # Handle position information
        if hasattr(data, 'pos') and data.pos is not None:
            pos = data.pos.to(self.device)
            pos_x = pos[:, 0:1] if pos.size(1) > 0 else torch.zeros(x.size(0), 1, device=self.device)
            pos_y = pos[:, 1:2] if pos.size(1) > 1 else torch.zeros(x.size(0), 1, device=self.device)
        else:
            pos_x = torch.zeros(x.size(0), 1, device=self.device)
            pos_y = torch.zeros(x.size(0), 1, device=self.device)
        
        # Handle global features
        if hasattr(data, 'global_attr') and data.global_attr is not None:
            global_attr = data.global_attr.to(self.device)
            if global_attr.dim() == 1:
                global_attr = global_attr.unsqueeze(0)
        else:
            global_attr = torch.zeros(1, 1, device=self.device)
        
        # Handle batch information
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch.to(self.device)
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        
        # Broadcast global features to nodes
        global_features = global_attr[batch]  # [N, global_input_size]
        
        # Combine all input features  
        node_input = torch.cat([x, pos_x, pos_y, global_features], dim=-1)
        
        # Encoder: embed raw features to hidden space
        h = self.embedding_mlp(node_input)
        
        # Processor: message passing layers
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_index, edge_attr)
        
        # Decoder: 1D CNN for temporal bundling
        # [N, hidden_features] -> [N, 1, hidden_features] -> [N, time_window]
        output = self.output_mlp(h.unsqueeze(1)).squeeze(1)
        
        return output
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------- Example usage ---------

if __name__ == "__main__":
    # Create dummy data for testing
    N, E = 100, 300
    node_input_size, edge_input_size, global_input_size = 6, 4, 2
    
    # Create sample data
    x = torch.randn(N, node_input_size)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, edge_input_size)
    pos = torch.randn(N, 2)
    global_attr = torch.randn(1, global_input_size)
    batch = torch.zeros(N, dtype=torch.long)
    
    data = Data(
        x=x, 
        edge_index=edge_index, 
        edge_attr=edge_attr, 
        pos=pos,
        global_attr=global_attr, 
        batch=batch
    )
    
    # Create model
    model = PIGNN(
        node_input_size=node_input_size,
        edge_input_size=edge_input_size,
        global_input_size=global_input_size,
        hidden_features=128,
        message_passing_steps=12,
        time_window=20,
        device="cpu"
    )
    
    # Forward pass
    output = model(data)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {PIGNN.count_parameters(model):,}")
