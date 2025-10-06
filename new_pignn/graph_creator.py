import torch
import numpy as np
import ngsolve as ng
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List, Dict, Tuple
from torch_geometric.data import Data

from mesh_utils import build_graph_from_mesh, create_free_node_subgraph

class GraphCreator:
    def __init__(self, mesh: ng.Mesh, n_neighbors: int = 8, dirichlet_names: List[str] = None, 
                 neumann_names: List[str] = None, connectivity_method: str = "fem"):
        """
        Initialize GraphCreator for PI-MGN graph construction.
        
        Args:
            mesh: NGSolve mesh object
            n_neighbors: Number of neighbors for k-NN connectivity (ignored for FEM)
            dirichlet_names: List of Dirichlet boundary names
            neumann_names: List of Neumann boundary names
            connectivity_method: "fem" (mesh connectivity) or "knn" (k-nearest neighbors)
        """
        self.mesh = mesh
        self.n_neighbors = n_neighbors
        self.dirichlet_names = dirichlet_names or []
        self.neumann_names = neumann_names or []
        self.connectivity_method = connectivity_method

    def create_graph(self, T_current: Optional[np.ndarray] = None, t_scalar: float = 0.0,
                   material_node_field: Optional[np.ndarray] = None, add_self_loops: bool = True,
                   device: Optional[torch.device] = None) -> Tuple[Data, Dict]:
        """
        Create PI-MGN graph from mesh according to the paper's methodology.
        
        Args:
            T_current: Current temperature field (N,) - optional
            t_scalar: Current time scalar
            material_node_field: Per-node material properties (N,) - optional  
            add_self_loops: Whether to add self-loops to all nodes
            device: Target device for tensors (CPU if None)
            
        Returns:
            data: PyTorch Geometric Data object with node/edge/global features
            aux: Auxiliary data dictionary with mappings and metadata
        """
        return build_graph_from_mesh(
            mesh=self.mesh,
            dirichlet_names=self.dirichlet_names,
            neumann_names=self.neumann_names,
            T_current=T_current,
            t_scalar=t_scalar,
            material_node_field=material_node_field,
            connectivity_method=self.connectivity_method,
            n_neighbors=self.n_neighbors,
            add_self_loops=add_self_loops,
            device=device
        )
    
    def create_free_node_subgraph(self, full_graph: Data, aux: Dict) -> Tuple[Data, torch.Tensor, Dict]:
        """
        Create a subgraph containing only free (non-Dirichlet) nodes.
        
        Args:
            full_graph: Full PyTorch Geometric Data object
            aux: Auxiliary data dictionary from create_graph
            
        Returns:
            free_graph: Subgraph with only free nodes
            node_mapping: Tensor mapping free node indices to original graph indices
            new_aux: Updated auxiliary data for the subgraph
        """
        return create_free_node_subgraph(full_graph, aux)

    def visualize_graph(self, data: Data, aux: Dict, figsize: Tuple[int, int] = (12, 5),
                       node_size: int = 50, edge_alpha: float = 0.6, save_path: Optional[str] = None, only_free_nodes: bool = False):
        """
        Visualize the graph using NetworkX and matplotlib.
        
        Args:
            data: PyTorch Geometric Data object
            aux: Auxiliary data from create_graph
            figsize: Figure size (width, height)
            node_size: Size of nodes in visualization
            edge_alpha: Transparency of edges
            save_path: Path to save visualization (optional)
            only_free_nodes: Whether this is a free node subgraph visualization
        """
        # Convert to NetworkX graph
        G = self._pyg_to_networkx(data)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Get positions from data
        pos_dict = {i: data.pos[i].numpy() for i in range(data.num_nodes)}
        
        # Color nodes by type - handle both full graph and subgraph cases
        if only_free_nodes:
            # For free node subgraphs, all nodes are interior (green)
            node_colors = ['green'] * data.num_nodes
            node_type_counts = {0: 0, 1: 0, 2: data.num_nodes}  # No Dirichlet/Neumann in free nodes
        else:
            # For full graphs, use the node types from aux
            node_types = aux['node_types'].numpy()
            # Ensure we only process as many node types as we have nodes in the graph
            node_types = node_types[:data.num_nodes]
            node_colors = []
            node_type_counts = {0: 0, 1: 0, 2: 0}
            
            for node_type in node_types:
                if node_type == 0:  # Dirichlet
                    node_colors.append('red')
                    node_type_counts[0] += 1
                elif node_type == 1:  # Neumann
                    node_colors.append('blue')
                    node_type_counts[1] += 1
                else:  # Interior
                    node_colors.append('green')
                    node_type_counts[2] += 1
        
        # Plot 1: Graph structure with node types
        nx.draw_networkx_nodes(G, pos_dict, node_color=node_colors, 
                              node_size=node_size, ax=ax1, alpha=0.8)
        nx.draw_networkx_edges(G, pos_dict, alpha=edge_alpha, ax=ax1, 
                              edge_color='gray', width=0.5)
        
        graph_type = "Free Nodes" if only_free_nodes else f'{self.connectivity_method.upper()} Graph'
        ax1.set_title(f'{graph_type} Structure\\n'
                     f'Nodes: {data.num_nodes}, Edges: {data.num_edges}')
        ax1.set_aspect('equal')
        ax1.axis('off')
        
        # Add legend for node types
        legend_elements = [
            mpatches.Patch(color='red', label=f'Dirichlet ({node_type_counts[0]})'),
            mpatches.Patch(color='blue', label=f'Neumann ({node_type_counts[1]})'),
            mpatches.Patch(color='green', label=f'Interior ({node_type_counts[2]})')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Plot 2: Temperature field visualization (if available)
        if data.x.shape[1] > 3:  # Has temperature data
            temp_values = data.x[:, 3].numpy()  # Temperature is 4th feature
            
            # Create scatter plot colored by temperature
            scatter = ax2.scatter(data.pos[:, 0].numpy(), data.pos[:, 1].numpy(), 
                                c=temp_values, cmap='coolwarm', s=node_size, alpha=0.8)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Temperature')
            
            ax2.set_title(f'Temperature Field\\nRange: [{temp_values.min():.3f}, {temp_values.max():.3f}]')
        else:
            ax2.text(0.5, 0.5, 'No temperature data\\navailable', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Temperature Field (N/A)')
        
        ax2.set_aspect('equal')
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to: {save_path}")
        
        plt.show()
    
    def print_graph_stats(self, data: Data, aux: Dict):
        """
        Print comprehensive statistics about the created graph.
        
        Args:
            data: PyTorch Geometric Data object
            aux: Auxiliary data from create_graph
        """
        node_types = aux['node_types']
        
        print(f"\\n=== Graph Statistics ({self.connectivity_method.upper()}) ===")
        print(f"Nodes: {data.num_nodes}")
        print(f"Edges: {data.num_edges}")
        print(f"Self-loops: {(data.edge_index[0] == data.edge_index[1]).sum().item()}")
        print(f"Average degree: {(2 * data.num_edges) / data.num_nodes:.2f}")
        
        print(f"\\nNode Types:")
        print(f"  Dirichlet: {(node_types == 0).sum().item()} ({100 * (node_types == 0).sum() / len(node_types):.1f}%)")
        print(f"  Neumann:   {(node_types == 1).sum().item()} ({100 * (node_types == 1).sum() / len(node_types):.1f}%)")
        print(f"  Interior:  {(node_types == 2).sum().item()} ({100 * (node_types == 2).sum() / len(node_types):.1f}%)")
        
        print(f"\\nFeature Dimensions:")
        print(f"  Node features: {data.x.shape}")
        print(f"  Edge features: {data.edge_attr.shape}")
        print(f"  Global features: {data.global_attr.shape}")
        print(f"  Positions: {data.pos.shape}")
        
        if data.x.shape[1] > 3:  # Has temperature data
            temp_values = data.x[:, 3]
            print(f"\\nTemperature Field:")
            print(f"  Range: [{temp_values.min():.4f}, {temp_values.max():.4f}]")
            print(f"  Mean: {temp_values.mean():.4f} ± {temp_values.std():.4f}")
    
    def _pyg_to_networkx(self, data: Data) -> nx.Graph:
        """
        Convert PyTorch Geometric Data to NetworkX graph.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            G: NetworkX graph
        """
        G = nx.Graph()
        
        # Add nodes
        for i in range(data.num_nodes):
            G.add_node(i)
        
        # Add edges (convert to undirected by removing duplicates)
        edge_list = data.edge_index.t().numpy()
        for src, dst in edge_list:
            if src != dst:  # Skip self-loops for cleaner visualization
                G.add_edge(src, dst)
        
        return G