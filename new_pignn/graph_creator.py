import torch
import numpy as np
import ngsolve as ng
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List, Dict, Tuple
from torch_geometric.data import Data

try:
    from mesh_utils import build_graph_from_mesh, create_free_node_subgraph, create_neumann_values, create_dirichlet_values, create_robin_values
except ModuleNotFoundError:
    from .mesh_utils import build_graph_from_mesh, create_free_node_subgraph, create_neumann_values, create_dirichlet_values, create_robin_values

class GraphCreator:
    def __init__(self, mesh: ng.Mesh, n_neighbors: int = 8, dirichlet_names: List[str] = None, 
                 neumann_names: List[str] = None, robin_names: List[str] = None, connectivity_method: str = "fem"):
        """
        Initialize GraphCreator for PI-MGN graph construction.
        
        Args:
            mesh: NGSolve mesh object
            n_neighbors: Number of neighbors for k-NN connectivity (ignored for FEM)
            dirichlet_names: List of Dirichlet boundary names
            neumann_names: List of Neumann boundary names
            robin_names: List of Robin boundary names
            connectivity_method: "fem" (mesh connectivity) or "knn" (k-nearest neighbors)
        """
        self.mesh = mesh
        self.n_neighbors = n_neighbors
        self.dirichlet_names = dirichlet_names or []
        self.neumann_names = neumann_names or []
        self.robin_names = robin_names or []
        self.connectivity_method = connectivity_method

    def create_graph(self, T_current: Optional[np.ndarray] = None, t_scalar: float = 0.0,
                   material_node_field: Optional[np.ndarray] = None, neumann_values: Optional[np.ndarray] = None,
                   dirichlet_values: Optional[np.ndarray] = None, robin_values: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                   source_values: Optional[np.ndarray] = None,
                   add_self_loops: bool = True, device: Optional[torch.device] = None) -> Tuple[Data, Dict]:
        """
        Create PI-MGN graph from mesh according to the paper's methodology.
        
        Args:
            T_current: Current temperature field (N,) - optional
            t_scalar: Current time scalar
            material_node_field: Per-node material properties (N,) - optional  
            neumann_values: Per-node Neumann boundary values (N,) - optional, h_N values for flux BC
            dirichlet_values: Per-node Dirichlet boundary values (N,) - optional, prescribed values for Dirichlet BC
            robin_values: Tuple of (h_values, amb_values) for Robin BCs - optional
            source_values: Per-node source term values (N,) - optional, volumetric heat source
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
            robin_names=self.robin_names,
            T_current=T_current,
            t_scalar=t_scalar,
            material_node_field=material_node_field,
            neumann_values=neumann_values,
            dirichlet_values=dirichlet_values,
            robin_values=robin_values,
            source_values=source_values,
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

    def create_neumann_values(self, pos, aux_data, neumann_names, flux_values=None, flux_magnitude=1.0, seed=None):
        return create_neumann_values(pos, aux_data, neumann_names, flux_values, flux_magnitude, seed)

    def create_dirichlet_values(self, pos, aux_data, dirichlet_names, boundary_values=None,
                               temperature_function=None, homogeneous_value=0.0, 
                               inhomogeneous_type="constant", seed=None):
        """
        Create Dirichlet boundary values according to PI-MGN methodology.
        
        For nodes that belong to multiple Dirichlet boundaries, higher values take priority
        over lower values. This ensures consistent boundary condition enforcement when
        boundaries intersect at corners or edges.
        
        Args:
            pos: Node positions (N, 2) - can be torch tensor or numpy array
            aux_data: Auxiliary data containing node type information
            dirichlet_names: List of Dirichlet boundary names or single boundary name
            boundary_values: Dict mapping boundary names to constant values, or single value for all boundaries
            temperature_function: Custom function f(x, y) -> temperature (optional)
            homogeneous_value: Default value if boundary_values not provided (backward compatibility)
            inhomogeneous_type: Type of BC - "constant" (use boundary_values), or legacy types
            seed: Random seed for reproducible values
            
        Returns:
            dirichlet_values: Per-node Dirichlet boundary values (N,) with higher values prioritized at corner nodes
        """
        return create_dirichlet_values(pos, aux_data, dirichlet_names, boundary_values,
                                     temperature_function, homogeneous_value, inhomogeneous_type, seed)

    def create_robin_values(self, pos, aux_data, robin_names, robin_values=None, h_default=1.0, amb_default=0.0, seed=None):
        return create_robin_values(pos, aux_data, robin_names, robin_values, h_default, amb_default, seed)

    def visualize_graph(self, data: Data, aux: Dict, figsize: Tuple[int, int] = (24, 6),
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
        
        # Create figure with subplots - add fourth subplot for Dirichlet values
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
        
        # Get positions from data
        pos_dict = {i: data.pos[i].numpy() for i in range(data.num_nodes)}
        
        # Color nodes by type - handle both full graph and subgraph cases
        if only_free_nodes:
            # For free node subgraphs, all nodes are interior (green)
            node_colors = ['green'] * data.num_nodes
            node_type_counts = {0: 0, 1: 0, 2: data.num_nodes, 3: 0}  # No Dirichlet/Neumann/Robin in free nodes
        else:
            # For full graphs, use the node types from aux
            node_types = aux['node_types'].numpy()
            # Ensure we only process as many node types as we have nodes in the graph
            node_types = node_types[:data.num_nodes]
            node_colors = []
            node_type_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            
            for node_type in node_types:
                if node_type == 0:  # Dirichlet
                    node_colors.append('red')
                    node_type_counts[0] += 1
                elif node_type == 1:  # Neumann
                    node_colors.append('blue')
                    node_type_counts[1] += 1
                elif node_type == 3:  # Robin
                    node_colors.append('orange')
                    node_type_counts[3] += 1
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
            mpatches.Patch(color='orange', label=f'Robin ({node_type_counts[3]})'),
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
        
        # Plot 3: Neumann values visualization (if available)
        if data.x.shape[1] > 6:  # Has Neumann data (7th feature)
            neumann_values = data.x[:, 6].numpy()  # Neumann values are 7th feature (index 6)
            neumann_mask = aux['neumann_mask']
            
            # Only show non-zero Neumann values or Neumann boundary nodes
            if neumann_mask.any() or (neumann_values != 0).any():
                # Create scatter plot colored by Neumann values
                scatter = ax3.scatter(data.pos[:, 0].numpy(), data.pos[:, 1].numpy(), 
                                    c=neumann_values, cmap='viridis', s=node_size, alpha=0.8)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax3)
                cbar.set_label('Neumann Values, ($h_N$)')
                
                # Highlight Neumann boundary nodes with red circles
                if neumann_mask.any():
                    neumann_indices = torch.where(neumann_mask)[0]
                    neumann_pos = data.pos[neumann_indices].numpy()
                    ax3.scatter(neumann_pos[:, 0], neumann_pos[:, 1], 
                               s=node_size*2, facecolors='none', edgecolors='red', linewidth=2)
                
                ax3.set_title(f'Neumann Values\\nRange: [{neumann_values.min():.3f}, {neumann_values.max():.3f}]')
            else:
                ax3.text(0.5, 0.5, 'No Neumann values\\navailable', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Neumann Values (N/A)')
        else:
            ax3.text(0.5, 0.5, 'No Neumann data\\navailable', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Neumann Values (N/A)')
        
        ax3.set_aspect('equal')
        ax3.axis('off')
        
        # Plot 4: Dirichlet values visualization (if available)
        if data.x.shape[1] > 7:  # Has Dirichlet data (8th feature)
            dirichlet_values = data.x[:, 7].numpy()  # Dirichlet values are 8th feature (index 7)
            dirichlet_mask = aux['dirichlet_mask']
            
            # Only show non-zero Dirichlet values or Dirichlet boundary nodes
            if dirichlet_mask.any() or (dirichlet_values != 0).any():
                # Create scatter plot colored by Dirichlet values
                scatter = ax4.scatter(data.pos[:, 0].numpy(), data.pos[:, 1].numpy(), 
                                    c=dirichlet_values, cmap='plasma', s=node_size, alpha=0.8)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax4)
                cbar.set_label('Dirichlet Values')
                
                # Highlight Dirichlet boundary nodes with blue circles
                if dirichlet_mask.any():
                    dirichlet_indices = torch.where(dirichlet_mask)[0]
                    dirichlet_pos = data.pos[dirichlet_indices].numpy()
                    ax4.scatter(dirichlet_pos[:, 0], dirichlet_pos[:, 1], 
                               s=node_size*2, facecolors='none', edgecolors='blue', linewidth=2)
                
                ax4.set_title(f'Dirichlet Values\\nRange: [{dirichlet_values.min():.3f}, {dirichlet_values.max():.3f}]')
            else:
                ax4.text(0.5, 0.5, 'No Dirichlet values\\navailable', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Dirichlet Values (N/A)')
        else:
            ax4.text(0.5, 0.5, 'No Dirichlet data\\navailable', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Dirichlet Values (N/A)')
        
        ax4.set_aspect('equal')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to: {save_path}")
        
        # plt.show()
    
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
        print(f"  Robin:     {(node_types == 3).sum().item()} ({100 * (node_types == 3).sum() / len(node_types):.1f}%)")
        
        print(f"\\nFeature Dimensions:")
        print(f"  Node features: {data.x.shape}")
        print(f"  Feature breakdown: [one_hot(4), T(1), t(1), material(1), neumann(1), dirichlet(1), robin_h(1), robin_amb(1)] = 11 total")
        print(f"  Edge features: {data.edge_attr.shape}")
        print(f"  Global features: {data.global_attr.shape}")
        print(f"  Positions: {data.pos.shape}")
        
        if data.x.shape[1] > 4:  # Has temperature data (index 4)
            temp_values = data.x[:, 4]
            print(f"\\nTemperature Field:")
            print(f"  Range: [{temp_values.min():.4f}, {temp_values.max():.4f}]")
            print(f"  Mean: {temp_values.mean():.4f} ± {temp_values.std():.4f}")
        
        if data.x.shape[1] > 7:  # Has Neumann data (index 7)
            neumann_values = data.x[:, 7]
            neumann_mask = aux['neumann_mask']
            print(f"\\nNeumann Values:")
            print(f"  Range: [{neumann_values.min():.4f}, {neumann_values.max():.4f}]")
            print(f"  Mean: {neumann_values.mean():.4f} ± {neumann_values.std():.4f}")
            print(f"  Non-zero values: {(neumann_values != 0).sum().item()}/{len(neumann_values)}")
            if neumann_mask.any():
                neumann_node_values = neumann_values[neumann_mask]
                print(f"  Neumann nodes range: [{neumann_node_values.min():.4f}, {neumann_node_values.max():.4f}]")
        
        if data.x.shape[1] > 8:  # Has Dirichlet data (index 8)
            dirichlet_values = data.x[:, 8]
            dirichlet_mask = aux['dirichlet_mask']
            print(f"\\nDirichlet Values:")
            print(f"  Range: [{dirichlet_values.min():.4f}, {dirichlet_values.max():.4f}]")
            print(f"  Mean: {dirichlet_values.mean():.4f} ± {dirichlet_values.std():.4f}")
            print(f"  Non-zero values: {(dirichlet_values != 0).sum().item()}/{len(dirichlet_values)}")
            if dirichlet_mask.any():
                dirichlet_node_values = dirichlet_values[dirichlet_mask]
                print(f"  Dirichlet nodes range: [{dirichlet_node_values.min():.4f}, {dirichlet_node_values.max():.4f}]")

        if data.x.shape[1] > 10:  # Has Robin data (indices 9, 10)
            h_values = data.x[:, 9]
            amb_values = data.x[:, 10]
            robin_mask = aux.get('robin_mask', torch.zeros_like(node_types, dtype=torch.bool))
            print(f"\\nRobin Values:")
            print(f"  h Range: [{h_values.min():.4f}, {h_values.max():.4f}]")
            print(f"  Amb Range: [{amb_values.min():.4f}, {amb_values.max():.4f}]")
            if robin_mask.any():
                h_node_values = h_values[robin_mask]
                amb_node_values = amb_values[robin_mask]
                print(f"  Robin nodes h range: [{h_node_values.min():.4f}, {h_node_values.max():.4f}]")
                print(f"  Robin nodes amb range: [{amb_node_values.min():.4f}, {amb_node_values.max():.4f}]")
    
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