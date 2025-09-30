import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import ngsolve as ng

class MeshProblem:
    """Container for a single training problem with mesh, initial condition, and physics parameters."""

    def __init__(self, mesh, graph_data, initial_condition, alpha, time_config, mesh_config, problem_id):
        self.mesh = mesh
        self.graph_data = graph_data
        self.initial_condition: np.ndarray = initial_condition
        self.alpha = alpha
        self.time_config: TimeConfig = time_config
        self.mesh_config: MeshConfig = mesh_config
        self.problem_id = problem_id
        
        # Statistics for logging
        self.n_nodes = graph_data.num_nodes
        self.n_edges = graph_data.num_edges

        self.boundary_values = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}  # Default BCs

        self.source_function = None  # Default: no source term

@dataclass
class TimeConfig:
    """Configuration for time integration."""
    
    dt: float = 0.01       # Time step size
    t_final: float = 1.0   # Final time
    time_steps: np.ndarray = None
    time_steps_export: np.ndarray = None
    num_steps: int = 0

    def __post_init__(self):
        self.time_steps = np.arange(self.dt, self.t_final + self.dt, self.dt)
        self.time_steps_export = np.arange(0.0, self.t_final + self.dt, self.dt)
        self.num_steps = int(self.t_final / self.dt)

@dataclass
class TrainingConfig:
    """Configuration for PI-GNN training."""
    
    # Time parameters
    time_config: TimeConfig = TimeConfig(
        t_final=1.0,
        dt=0.01
    )

    # Training parameters
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 2  # Number of problems per batch
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100

    patience: int = 30  # Early stopping patience

@dataclass
class MeshConfig:
    """Configuration for mesh generation."""
    
    maxh: float = 0.1  # Maximum element size
    order: int = 1     # Finite element order
    dim: int = 2       # Spatial dimension (2D or 3D)
    dirichlet_boundaries: Optional[List[str]] = None  # Names of boundaries with Dirichlet BCs

    mesh_type: str = "rectangle"  # Type of mesh: rectangle, circle, etc.