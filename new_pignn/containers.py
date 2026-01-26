import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import torch
import ngsolve as ng

r_star = 70 * 1e-3  # m
A_star = 4.8 * 1e-4  # Wb/m
mu_star = 4 * 3.1415926535e-7 # H/m
J_star = A_star / (r_star**2 * mu_star)

class MeshProblem:
    """Container for a single training problem with mesh, initial condition, and physics parameters."""

    def __init__(
        self,
        mesh,
        graph_data,
        initial_condition,
        alpha,
        time_config,
        mesh_config,
        problem_id,
    ):
        self.mesh = mesh
        self.graph_data = graph_data
        self.initial_condition: np.ndarray = initial_condition
        self.alpha = alpha
        self.alpha_coefficient = None  # Optional spatially varying coefficient function
        self.time_config: TimeConfig = time_config
        self.mesh_config: MeshConfig = mesh_config
        self.problem_id = problem_id

        # Statistics for logging
        self.n_nodes = graph_data.num_nodes
        self.n_edges = graph_data.num_edges

        self.boundary_values = {
            "left": 0.0,
            "right": 0.0,
            "top": 0.0,
            "bottom": 0.0,
        }  # Default Dirichlet BCs
        self.dirichlet_values_array = None  # To be set
        self.neumann_values = (
            {}
        )  # Neumann boundary conditions: {boundary_name: flux_value}
        self.neumann_values_array = None  # To be set
        self.robin_values = {}  # Robin boundary conditions: {boundary_name: (h, T_amb)}
        self.robin_values_array = (
            None  # To be set (tuple of arrays: (h_array, amb_array))
        )

        self.source_function = None  # Default: no source term
        self.nonlinear_source_params = (
            None  # Optional parameters for nonlinear heat sources
        )
        self.material_fraction_field = None  # Optional per-node material descriptor
        self.material_field = (
            None  # Optional per-node physical property (e.g., diffusivity)
        )

    def set_dirichlet_values(self, boundary_values: dict):
        """Set Dirichlet boundary conditions."""
        self.boundary_values = boundary_values

    def set_neumann_values(self, neumann_values: dict):
        """Set Neumann boundary conditions."""
        # example: {"top": 5.0} for a flux of 5.0 on the "top" boundary
        self.neumann_values = neumann_values

    def set_neumann_values_array(self, neumann_values_array: np.ndarray):
        """Set Neumann values array for all nodes."""
        self.neumann_values_array = neumann_values_array

    def set_robin_values(self, robin_values: dict):
        """Set Robin boundary conditions."""
        # example: {"top": (10.0, 20.0)} for h=10.0, T_amb=20.0 on "top"
        self.robin_values = robin_values

    def set_robin_values_array(self, robin_values_array: Tuple[np.ndarray, np.ndarray]):
        """Set Robin values arrays for all nodes (h_array, amb_array)."""
        self.robin_values_array = robin_values_array

    def set_dirichlet_values_array(self, dirichlet_values_array: np.ndarray):
        """Set Dirichlet values array for all nodes."""
        self.dirichlet_values_array = dirichlet_values_array

    def set_source_function(self, source_function):
        """Set source function."""
        self.source_function = source_function


class MeshProblemEM:
    """Container for a single training problem for electromagnetic simulations."""

    def __init__(self, mesh, graph_data, mesh_config, problem_id):
        self.mesh = mesh
        self.graph_data = graph_data
        self.mesh_config: MeshConfig = mesh_config
        self.problem_id = problem_id

        # Statistics for logging
        self.n_nodes = graph_data.num_nodes
        self.n_edges = graph_data.num_edges

        self.material_properties = {}
        self.dirichlet_values = {}
        self.dirichlet_values_array = None  # To be set
        self.material_field = None  # To be set
        self.sigma_field = None
        self.current_density_field = None  # To be set (current density at each node)

        ###
        
        self.profile_width_phys = 7 * 1e-3
        self.profile_height_phys = 7 * 1e-3

        self.profile_width = 7 * 1e-3 / r_star  # m
        self.profile_height = 7 * 1e-3 / r_star  # m

        # In the nondimensionalized system, mu0 = 1 (dimensionless)
        # because we scaled by mu_star = mu0
        self.mu0 = 1.0  # Normalized permeability of free space
        self.mu_r_workpiece = 100.0  # Relative permeability of workpiece
        self.mu_r_air = 1.0
        self.mu_r_coil = 1.0

        self.sigma_workpiece = 1e6  # Electrical conductivity [S/m]
        self.sigma_air = 0.0
        self.sigma_coil = 5.8e7

        # Coil parameters
        self.N_turns = 1  # Number of turns
        self.I_coil = 1000  # A
        self.frequency = 1000  # Hz
        self.omega = 2 * ng.pi * self.frequency  # rad/s
        self.complex = True

    def set_material_properties(self, material_properties: dict):
        """Set material properties."""
        self.material_properties = material_properties

    def set_dirichlet_values(self, dirichlet_values: dict):
        """Set Dirichlet boundary conditions."""
        self.dirichlet_values = dirichlet_values

    def set_dirichlet_values_array(self, dirichlet_values_array: np.ndarray):
        """Set Dirichlet values array for all nodes."""
        self.dirichlet_values_array = dirichlet_values_array


@dataclass
class TimeConfig:
    """Configuration for time integration."""

    dt: float = 0.01  # Time step size
    t_final: float = 1.0  # Final time
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
    time_config: TimeConfig = field(
        default_factory=lambda: TimeConfig(t_final=1.0, dt=0.01)
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
    order: int = 1  # Finite element order
    dim: int = 2  # Spatial dimension (2D or 3D)
    dirichlet_boundaries: Optional[List[str]] = (
        None  # Names of boundaries with Dirichlet BCs
    )
    neumann_boundaries: Optional[List[str]] = (
        None  # Names of boundaries with Neumann BCs
    )
    robin_boundaries: Optional[List[str]] = None  # Names of boundaries with Robin BCs
    dirichlet_pipe: Optional[str] = None
    neumann_pipe: Optional[str] = None
    robin_pipe: Optional[str] = None

    mesh_type: str = "rectangle"  # Type of mesh: rectangle, circle, etc.

    def __post_init__(self):
        if self.dirichlet_boundaries:
            self.dirichlet_pipe = "|".join(self.dirichlet_boundaries)
        if self.neumann_boundaries:
            self.neumann_pipe = "|".join(self.neumann_boundaries)
        if self.robin_boundaries:
            self.robin_pipe = "|".join(self.robin_boundaries)
