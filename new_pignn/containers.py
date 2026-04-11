from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union
import torch
import ngsolve as ng
from pydantic import BaseModel

AIR_CONV = 10.0  # Convection coefficient for air (W/m^2K)


class Table1D(BaseModel):
    """
    1-D lookup table with linear interpolation.
    """
    args: list[float]
    values: list[float]

    def __init__(self, **data):
        super().__init__(**data)
        if len(self.args) != len(self.values):
            raise ValueError("args and values must have the same length")
        if len(self.args) < 2:
            raise ValueError("Table1D requires at least 2 breakpoints")
        for i in range(len(self.args) - 1):
            if self.args[i] >= self.args[i + 1]:
                raise ValueError("args must be strictly increasing")

    def evaluate(self, x: float) -> float:
        """Linearly interpolate / extrapolate at *x*."""
        return float(np.interp(x, self.args, self.values))

    def evaluate_array(self, x: np.ndarray) -> np.ndarray:
        """Vectorised linear interpolation."""
        return np.interp(x, self.args, self.values)

    def sample_at_references(self, ref_temps: Optional[list[float]] = None) -> np.ndarray:
        """Return k values sampled at reference temperatures.

        If *ref_temps* is ``None`` the table's own breakpoints are used.
        The returned array has shape ``(len(ref_temps),)``.
        """
        if ref_temps is None:
            return np.array(self.values, dtype=np.float64)
        return np.interp(ref_temps, self.args, self.values).astype(np.float64)

    def to_ngsolve_cf(self, temperature_gf: ng.GridFunction) -> ng.CoefficientFunction:
        """Build a piecewise-linear NGSolve CoefficientFunction of *temperature_gf*.

        Uses nested ng.IfPos to build a branchless CF that NGSolve can
        evaluate at quadrature points during assembly.
        """
        t = temperature_gf
        xs = self.args
        ys = self.values
        # Start with constant extrapolation below the first breakpoint
        cf = ng.CoefficientFunction(ys[0])
        for i in range(len(xs) - 1):
            slope = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
            segment_cf = ys[i] + slope * (t - xs[i])
            # Replace cf for t >= xs[i]
            cf = ng.IfPos(t - xs[i], segment_cf, cf)
        # Clamp: for t >= xs[-1] use the last value
        cf = ng.IfPos(t - xs[-1], ys[-1], cf)
        return cf


class FieldValue(BaseModel):
    """A material property that is either a constant or temperature-dependent table.

    Usage
    -----
    >>> FieldValue(constant=1.0)          # constant property
    >>> FieldValue(table=Table1D(...))     # temperature-dependent property
    """
    constant: Optional[float] = None
    table: Optional[Table1D] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.constant is None and self.table is None:
            raise ValueError("Either constant or table must be provided")
        if self.constant is not None and self.table is not None:
            raise ValueError("Cannot specify both constant and table")

    def is_temperature_dependent(self) -> bool:
        return self.table is not None

    def get_value(self) -> float:
        """
        Return a representative scalar value.
        """
        if self.constant is not None:
            return self.constant
        return self.table.values[0]

    def get_table(self) -> Optional[Table1D]:
        return self.table


def _coerce_to_field_value(v) -> FieldValue:
    """Accept float, int, or FieldValue and return a FieldValue."""
    if isinstance(v, FieldValue):
        return v
    if isinstance(v, (int, float)):
        return FieldValue(constant=float(v))
    raise TypeError(f"Cannot coerce {type(v).__name__} to FieldValue")


class BoundaryCondition(BaseModel):
    type: str

class DirichletBC(BoundaryCondition):
    type: str = "Dirichlet"
    value: float

class NeumannBC(BoundaryCondition):
    type: str = "Neumann"
    value: float

class RobinBC(BoundaryCondition):
    type: str = "Robin"
    value: tuple # (h, T_amb)

class ConvectionBC(BoundaryCondition):
    type: str = "Convection"
    value: tuple # (h, T_amb)

class RadiationBC(BoundaryCondition):
    type: str = "Radiation"
    value: tuple # (eps, T_amb)

class CombinedBC(BoundaryCondition):
    type: str = "Combined"
    value: dict # {"convection": (h, T_amb), "radiation": (eps, T_amb)}

class MaterialPropertiesHeat(BaseModel):
    """Thermal material properties.  Each of *rho*, *cp*, *k* may be a scalar
    or a :class:`FieldValue` wrapping a :class:`Table1D` for temperature
    dependence."""
    model_config = {"arbitrary_types_allowed": True}

    rho: Union[float, FieldValue]  # mass density
    cp: Union[float, FieldValue]   # specific heat capacity
    k: Union[float, FieldValue]    # thermal conductivity
    h_conv: Optional[float] = None
    thermal_diffusivity: Optional[float] = None  # alpha = k / (rho * cp)

    def __init__(self, **data):
        # Coerce plain numbers into FieldValue before Pydantic validation
        for key in ("rho", "cp", "k"):
            if key in data:
                data[key] = _coerce_to_field_value(data[key])
        super().__init__(**data)
        if self.thermal_diffusivity is None:
            self.thermal_diffusivity = (
                self.k.get_value() / (self.rho.get_value() * self.cp.get_value())
            )
        if self.h_conv is None:
            self.h_conv = AIR_CONV

class MaterialPropertiesEM(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    sigma: Union[float, FieldValue] # electrical conductivity
    mu: Union[float, FieldValue] # magnetic permeability

class SourceProperties(BaseModel):
    frequency: float
    current: float
    fill_factor: float = 1.0
    n_turns: int = 1

class MeshProblem:
    """Container for a single training problem with mesh, initial condition, and physics parameters."""

    def __init__(
        self,
        mesh,
        graph_data,
        initial_condition,
        rho_cp,
        k,
        time_config,
        mesh_config,
        problem_id,
        boundary_conditions=None,
    ):
        self.mesh = mesh
        self.graph_data = graph_data
        self.initial_condition: np.ndarray = initial_condition
        self.rho_cp = rho_cp  # Density times specific heat capacity [J/(m^3·K)]
        self.k = k  # Thermal conductivity [W/(m·K)] — scalar representative value
        self.k_table = None  # Optional Table1D for temperature-dependent k
        self.k_table_ref_values = None  # Optional [N_ref] array: k sampled at reference temperatures
        self.k_coefficient = None  # Optional spatially varying conductivity (NGSolve CoefficientFunction)
        self.rho_cp_coefficient = None  # Optional spatially varying rho*cp (NGSolve CoefficientFunction)
        self.time_config: TimeConfig = time_config
        self.mesh_config: MeshConfig = mesh_config
        self.problem_id = problem_id
        self.boundary_conditions = boundary_conditions or {}

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

        self.source_function = None  # Default: no source term (nodal array)
        self.source_coefficient = None  # Optional: NGSolve CoefficientFunction for accurate FEM assembly
        self.nonlinear_source_params = (
            None  # Optional parameters for nonlinear heat sources
        )
        self.nonlinear_bc_tol = 1e-3
        self.nonlinear_bc_max_iters = 15
        self.nonlinear_bc_min_iters = 1
        self.material_fraction_field = None  # Optional per-node material descriptor
        self.material_field = (
            None  # Optional per-node physical property (e.g., diffusivity)
        )

        # Workpiece-only thermal domain support
        self.wp_node_mask = None  # Boolean mask: True for workpiece nodes
        self.material_region = None  # NGSolve material region name (e.g. "mat_workpiece")
        self.axisymmetric = False  # If True, multiply weak-form integrals by r=ng.x

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
        """Set source function (nodal values array for GNN features)."""
        self.source_function = source_function

    def set_source_coefficient(self, source_coefficient):
        """Set NGSolve CoefficientFunction for accurate FEM source integration."""
        self.source_coefficient = source_coefficient

    def set_material_region(self, region_name: str):
        """Set the material region name for domain-restricted FEM integration."""
        self.material_region = region_name

    def set_axisymmetric(self, axisymmetric: bool = True):
        """Enable axisymmetric (cylindrical) weak form with Jacobian r = ng.x."""
        self.axisymmetric = axisymmetric

    def set_wp_node_mask(self, wp_node_mask):
        """Set workpiece node mask (boolean array)."""
        self.wp_node_mask = wp_node_mask


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
        self.sigma_nodal = None  # Per-node sigma for FEM assembly (nondimensionalized)
        self.current_density_field = None  # To be set (current density at each node)

        ###

        # Coil parameters
        self.N_turns = 1  # Number of turns
        self.I_coil = 1000  # A
        self.frequency = 8000  # Hz
        self.fill_factor = 1.0

        # Normalisation factors
        self.r_star = 70 * 1e-3  # m
        self.A_star = 4.8 * 1e-4  # Wb/m
        self.mu_star = 4 * 3.1415926535e-7  # H/m
        self.J_star = self.A_star / (self.r_star**2 * self.mu_star)

        self.profile_width_phys = 7 * 1e-3
        self.profile_height_phys = 7 * 1e-3

        self.profile_width = 7 * 1e-3  # m
        self.profile_height = 7 * 1e-3  # m

        # In the nondimensionalized system, mu0 = 1 (dimensionless)
        # because we scaled by mu_star = mu0
        self.mu0 = 1.0  # Normalized permeability of free space
        self.mu_r_workpiece = 100.0  # Relative permeability of workpiece
        self.mu_r_air = 1.0
        self.mu_r_coil = 1.0

        self.sigma_workpiece = 1e6  # Electrical conductivity [S/m]
        self.sigma_air = 0.0
        self.sigma_coil = 5.8e7

        self.refresh_derived_quantities()
        self.complex = True
        self.mixed = False

    def refresh_derived_quantities(self):
        """Refresh derived nondimensional EM quantities after parameter updates."""
        self.omega = float(2 * np.pi * self.frequency)  # rad/s
        if self.omega == 0.0:
            self.sigma_star = np.inf
        else:
            self.sigma_star = self.J_star / (self.omega * self.A_star)
        self.kappa = self.omega * self.mu_star * (self.r_star**2)
        self.normalized_current = self.N_turns * self.I_coil / self.J_star

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
        self.num_steps = int(round(self.t_final / self.dt))
        if not np.isclose(self.num_steps * self.dt, self.t_final, rtol=0.0, atol=1e-12):
            raise ValueError("t_final must be an integer multiple of dt")

        self.time_steps = np.linspace(self.dt, self.t_final, self.num_steps)
        self.time_steps_export = np.linspace(0.0, self.t_final, self.num_steps + 1)


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
