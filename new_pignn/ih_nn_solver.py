
from pathlib import Path
from typing import Optional
import torch
import ngsolve as ng
import numpy as np
from pydantic import BaseModel
from mesh_utils import create_ih_mesh
from containers import MeshConfig, MeshProblem, MeshProblemEM, TimeConfig
from graph_creator import GraphCreator
from graph_creator_em import GraphCreatorEM
from meshgraphnet import MeshGraphNet
from trainer import PIMGNTrainer
from trainer_em import PIMGNTrainerEM

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

class MaterialPropertiesHeat(BaseModel):
    rho: float # mass density
    cp: float # specific heat capacity
    k: float # thermal conductivity
    h_conv: Optional[float] = None
    thermal_diffusivity: Optional[float] = None # alpha = k / (rho * cp)

    def __init__(self, **data):
        super().__init__(**data)
        if self.thermal_diffusivity is None:
            self.thermal_diffusivity = self.k / (self.rho * self.cp)
        if self.h_conv is None:
            self.h_conv = 10.0 / (self.rho * self.cp)

class MaterialPropertiesEM(BaseModel):
    sigma: float # electrical conductivity
    mu: float # magnetic permeability

class SourceProperties(BaseModel):
    frequency: float
    current: float
    fill_factor: float = 1.0
    n_turns: int = 1

class IHNNSolver:
    def __init__(
        self, 
        path_to_thermal_model: Path, 
        path_to_em_model: Path,
        mesh: ng.Mesh,
        boundary_conditions_heat: dict[str, BoundaryCondition],
        boundary_conditions_em: dict[str, BoundaryCondition],
        initial_condition: float,
        material_properties: MaterialPropertiesHeat,
        material_properties_em: dict[MaterialPropertiesEM],
        source_properties: SourceProperties,
        time_config: TimeConfig,
    ):
        self.path_to_thermal_model = path_to_thermal_model
        self.path_to_em_model = path_to_em_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mesh = mesh
        self.boundary_conditions_heat = boundary_conditions_heat
        self.boundary_conditions_em = boundary_conditions_em
        self.initial_condition = initial_condition
        self.material_properties = material_properties
        self.material_properties_em = material_properties_em
        self.source_properties = source_properties
        self.time_config = time_config
        self.em_model = None
        self.thermal_model = None


    def _load_checkpoint(self, path_to_model: Path):
        """
        PyTorch 2.6 changed the default `weights_only` of torch.load from False -> True.
        We store a full training checkpoint (optimizer/scheduler/etc.), so we must load
        with `weights_only=False`. Only do this for trusted checkpoint files.
        """
        try:
            checkpoint = torch.load(
                path_to_model, map_location=self.device, weights_only=False
            )
        except TypeError:
            # Backward compatibility with older PyTorch versions.
            checkpoint = torch.load(path_to_model, map_location=self.device)
        return checkpoint

    def _infer_time_window(self, checkpoint: dict) -> int:
        """
        Get time window from checkpoint.
        """
        state = checkpoint if "model_state_dict" not in checkpoint else checkpoint["model_state_dict"]

        if not "time_window" in checkpoint:
            print("Warning: time_window not found in checkpoint, inferring from model weights. This may be unreliable.")
        else:
            return checkpoint["time_window"]

        # CNN-based output head (time_window in {5, 10, 20})
        key_conv2 = "output_head.2.weight"
        if key_conv2 in state:
            k2 = state[key_conv2].shape[2]
            # k1=15, s1=4 → L1 = (128-15)//4 + 1 = 29; T = L1 - k2 + 1
            return 29 - k2 + 1

        # Linear output head (time_window == 1 or fallback)
        key_linear = "output_head.weight"
        if key_linear in state:
            return state[key_linear].shape[0]

        return 20 # default

    def _set_thermal_model(self, path_to_model: Path, joule_heating: np.ndarray):
        checkpoint = self._load_checkpoint(path_to_model)
        time_window = self._infer_time_window(checkpoint)
        bcs = self.boundary_conditions_heat
        dirichlet_names = [name for name, bc in bcs.items() if bc.type == "Dirichlet"]
        neumann_names = [name for name, bc in bcs.items() if bc.type == "Neumann"]
        robin_names = [name for name, bc in bcs.items() if bc.type == "Robin"]

        dirichlet_boundaries_dict = {name: bc.value for name, bc in bcs.items() if bc.type == "Dirichlet"}
        neumann_boundaries_dict = {name: bc.value for name, bc in bcs.items() if bc.type == "Neumann"}
        robin_boundaries_dict = {name: bc.value for name, bc in bcs.items() if bc.type == "Robin"}

        # Create graph to get node positions
        graph_creator = GraphCreator(
            mesh=self.mesh,
            dirichlet_names=dirichlet_names,
            neumann_names=neumann_names,
            robin_names=robin_names,
        )
        # First create a temporary graph to get positions and aux data
        temp_data, temp_aux = graph_creator.create_graph()

        # Create Neumann values based on the temporary data
        neumann_vals = graph_creator.create_neumann_values(
            pos=temp_data.pos,
            aux_data=temp_aux,
            neumann_names=neumann_names,
            flux_values=neumann_boundaries_dict,
            seed=42,
        )
        dirichlet_vals = graph_creator.create_dirichlet_values(
            pos=temp_data.pos,
            aux_data=temp_aux,
            dirichlet_names=dirichlet_names,
            boundary_values=dirichlet_boundaries_dict,
        )
        h_vals, amb_vals = graph_creator.create_robin_values(
            pos=temp_data.pos,
            aux_data=temp_aux,
            robin_names=robin_names,
            robin_values=robin_boundaries_dict,
        )

        # Create the final graph with Neumann values
        temp_data, _ = graph_creator.create_graph(
            neumann_values=neumann_vals,
            dirichlet_values=dirichlet_vals,
            robin_values=(h_vals, amb_vals),
        )

        initial_condition = np.ones_like(temp_data.pos[:, 0]) * self.initial_condition

        mesh_config = MeshConfig(
            maxh=1,
            order=1,
            dim=2,
            dirichlet_boundaries=dirichlet_names,
            mesh_type="ih_mesh",
        )

        # Create problem
        problem = MeshProblem(
            mesh=self.mesh,
            graph_data=temp_data,
            initial_condition=initial_condition,
            alpha=self.material_properties.thermal_diffusivity,  # Thermal diffusivity
            time_config=self.time_config,
            mesh_config=mesh_config,
            problem_id=0,
        )

        problem.set_source_function(joule_heating)

        config = {
            "epochs": 10000,
            "lr": 1e-3,
            "time_window": time_window,
            "generate_ground_truth_for_validation": False,
            "save_dir": "results/coupled_physics_informed/test_ih_problem",
            "resume_from": str(path_to_model),
            # "save_interval": 1000,
            # "phi_weight": 0.1,
            # "data_weight": 0.0,
            # "data_weight_decay": 0.9995,
        }

        trainer = PIMGNTrainer([problem], config)

        return trainer

    def _set_em_model(self, path_to_model: Path):
        bcs = self.boundary_conditions_em

        dirichlet_boundaries = [name for name, bc in bcs.items() if bc.type == "Dirichlet"]
        dirichlet_boundaries_dict = {name: bc.value for name, bc in bcs.items() if bc.type == "Dirichlet"}
        # Mesh configuration
        mesh_config = MeshConfig(
            maxh=1,
            order=1,
            dim=2,
            dirichlet_boundaries=dirichlet_boundaries,
            mesh_type="ih_mesh",
        )
        graph_creator = GraphCreatorEM(
            mesh=self.mesh,
            n_neighbors=2,
            dirichlet_names=dirichlet_boundaries,
        )
        # First create a temporary graph to get positions and aux data
        temp_data, temp_aux = graph_creator.create_graph()

        dirichlet_vals = graph_creator.create_dirichlet_values(
            pos=temp_data.pos,
            aux_data=temp_aux,
            dirichlet_names=dirichlet_boundaries,
            boundary_values=dirichlet_boundaries_dict,
        )

        temp_data, _ = graph_creator.create_graph(
            dirichlet_values=dirichlet_vals,
        )
        problem = MeshProblemEM(
            mesh=self.mesh,
            graph_data=temp_data,
            mesh_config=mesh_config,
            problem_id=0,
        )
        problem.set_dirichlet_values_array(dirichlet_vals)
        problem.set_dirichlet_values(dirichlet_boundaries_dict)
        problem.complex = True
        problem.mixed = False
        problem.sigma_workpiece = 6289308 / problem.sigma_star
        problem.sigma_air = 0
        problem.sigma_coil = 0
        problem.frequency = self.source_properties.frequency
        problem.I_coil = self.source_properties.current
        problem.N_turns = self.source_properties.n_turns
        problem.fill_factor = self.source_properties.fill_factor

        # Create material fields (mu_r at each node) based on material subdomain
        n_nodes = temp_data.pos.shape[0]
        mu_r_field = np.ones(n_nodes, dtype=np.float64)  # Default to air (mu_r = 1)
        sigma_field = np.zeros(n_nodes, dtype=np.float64) # Default to 0 conductivity
        current_density = np.zeros(n_nodes, dtype=np.float64)

        # Calculate current density in the coil: J = N * I / A_coil
        Acoil = problem.profile_width_phys * problem.profile_height_phys
        Js_phi = problem.N_turns * problem.I_coil / Acoil
        Js_phi = Js_phi / problem.J_star  # Normalize current density

        config = {
            "epochs": 10000,
            "lr": 1e-3,
            "generate_ground_truth_for_validation": False,
            "save_dir": "results/coupled_physics_informed/test_ih_problem",
            "resume_from": str(path_to_model),
            # "save_interval": 1000,
            # "phi_weight": 0.1,
            # "data_weight": 0.0,
            # "data_weight_decay": 0.9995,
        }

        trainer = PIMGNTrainerEM([problem], config)
        return trainer

    def solve_em(self):
        em_solution = self.em_model.predict(problem_idx=0)
        return em_solution

    def compute_joule_heat(self, em_solution):
        n_nodes = em_solution.shape[0]
        heat_source = np.zeros(n_nodes)
        return heat_source

    def solve_thermal(self, heat_source):
        thermal_solution = self.thermal_model.rollout(problem_idx=0)
        return thermal_solution

    def solve_coupled(self):
        self.em_model = self._set_em_model(self.path_to_em_model)
        joule_heat = self.compute_joule_heat(self.solve_em())
        self.thermal_model = self._set_thermal_model(self.path_to_thermal_model, joule_heat)
        thermal_solution = self.solve_thermal(joule_heat)
        return thermal_solution

if __name__ == "__main__":
    path_to_thermal = Path("results/physics_informed/em_to_thermal/pimgn_trained_model.pth")
    path_to_em = Path("results/physics_informed/test_em_problem_complex/pimgn_trained_model.pth")
    mesh = create_ih_mesh()
    material_properties_heat = MaterialPropertiesHeat(
        rho=7870,
        cp=461,
        k=86,
    )
    material_properties_em = {
        "mat_workpiece": MaterialPropertiesEM(sigma=6289308, mu=4 * 3.1415926535e-7),
        "mat_air": MaterialPropertiesEM(sigma=0.0, mu=4 * 3.1415926535e-7),
        "mat_coil": MaterialPropertiesEM(sigma=58823529, mu=4 * 3.1415926535e-7),
    }
    boundary_conditions_heat = {
        "bc_workpiece_top": RobinBC(value=(material_properties_heat.h_conv, 22.0)),
        "bc_workpiece_right": RobinBC(value=(material_properties_heat.h_conv, 22.0)),
        "bc_workpiece_bottom": RobinBC(value=(material_properties_heat.h_conv, 22.0)),
    }
    boundary_conditions_em = {
        "bc_air": DirichletBC(value=0.0),
        "bc_axis": DirichletBC(value=0.0),
        "bc_workpiece_left": DirichletBC(value=0.0),
    }
    source_properties = SourceProperties(
        frequency=8000,
        current=1000,
        fill_factor=1.0,
    )
    time_config = TimeConfig(
        dt=0.1,
        t_final=1.0,
    )

    solver = IHNNSolver(
        path_to_thermal,
        path_to_em,
        mesh,
        boundary_conditions_heat,
        boundary_conditions_em,
        22,
        material_properties_heat,
        material_properties_em,
        source_properties,
        time_config,
    )
    thermal_solution = solver.solve_coupled()
