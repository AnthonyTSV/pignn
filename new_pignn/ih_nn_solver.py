from pathlib import Path
from typing import Optional
import torch
import ngsolve as ng
import numpy as np

try:
    from .mesh_utils import create_ih_mesh
    from .containers import (
        MeshConfig,
        MeshProblem,
        MeshProblemEM,
        TimeConfig,
        BoundaryCondition,
        DirichletBC,
        ConvectionBC,
        CombinedBC,
        MaterialPropertiesHeat,
        MaterialPropertiesEM,
        SourceProperties,
    )
    from .graph_creator import GraphCreator
    from .graph_creator_em import GraphCreatorEM
    from .meshgraphnet import MeshGraphNet
    from .trainer import PIMGNTrainer
    from .trainer_em import PIMGNTrainerEM
    from .em_eddy_problems import GenericEddyCurrentProblem
    from .thermal_problems import GenericHeatEquationProblem
    from .ih_geometry_and_mesh import (
        BilletParams,
        RectangularInductorParams,
        IHGeometryAndMesh,
    )
except ImportError:
    from mesh_utils import create_ih_mesh
    from containers import (
        MeshConfig,
        MeshProblem,
        MeshProblemEM,
        TimeConfig,
        BoundaryCondition,
        DirichletBC,
        ConvectionBC,
        CombinedBC,
        MaterialPropertiesHeat,
        MaterialPropertiesEM,
        SourceProperties,
    )
    from graph_creator import GraphCreator
    from graph_creator_em import GraphCreatorEM
    from meshgraphnet import MeshGraphNet
    from trainer import PIMGNTrainer
    from trainer_em import PIMGNTrainerEM
    from em_eddy_problems import GenericEddyCurrentProblem, eddy_current_problem_different_currents, eddy_current_problem_different_mu_r
    from thermal_problems import GenericHeatEquationProblem, create_ih_problem
    from ih_geometry_and_mesh import (
        BilletParams,
        RectangularInductorParams,
        IHGeometryAndMesh,
    )


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
        material_properties_em: dict[str, MaterialPropertiesEM],
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

        self.em_problem: Optional[MeshProblemEM] = None
        self.thermal_problem: Optional[MeshProblem] = None

        self.em_solution = None
        self.thermal_solution = None

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
        state = (
            checkpoint
            if "model_state_dict" not in checkpoint
            else checkpoint["model_state_dict"]
        )

        if not "time_window" in checkpoint:
            print(
                "Warning: time_window not found in checkpoint, inferring from model weights. This may be unreliable."
            )
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

        return 20  # default

    @staticmethod
    def _require_model_checkpoint(path_to_model: Path, model_name: str):
        path_to_model = Path(path_to_model)
        if not path_to_model.exists():
            raise FileNotFoundError(
                f"{model_name} checkpoint not found: {path_to_model}"
            )

    def _em_dirichlet_values(self) -> tuple[list[str], dict[str, float]]:
        dirichlet_boundaries = []
        dirichlet_values = {}

        for name, bc in self.boundary_conditions_em.items():
            if getattr(bc, "type", None) != "Dirichlet":
                continue
            dirichlet_boundaries.append(name)
            dirichlet_values[name] = float(getattr(bc, "value", 0.0))

        if not dirichlet_boundaries:
            raise ValueError("At least one EM Dirichlet boundary is required")

        return dirichlet_boundaries, dirichlet_values

    def _set_thermal_model(self, path_to_model: Path, joule_heating: np.ndarray):
        self._require_model_checkpoint(path_to_model, "Thermal")
        time_window = 20

        # problem = GenericHeatEquationProblem(
        #     mesh=self.mesh,
        #     material_properties=self.material_properties,
        #     time_config=self.time_config,
        #     boundary_conditions=self.boundary_conditions_heat,
        #     source_function=joule_heating,
        #     thermal_domain_materials=["mat_workpiece"],
        #     axisymmetric=True,
        #     mesh_type="ih_mesh",
        # ).get_problem()

        problem = create_ih_problem()

        self.thermal_problem = problem

        config = {
            "epochs": 1,
            "lr": 1e-3,
            "time_window": time_window,
            "generate_ground_truth_for_validation": False,
            "save_dir": "results/coupled_physics_informed/test_ih_problem",
            "resume_from": str(path_to_model),
        }

        trainer = PIMGNTrainer([problem], config)

        return trainer

    def _set_em_model(self, path_to_model: Path):
        self._require_model_checkpoint(path_to_model, "EM")

        dirichlet_boundaries, dirichlet_boundaries_dict = self._em_dirichlet_values()

        problem_generator = GenericEddyCurrentProblem(
            mesh=self.mesh,
            dirichlet_boundaries=dirichlet_boundaries,
            dirichlet_boundaries_dict=dirichlet_boundaries_dict,
            material_properties=self.material_properties_em,
            A_star=3.4e-3,
        )
        problem = problem_generator.get_problem(
            current=self.source_properties.current,
            frequency=self.source_properties.frequency,
        )
        problem.N_turns = int(
            getattr(self.source_properties, "n_turns", problem.N_turns)
        )
        problem.refresh_derived_quantities()

        if (
            problem.current_density_field is not None
            and problem.material_field is not None
        ):
            current_density_field = np.asarray(
                problem.current_density_field, dtype=np.float64
            ).copy()
            coil_mask = problem.material_field == 1
            current_density_field[coil_mask] = (
                problem.N_turns * problem.I_coil / problem.area_coil / problem.J_star
            )
            problem.current_density_field = current_density_field

        self.em_problem = problem

        config = {
            "epochs": 1,
            "lr": 1e-3,
            "generate_ground_truth_for_validation": False,
            "save_dir": "results/coupled_physics_informed/test_ih_problem",
            "enforce_axis_regularity": True,
            "require_checkpoint": True,
            "strict_checkpoint": True,
            "resume_from": str(path_to_model),
        }

        trainer = PIMGNTrainerEM([problem], config)
        return trainer

    def solve_em(self):
        em_solution = self.em_model.predict(problem_idx=0)
        return em_solution

    def get_source_function(self, heat_source, n_nodes):
        source_function = np.zeros(n_nodes, dtype=np.float64)
        ngmesh = self.mesh.ngmesh
        for i, elem in enumerate(ngmesh.Elements2D()):
            mat_index = elem.index
            mat_name = ngmesh.GetMaterial(mat_index)

            # Get vertices of this element
            vertices = elem.vertices
            for v in vertices:
                node_idx = v.nr - 1 if hasattr(v, "nr") else int(v) - 1
                if 0 <= node_idx < n_nodes:
                    if mat_name == "mat_workpiece":
                        p = ngmesh.Points()[v.nr].p
                        x, y = p[0], p[1]
                        q_val = heat_source(self.mesh(x, y))
                        source_function[node_idx] = q_val
        return source_function

    def compute_joule_heat(self, A):
        """
        Compute Joule heating: Q = 1/2 * sigma * |E|^2,
        where E is the electric field derived from the potential A.
        """
        gfu = ng.GridFunction(self.em_model.all_fem_solvers[0].fes)
        gfu.vec.data = A * self.em_problem.A_star
        omega = 2 * np.pi * self.source_properties.frequency
        E_phi = -1j * omega * gfu
        heat_source_gf = (
            0.5
            * self.material_properties_em["mat_workpiece"].sigma
            * ng.Norm(E_phi) ** 2
        )
        heat_source = self.get_source_function(
            heat_source_gf, self.em_model.problems[0].n_nodes
        )
        return heat_source

    def solve_thermal(self):
        thermal_solution = self.thermal_model.rollout(problem_idx=0)
        return thermal_solution

    def solve_coupled(self):
        self.em_model = self._set_em_model(self.path_to_em_model)
        self.em_solution = self.solve_em()
        joule_heat = self.compute_joule_heat(self.em_solution)
        self.thermal_model = self._set_thermal_model(
            self.path_to_thermal_model, joule_heat
        )
        self.thermal_solution = self.solve_thermal()
        return self.thermal_solution

    def export_to_vtk(self):
        em_fes = ng.H1(self.mesh, order=1, dirichlet="bc_air|bc_axis|bc_workpiece_left", complex=False)
        thermal_fes = ng.H1(self.mesh, order=1, complex=False, definedon="mat_workpiece")

        gfu = ng.GridFunction(self.em_model.all_fem_solvers[0].fes)
        gfu.vec.data = self.em_solution * self.em_problem.A_star
        omega = 2 * np.pi * self.source_properties.frequency
        E_phi = -1j * omega * gfu
        joule_heat = (
            0.5
            * self.material_properties_em["mat_workpiece"].sigma
            * ng.Norm(E_phi) ** 2
        )

        thermal_gfu = ng.GridFunction(thermal_fes)
        thermal_gfu.vec.data = self.thermal_solution[0] # First time step

        path_to_save = "results/coupled_physics_informed/test_ih_problem/vtk"
        path_to_save = Path(path_to_save)
        path_to_save.mkdir(parents=True, exist_ok=True)

        # Export to VTK
        vtk_out = ng.VTKOutput(
            self.mesh,
            coefs=[thermal_gfu, joule_heat, ng.Norm(gfu)],
            names=["Temperature", "Joule_heating", "Magnetic_vector_potential"],
            filename=str(path_to_save / "solution"),
            order=1,
        )

        time_steps = np.asarray(self.time_config.time_steps_export, dtype=np.float64)

        for idx, time in enumerate(time_steps):
            thermal_gfu.vec.FV().NumPy()[:] = self.thermal_solution[idx]
            vtk_out.Do(time)

        print(f"Exported thermal and EM solutions to VTK at {path_to_save}")

if __name__ == "__main__":
    path_to_thermal = Path(
        "results/physics_informed/thermal_ih_problem/pimgn_trained_model.pth"
    )
    path_to_em = Path(
        "results/physics_informed/em_different_mu_r_sigma/pimgn_trained_model.pth"
    )
    # wp = BilletParams(diameter=0.030, height=0.070)
    # ind = RectangularInductorParams(
    #     coil_inner_diameter=0.050,
    #     coil_height=0.040,
    #     winding_count=1,
    #     profile_width=0.007,
    #     profile_height=0.007,
    # )
    # kw = dict(h_workpiece=1e-3, h_air=60e-3, h_coil=1e-3)
    # builder = IHGeometryAndMesh(wp, ind, **kw)
    # mesh = builder.generate()
    em_problem = eddy_current_problem_different_mu_r(mu_r_workpiece=1)
    mesh = em_problem.mesh
    material_properties_heat = MaterialPropertiesHeat(
        rho=7870,
        cp=461,
        k=86,
    )
    material_properties_em = {
        "mat_workpiece": MaterialPropertiesEM(sigma=6289308, mu=1),
        "mat_air": MaterialPropertiesEM(sigma=0, mu=1),
        "mat_coil": MaterialPropertiesEM(sigma=0, mu=1),
    }
    boundary_conditions_heat = {
        "bc_workpiece_top": ConvectionBC(value=(10, 20)),
        "bc_workpiece_right": ConvectionBC(value=(10, 20)),
        "bc_workpiece_bottom": ConvectionBC(value=(10, 20)),
    }
    boundary_conditions_em = {
        "bc_air": DirichletBC(value=0.0),
        "bc_axis": DirichletBC(value=0.0),
        "bc_workpiece_left": DirichletBC(value=0.0),
    }
    source_properties = SourceProperties(
        frequency=3000,
        current=3000,
        fill_factor=1.0,
    )
    time_config = TimeConfig(
        dt=0.1,
        t_final=10.0,
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
    print(len(thermal_solution))
    solver.export_to_vtk()

    from fem import FEMSolver
    fem_solver = FEMSolver(mesh, problem=solver.thermal_problem)
    fem_solution = fem_solver.solve_transient_problem(solver.thermal_problem)
    fem_solver.export_to_vtk(
        array_true=fem_solution,
        array_pred=thermal_solution,
        time_steps=solver.time_config.time_steps_export,
        filename="results/coupled_physics_informed/test_ih_problem/vtk/fem_solution",
    )

    from fem_em import FEMSolverEM
    fem_em_solver = FEMSolverEM(mesh, problem=solver.em_problem)
    fem_em_solution = fem_em_solver.solve(problem=solver.em_problem)
    fem_em_solver.export_to_vtk_complex(
        array_true=fem_em_solution*solver.em_problem.A_star,
        array_pred=solver.em_solution*solver.em_problem.A_star,
        filename="results/coupled_physics_informed/test_ih_problem/vtk/fem_em_solution",
    )
