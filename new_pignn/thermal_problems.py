from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import ngsolve as ng

try:
    from .containers import (
        MeshConfig,
        MeshProblem,
        TimeConfig,
        BoundaryCondition,
        DirichletBC,
        NeumannBC,
        RobinBC,
        ConvectionBC,
        RadiationBC,
        CombinedBC,
        MaterialPropertiesHeat,
        FieldValue,
        Table1D,
    )
    from .graph_creator import GraphCreator, _normalize_ids_to_idx
    from .mesh_utils import create_rectangular_mesh
    from .ih_geometry_and_mesh import (
        IHGeometryAndMesh,
        BilletParams,
        TubeParams,
        SteppedShaftParams,
        MultiBilletParams,
        CircularInductorParams,
        RectangularInductorParams,
    )
    from .em_eddy_problems import (
        eddy_current_problem_1,
        eddy_current_problem_different_currents,
        em_team_36_problem,
        eddy_current_problem_different_mu_r
    )
    from .fem_em import FEMSolverEM
except ImportError:
    from containers import (
        MeshConfig,
        MeshProblem,
        TimeConfig,
        BoundaryCondition,
        DirichletBC,
        NeumannBC,
        RobinBC,
        ConvectionBC,
        RadiationBC,
        CombinedBC,
        MaterialPropertiesHeat,
        FieldValue,
        Table1D,
    )
    from graph_creator import GraphCreator, _normalize_ids_to_idx
    from mesh_utils import create_rectangular_mesh
    from ih_geometry_and_mesh import (
        IHGeometryAndMesh,
        BilletParams,
        TubeParams,
        SteppedShaftParams,
        MultiBilletParams,
        CircularInductorParams,
        RectangularInductorParams,
    )
    from em_eddy_problems import (
        eddy_current_problem_1,
        eddy_current_problem_different_currents,
        em_team_36_problem,
        eddy_current_problem_different_mu_r
    )
    from fem_em import FEMSolverEM


class GenericHeatEquationProblem:
    """?"""

    STEFAN_BOLTZMANN = 5.670374419e-8
    KELVIN_OFFSET = 273.15

    def __init__(
        self,
        mesh,
        time_config: TimeConfig,
        boundary_conditions: dict[str, BoundaryCondition],
        material_properties: Optional[MaterialPropertiesHeat] = None,
        rho_cp: Optional[float] = None,
        k: Optional[float] = None,
        initial_condition=0.0,
        material_field=None,
        source_function=None,
        source_coefficient=None,
        thermal_domain_materials: Optional[str | Iterable[str]] = None,
        source_materials: Optional[str | Iterable[str]] = None,
        material_region: Optional[str] = None,
        axisymmetric: bool = False,
        mesh_type: str = "generic",
        maxh: float = 1.0,
        order: int = 1,
        dim: int = 2,
        problem_id: int = 0,
    ):
        self.mesh = mesh
        self.time_config = time_config
        self.material_properties = material_properties

        if material_properties is not None:
            self.rho_cp = material_properties.rho.get_value() * material_properties.cp.get_value()
            self.k = material_properties.k.get_value()
        elif rho_cp is not None and k is not None:
            self.rho_cp = rho_cp
            self.k = k
        else:
            raise ValueError("Either material_properties or (rho_cp and k) must be provided")

        self.initial_condition = initial_condition

        # Extract boundary names and value dicts from structured BCs
        self.dirichlet_boundaries = [
            name
            for name, bc in boundary_conditions.items()
            if isinstance(bc, DirichletBC)
        ]
        self.dirichlet_boundaries_dict = {
            name: bc.value
            for name, bc in boundary_conditions.items()
            if isinstance(bc, DirichletBC)
        }
        self.neumann_boundaries = [
            name
            for name, bc in boundary_conditions.items()
            if isinstance(bc, NeumannBC)
        ]
        self.neumann_boundaries_dict = {
            name: bc.value
            for name, bc in boundary_conditions.items()
            if isinstance(bc, NeumannBC)
        }
        self.robin_boundaries = [
            name
            for name, bc in boundary_conditions.items()
            if self._is_robin_like(bc)
        ]
        self.robin_boundaries_dict = {
            name: self._collapse_robin_bc(bc)
            for name, bc in boundary_conditions.items()
            if self._is_robin_like(bc)
        }
        self.material_field = material_field
        self.source_function = source_function
        self.source_coefficient = source_coefficient
        self.thermal_domain_materials = self._normalize_material_names(
            thermal_domain_materials
        )
        self.source_materials = self._normalize_material_names(source_materials)
        self.material_region = material_region
        self.axisymmetric = axisymmetric
        self.mesh_type = mesh_type
        self.maxh = maxh
        self.order = order
        self.dim = dim
        self.problem_id = problem_id
        self.boundary_conditions = boundary_conditions or {}

    @staticmethod
    def _is_robin_like(bc: BoundaryCondition) -> bool:
        return isinstance(bc, RobinBC | ConvectionBC | RadiationBC | CombinedBC)

    def _radiation_to_robin(self, emissivity: float, t_surroundings: float) -> tuple[float, float]:
        # Linearize radiation around the surroundings temperature so the term can
        # be assembled with the same constant Robin operator used by the FEM residual.
        t_surroundings = float(t_surroundings)
        t_kelvin = t_surroundings + self.KELVIN_OFFSET
        h_rad = float(emissivity) * self.STEFAN_BOLTZMANN * 4.0 * t_kelvin**3
        return h_rad, t_surroundings

    def _collapse_robin_bc(self, bc: BoundaryCondition) -> tuple[float, float]:
        if isinstance(bc, RobinBC | ConvectionBC):
            h_val, t_amb = bc.value
            return float(h_val), float(t_amb)

        if isinstance(bc, RadiationBC):
            emissivity, t_surroundings = bc.value
            return self._radiation_to_robin(emissivity, t_surroundings)

        if isinstance(bc, CombinedBC):
            convection = bc.value.get("convection")
            radiation = bc.value.get("radiation")

            h_eff = 0.0
            rhs_eff = 0.0

            if convection is not None:
                h_conv, t_conv = convection
                h_eff += float(h_conv)
                rhs_eff += float(h_conv) * float(t_conv)

            if radiation is not None:
                emissivity, t_surroundings = radiation
                h_rad, t_rad = self._radiation_to_robin(emissivity, t_surroundings)
                h_eff += h_rad
                rhs_eff += h_rad * t_rad

            if h_eff <= 0.0:
                raise ValueError(
                    "CombinedBC must contain at least one positive convection or radiation contribution"
                )

            return h_eff, rhs_eff / h_eff

        raise TypeError(f"Unsupported Robin boundary condition type: {type(bc)!r}")

    @staticmethod
    def _normalize_material_names(
        material_names: Optional[str | Iterable[str]],
    ) -> list[str]:
        if material_names is None:
            return []
        if isinstance(material_names, str):
            return [material_names]
        return list(material_names)

    def _build_material_mask(
        self, pnum_to_idx: dict[int, int], material_names: list[str]
    ) -> np.ndarray:
        n_nodes = len(pnum_to_idx)
        mask = np.zeros(n_nodes, dtype=bool)
        if not material_names:
            return mask

        ngmesh = self.mesh.ngmesh
        target_materials = set(material_names)
        found_materials = set()
        for elem in ngmesh.Elements2D():
            mat_name = ngmesh.GetMaterial(elem.index)
            if mat_name not in target_materials:
                continue
            found_materials.add(mat_name)

            vertex_indices = _normalize_ids_to_idx(
                elem.vertices,
                pnum_to_idx,
                n_nodes,
                context=f"material_nodes:{mat_name}",
            )
            mask[vertex_indices] = True

        missing_materials = sorted(target_materials - found_materials)
        if missing_materials:
            raise ValueError(f"Materials not found on mesh: {missing_materials}")

        return mask

    def _evaluate_on_node(self, values, x: float, y: float):
        if hasattr(values, "__call__"):
            try:
                return values(self.mesh(x, y))
            except TypeError:
                return values(x, y)
        raise TypeError("Unsupported nodal value specification")

    def _as_nodal_array(
        self,
        values,
        pos,
        default_value: float = 0.0,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if hasattr(pos, "detach"):
            pos_np = pos.detach().cpu().numpy()
        elif hasattr(pos, "cpu"):
            pos_np = pos.cpu().numpy()
        else:
            pos_np = np.asarray(pos)

        n_nodes = pos_np.shape[0]
        array = np.full(n_nodes, default_value, dtype=np.float64)

        if values is None:
            pass
        elif np.isscalar(values):
            array.fill(float(values))
        elif hasattr(values, "detach"):
            values_np = values.detach().cpu().numpy()
            if values_np.shape != (n_nodes,):
                raise ValueError(
                    f"Expected nodal tensor with shape ({n_nodes},), got {values_np.shape}"
                )
            array = values_np.astype(np.float64, copy=True)
        elif isinstance(values, (list, tuple, np.ndarray)):
            values_np = np.asarray(values)
            if values_np.shape != (n_nodes,):
                raise ValueError(
                    f"Expected nodal array with shape ({n_nodes},), got {values_np.shape}"
                )
            array = values_np.astype(np.float64, copy=True)
        else:
            for i, (x, y) in enumerate(pos_np[:, :2]):
                array[i] = float(self._evaluate_on_node(values, float(x), float(y)))

        if mask is not None:
            array = np.where(mask, array, 0.0)

        return array

    def _build_initial_condition(self, pos) -> np.ndarray:
        return self._as_nodal_array(self.initial_condition, pos, default_value=0.0)

    def _project_initial_condition_to_dirichlet(self, problem: MeshProblem):
        if not problem.mesh_config.dirichlet_boundaries:
            return

        fes = ng.H1(
            self.mesh,
            order=problem.mesh_config.order,
            dirichlet=problem.mesh_config.dirichlet_pipe,
        )
        gfu = ng.GridFunction(fes)
        gfu_initial = ng.GridFunction(fes)

        gfu_initial.vec.FV().NumPy()[:] = problem.initial_condition

        boundary_cf = self.mesh.BoundaryCF(problem.boundary_values, default=0)
        gfu.Set(
            boundary_cf,
            definedon=self.mesh.Boundaries(problem.mesh_config.dirichlet_pipe),
        )

        free_dofs = fes.FreeDofs()
        for dof in range(fes.ndof):
            if free_dofs[dof]:
                gfu.vec[dof] = gfu_initial.vec[dof]

        problem.initial_condition = gfu.vec.FV().NumPy().copy()

    def get_problem(self) -> MeshProblem:
        mesh_config = MeshConfig(
            maxh=self.maxh,
            order=self.order,
            dim=self.dim,
            dirichlet_boundaries=self.dirichlet_boundaries,
            neumann_boundaries=self.neumann_boundaries,
            robin_boundaries=self.robin_boundaries,
            mesh_type=self.mesh_type,
        )

        graph_creator = GraphCreator(
            mesh=self.mesh,
            dirichlet_names=self.dirichlet_boundaries,
            neumann_names=self.neumann_boundaries,
            robin_names=self.robin_boundaries,
        )
        temp_data, temp_aux = graph_creator.create_graph()

        pnum_to_idx = temp_aux["pnum_to_idx"]
        thermal_domain_mask = self._build_material_mask(
            pnum_to_idx, self.thermal_domain_materials
        )
        source_mask = (
            self._build_material_mask(pnum_to_idx, self.source_materials)
            if self.source_materials
            else thermal_domain_mask if self.thermal_domain_materials else None
        )

        dirichlet_vals = graph_creator.create_dirichlet_values(
            pos=temp_data.pos,
            aux_data=temp_aux,
            dirichlet_names=self.dirichlet_boundaries,
            boundary_values=self.dirichlet_boundaries_dict,
        )
        neumann_vals = graph_creator.create_neumann_values(
            pos=temp_data.pos,
            aux_data=temp_aux,
            neumann_names=self.neumann_boundaries,
            flux_values=self.neumann_boundaries_dict,
        )
        robin_vals = graph_creator.create_robin_values(
            pos=temp_data.pos,
            aux_data=temp_aux,
            robin_names=self.robin_boundaries,
            robin_values=self.robin_boundaries_dict,
        )

        if self.material_field is None:
            if self.thermal_domain_materials:
                material_field = np.where(thermal_domain_mask, self.k, 0.0).astype(
                    np.float64
                )
            else:
                material_field = np.full(
                    temp_data.pos.shape[0], self.k, dtype=np.float64
                )
        else:
            mask = thermal_domain_mask if self.thermal_domain_materials else None
            material_field = self._as_nodal_array(
                self.material_field,
                temp_data.pos,
                default_value=self.k,
                mask=mask,
            )

        # For temperature-dependent k, evaluate k at the initial temperature
        # so the initial graph encodes the correct instantaneous conductivity.
        k_table_ref_values = None
        if self.material_properties is not None:
            k_fv = self.material_properties.k
            if k_fv.is_temperature_dependent():
                initial_T = self._as_nodal_array(
                    self.initial_condition, temp_data.pos, default_value=0.0
                )
                material_field = k_fv.get_table().evaluate_array(initial_T)
                k_table_ref_values = k_fv.get_table().sample_at_references()

        nodal_source = self._as_nodal_array(
            (
                self.source_function
                if self.source_function is not None
                else self.source_coefficient
            ),
            temp_data.pos,
            default_value=0.0,
            mask=source_mask,
        )
        wp_node_mask = thermal_domain_mask if self.thermal_domain_materials else None

        temp_data, _ = graph_creator.create_graph(
            material_node_field=material_field,
            neumann_values=neumann_vals,
            dirichlet_values=dirichlet_vals,
            robin_values=robin_vals,
            source_values=nodal_source,
            wp_node_mask=wp_node_mask,
            k_table_ref_values=k_table_ref_values,
        )

        initial_condition = self._build_initial_condition(temp_data.pos)
        problem = MeshProblem(
            mesh=self.mesh,
            graph_data=temp_data,
            initial_condition=initial_condition,
            rho_cp=self.rho_cp,
            k=self.k,
            time_config=self.time_config,
            mesh_config=mesh_config,
            problem_id=self.problem_id,
            boundary_conditions=self.boundary_conditions,
        )

        problem.material_field = material_field
        problem.set_dirichlet_values_array(dirichlet_vals)
        problem.set_neumann_values_array(neumann_vals)
        problem.set_robin_values_array(robin_vals)
        problem.set_dirichlet_values(self.dirichlet_boundaries_dict)
        problem.set_neumann_values(self.neumann_boundaries_dict)
        problem.set_robin_values(self.robin_boundaries_dict)

        if np.any(nodal_source):
            problem.set_source_function(nodal_source)
        if self.source_coefficient is not None:
            problem.set_source_coefficient(self.source_coefficient)
        if wp_node_mask is not None and np.any(wp_node_mask):
            problem.set_wp_node_mask(wp_node_mask)
        if self.material_region is not None:
            problem.set_material_region(self.material_region)
        elif len(self.thermal_domain_materials) == 1:
            problem.set_material_region(self.thermal_domain_materials[0])
        if self.axisymmetric:
            problem.set_axisymmetric(True)

        # Propagate temperature-dependent material tables to MeshProblem
        if self.material_properties is not None:
            k_fv = self.material_properties.k
            if k_fv.is_temperature_dependent():
                problem.k_table = k_fv.get_table()
                problem.k_table_ref_values = k_fv.get_table().sample_at_references()

        self._project_initial_condition_to_dirichlet(problem)
        return problem


def create_simple_problem():
    """
    Method of manufactured solutions for the 2D heat equation with homogeneous Dirichlet BCs on a unit square domain.
    """
    mesh = create_rectangular_mesh()
    material_properties = MaterialPropertiesHeat(
        rho=1,
        cp=1,
        k=0.1,
    )
    boundary_conditions = {
        "left": DirichletBC(value=0),
        "top": DirichletBC(value=0),
        "right": DirichletBC(value=0),
        "bottom": DirichletBC(value=0),
    }
    problem = GenericHeatEquationProblem(
        mesh=mesh,
        time_config=TimeConfig(dt=0.01, t_final=1.0),
        boundary_conditions=boundary_conditions,
        material_properties=material_properties,
        initial_condition=100 * ng.sin(ng.pi * ng.x) * ng.sin(ng.pi * ng.y),
        thermal_domain_materials=None,
        source_coefficient=None,
        axisymmetric=False,
        mesh_type="rectangular_mesh",
    )
    return problem.get_problem()

def create_bc_verification_problem():
    mesh = create_rectangular_mesh(width=2, height=1, maxh=0.1)
    material_properties = MaterialPropertiesHeat(
        rho=1,
        cp=1,
        k=1,
    )
    boundary_conditions = {
        "left": DirichletBC(value=100),
        "top": CombinedBC(value={
            "convection": (10, 20),
            "radiation": (0.8, 20),
        }),
        "right": DirichletBC(value=100),
        "bottom": NeumannBC(value=0),
    }
    problem = GenericHeatEquationProblem(
        mesh=mesh,
        time_config=TimeConfig(dt=0.01, t_final=1.0),
        boundary_conditions=boundary_conditions,
        material_properties=material_properties,
        initial_condition=20,
        thermal_domain_materials=None,
        source_coefficient=None,
        axisymmetric=False,
        mesh_type="rectangular_mesh",
    )
    return problem.get_problem()

def create_volumetric_heat_source_problem():
    r"""
    q(x, y) = q_0 \exp \left (- \frac{D-x}{\delta} \right ) 
    \exp \left (-\frac{(y-H/2)^2}{2\sigma_{y}^2} \right)
    """
    mesh = create_rectangular_mesh(width=1, height=1, maxh=0.1)
    material_properties = MaterialPropertiesHeat(
        rho=1,
        cp=1,
        k=3,
    )
    boundary_conditions = {
        "left": NeumannBC(value=0),
        "top": ConvectionBC(value=(10, 20)),
        "right": ConvectionBC(value=(10, 20)),
        "bottom": ConvectionBC(value=(10, 20)),
    }
    problem = GenericHeatEquationProblem(
        mesh=mesh,
        time_config=TimeConfig(dt=0.01, t_final=1.0),
        boundary_conditions=boundary_conditions,
        material_properties=material_properties,
        initial_condition=20,
        source_function=5e4 * ng.exp(- (1 - ng.x) / 0.2) * ng.exp(- ((ng.y - 0.5) ** 2) / (2 * 0.1**2)),
        thermal_domain_materials=None,
        axisymmetric=False,
        mesh_type="rectangular_mesh",
    )
    return problem.get_problem()

def create_temp_dependent_material_problem():
    mesh = create_rectangular_mesh(width=1, height=1, maxh=0.1)
    material_properties = MaterialPropertiesHeat(
        rho=1,
        cp=1,
        k=FieldValue(table=Table1D(args=[0, 100, 300], values=[3.0, 2.0, 1.0])),
    )
    boundary_conditions = {
        "left": NeumannBC(value=0),
        "top": ConvectionBC(value=(10, 20)),
        "right": ConvectionBC(value=(10, 20)),
        "bottom": ConvectionBC(value=(10, 20)),
    }
    problem = GenericHeatEquationProblem(
        mesh=mesh,
        time_config=TimeConfig(dt=0.01, t_final=1.0),
        boundary_conditions=boundary_conditions,
        material_properties=material_properties,
        initial_condition=20,
        thermal_domain_materials=None,
        source_function=5e4 * ng.exp(- (1 - ng.x) / 0.2) * ng.exp(- ((ng.y - 0.5) ** 2) / (2 * 0.1**2)),
        axisymmetric=False,
        mesh_type="rectangular_mesh",
    )
    return problem.get_problem()

def get_source_function(mesh, heat_source, n_nodes):
    source_function = np.zeros(n_nodes, dtype=np.float64)
    ngmesh = mesh.ngmesh
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
                    q_val = heat_source(mesh(x, y))
                    source_function[node_idx] = q_val
    return source_function

def create_ih_problem():
    material_properties = MaterialPropertiesHeat(
        rho=2700,
        cp=933.3,
        k=211,
    )
    boundary_conditions = {
        "bc_workpiece_top": ConvectionBC(value=(10, 20)),
        "bc_workpiece_right": ConvectionBC(value=(10, 20)),
        "bc_workpiece_bottom": ConvectionBC(value=(10, 20)),
    }

    em_problem = eddy_current_problem_different_mu_r(mu_r_workpiece=1, sigma_workpiece=37037037, a_star=2e-3)
    mesh = em_problem.mesh
    fem_solver = FEMSolverEM(mesh, order=1, problem=em_problem)
    gfA_unscaled = fem_solver.solve(em_problem)
    gfA = gfA_unscaled * em_problem.A_star

    gfu = ng.GridFunction(fem_solver.fes)
    gfu.vec.data = gfA
    omega = 2 * np.pi * em_problem.frequency
    E_phi = -1j * omega * gfu
    heat_source_gf = (
        0.5
        * em_problem.sigma_workpiece * em_problem.sigma_star
        * ng.Norm(E_phi) ** 2
    )

    source_function = get_source_function(mesh, heat_source_gf, mesh.nv)

    problem = GenericHeatEquationProblem(
        mesh=mesh,
        time_config=TimeConfig(dt=0.1, t_final=10),
        boundary_conditions=boundary_conditions,
        material_properties=material_properties,
        initial_condition=22.0,
        thermal_domain_materials=["mat_workpiece"],
        source_function=source_function,
        axisymmetric=True,
        mesh_type="ih_mesh",
    )
    return problem.get_problem()

def create_ih_problem_mu_r_sigma(mu_r, sigma):
    material_properties = MaterialPropertiesHeat(
        rho=7870,
        cp=461,
        k=86,
    )
    boundary_conditions = {
        "bc_workpiece_top": ConvectionBC(value=(10, 20)),
        "bc_workpiece_right": ConvectionBC(value=(10, 20)),
        "bc_workpiece_bottom": ConvectionBC(value=(10, 20)),
    }

    em_problem = eddy_current_problem_different_mu_r(mu_r_workpiece=mu_r, sigma_workpiece=sigma)
    mesh = em_problem.mesh
    fem_solver = FEMSolverEM(mesh, order=1, problem=em_problem)
    gfA_unscaled = fem_solver.solve(em_problem)
    gfA = gfA_unscaled * em_problem.A_star

    gfu = ng.GridFunction(fem_solver.fes)
    gfu.vec.data = gfA
    omega = 2 * np.pi * em_problem.frequency
    E_phi = -1j * omega * gfu
    heat_source_gf = (
        0.5
        * em_problem.sigma_workpiece * em_problem.sigma_star
        * ng.Norm(E_phi) ** 2
    )

    source_function = get_source_function(mesh, heat_source_gf, mesh.nv)

    problem = GenericHeatEquationProblem(
        mesh=mesh,
        time_config=TimeConfig(dt=0.05, t_final=5),
        boundary_conditions=boundary_conditions,
        material_properties=material_properties,
        initial_condition=22.0,
        thermal_domain_materials=["mat_workpiece"],
        source_function=source_function,
        axisymmetric=True,
        mesh_type="ih_mesh",
    )
    return problem.get_problem()

def test_boundary_layer():
    wp = BilletParams(diameter=0.030, height=0.070)
    ind = RectangularInductorParams(
        coil_inner_diameter=0.050,
        coil_height=0.040,
        winding_count=1,
        profile_width=0.007,
        profile_height=0.007,
    )
    kw = dict(h_workpiece=2e-3, h_air=60e-3, h_coil=1e-3, workpiece_boundary_layer_thicknesses=[1e-3, 2e-3, 3e-3, 4e-3, 5e-3])
    builder = IHGeometryAndMesh(wp, ind, **kw)
    mesh = builder.generate()
    material_properties = MaterialPropertiesHeat(
        rho=7870,
        cp=461,
        k=86,
    )
    boundary_conditions = {
        "bc_workpiece_top": ConvectionBC(value=(10, 20)),
        "bc_workpiece_right": ConvectionBC(value=(10, 20)),
        "bc_workpiece_bottom": ConvectionBC(value=(10, 20)),
    }

    em_problem = eddy_current_problem_different_currents(mesh, frequency=3000, current=3000)
    fem_solver = FEMSolverEM(em_problem.mesh, order=1, problem=em_problem)
    gfA_unscaled = fem_solver.solve(em_problem)
    gfA = gfA_unscaled * em_problem.A_star

    gfu = ng.GridFunction(fem_solver.fes)
    gfu.vec.data = gfA
    omega = 2 * np.pi * em_problem.frequency
    E_phi = -1j * omega * gfu
    heat_source_gf = (
        0.5
        * em_problem.sigma_workpiece * em_problem.sigma_star
        * ng.Norm(E_phi) ** 2
    )

    source_function = get_source_function(mesh, heat_source_gf, mesh.nv)

    problem = GenericHeatEquationProblem(
        mesh=mesh,
        time_config=TimeConfig(dt=0.01, t_final=1),
        boundary_conditions=boundary_conditions,
        material_properties=material_properties,
        initial_condition=22.0,
        thermal_domain_materials=["mat_workpiece"],
        source_function=source_function,
        axisymmetric=True,
        mesh_type="ih_mesh",
    )
    return problem.get_problem()

def ih_team_36_problem():
    mm = 1e-3
    thicknesses = [
        0.04899417051926398 * mm,
        0.06859183872696957 * mm,
        0.09602857421775739 * mm,
        0.13444000390486033 * mm,
        0.18821600546680448 * mm,
        0.2635024076535262 * mm
    ]
    builder = IHGeometryAndMesh(
        BilletParams(diameter=60 * mm, height=500 * mm),
        RectangularInductorParams(
            coil_inner_diameter=48 * 2 * mm,
            coil_height=500 * mm,
            winding_count=10,
            profile_width=20 * mm,
            profile_height=40 * mm,
            is_hollow=True,
            wall_thickness=3 * mm,
        ),
        h_workpiece=5 * mm,
        h_coil=8 * mm,
        h_air=100 * mm,
        air_width=300 * mm,
        air_height_factor=2.0,
        workpiece_boundary_layer_thicknesses=thicknesses
    )
    mesh = builder.generate()
    material_properties = MaterialPropertiesHeat(
        rho=7870,
        cp=461,
        k=FieldValue(table=Table1D(
            args=[0, 100, 200, 300, 400, 500, 600, 700, 750, 800, 900, 1000, 1100, 1200, 1400, 1470, 1800], 
            values=[48.1, 48.1, 46.5, 44.0, 41.0, 38.5, 36.0, 31.4, 28.5, 26.7, 25.9, 26.7, 28.0, 29.8, 35, 39, 39])),
    )
    boundary_conditions = {
        "bc_workpiece_top": CombinedBC(value={
            "convection": (7, 70),
            "radiation": (0.8, 70),
        }),
        "bc_workpiece_right": CombinedBC(value={
            "convection": (7, 25),
            "radiation": (0.8, 25),
        }),
        "bc_workpiece_bottom": CombinedBC(value={
            "convection": (7, 70),
            "radiation": (0.8, 70),
        }),
    }
    # boundary_conditions = {
    #     "bc_workpiece_top": ConvectionBC(value=(10, 70)),
    #     "bc_workpiece_right": ConvectionBC(value=(10, 25)),
    #     "bc_workpiece_bottom": ConvectionBC(value=(10, 70)),
    # }

    em_problem = em_team_36_problem(mesh=mesh)
    fem_solver = FEMSolverEM(em_problem.mesh, order=1, problem=em_problem)
    gfA_unscaled = fem_solver.solve(em_problem)
    gfA = gfA_unscaled * em_problem.A_star

    gfu = ng.GridFunction(fem_solver.fes)
    gfu.vec.data = gfA
    omega = 2 * np.pi * em_problem.frequency
    E_phi = -1j * omega * gfu
    heat_source_gf = (
        0.5
        * em_problem.sigma_workpiece * em_problem.sigma_star
        * ng.Norm(E_phi) ** 2
    )

    source_function = get_source_function(mesh, heat_source_gf, mesh.nv)

    problem = GenericHeatEquationProblem(
        mesh=mesh,
        time_config=TimeConfig(dt=0.1, t_final=25),
        boundary_conditions=boundary_conditions,
        material_properties=material_properties,
        initial_condition=22.0,
        thermal_domain_materials=["mat_workpiece"],
        source_function=source_function,
        axisymmetric=True,
        mesh_type="ih_mesh",
    )
    return problem.get_problem()

if __name__ == "__main__":
    problem = ih_team_36_problem()
    print("Problem created successfully")
