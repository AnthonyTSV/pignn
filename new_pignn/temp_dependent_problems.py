from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import ngsolve as ng

try:
    from .containers import MeshConfig, MeshProblem, TimeConfig
    from .graph_creator import GraphCreator, _normalize_ids_to_idx
except ImportError:
    from containers import MeshConfig, MeshProblem, TimeConfig
    from graph_creator import GraphCreator, _normalize_ids_to_idx


class GenericHeatEquationProblem:
    """?"""

    def __init__(
        self,
        mesh,
        time_config: TimeConfig,
        alpha: float,
        initial_condition=0.0,
        dirichlet_boundaries: Optional[list[str]] = None,
        dirichlet_boundaries_dict: Optional[dict[str, float]] = None,
        neumann_boundaries: Optional[list[str]] = None,
        neumann_boundaries_dict: Optional[dict[str, float]] = None,
        robin_boundaries: Optional[list[str]] = None,
        robin_boundaries_dict: Optional[dict[str, tuple[float, float]]] = None,
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
        n_neighbors: int = 2,
        connectivity_method: str = "fem",
        problem_id: int = 0,
    ):
        self.mesh = mesh
        self.time_config = time_config
        self.alpha = alpha
        self.initial_condition = initial_condition
        self.dirichlet_boundaries = dirichlet_boundaries or []
        self.dirichlet_boundaries_dict = dirichlet_boundaries_dict or {}
        self.neumann_boundaries = neumann_boundaries or []
        self.neumann_boundaries_dict = neumann_boundaries_dict or {}
        self.robin_boundaries = robin_boundaries or []
        self.robin_boundaries_dict = robin_boundaries_dict or {}
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
        self.n_neighbors = n_neighbors
        self.connectivity_method = connectivity_method
        self.problem_id = problem_id

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
            raise ValueError(
                f"Materials not found on mesh: {missing_materials}"
            )

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
            n_neighbors=self.n_neighbors,
            dirichlet_names=self.dirichlet_boundaries,
            neumann_names=self.neumann_boundaries,
            robin_names=self.robin_boundaries,
            connectivity_method=self.connectivity_method,
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
                material_field = np.where(thermal_domain_mask, self.alpha, 0.0).astype(
                    np.float64
                )
            else:
                material_field = np.full(temp_data.pos.shape[0], self.alpha, dtype=np.float64)
        else:
            mask = thermal_domain_mask if self.thermal_domain_materials else None
            material_field = self._as_nodal_array(
                self.material_field,
                temp_data.pos,
                default_value=self.alpha,
                mask=mask,
            )

        nodal_source = self._as_nodal_array(
            self.source_function if self.source_function is not None else self.source_coefficient,
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
        )

        initial_condition = self._build_initial_condition(temp_data.pos)
        problem = MeshProblem(
            mesh=self.mesh,
            graph_data=temp_data,
            initial_condition=initial_condition,
            alpha=self.alpha,
            time_config=self.time_config,
            mesh_config=mesh_config,
            problem_id=self.problem_id,
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

        self._project_initial_condition_to_dirichlet(problem)
        return problem


def create_temp_dependent_material_problem():
    GenericHeatEquationProblem(
        mesh=mesh,
        time_config=TimeConfig(dt=0.1, t_final=1.0),
        alpha=alpha,
        initial_condition=22.0,
        robin_boundaries=["bc_workpiece_top", "bc_workpiece_right", "bc_workpiece_bottom"],
        robin_boundaries_dict={
            "bc_workpiece_top": (h_conv, 22.0),
            "bc_workpiece_right": (h_conv, 22.0),
            "bc_workpiece_bottom": (h_conv, 22.0),
        },
        thermal_domain_materials="mat_workpiece",
        source_coefficient=Q,
        axisymmetric=True,
        mesh_type="ih_mesh",
    )