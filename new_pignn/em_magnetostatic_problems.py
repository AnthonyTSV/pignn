import numpy as np
import random
import ngsolve as ng
from mesh_utils import (
    create_rectangular_mesh,
    create_lshape_mesh,
    create_gaussian_initial_condition,
    create_ih_mesh,
)
from graph_creator_em import GraphCreatorEM
from containers import TimeConfig, MeshConfig, MeshProblem, MeshProblemEM
from ih_geometry_and_mesh import (
    IHGeometryAndMesh,
    BilletParams,
    TubeParams,
    SteppedShaftParams,
    MultiBilletParams,
    CircularInductorParams,
    RectangularInductorParams,
)


class GenericMagnetostaticProblem:
    def __init__(self, mesh, dirichlet_boundaries, dirichlet_boundaries_dict):
        self.r_star = 70 * 1e-3  # m
        self.A_star = 4.8 * 1e-4  # Wb
        self.mu_star = 4 * 3.1415926535e-7  # H/m
        self.J_star = self.A_star / (self.r_star**2 * self.mu_star)
        self.mesh = mesh
        self.dirichlet_boundaries = dirichlet_boundaries
        self.dirichlet_boundaries_dict = dirichlet_boundaries_dict

    def get_problem(self):
        mesh_config = MeshConfig(
            maxh=1,
            order=1,
            dim=2,
            dirichlet_boundaries=self.dirichlet_boundaries,
            mesh_type="ih_mesh",
        )
        graph_creator = GraphCreatorEM(
            mesh=self.mesh,
            n_neighbors=2,
            dirichlet_names=self.dirichlet_boundaries,
        )
        # First create a temporary graph to get positions and aux data
        temp_data, temp_aux = graph_creator.create_graph()

        dirichlet_vals = graph_creator.create_dirichlet_values(
            pos=temp_data.pos,
            aux_data=temp_aux,
            dirichlet_names=self.dirichlet_boundaries,
            boundary_values=self.dirichlet_boundaries_dict,
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
        problem.set_dirichlet_values(self.dirichlet_boundaries_dict)
        problem.complex = False
        problem.mixed = False
        problem.sigma_workpiece = 0
        problem.sigma_air = 0
        problem.sigma_coil = 0

        # Create material fields (mu_r at each node) based on material subdomain
        n_nodes = temp_data.pos.shape[0]
        mu_r_field = np.ones(n_nodes, dtype=np.float64)  # Default to air (mu_r = 1)
        current_density = np.zeros(n_nodes, dtype=np.float64)

        # Calculate current density in the coil: J = N * I / A_coil
        Acoil = problem.profile_width_phys * problem.profile_height_phys
        Js_phi = problem.N_turns * problem.I_coil / Acoil
        Js_phi = Js_phi / self.J_star  # Normalize current density

        # Material property mapping (mu_r values)
        mu_r_map = {
            "mat_workpiece": problem.mu_r_workpiece,
            "mat_air": problem.mu_r_air,
            "mat_coil": problem.mu_r_coil,
        }

        # Identify nodes in each material region and assign properties
        ngmesh = self.mesh.ngmesh
        for i, elem in enumerate(ngmesh.Elements2D()):
            mat_index = elem.index
            mat_name = ngmesh.GetMaterial(mat_index)

            # Get vertices of this element
            vertices = elem.vertices
            for v in vertices:
                node_idx = v.nr - 1 if hasattr(v, "nr") else int(v) - 1
                if 0 <= node_idx < n_nodes:
                    # Assign mu_r based on material region
                    if mat_name in mu_r_map:
                        mu_r_field[node_idx] = mu_r_map[mat_name]

                    # Assign current density (only in coil)
                    if mat_name == "mat_coil":
                        current_density[node_idx] = Js_phi

        problem.material_field = mu_r_field

        problem.current_density_field = current_density

        return problem


def magnetostatic_problem_1():

    wp = BilletParams(diameter=0.030, height=0.070)
    ind = RectangularInductorParams(
        coil_inner_diameter=0.050,
        coil_height=0.040,
        winding_count=1,
        profile_width=0.007,
        profile_height=0.007,
    )
    kw = dict(h_workpiece=5e-3, h_air=60e-3, h_coil=2e-3)
    builder = IHGeometryAndMesh(wp, ind, **kw)
    mesh = builder.generate()

    dirichlet_boundaries = ["bc_air", "bc_axis", "bc_workpiece_left"]
    dirichlet_boundaries_dict = {"bc_air": 0, "bc_axis": 0, "bc_workpiece_left": 0}

    problem_generator = GenericMagnetostaticProblem(
        mesh, dirichlet_boundaries, dirichlet_boundaries_dict
    )

    problem = problem_generator.get_problem()

    return problem

def magnetostatic_problem_2():
    wp = BilletParams(diameter=0.030, height=0.070)
    ind = CircularInductorParams(
        coil_inner_diameter=0.050,
        coil_height=0.040,
        winding_count=1,
        profile_diameter=0.007,
    )
    kw = dict(h_workpiece=5e-3, h_air=60e-3, h_coil=2e-3)
    builder = IHGeometryAndMesh(wp, ind, **kw)
    mesh = builder.generate()

    dirichlet_boundaries = ["bc_air", "bc_axis", "bc_workpiece_left"]
    dirichlet_boundaries_dict = {"bc_air": 0, "bc_axis": 0, "bc_workpiece_left": 0}

    problem_generator = GenericMagnetostaticProblem(
        mesh, dirichlet_boundaries, dirichlet_boundaries_dict
    )

    problem = problem_generator.get_problem()

    return problem

def magnetostatic_problem_3():
    wp = TubeParams(outer_diameter=0.03, inner_diameter=0.016, height=0.070)
    ind = RectangularInductorParams(
        coil_inner_diameter=0.050,
        coil_height=0.040,
        winding_count=1,
        profile_width=0.007,
        profile_height=0.007,
    )
    kw = dict(h_workpiece=5e-3, h_air=60e-3, h_coil=2e-3)
    builder = IHGeometryAndMesh(wp, ind, **kw)
    mesh = builder.generate()

    dirichlet_boundaries = ["bc_air", "bc_axis", "bc_workpiece_left"]
    dirichlet_boundaries_dict = {"bc_air": 0, "bc_axis": 0, "bc_workpiece_left": 0}

    problem_generator = GenericMagnetostaticProblem(
        mesh, dirichlet_boundaries, dirichlet_boundaries_dict
    )

    problem = problem_generator.get_problem()

    return problem

def magnetostatic_problem_4():
    wp = SteppedShaftParams(
        diameter1=0.030, diameter2=0.016, height1=0.035, height2=0.035
    )
    ind = CircularInductorParams(
        coil_inner_diameter=0.050,
        coil_height=0.040,
        winding_count=1,
        profile_diameter=0.007,
    )
    kw = dict(h_workpiece=1e-3, h_air=60e-3, h_coil=0.5e-3)
    builder = IHGeometryAndMesh(wp, ind, **kw)
    mesh = builder.generate()

    dirichlet_boundaries = ["bc_air", "bc_axis", "bc_workpiece_left"]
    dirichlet_boundaries_dict = {"bc_air": 0, "bc_axis": 0, "bc_workpiece_left": 0}

    problem_generator = GenericMagnetostaticProblem(
        mesh, dirichlet_boundaries, dirichlet_boundaries_dict
    )

    problem = problem_generator.get_problem()

    return problem
