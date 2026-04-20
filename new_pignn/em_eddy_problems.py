import numpy as np
import random
import ngsolve as ng
from typing import Optional

try:
    from mesh_utils import (
        create_rectangular_mesh,
        create_lshape_mesh,
        create_gaussian_initial_condition,
        create_ih_mesh,
    )
    from graph_creator_em import GraphCreatorEM
    from containers import (
        MeshConfig,
        MeshProblemEM,
        TimeConfig,
        BoundaryCondition,
        DirichletBC,
        NeumannBC,
        RobinBC,
        ConvectionBC,
        RadiationBC,
        CombinedBC,
        MaterialPropertiesEM,
        FieldValue,
        Table1D,
    )
    from ih_geometry_and_mesh import (
        IHGeometryAndMesh,
        BilletParams,
        TubeParams,
        SteppedShaftParams,
        MultiBilletParams,
        CircularInductorParams,
        RectangularInductorParams,
    )
except ImportError:
    from .mesh_utils import (
        create_rectangular_mesh,
        create_lshape_mesh,
        create_gaussian_initial_condition,
        create_ih_mesh,
    )
    from .graph_creator_em import GraphCreatorEM
    from .containers import (
        MeshConfig,
        MeshProblemEM,
        TimeConfig,
        BoundaryCondition,
        DirichletBC,
        NeumannBC,
        RobinBC,
        ConvectionBC,
        RadiationBC,
        CombinedBC,
        MaterialPropertiesEM,
        FieldValue,
        Table1D,
    )
    from .ih_geometry_and_mesh import (
        IHGeometryAndMesh,
        BilletParams,
        TubeParams,
        SteppedShaftParams,
        MultiBilletParams,
        CircularInductorParams,
        RectangularInductorParams,
    )


class GenericEddyCurrentProblem:
    def __init__(
        self,
        mesh,
        dirichlet_boundaries,
        dirichlet_boundaries_dict,
        material_properties: Optional[dict[str, MaterialPropertiesEM]] = None,
        sigma_nodal=None,
        A_star=4.8 * 1e-4,  # Wb
        r_star=70 * 1e-3,  # m
        coil_area: Optional[float] = None,
    ):
        self.r_star = r_star  # m
        self.A_star = A_star  # Wb
        self.mu_star = 4 * 3.1415926535e-7  # H/m
        self.J_star = self.A_star / (self.r_star**2 * self.mu_star)
        self.mesh = mesh
        self.dirichlet_boundaries = dirichlet_boundaries
        self.dirichlet_boundaries_dict = dirichlet_boundaries_dict
        self.material_properties = material_properties
        self.sigma_nodal = sigma_nodal  # Optional per-node sigma [S/m] (physical units)
        self.coil_area = coil_area

    def get_problem(self, current=1000, frequency=8000) -> MeshProblemEM:
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
            r_star=self.r_star,
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
        problem.complex = True
        problem.mixed = False
        problem.r_star = self.r_star
        problem.A_star = self.A_star
        problem.mu_star = self.mu_star
        problem.J_star = problem.A_star / (problem.r_star**2 * problem.mu_star)
        problem.frequency = frequency
        problem.I_coil = current
        if self.coil_area:
            problem.area_coil = self.coil_area
        else:
            problem.area_coil = problem.profile_width_phys * problem.profile_height_phys
        problem.refresh_derived_quantities()
        if self.material_properties is not None:
            problem.sigma_workpiece = (
                self.material_properties["mat_workpiece"].sigma / problem.sigma_star
            )
        else:
            problem.sigma_workpiece = (
                6289308 / problem.sigma_star
            )  # Default value (nondimensionalized)
        problem.sigma_air = 0
        problem.sigma_coil = 0

        # Create material fields (mu_r at each node) based on material subdomain
        n_nodes = temp_data.pos.shape[0]
        mu_r_field = np.ones(n_nodes, dtype=np.float64)  # Default to air (mu_r = 1)
        current_density = np.zeros(n_nodes, dtype=np.float64)

        # Calculate current density in the coil: J = N * I / A_coil
        Js_phi = problem.N_turns * problem.I_coil / problem.area_coil
        Js_phi = Js_phi / problem.J_star  # Keep graph features consistent with FEM RHS

        # Material property mapping (mu_r values)
        n_nodes = temp_data.pos.shape[0]
        mu_r_field = np.ones(n_nodes, dtype=np.float64)  # Default to air (mu_r = 1)
        sigma_field = np.zeros(n_nodes, dtype=np.float64)  # Default to 0 conductivity
        current_density = np.zeros(n_nodes, dtype=np.float64)

        material_encode = {
            "mat_workpiece": 2,
            "mat_air": 0,
            "mat_coil": 1,
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
                    mu_r_field[node_idx] = material_encode[mat_name]
                    if mat_name == "mat_workpiece":
                        sigma_field[node_idx] = problem.sigma_workpiece
                    elif mat_name == "mat_coil":
                        current_density[node_idx] = Js_phi

        # If a spatially varying sigma was provided, nondimensionalize and
        # overlay it onto the workpiece nodes (air/coil keep their defaults).
        if self.sigma_nodal is not None:
            sigma_nodal_nd = (
                np.asarray(self.sigma_nodal, dtype=np.float64) / problem.sigma_star
            )
            # Only overwrite workpiece nodes (material_encode == 2)
            wp_mask = mu_r_field == material_encode["mat_workpiece"]
            sigma_field[wp_mask] = sigma_nodal_nd[wp_mask]
            problem.sigma_nodal = sigma_nodal_nd  # per-node for FEM assembly

        problem.material_field = mu_r_field
        problem.sigma_field = sigma_field
        problem.current_density_field = current_density

        return problem

def eddy_current_problem_1(mesh=None):
    if mesh is None:
        wp = BilletParams(diameter=0.030, height=0.070)
        ind = RectangularInductorParams(
            coil_inner_diameter=0.050,
            coil_height=0.040,
            winding_count=1,
            profile_width=0.007,
            profile_height=0.007,
        )
        kw = dict(h_workpiece=1e-3, h_air=60e-3, h_coil=1e-3)
        builder = IHGeometryAndMesh(wp, ind, **kw)
        mesh = builder.generate()

    dirichlet_boundaries = ["bc_air", "bc_axis", "bc_workpiece_left"]
    dirichlet_boundaries_dict = {"bc_air": 0, "bc_axis": 0, "bc_workpiece_left": 0}

    problem_generator = GenericEddyCurrentProblem(
        mesh, dirichlet_boundaries, dirichlet_boundaries_dict
    )

    problem = problem_generator.get_problem(current=10000)

    return problem


def eddy_current_problem_2():

    wp = BilletParams(diameter=0.030, height=0.070)
    ind = RectangularInductorParams(
        coil_inner_diameter=0.050,
        coil_height=0.040,
        winding_count=2,
        profile_width=0.007,
        profile_height=0.007,
    )
    kw = dict(h_workpiece=1e-3, h_air=60e-3, h_coil=1e-3)
    builder = IHGeometryAndMesh(wp, ind, **kw)
    mesh = builder.generate()

    dirichlet_boundaries = ["bc_air", "bc_axis", "bc_workpiece_left"]
    dirichlet_boundaries_dict = {"bc_air": 0, "bc_axis": 0, "bc_workpiece_left": 0}

    problem_generator = GenericEddyCurrentProblem(
        mesh, dirichlet_boundaries, dirichlet_boundaries_dict
    )

    problem = problem_generator.get_problem()

    return problem


def eddy_current_problem_different_currents(
    mesh=None, current=1000, frequency=8000, winding_count=1
):
    if mesh is None:
        wp = BilletParams(diameter=0.030, height=0.070)
        ind = RectangularInductorParams(
            coil_inner_diameter=0.050,
            coil_height=0.040,
            winding_count=winding_count,
            profile_width=0.007,
            profile_height=0.007,
        )
        kw = dict(h_workpiece=1e-3, h_air=60e-3, h_coil=1e-3)
        builder = IHGeometryAndMesh(wp, ind, **kw)
        mesh = builder.generate()

    dirichlet_boundaries = ["bc_air", "bc_axis", "bc_workpiece_left"]
    dirichlet_boundaries_dict = {"bc_air": 0, "bc_axis": 0, "bc_workpiece_left": 0}

    problem_generator = GenericEddyCurrentProblem(
        mesh, dirichlet_boundaries, dirichlet_boundaries_dict, A_star=0.0022536
    )

    problem = problem_generator.get_problem(current=current, frequency=frequency)

    return problem


def eddy_current_problem_different_meshes(setting="default"):

    h_workpiece = 1e-3
    h_air = 60e-3
    h_coil = 2e-3
    # baseline_h_max = {
    #     "h_workpiece": 1e-3,
    #     "h_air": 60e-3,
    #     "h_coil": 1e-3,
    # }

    match setting:
        case "coarse":
            h_workpiece *= 2
            h_coil *= 2
        case "fine":
            h_workpiece /= 2
            h_coil /= 2
        case "very_fine":
            h_workpiece /= 4
            h_coil /= 4
        case _:
            pass  # Use default values

    wp = BilletParams(diameter=0.030, height=0.070)
    ind = RectangularInductorParams(
        coil_inner_diameter=0.050,
        coil_height=0.040,
        winding_count=1,
        profile_width=0.007,
        profile_height=0.007,
    )
    kw = dict(h_workpiece=h_workpiece, h_air=h_air, h_coil=h_coil)
    builder = IHGeometryAndMesh(wp, ind, **kw)
    mesh = builder.generate()

    dirichlet_boundaries = ["bc_air", "bc_axis", "bc_workpiece_left"]
    dirichlet_boundaries_dict = {"bc_air": 0, "bc_axis": 0, "bc_workpiece_left": 0}

    problem_generator = GenericEddyCurrentProblem(
        mesh, dirichlet_boundaries, dirichlet_boundaries_dict, A_star=0.0022536
    )

    problem = problem_generator.get_problem(current=3000, frequency=3000)

    return problem


def em_team_36_problem(mesh=None):

    mm = 1e-3
    if mesh is None:
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
            h_workpiece=1 * mm,
            h_coil=8 * mm,
            h_air=100 * mm,
            air_width=300 * mm,
            air_height_factor=2.0,
        )
        mesh = builder.generate()

    dirichlet_boundaries = ["bc_air", "bc_axis", "bc_workpiece_left"]
    dirichlet_boundaries_dict = {"bc_air": 0, "bc_axis": 0, "bc_workpiece_left": 0}

    full_coil_area = 20 * mm * 40 * mm
    hollow_coil_area = (20 - 2 * 3) * mm * (40 - 2 * 3) * mm
    area_coil = full_coil_area - hollow_coil_area
    print(f"Coil area used for current density calculation: {area_coil:.6e} m^2")

    material_properties = {
        "mat_workpiece": MaterialPropertiesEM(
            mu=100.0,
            sigma=4761904,
        ),
        "mat_air": MaterialPropertiesEM(mu=1.0, sigma=0),
        "mat_coil": MaterialPropertiesEM(mu=1.0, sigma=0),
    }

    problem_generator = GenericEddyCurrentProblem(
        mesh,
        dirichlet_boundaries,
        dirichlet_boundaries_dict,
        material_properties=material_properties,
        A_star=3.9e-3,
        r_star=600e-3,
        coil_area=area_coil,
    )

    problem = problem_generator.get_problem(current=4950, frequency=2000)

    print("Skin depth in workpiece (m):", problem.calculate_skin_depth())

    return problem


def eddy_current_problem_temp_dependent_conductivity():
    """
    Not finished!
    """
    wp = BilletParams(diameter=0.030, height=0.070)
    ind = RectangularInductorParams(
        coil_inner_diameter=0.050,
        coil_height=0.040,
        winding_count=1,
        profile_width=0.007,
        profile_height=0.007,
    )
    kw = dict(h_workpiece=1e-3, h_air=60e-3, h_coil=1e-3)
    builder = IHGeometryAndMesh(wp, ind, **kw)
    mesh = builder.generate()

    dirichlet_boundaries = ["bc_air", "bc_axis", "bc_workpiece_left"]
    dirichlet_boundaries_dict = {"bc_air": 0, "bc_axis": 0, "bc_workpiece_left": 0}

    temp_creator = GraphCreatorEM(mesh)
    temp_data, temp_aux = temp_creator.create_graph()

    electrical_resistivity = {
        "0": 1.77e-7,
        "100": 2.38e-7,
        "200": 3.12e-7,
        "300": 4e-7,
        "400": 5.1e-7,
        "500": 6.35e-7,
        "600": 7.55e-7,
        "700": 9.5e-6,
        "800": 1.11e-6,
        "900": 1.16e-6,
        "1000": 1.19e-6,
    }
    electrical_conductivity = {T: 1 / rho for T, rho in electrical_resistivity.items()}

    print(electrical_conductivity)

    def sigma_from_temperature_builder(T):
        # Simple linear interpolation of conductivity based on the provided table
        T_values = np.array(list(electrical_conductivity.keys()), dtype=np.float64)
        sigma_values = np.array(
            list(electrical_conductivity.values()), dtype=np.float64
        )
        return np.interp(T, T_values, sigma_values)

    # sigma_from_temperature = sigma_from_temperature_builder(temperature)

    ngmesh = mesh.ngmesh
    n_nodes = temp_data.pos.shape[0]
    temperature_field_nodes = np.zeros(n_nodes, dtype=np.float64)
    sigma_from_temperature = np.zeros(n_nodes, dtype=np.float64)
    for i, elem in enumerate(ngmesh.Elements2D()):
        mat_index = elem.index
        mat_name = ngmesh.GetMaterial(mat_index)

        # Get vertices of this element
        vertices = elem.vertices
        for v in vertices:
            node_idx = v.nr - 1 if hasattr(v, "nr") else int(v) - 1
            if 0 <= node_idx < n_nodes:
                if mat_name == "mat_workpiece":
                    pos_x, pos_y = temp_data.pos[node_idx]
                    temperature_at_node = temperature_field(pos_x, pos_y)
                    temperature_field_nodes[node_idx] = temperature_at_node
                    sigma_from_temperature[node_idx] = sigma_from_temperature_builder(
                        temperature_at_node
                    )

    import matplotlib.pyplot as plt

    # plt.figure(figsize=(6, 5))
    # scatter = plt.scatter(temp_data.pos[:, 0], temp_data.pos[:, 1], c=sigma_from_temperature, cmap="viridis", vmin=np.min(sigma_from_temperature[np.nonzero(sigma_from_temperature)]), vmax=np.max(sigma_from_temperature))
    # plt.colorbar(scatter, label="Conductivity (S/m)")
    # plt.show()

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        temp_data.pos[:, 0],
        temp_data.pos[:, 1],
        c=temperature_field_nodes,
        cmap="inferno",
        vmin=np.min(temperature_field_nodes[np.nonzero(temperature_field_nodes)]),
        vmax=np.max(temperature_field_nodes),
    )
    plt.colorbar(scatter, label="Temperature (°C)")
    plt.show()

    material_properties = {
        "mat_workpiece": MaterialPropertiesEM(
            mu=100.0,
            sigma=sigma_from_temperature_builder(0),
        ),
        "mat_air": MaterialPropertiesEM(mu=1.0, sigma=0),
        "mat_coil": MaterialPropertiesEM(mu=1.0, sigma=0),
    }
    problem_generator = GenericEddyCurrentProblem(
        mesh,
        dirichlet_boundaries,
        dirichlet_boundaries_dict,
        material_properties=material_properties,
        sigma_nodal=sigma_from_temperature,
    )

    problem = problem_generator.get_problem(current=10000)

    return problem
