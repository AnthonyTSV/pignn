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
        sigma_nodal=None
    ):
        self.r_star = 70 * 1e-3  # m
        self.A_star = 4.8 * 1e-4  # Wb
        self.mu_star = 4 * 3.1415926535e-7  # H/m
        self.J_star = self.A_star / (self.r_star**2 * self.mu_star)
        self.mesh = mesh
        self.dirichlet_boundaries = dirichlet_boundaries
        self.dirichlet_boundaries_dict = dirichlet_boundaries_dict
        self.material_properties = material_properties
        self.sigma_nodal = sigma_nodal  # Optional per-node sigma [S/m] (physical units)

    def get_problem(self, current=1000, frequency=8000):
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
        problem.complex = True
        problem.mixed = False
        problem.frequency = frequency
        problem.I_coil = current
        problem.refresh_derived_quantities()
        if self.material_properties is not None:
            problem.sigma_workpiece = self.material_properties["mat_workpiece"].sigma / problem.sigma_star
        else:
            problem.sigma_workpiece = 6289308 / problem.sigma_star  # Default value (nondimensionalized)
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


def eddy_current_problem_different_currents(mesh = None, current=1000, frequency=8000):
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

    problem = problem_generator.get_problem(current=current, frequency=frequency)

    return problem

def temperature_field(Xn, Yn, T0=600.0, A=600.0, sigma_y=0.18, delta_x=0.08, lambda_x=0.55):

    xn = np.clip(Xn / 0.03, 0.0, 1.0)
    yn = np.clip(Yn / 0.07, 0.0, 1.0)

    return T0 + A * np.exp(-((yn - 0.5) ** 2) / (2 * sigma_y**2)) * (
        1.0 - np.exp(-(1.0 - xn) / delta_x)
    ) * np.exp(-(1.0 - xn) / lambda_x)

def approx_temperature(
    x,
    y,
    width=0.03,
    height=0.07,
    T0=60.0,
    A=600.0,
    sigma_y=0.18,
    delta_x=0.08,
    lambda_x=0.55,
    cmap="coolwarm",
    plot=False,
):

    import matplotlib.pyplot as plt

    x = np.asarray(x)
    y = np.asarray(y)

    X, Y = x, y

    Xn = np.clip(X / width, 0.0, 1.0)
    Yn = np.clip(Y / height, 0.0, 1.0)

    # Approximate analytical temperature field
    T = temperature_field(X, Y)

    if plot:
        Xplot, Yplot = np.meshgrid(np.sort(Xn), np.sort(Yn))
        Tplot = temperature_field(Xplot, Yplot)

        fig, ax = plt.subplots(figsize=(6, 5))
        contour = ax.contourf(Xplot, Yplot, Tplot, levels=50, cmap=cmap)
        plt.colorbar(contour, ax=ax, label="Temperature (°C)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Approximate Temperature Field")

        plt.tight_layout()
        plt.show()

    return T


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
        "1000": 1.19e-6
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
                    sigma_from_temperature[node_idx] = sigma_from_temperature_builder(temperature_at_node)

    import matplotlib.pyplot as plt
    # plt.figure(figsize=(6, 5))
    # scatter = plt.scatter(temp_data.pos[:, 0], temp_data.pos[:, 1], c=sigma_from_temperature, cmap="viridis", vmin=np.min(sigma_from_temperature[np.nonzero(sigma_from_temperature)]), vmax=np.max(sigma_from_temperature))
    # plt.colorbar(scatter, label="Conductivity (S/m)")
    # plt.show()

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(temp_data.pos[:, 0], temp_data.pos[:, 1], c=temperature_field_nodes, cmap="inferno", vmin=np.min(temperature_field_nodes[np.nonzero(temperature_field_nodes)]), vmax=np.max(temperature_field_nodes))
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
