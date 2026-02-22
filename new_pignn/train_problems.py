import numpy as np
import random
import ngsolve as ng
from mesh_utils import (
    create_rectangular_mesh,
    create_lshape_mesh,
    create_gaussian_initial_condition,
    create_ih_mesh,
)
from graph_creator import GraphCreator
from containers import TimeConfig, MeshConfig, MeshProblem, MeshProblemEM


def create_test_problem(maxh=0.2, alpha=1.0):
    """Create a simple test problem for PIMGN training."""
    print("Creating test problem for Physics-Informed training...")

    # Time configuration
    time_config = TimeConfig(dt=0.01, t_final=1.0)
    # Create rectangular mesh
    mesh = create_rectangular_mesh(width=2, height=1, maxh=maxh)

    dirichlet_boundaries = ["left", "right"]
    neumann_boundaries = ["bottom"]
    dirichlet_boundaries_dict = {"left": 100, "right": 100}
    neumann_boundaries_dict = {"bottom": 0}
    robin_boundaries = ["top"]
    robin_boundaries_dict = {"top": (10, 20)}  # h=10.0, T_amb=20.0

    # Mesh configuration
    mesh_config = MeshConfig(
        maxh=maxh,
        order=1,
        dim=2,
        dirichlet_boundaries=dirichlet_boundaries,
        neumann_boundaries=neumann_boundaries,
        robin_boundaries=robin_boundaries,
        mesh_type="rectangle",
    )

    # Create graph to get node positions
    graph_creator = GraphCreator(
        mesh=mesh,
        n_neighbors=2,
        dirichlet_names=dirichlet_boundaries,
        neumann_names=neumann_boundaries,
        robin_names=robin_boundaries,
        connectivity_method="fem",
    )
    # First create a temporary graph to get positions and aux data
    temp_data, temp_aux = graph_creator.create_graph()

    # Create Neumann values based on the temporary data
    neumann_vals = graph_creator.create_neumann_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        neumann_names=neumann_boundaries,
        flux_values=neumann_boundaries_dict,
        seed=42,
    )
    dirichlet_vals = graph_creator.create_dirichlet_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        dirichlet_names=dirichlet_boundaries,
        boundary_values=dirichlet_boundaries_dict,
    )
    h_vals, amb_vals = graph_creator.create_robin_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        robin_names=robin_boundaries,
        robin_values=robin_boundaries_dict,
        seed=42,
    )
    material_node_field = np.ones(temp_data.pos.shape[0], dtype=np.float32) * alpha
    # Create the final graph with Neumann values
    temp_data, _ = graph_creator.create_graph(
        neumann_values=neumann_vals,
        dirichlet_values=dirichlet_vals,
        material_node_field=material_node_field,
    )

    # Create Gaussian initial condition
    initial_condition = create_gaussian_initial_condition(
        pos=temp_data.pos,
        num_gaussians=1,
        amplitude_range=(10.0, 10.0),
        sigma_fraction_range=(0.2, 0.2),
        seed=42,
        centered=True,
        enforce_boundary_conditions=True,
    )
    initial_condition = np.ones_like(initial_condition) * 20.0

    # Create problem
    problem = MeshProblem(
        mesh=mesh,
        graph_data=temp_data,
        initial_condition=initial_condition,
        alpha=alpha,  # Thermal diffusivity
        time_config=time_config,
        mesh_config=mesh_config,
        problem_id=0,
    )

    problem.material_field = material_node_field

    # Store the Neumann values array for later use
    problem.set_neumann_values_array(neumann_vals)
    problem.set_dirichlet_values_array(dirichlet_vals)

    # Set boundary conditions
    problem.set_neumann_values(neumann_boundaries_dict)
    problem.set_dirichlet_values(dirichlet_boundaries_dict)
    problem.set_robin_values(robin_boundaries_dict)
    problem.set_robin_values_array((h_vals, amb_vals))
    # source = create_gaussian_initial_condition(
    #     pos=temp_data.pos,
    #     num_gaussians=1,
    #     amplitude_range=(100.0, 100.0),
    #     sigma_fraction_range=(0.5, 0.5),
    #     seed=42,
    #     centered=True,
    #     enforce_boundary_conditions=True,
    # )
    # problem.set_source_function(source)
    # project initial condition onto FEM space to enforce Dirichlet BCs
    import ngsolve as ng

    fes = ng.H1(mesh, order=1, dirichlet=problem.mesh_config.dirichlet_pipe)
    gfu = ng.GridFunction(fes)
    gfu_initial = ng.GridFunction(fes)

    # Set initial condition on the interior
    gfu_initial.vec.FV().NumPy()[:] = problem.initial_condition

    # Set Dirichlet boundary conditions
    boundary_cf = mesh.BoundaryCF(problem.boundary_values, default=0)
    gfu.Set(boundary_cf, definedon=mesh.Boundaries(problem.mesh_config.dirichlet_pipe))

    # Copy initial condition values for free DOFs only
    free_dofs = fes.FreeDofs()
    for dof in range(fes.ndof):
        if free_dofs[dof]:
            gfu.vec[dof] = gfu_initial.vec[dof]
    problem.initial_condition = gfu.vec.FV().NumPy()

    print(f"Problem created with {problem.n_nodes} nodes and {problem.n_edges} edges")
    print(f"Time steps: {len(time_config.time_steps)}, dt: {time_config.dt}")

    return problem, time_config


def generate_multiple_problems(n_problems=20, seed=42):
    """Generate multiple problems with varying mesh sizes and Gaussian initial conditions."""
    print(f"Generating {n_problems} problems with varying parameters...")

    np.random.seed(seed)
    problems = []

    maxh_values = np.random.uniform(0.1, 0.3, size=n_problems)

    for i in range(n_problems):
        maxh = maxh_values[i]
        problem, time_config = create_test_problem(maxh=maxh)
        problems.append(problem)
        print(f"Generated problem {i+1}/{n_problems} with maxh={maxh:.3f}")

    print(f"Generated {len(problems)} problems successfully!")
    return problems, time_config


def create_lshaped_problem(maxh=0.2):
    # Time configuration
    time_config = TimeConfig(dt=0.01, t_final=0.2)
    # Create rectangular mesh
    length = random.uniform(0.5, 1)
    height = random.uniform(0.5, 1)
    a_l = random.uniform(1 / 3, 2 / 3)
    a_h = random.uniform(1 / 3, 2 / 3)
    corner = random.randint(1, 4)
    mesh = create_lshape_mesh(
        maxh=maxh, length=length, height=height, a_l=a_l, a_h=a_h, corner=corner
    )

    dirichlet_boundaries = ["outer"]
    neumann_boundaries = []
    dirichlet_boundaries_dict = {"outer": 0}
    neumann_boundaries_dict = {}

    # Mesh configuration
    mesh_config = MeshConfig(
        maxh=maxh,
        order=1,
        dim=2,
        dirichlet_boundaries=dirichlet_boundaries,
        neumann_boundaries=neumann_boundaries,
        mesh_type="rectangle",
    )

    # Create graph to get node positions
    graph_creator = GraphCreator(
        mesh=mesh,
        n_neighbors=2,
        dirichlet_names=dirichlet_boundaries,
        neumann_names=neumann_boundaries,
        connectivity_method="fem",
    )
    # First create a temporary graph to get positions and aux data
    temp_data, temp_aux = graph_creator.create_graph()

    # Create Neumann values based on the temporary data
    neumann_vals = graph_creator.create_neumann_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        neumann_names=neumann_boundaries,
        flux_values=neumann_boundaries_dict,
        seed=42,
    )
    dirichlet_vals = graph_creator.create_dirichlet_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        dirichlet_names=dirichlet_boundaries,
        boundary_values=dirichlet_boundaries_dict,
    )

    # Create the final graph with Neumann values
    temp_data, _ = graph_creator.create_graph(
        neumann_values=neumann_vals, dirichlet_values=dirichlet_vals
    )

    # Create Gaussian initial condition
    initial_condition = create_gaussian_initial_condition(
        pos=temp_data.pos,
        num_gaussians=10,
        amplitude_range=(0.5, 1),
        sigma_fraction_range=(1 / 12, 1 / 6),
        seed=42,
        centered=False,
        enforce_boundary_conditions=False,
    )
    # initial_condition = np.zeros_like(initial_condition)

    # Create problem
    problem = MeshProblem(
        mesh=mesh,
        graph_data=temp_data,
        initial_condition=initial_condition,
        alpha=0.5e-2,  # Thermal diffusivity
        time_config=time_config,
        mesh_config=mesh_config,
        problem_id=0,
    )

    # Store the Neumann values array for later use
    problem.set_neumann_values_array(neumann_vals)
    problem.set_dirichlet_values_array(dirichlet_vals)

    # Set boundary conditions
    problem.set_neumann_values(neumann_boundaries_dict)
    problem.set_dirichlet_values(dirichlet_boundaries_dict)
    # source = create_gaussian_initial_condition(
    #     pos=temp_data.pos,
    #     num_gaussians=1,
    #     amplitude_range=(100.0, 100.0),
    #     sigma_fraction_range=(0.5, 0.5),
    #     seed=42,
    #     centered=True,
    #     enforce_boundary_conditions=True,
    # )
    # problem.set_source_function(source)
    # project initial condition onto FEM space to enforce Dirichlet BCs
    import ngsolve as ng

    fes = ng.H1(mesh, order=1, dirichlet=problem.mesh_config.dirichlet_pipe)
    gfu = ng.GridFunction(fes)
    gfu_initial = ng.GridFunction(fes)

    # Set initial condition on the interior
    gfu_initial.vec.FV().NumPy()[:] = problem.initial_condition

    # Set Dirichlet boundary conditions
    boundary_cf = mesh.BoundaryCF(problem.boundary_values, default=0)
    gfu.Set(boundary_cf, definedon=mesh.Boundaries(problem.mesh_config.dirichlet_pipe))

    # Copy initial condition values for free DOFs only
    free_dofs = fes.FreeDofs()
    for dof in range(fes.ndof):
        if free_dofs[dof]:
            gfu.vec[dof] = gfu_initial.vec[dof]
    problem.initial_condition = gfu.vec.FV().NumPy()

    print(f"Problem created with {problem.n_nodes} nodes and {problem.n_edges} edges")
    print(f"Time steps: {len(time_config.time_steps)}, dt: {time_config.dt}")

    return problem, time_config


def create_mms_problem(maxh=0.2, alpha=0.1, problem_id=0):
    """Create a manufactured solution problem for testing."""
    # Time configuration
    time_config = TimeConfig(dt=0.01, t_final=1.0)
    # Create rectangular mesh
    mesh = create_rectangular_mesh(width=1, height=1, maxh=maxh)

    dirichlet_boundaries = ["left", "right", "bottom", "top"]
    neumann_boundaries = []
    dirichlet_boundaries_dict = {"left": 0, "right": 0, "bottom": 0, "top": 0}
    neumann_boundaries_dict = {}

    order = 1

    fes = ng.H1(mesh, order=order, dirichlet="|".join(dirichlet_boundaries))

    # Mesh configuration
    mesh_config = MeshConfig(
        maxh=maxh,
        order=order,
        dim=2,
        dirichlet_boundaries=dirichlet_boundaries,
        neumann_boundaries=neumann_boundaries,
        mesh_type="rectangle",
    )

    # Create graph to get node positions
    graph_creator = GraphCreator(
        mesh=mesh,
        n_neighbors=2,
        dirichlet_names=dirichlet_boundaries,
        neumann_names=neumann_boundaries,
        connectivity_method="fem",
        fes=fes,
    )
    # First create a temporary graph to get positions and aux data
    temp_data, temp_aux = graph_creator.create_graph()

    # Create Neumann values based on the temporary data
    neumann_vals = graph_creator.create_neumann_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        neumann_names=neumann_boundaries,
        flux_values=neumann_boundaries_dict,
        seed=42,
    )
    dirichlet_vals = graph_creator.create_dirichlet_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        dirichlet_names=dirichlet_boundaries,
        boundary_values=dirichlet_boundaries_dict,
    )

    # Create the final graph with Neumann values
    temp_data, _ = graph_creator.create_graph(
        neumann_values=neumann_vals, dirichlet_values=dirichlet_vals
    )

    # Manufactured solution initial condition
    initial_condition = (
        100 * np.sin(np.pi * temp_data.pos[:, 0]) * np.sin(np.pi * temp_data.pos[:, 1])
    )

    # Create problem
    problem = MeshProblem(
        mesh=mesh,
        graph_data=temp_data,
        initial_condition=initial_condition,
        alpha=alpha,  # Thermal diffusivity
        time_config=time_config,
        mesh_config=mesh_config,
        problem_id=0,
    )

    # Store the Neumann values array for later use
    problem.set_neumann_values_array(neumann_vals)
    problem.set_dirichlet_values_array(dirichlet_vals)

    # Set boundary conditions
    problem.set_neumann_values(neumann_boundaries_dict)
    problem.set_dirichlet_values(dirichlet_boundaries_dict)

    gfu = ng.GridFunction(fes)
    gfu_initial = ng.GridFunction(fes)

    # Set initial condition on the interior
    gfu_initial.vec.FV().NumPy()[:] = problem.initial_condition

    # Set Dirichlet boundary conditions
    boundary_cf = mesh.BoundaryCF(problem.boundary_values, default=0)
    gfu.Set(boundary_cf, definedon=mesh.Boundaries(problem.mesh_config.dirichlet_pipe))

    # Copy initial condition values for free DOFs only
    free_dofs = fes.FreeDofs()
    for dof in range(fes.ndof):
        if free_dofs[dof]:
            gfu.vec[dof] = gfu_initial.vec[dof]
    problem.initial_condition = gfu.vec.FV().NumPy()

    print(f"Problem created with {problem.n_nodes} nodes and {problem.n_edges} edges")
    print(f"Time steps: {len(time_config.time_steps)}, dt: {time_config.dt}")

    return problem, time_config


def create_source_test_problem(maxh=0.2, alpha=0.1, problem_id=0):
    # Time configuration
    time_config = TimeConfig(dt=0.01, t_final=1.0)
    # Create rectangular mesh
    mesh = create_rectangular_mesh(width=1, height=1, maxh=maxh)

    dirichlet_boundaries = ["right", "bottom", "top"]
    neumann_boundaries = ["left"]
    dirichlet_boundaries_dict = {"right": 0, "bottom": 0, "top": 0}
    neumann_boundaries_dict = {"left": 0}

    # Mesh configuration
    mesh_config = MeshConfig(
        maxh=maxh,
        order=1,
        dim=2,
        dirichlet_boundaries=dirichlet_boundaries,
        neumann_boundaries=neumann_boundaries,
        mesh_type="rectangle",
    )

    # Create graph to get node positions
    graph_creator = GraphCreator(
        mesh=mesh,
        n_neighbors=2,
        dirichlet_names=dirichlet_boundaries,
        neumann_names=neumann_boundaries,
        connectivity_method="fem",
    )
    # First create a temporary graph to get positions and aux data
    temp_data, temp_aux = graph_creator.create_graph()

    # Create Neumann values based on the temporary data
    neumann_vals = graph_creator.create_neumann_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        neumann_names=neumann_boundaries,
        flux_values=neumann_boundaries_dict,
        seed=42,
    )
    dirichlet_vals = graph_creator.create_dirichlet_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        dirichlet_names=dirichlet_boundaries,
        boundary_values=dirichlet_boundaries_dict,
    )

    # Create the final graph with Neumann values
    temp_data, _ = graph_creator.create_graph(
        neumann_values=neumann_vals, dirichlet_values=dirichlet_vals
    )

    # Manufactured solution initial condition
    initial_condition = np.zeros_like(temp_data.pos[:, 0])

    # Create problem
    problem = MeshProblem(
        mesh=mesh,
        graph_data=temp_data,
        initial_condition=initial_condition,
        alpha=alpha,  # Thermal diffusivity
        time_config=time_config,
        mesh_config=mesh_config,
        problem_id=0,
    )

    # Store the Neumann values array for later use
    problem.set_neumann_values_array(neumann_vals)
    problem.set_dirichlet_values_array(dirichlet_vals)

    # Set boundary conditions
    problem.set_neumann_values(neumann_boundaries_dict)
    problem.set_dirichlet_values(dirichlet_boundaries_dict)

    x = temp_data.pos[:, 0]
    y = temp_data.pos[:, 1]
    x_depth = 1 - x
    depth = 0.1
    profile_x = np.exp(-x_depth / depth)
    y_center = 1 / 2.0
    sigma_y_mm = 1 / 4.0
    profile_y = np.exp(-((y - y_center) ** 2) / (2 * sigma_y_mm**2))
    s0 = 500.0
    source_function = s0 * profile_x * profile_y
    problem.set_source_function(source_function)

    import ngsolve as ng

    fes = ng.H1(mesh, order=1, dirichlet=problem.mesh_config.dirichlet_pipe)
    gfu = ng.GridFunction(fes)
    gfu_initial = ng.GridFunction(fes)

    # Set initial condition on the interior
    gfu_initial.vec.FV().NumPy()[:] = problem.initial_condition

    # Set Dirichlet boundary conditions
    boundary_cf = mesh.BoundaryCF(problem.boundary_values, default=0)
    gfu.Set(boundary_cf, definedon=mesh.Boundaries(problem.mesh_config.dirichlet_pipe))

    # Copy initial condition values for free DOFs only
    free_dofs = fes.FreeDofs()
    for dof in range(fes.ndof):
        if free_dofs[dof]:
            gfu.vec[dof] = gfu_initial.vec[dof]
    problem.initial_condition = gfu.vec.FV().NumPy()

    print(f"Problem created with {problem.n_nodes} nodes and {problem.n_edges} edges")
    print(f"Time steps: {len(time_config.time_steps)}, dt: {time_config.dt}")

    return problem, time_config


def create_industrial_heating_problem(maxh=0.1):
    density = 7850  # kg/m^3
    specific_heat = 450  # J/(kg·K)
    k = 45  # W/(m·K)
    # Time configuration
    time_config = TimeConfig(dt=0.01, t_final=1.0)
    h_conv = 10  # Air convective heat transfer coefficient
    T_amb = 23  # Ambient temperature
    alpha = k / (density * specific_heat)  # Thermal diffusivity
    # Create rectangular mesh
    L = 30e-3  # m full billet length
    D = L / 2  # m half of the billet
    H = 100e-3  # m
    mesh = create_rectangular_mesh(width=D, height=H, maxh=maxh)

    dirichlet_boundaries = ["bottom"]
    neumann_boundaries = ["left"]
    robin_boundaries = ["top", "right"]
    dirichlet_boundaries_dict = {"bottom": 23}
    neumann_boundaries_dict = {"left": 0}
    robin_boundaries_dict = {"top": (h_conv, T_amb), "right": (h_conv, T_amb)}

    # Mesh configuration
    mesh_config = MeshConfig(
        maxh=maxh,
        order=1,
        dim=2,
        dirichlet_boundaries=dirichlet_boundaries,
        neumann_boundaries=neumann_boundaries,
        mesh_type="rectangle",
    )

    # Create graph to get node positions
    graph_creator = GraphCreator(
        mesh=mesh,
        n_neighbors=2,
        dirichlet_names=dirichlet_boundaries,
        neumann_names=neumann_boundaries,
        connectivity_method="fem",
    )
    # First create a temporary graph to get positions and aux data
    temp_data, temp_aux = graph_creator.create_graph()

    # Create Neumann values based on the temporary data
    neumann_vals = graph_creator.create_neumann_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        neumann_names=neumann_boundaries,
        flux_values=neumann_boundaries_dict,
        seed=42,
    )
    dirichlet_vals = graph_creator.create_dirichlet_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        dirichlet_names=dirichlet_boundaries,
        boundary_values=dirichlet_boundaries_dict,
    )
    h_vals, amb_vals = graph_creator.create_robin_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        robin_names=robin_boundaries,
        robin_values=robin_boundaries_dict,
    )

    # Create the final graph with Neumann values
    temp_data, _ = graph_creator.create_graph(
        neumann_values=neumann_vals,
        dirichlet_values=dirichlet_vals,
        robin_values=(h_vals, amb_vals),
    )

    initial_condition = np.ones_like(temp_data.pos[:, 0]) * T_amb

    # Create problem
    problem = MeshProblem(
        mesh=mesh,
        graph_data=temp_data,
        initial_condition=initial_condition,
        alpha=alpha,  # Thermal diffusivity
        time_config=time_config,
        mesh_config=mesh_config,
        problem_id=0,
    )

    # Store the Neumann values array for later use
    problem.set_neumann_values_array(neumann_vals)
    problem.set_dirichlet_values_array(dirichlet_vals)
    problem.set_robin_values_array((h_vals, amb_vals))

    # Set boundary conditions
    problem.set_neumann_values(neumann_boundaries_dict)
    problem.set_dirichlet_values(dirichlet_boundaries_dict)
    problem.set_robin_values(robin_boundaries_dict)

    x = temp_data.pos[:, 0]
    y = temp_data.pos[:, 1]
    x_depth = D - x
    delta_m = 0.2 * D
    profile_x = np.exp(-x_depth / delta_m)
    y_center = H / 2.0
    sigma_y_m = H / 4.0
    profile_y = np.exp(-((y - y_center) ** 2) / (2 * sigma_y_m**2))
    s0 = 100
    source_function = s0 * profile_x * profile_y
    problem.set_source_function(source_function)

    import ngsolve as ng

    fes = ng.H1(mesh, order=1, dirichlet=problem.mesh_config.dirichlet_pipe)
    gfu = ng.GridFunction(fes)
    gfu_initial = ng.GridFunction(fes)

    # Set initial condition on the interior
    gfu_initial.vec.FV().NumPy()[:] = problem.initial_condition

    # Set Dirichlet boundary conditions
    boundary_cf = mesh.BoundaryCF(problem.boundary_values, default=0)
    gfu.Set(boundary_cf, definedon=mesh.Boundaries(problem.mesh_config.dirichlet_pipe))

    # Copy initial condition values for free DOFs only
    free_dofs = fes.FreeDofs()
    for dof in range(fes.ndof):
        if free_dofs[dof]:
            gfu.vec[dof] = gfu_initial.vec[dof]
    problem.initial_condition = gfu.vec.FV().NumPy()

    print(f"Problem created with {problem.n_nodes} nodes and {problem.n_edges} edges")
    print(f"Time steps: {len(time_config.time_steps)}, dt: {time_config.dt}")

    return problem, time_config

def create_em_to_thermal():
    """
    This function will create an electromagnetic problem, solve it, 
    and then use the resulting fields to create a thermal problem with
    a source term based on the electromagnetic solution.
    """
    from pathlib import Path
    import sys
    project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root))
    from fem_tests.induction_heating import solve_em
    import ngsolve as ng
    mesh = create_ih_mesh(overwrite_r_star=True)
    A, phi, r1 = solve_em()
    density = 7870  # kg/m^3
    specific_heat = 461  # J/(kg·K)
    k = 86  # W/(m·K)
    # Time configuration
    time_config = TimeConfig(dt=0.1, t_final=1.0)
    h_conv_air = 10  # Air convective heat transfer coefficient
    h_conv = h_conv_air / (density * specific_heat)
    T_amb = 22  # Ambient temperature
    alpha = k / (density * specific_heat)  # Thermal diffusivity
    dirichlet_boundaries = []
    neumann_boundaries = []
    robin_boundaries = ["bc_workpiece_top", "bc_workpiece_right", "bc_workpiece_bottom"]
    dirichlet_boundaries_dict = {}
    neumann_boundaries_dict = {}
    robin_boundaries_dict = {"bc_workpiece_top": (h_conv, T_amb), "bc_workpiece_right": (h_conv, T_amb), "bc_workpiece_bottom": (h_conv, T_amb)}

    # Mesh configuration
    mesh_config = MeshConfig(
        maxh=1e-3,
        order=1,
        dim=2,
        dirichlet_boundaries=dirichlet_boundaries,
        neumann_boundaries=neumann_boundaries,
        robin_boundaries=robin_boundaries,
        mesh_type="ih",
    )

    # Create graph to get node positions
    graph_creator = GraphCreator(
        mesh=mesh,
        n_neighbors=2,
        dirichlet_names=dirichlet_boundaries,
        neumann_names=neumann_boundaries,
        robin_names=robin_boundaries,
        connectivity_method="fem",
    )
    # First create a temporary graph to get positions and aux data
    temp_data, temp_aux = graph_creator.create_graph()

    # Create Neumann values based on the temporary data
    neumann_vals = graph_creator.create_neumann_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        neumann_names=neumann_boundaries,
        flux_values=neumann_boundaries_dict,
        seed=42,
    )
    dirichlet_vals = graph_creator.create_dirichlet_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        dirichlet_names=dirichlet_boundaries,
        boundary_values=dirichlet_boundaries_dict,
    )
    h_vals, amb_vals = graph_creator.create_robin_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        robin_names=robin_boundaries,
        robin_values=robin_boundaries_dict,
    )

    # Create the final graph with Neumann values
    temp_data, _ = graph_creator.create_graph(
        neumann_values=neumann_vals,
        dirichlet_values=dirichlet_vals,
        robin_values=(h_vals, amb_vals),
    )

    initial_condition = np.ones_like(temp_data.pos[:, 0]) * T_amb

    # Create problem
    problem = MeshProblem(
        mesh=mesh,
        graph_data=temp_data,
        initial_condition=initial_condition,
        alpha=alpha,  # Thermal diffusivity
        time_config=time_config,
        mesh_config=mesh_config,
        problem_id=0,
    )

    # Store the Neumann values array for later use
    problem.set_neumann_values_array(neumann_vals)
    problem.set_dirichlet_values_array(dirichlet_vals)
    problem.set_robin_values_array((h_vals, amb_vals))

    # Set boundary conditions
    problem.set_neumann_values(neumann_boundaries_dict)
    problem.set_dirichlet_values(dirichlet_boundaries_dict)
    problem.set_robin_values(robin_boundaries_dict)

    x = temp_data.pos[:, 0]
    y = temp_data.pos[:, 1]
    omega = 2 * np.pi * 8000
    sigma = 6289308
    E_phi = -1j * omega * (A + phi * r1)

    Q = 0.5 * sigma * ng.Norm(E_phi) ** 2 / (density * specific_heat)

    ngmesh = mesh.ngmesh
    n_nodes = temp_data.pos.shape[0]
    source_function = np.zeros(n_nodes, dtype=np.float64)
    wp_node_mask = np.zeros(n_nodes, dtype=bool)  # Track which nodes are in the workpiece
    for i, elem in enumerate(ngmesh.Elements2D()):
        mat_index = elem.index
        mat_name = ngmesh.GetMaterial(mat_index)

        # Get vertices of this element
        vertices = elem.vertices
        for v in vertices:
            node_idx = v.nr - 1 if hasattr(v, "nr") else int(v) - 1
            if 0 <= node_idx < n_nodes:
                if mat_name == "mat_workpiece":
                    p = mesh.ngmesh.Points()[v.nr].p
                    x, y = p[0], p[1]
                    q_val = Q(mesh(x, y))
                    source_function[node_idx] = q_val
                    wp_node_mask[node_idx] = True

    problem.set_source_function(source_function)
    problem.set_source_coefficient(Q)  # CF for accurate FEM quadrature integration
    problem.set_wp_node_mask(wp_node_mask)
    problem.set_material_region("mat_workpiece")
    problem.set_axisymmetric(True)
    if dirichlet_boundaries:

        fes = ng.H1(mesh, order=1, dirichlet=problem.mesh_config.dirichlet_pipe)
        gfu = ng.GridFunction(fes)
        gfu_initial = ng.GridFunction(fes)

        # Set initial condition on the interior
        gfu_initial.vec.FV().NumPy()[:] = problem.initial_condition

        # Set Dirichlet boundary conditions
        boundary_cf = mesh.BoundaryCF(problem.boundary_values, default=0)
        gfu.Set(boundary_cf, definedon=mesh.Boundaries(problem.mesh_config.dirichlet_pipe))

        # Copy initial condition values for free DOFs only
        free_dofs = fes.FreeDofs()
        for dof in range(fes.ndof):
            if free_dofs[dof]:
                gfu.vec[dof] = gfu_initial.vec[dof]
        problem.initial_condition = gfu.vec.FV().NumPy()

    print(f"Problem created with {problem.n_nodes} nodes and {problem.n_edges} edges")
    print(f"Time steps: {len(time_config.time_steps)}, dt: {time_config.dt}")

    return problem, time_config

def create_em_problem():

    r_star = 70 * 1e-3  # m
    A_star = 4.8 * 1e-4  # Wb/m
    mu_star = 4 * 3.1415926535e-7  # H/m
    J_star = A_star / (r_star**2 * mu_star)

    from graph_creator_em import GraphCreatorEM

    mesh = create_ih_mesh()

    dirichlet_boundaries = ["bc_air", "bc_axis", "bc_workpiece_left"]
    dirichlet_boundaries_dict = {"bc_air": 0, "bc_axis": 0, "bc_workpiece_left": 0}
    fes = ng.H1(mesh, order=1, dirichlet="|".join(dirichlet_boundaries))
    # Mesh configuration
    mesh_config = MeshConfig(
        maxh=1,
        order=1,
        dim=2,
        dirichlet_boundaries=dirichlet_boundaries,
        mesh_type="ih_mesh",
    )
    graph_creator = GraphCreatorEM(
        mesh=mesh,
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
        mesh=mesh,
        graph_data=temp_data,
        mesh_config=mesh_config,
        problem_id=0,
    )
    problem.set_dirichlet_values_array(dirichlet_vals)
    problem.set_dirichlet_values(dirichlet_boundaries_dict)
    problem.complex = False
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
    Js_phi = Js_phi / J_star  # Normalize current density

    # Material property mapping (mu_r values)
    mu_r_map = {
        "mat_workpiece": problem.mu_r_workpiece,
        "mat_air": problem.mu_r_air,
        "mat_coil": problem.mu_r_coil,
    }

    # Identify nodes in each material region and assign properties
    ngmesh = mesh.ngmesh
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

def create_em_problem_complex():

    r_star = 70 * 1e-3  # m
    A_star = 4.8 * 1e-4  # Wb/m
    mu_star = 4 * 3.1415926535e-7  # H/m
    J_star = A_star / (r_star**2 * mu_star)
    frequency = 1000  # Hz
    omega = 2 * ng.pi * frequency  # rad/s
    sigma_star = J_star / (omega * A_star)

    from graph_creator_em import GraphCreatorEM

    mesh = create_ih_mesh()

    dirichlet_boundaries = ["bc_air", "bc_axis", "bc_workpiece_left"]
    dirichlet_boundaries_dict = {"bc_air": 0, "bc_axis": 0, "bc_workpiece_left": 0}
    fes = ng.H1(mesh, order=1, dirichlet="|".join(dirichlet_boundaries))
    # Mesh configuration
    mesh_config = MeshConfig(
        maxh=1,
        order=1,
        dim=2,
        dirichlet_boundaries=dirichlet_boundaries,
        mesh_type="ih_mesh",
    )
    graph_creator = GraphCreatorEM(
        mesh=mesh,
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
        mesh=mesh,
        graph_data=temp_data,
        mesh_config=mesh_config,
        problem_id=0,
    )
    problem.set_dirichlet_values_array(dirichlet_vals)
    problem.set_dirichlet_values(dirichlet_boundaries_dict)
    problem.complex = True
    problem.sigma_workpiece = 1e6 / sigma_star
    problem.sigma_air = 0.0 / sigma_star
    problem.sigma_coil = 5.8e7 / sigma_star

    # Create material fields (mu_r at each node) based on material subdomain
    n_nodes = temp_data.pos.shape[0]
    mu_r_field = np.ones(n_nodes, dtype=np.float64)  # Default to air (mu_r = 1)
    sigma_field = np.zeros(n_nodes, dtype=np.float64) # Default to 0 conductivity
    current_density = np.zeros(n_nodes, dtype=np.float64)

    # Calculate current density in the coil: J = N * I / A_coil
    Acoil = problem.profile_width_phys * problem.profile_height_phys
    Js_phi = problem.N_turns * problem.I_coil / Acoil
    Js_phi = Js_phi / J_star  # Normalize current density

    # Material property mapping (mu_r values)
    mu_r_map = {
        "mat_workpiece": problem.mu_r_workpiece,
        "mat_air": problem.mu_r_air,
        "mat_coil": problem.mu_r_coil,
    }

    sigma_map = {
        "mat_workpiece": problem.sigma_workpiece,
        "mat_air": problem.sigma_air,
        "mat_coil": problem.sigma_coil,
    }

    material_encode = {
        "mat_workpiece": 2,
        "mat_air": 0,
        "mat_coil": 1,
    }

    # Identify nodes in each material region and assign properties
    ngmesh = mesh.ngmesh
    for i, elem in enumerate(ngmesh.Elements2D()):
        mat_index = elem.index
        mat_name = ngmesh.GetMaterial(mat_index)

        # Get vertices of this element
        vertices = elem.vertices
        for v in vertices:
            node_idx = v.nr - 1 if hasattr(v, "nr") else int(v) - 1
            if 0 <= node_idx < n_nodes:
                # # Assign mu_r based on material region
                # if mat_name in mu_r_map:
                #     mu_r_field[node_idx] = mu_r_map[mat_name]
                # if mat_name in sigma_map:
                #     sigma_field[node_idx] = sigma_map[mat_name]

                # # Assign current density (only in coil)
                # if mat_name == "mat_coil":
                #     current_density[node_idx] = Js_phi
                mu_r_field[node_idx] = material_encode[mat_name]

    problem.material_field = mu_r_field
    problem.sigma_field = sigma_field
    problem.current_density_field = current_density

    return problem

def create_em_mixed(i_coil: float = 1000):

    r_star = 70 * 1e-3  # m
    A_star = 4.8 * 1e-4  # Wb/m
    mu_star = 4 * 3.1415926535e-7  # H/m
    J_star = A_star / (r_star**2 * mu_star)
    frequency = 1000  # Hz
    omega = 2 * ng.pi * frequency  # rad/s
    sigma_star = J_star / (omega * A_star)

    from graph_creator_em import GraphCreatorEM

    mesh = create_ih_mesh()

    dirichlet_boundaries = ["bc_air", "bc_axis", "bc_workpiece_left"]
    dirichlet_boundaries_dict = {"bc_air": 0, "bc_axis": 0, "bc_workpiece_left": 0}
    fes = ng.H1(mesh, order=1, dirichlet="|".join(dirichlet_boundaries))
    # Mesh configuration
    mesh_config = MeshConfig(
        maxh=1,
        order=1,
        dim=2,
        dirichlet_boundaries=dirichlet_boundaries,
        mesh_type="ih_mesh",
    )
    graph_creator = GraphCreatorEM(
        mesh=mesh,
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
        mesh=mesh,
        graph_data=temp_data,
        mesh_config=mesh_config,
        problem_id=0,
    )
    problem.set_dirichlet_values_array(dirichlet_vals)
    problem.set_dirichlet_values(dirichlet_boundaries_dict)
    problem.complex = True
    problem.sigma_workpiece = 6289308 / sigma_star
    problem.sigma_air = 0.0
    problem.sigma_coil = 58823529 / sigma_star
    problem.omega = omega
    problem.I_coil = i_coil

    # Create material fields (mu_r at each node) based on material subdomain
    n_nodes = temp_data.pos.shape[0]
    mu_r_field = np.ones(n_nodes, dtype=np.float64)  # Default to air (mu_r = 1)
    sigma_field = np.zeros(n_nodes, dtype=np.float64)  # Default to 0 conductivity
    current_density = np.zeros(n_nodes, dtype=np.float64)
    coil_node_mask = np.zeros(n_nodes, dtype=bool)  # Track which nodes are in the coil

    # Calculate current density in the coil: J = N * I / A_coil
    Acoil = problem.profile_width_phys * problem.profile_height_phys
    Js_phi = problem.N_turns * problem.I_coil / Acoil
    Js_phi = Js_phi / J_star  # Normalize current density

    # Material property mapping (mu_r values)
    mu_r_map = {
        "mat_workpiece": problem.mu_r_workpiece,
        "mat_air": problem.mu_r_air,
        "mat_coil": problem.mu_r_coil,
    }

    sigma_map = {
        "mat_workpiece": problem.sigma_workpiece,
        "mat_air": problem.sigma_air,
        "mat_coil": problem.sigma_coil,
    }

    material_encode = {
        "mat_workpiece": 2,
        "mat_air": 0,
        "mat_coil": 1,
    }

    # Identify nodes in each material region and assign properties
    ngmesh = mesh.ngmesh
    for i, elem in enumerate(ngmesh.Elements2D()):
        mat_index = elem.index
        mat_name = ngmesh.GetMaterial(mat_index)

        # Get vertices of this element
        vertices = elem.vertices
        for v in vertices:
            node_idx = v.nr - 1 if hasattr(v, "nr") else int(v) - 1
            if 0 <= node_idx < n_nodes:
                # # Assign mu_r based on material region
                # if mat_name in mu_r_map:
                #     mu_r_field[node_idx] = mu_r_map[mat_name]
                # if mat_name in sigma_map:
                #     sigma_field[node_idx] = sigma_map[mat_name]

                # # Assign current density (only in coil)
                # if mat_name == "mat_coil":
                #     current_density[node_idx] = Js_phi
                mu_r_field[node_idx] = material_encode[mat_name]

                # Track coil nodes for phi computation
                if mat_name == "mat_coil":
                    coil_node_mask[node_idx] = True

    problem.material_field = mu_r_field
    problem.sigma_field = sigma_field
    problem.current_density_field = current_density
    problem.coil_node_mask = coil_node_mask  # Boolean mask for coil nodes

    return problem