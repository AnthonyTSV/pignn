import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import time
from argparse import Namespace
from typing import List

# Import our modules
from meshgraphnet import MeshGraphNet
from fem import FEMSolver
from mesh_utils import create_dirichlet_values, create_rectangular_mesh, create_lshape_mesh, create_gaussian_initial_condition
from graph_creator import GraphCreator
from containers import TimeConfig, MeshConfig, MeshProblem
from torch_geometric.data import Data

def create_test_problem(maxh=0.2, alpha=1.0):
    """Create a simple test problem for PIMGN training."""
    print("Creating test problem for Physics-Informed training...")
    
    # Time configuration
    time_config = TimeConfig(
        dt=0.01,
        t_final=2.0
    )
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
        mesh_type="rectangle"
    )
    
    # Create graph to get node positions
    graph_creator = GraphCreator(
        mesh=mesh,
        n_neighbors=2,
        dirichlet_names=dirichlet_boundaries,
        neumann_names=neumann_boundaries,
        robin_names=robin_boundaries,
        connectivity_method="fem"
    )
    # First create a temporary graph to get positions and aux data
    temp_data, temp_aux = graph_creator.create_graph()
    
    # Create Neumann values based on the temporary data
    neumann_vals = graph_creator.create_neumann_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        neumann_names=neumann_boundaries,
        flux_values=neumann_boundaries_dict,
        seed=42
    )
    dirichlet_vals = create_dirichlet_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        dirichlet_names=dirichlet_boundaries,
        boundary_values=dirichlet_boundaries_dict
    )
    h_vals, amb_vals = graph_creator.create_robin_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        robin_names=robin_boundaries,
        robin_values=robin_boundaries_dict,
        seed=42
    )
    material_node_field = np.ones(temp_data.pos.shape[0], dtype=np.float32) * alpha
    # Create the final graph with Neumann values
    temp_data, _ = graph_creator.create_graph(neumann_values=neumann_vals, dirichlet_values=dirichlet_vals, material_node_field=material_node_field)
    
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
    initial_condition = np.zeros_like(initial_condition)
    
    # Create problem
    problem = MeshProblem(
        mesh=mesh,
        graph_data=temp_data,
        initial_condition=initial_condition,
        alpha=alpha,  # Thermal diffusivity
        time_config=time_config,
        mesh_config=mesh_config,
        problem_id=0
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
    time_config = TimeConfig(
        dt=0.01,
        t_final=0.2
    )
    # Create rectangular mesh
    length = random.uniform(0.5, 1)
    height = random.uniform(0.5, 1)
    a_l = random.uniform(1/3, 2/3)
    a_h = random.uniform(1/3, 2/3)
    corner = random.randint(1, 4)
    mesh = create_lshape_mesh(
        maxh=maxh,
        length=length,
        height=height,
        a_l=a_l,
        a_h=a_h,
        corner=corner
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
        mesh_type="rectangle"
    )
    
    # Create graph to get node positions
    graph_creator = GraphCreator(
        mesh=mesh,
        n_neighbors=2,
        dirichlet_names=dirichlet_boundaries,
        neumann_names=neumann_boundaries,
        connectivity_method="fem"
    )
    # First create a temporary graph to get positions and aux data
    temp_data, temp_aux = graph_creator.create_graph()
    
    # Create Neumann values based on the temporary data
    neumann_vals = graph_creator.create_neumann_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        neumann_names=neumann_boundaries,
        flux_values=neumann_boundaries_dict,
        seed=42
    )
    dirichlet_vals = create_dirichlet_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        dirichlet_names=dirichlet_boundaries,
        boundary_values=dirichlet_boundaries_dict
    )
    
    # Create the final graph with Neumann values
    temp_data, _ = graph_creator.create_graph(neumann_values=neumann_vals, dirichlet_values=dirichlet_vals)
    
    # Create Gaussian initial condition
    initial_condition = create_gaussian_initial_condition(
        pos=temp_data.pos,
        num_gaussians=10,
        amplitude_range=(0.5, 1),
        sigma_fraction_range=(1/12, 1/6),
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
        problem_id=0
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

def create_nonlinear_rectangular_problem(maxh=0.1, seed=42, n_functions=10):
    """Create 2D nonlinear inhomogeneous heating problem on a rectangle."""
    import ngsolve as ng

    rng = np.random.default_rng(seed)

    # Experiment constants from paper
    t0 = 1.0
    T0 = 100.0
    q0 = 20.0
    alpha_f = 0.1
    alpha_m = 0.01
    vf0 = 0.5
    dt = 0.01
    t_final = 0.4

    time_config = TimeConfig(dt=dt, t_final=t_final)

    mesh = create_rectangular_mesh(width=1.0, height=1.0, maxh=maxh)

    dirichlet_boundaries = ["bottom", "right", "top", "left"]
    neumann_boundaries = []
    dirichlet_boundaries_dict = {name: T0 for name in dirichlet_boundaries}
    neumann_boundaries_dict = {}

    mesh_config = MeshConfig(
        maxh=maxh,
        order=1,
        dim=2,
        dirichlet_boundaries=dirichlet_boundaries,
        neumann_boundaries=neumann_boundaries,
        mesh_type="rectangle",
    )

    graph_creator = GraphCreator(
        mesh=mesh,
        n_neighbors=2,
        dirichlet_names=dirichlet_boundaries,
        neumann_names=neumann_boundaries,
        connectivity_method="fem",
    )

    temp_data, temp_aux = graph_creator.create_graph()

    neumann_vals = graph_creator.create_neumann_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        neumann_names=neumann_boundaries,
        flux_values=neumann_boundaries_dict,
        seed=seed,
    )
    dirichlet_vals = create_dirichlet_values(
        pos=temp_data.pos,
        aux_data=temp_aux,
        dirichlet_names=dirichlet_boundaries,
        boundary_values=dirichlet_boundaries_dict,
    )

    pos_np = temp_data.pos.numpy()
    vf_field = np.full(pos_np.shape[0], vf0, dtype=np.float64)

    coeff_params = []
    for _ in range(n_functions):
        ai = rng.uniform(0.0, 1.0 / 20.0)
        kx = rng.uniform(0.0, 8.0 * np.pi)
        ky = rng.uniform(0.0, 8.0 * np.pi)
        dx = rng.uniform(0.0, 2.0 * np.pi)
        dy = rng.uniform(0.0, 2.0 * np.pi)
        coeff_params.append((ai, kx, ky, dx, dy))
        vf_field += ai * np.sin(kx * pos_np[:, 0] + dx) * np.sin(ky * pos_np[:, 1] + dy)

    vf_field = np.clip(vf_field, 0.05, 0.95)
    alpha_field = 1.0 / (vf_field / alpha_f + (1.0 - vf_field) / alpha_m)

    # Build coefficient function for FEM assembly
    x, y = ng.x, ng.y
    vf_cf = ng.CoefficientFunction(vf0)
    for ai, kx, ky, dx, dy in coeff_params:
        vf_cf += ai * ng.sin(kx * x + dx) * ng.sin(ky * y + dy)
    alpha_cf = 1.0 / (vf_cf / alpha_f + (1 - vf_cf) / alpha_m)

    initial_condition = np.ones_like(pos_np[:, 0]) * T0

    gaussians = create_gaussian_initial_condition(
        pos=pos_np,
        num_gaussians=5,
        amplitude_range=(1, 10),
        sigma_fraction_range=(0.1, 0.2),
        seed=seed,
        centered=False,
        enforce_boundary_conditions=False,
    )
    initial_condition += gaussians

    temp_data, _ = graph_creator.create_graph(
        T_current=initial_condition,
        t_scalar=0.0,
        material_node_field=vf_field,
        neumann_values=neumann_vals,
        dirichlet_values=dirichlet_vals,
    )

    problem = MeshProblem(
        mesh=mesh,
        graph_data=temp_data,
        initial_condition=initial_condition,
        alpha=float(alpha_field.mean()),
        time_config=time_config,
        mesh_config=mesh_config,
        problem_id=0,
    )

    problem.alpha_coefficient = alpha_cf
    problem.material_fraction_field = vf_field
    problem.material_field = alpha_field
    problem.nonlinear_source_params = {
        "q0": q0,
        "t0": t0,
        "C": 8.0 / T0,
        "T0": T0,
    }

    problem.set_neumann_values(neumann_boundaries_dict)
    problem.set_dirichlet_values(dirichlet_boundaries_dict)
    problem.set_neumann_values_array(neumann_vals)
    problem.set_dirichlet_values_array(dirichlet_vals)

    # Project initial condition to FEM space to enforce BCs
    fes = ng.H1(mesh, order=1, dirichlet=problem.mesh_config.dirichlet_pipe)
    gfu = ng.GridFunction(fes)
    gfu_initial = ng.GridFunction(fes)

    gfu_initial.vec.FV().NumPy()[:] = problem.initial_condition
    boundary_cf = mesh.BoundaryCF(problem.boundary_values, default=0)
    gfu.Set(boundary_cf, definedon=mesh.Boundaries(problem.mesh_config.dirichlet_pipe))

    free_dofs = fes.FreeDofs()
    for dof in range(fes.ndof):
        if free_dofs[dof]:
            gfu.vec[dof] = gfu_initial.vec[dof]

    problem.initial_condition = gfu.vec.FV().NumPy()

    print(
        f"Nonlinear rectangular problem created with {problem.n_nodes} nodes, maxh={maxh}, seed={seed}"
    )

    return problem, time_config