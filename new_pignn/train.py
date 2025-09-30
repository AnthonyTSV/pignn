import numpy as np
import torch
from trainer import PIGNNTrainer
from graph_creator import GraphCreator
from containers import TimeConfig, TrainingConfig, MeshProblem, MeshConfig
from fem import FEMSolver
from mesh_utils import create_rectangular_mesh, build_graph_from_mesh, create_gaussian_initial_condition

def create_training_problem(problem_id=0, time_config: TimeConfig = TimeConfig(), width=1.0, height=1.0, maxh=0.1, alpha=1.0):
    """Create a single rectangular training problem with Gaussian initial condition."""
    
    # Create rectangular mesh
    mesh = create_rectangular_mesh(
        width=width, 
        height=height, 
        maxh=maxh,
    )

    dir_names = ["left", "right", "top", "bottom"]
    neu_names = []
    n_nodes = len(list(mesh.ngmesh.Points()))
    Tn = np.random.randn(n_nodes).astype(np.float64) * 0.01
    
    # Build graph data from mesh
    t_n = 0.0

    # Build graph
    graph_creator = GraphCreator(
        mesh=mesh,
        n_neighbors=2,
        dirichlet_names=dir_names,
        neumann_names=neu_names,
        connectivity_method="fem"
    )
    temp_data, _ = graph_creator.create_graph()

    # Create mesh configuration
    mesh_config = MeshConfig(
        maxh=maxh,
        order=1,
        dim=2,
        dirichlet_boundaries=["left", "right", "top", "bottom"],
        mesh_type="rectangle"
    )
    
    # Create Gaussian initial condition that satisfies boundary conditions
    initial_condition = create_gaussian_initial_condition(
        pos=temp_data['pos'],
        num_gaussians=1,
        amplitude_range=(0.5, 1.0),
        sigma_fraction_range=(0.1, 0.2),
        seed=42,  # Different seed for each problem
        centered=True,
        enforce_boundary_conditions=True
    )

    graph_data, aux_data = graph_creator.create_graph(T_current=initial_condition, t_scalar=0.0)
    # graph_creator.visualize_graph(graph_data, aux_data)
    # Create the problem
    problem = MeshProblem(
        mesh=mesh,
        graph_data=graph_data,
        initial_condition=initial_condition,
        alpha=alpha,
        time_config=time_config,
        mesh_config=mesh_config,
        problem_id=problem_id
    )
    
    # Set boundary conditions (homogeneous Dirichlet)
    problem.boundary_values = {
        "left": 0.0, 
        "right": 0.0, 
        "top": 0.0, 
        "bottom": 0.0
    }
    
    # No source function (homogeneous heat equation)
    problem.source_function = None
    
    return problem

def e_l2_norm(T_pred: np.ndarray, T_true: np.ndarray, time_steps: np.ndarray) -> float:
    errors = []
    for idx, time_step in enumerate(time_steps):
        error = np.linalg.norm(T_pred[idx] - T_true[idx]) / np.linalg.norm(T_true[idx])
        errors.append(error)
    return errors
    

if __name__ == "__main__":
    print("Creating simple PI-GNN training script...")
    
    time_config = TimeConfig(
        dt=0.5,
        t_final=1.0
    )

    # Create training problems
    print("Creating training problem...")
    training_problem = create_training_problem(
        problem_id=0,
        time_config=time_config,
        width=1.0, 
        height=1.0, 
        maxh=0.3,  # Coarser mesh for faster training
        alpha=1.0
    )
    training_problems = [training_problem]
    
    # Create validation problem (slightly different mesh size)
    print("Creating validation problem...")
    validation_problem = create_training_problem(
        problem_id=1, 
        time_config=time_config,
        width=1.0, 
        height=1.0, 
        maxh=0.3,  # Slightly finer mesh
        alpha=1.0
    )
    validation_problems = [validation_problem]
    
    # Print problem statistics
    print(f"\nTraining problem statistics:")
    print(f"  Nodes: {training_problem.n_nodes}")
    print(f"  Edges: {training_problem.n_edges}")
    
    print(f"\nValidation problem statistics:")
    print(f"  Nodes: {validation_problem.n_nodes}")
    print(f"  Edges: {validation_problem.n_edges}")
    
    # Configure training
    config = TrainingConfig(
        time_config=time_config,
        epochs=500,
        lr=1e-3,
        weight_decay=0.0,
        batch_size=1,  # Single problem per batch
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_interval=1,
        save_interval=10
    )
    
    print(f"\nTraining configuration:")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Time steps: {config.time_config.num_steps}")
    
    # Create and start training
    print(f"\nInitializing trainer...")
    trainer = PIGNNTrainer(
        training_problems=training_problems,
        validation_problems=validation_problems,
        config=config
    )
    
    print("Starting training...")
    trainer.train()

    # evaluate on validation problem
    print("Evaluating on validation problem...")
    fem_solver = FEMSolver(validation_problem.mesh, order=1, problem=validation_problem)
    transient_solution = fem_solver.solve_transient_problem(validation_problem)
    t_steps = time_config.time_steps_export
    T_pred = trainer.rollout(validation_problem)
    T_true = transient_solution
    # export to vtk
    fem_solver.export_to_vtk(T_true, T_pred, t_steps, filename="results/vtk/validation_results.vtk")

    l2_errors = e_l2_norm(T_pred, T_true, t_steps)
    print(f"L2 errors: {l2_errors}")
