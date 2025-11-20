import numpy as np
import torch
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
from mesh_utils import create_dirichlet_values, create_rectangular_mesh, create_free_node_subgraph, create_gaussian_initial_condition
from graph_creator import GraphCreator
from containers import TimeConfig, MeshConfig, MeshProblem
from torch_geometric.data import Data
from train_problems import create_test_problem, create_nonlinear_rectangular_problem

def load_trained_model(model_path, problem):
    """Load the trained PIMGN model with proper architecture."""
    print(f"Loading trained model from: {model_path}")
    
    # Create the same model architecture as used in training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get model dimensions from a sample graph
    graph_creator = GraphCreator(
        mesh=problem.mesh,
        n_neighbors=2,
        dirichlet_names=problem.mesh_config.dirichlet_boundaries,
        neumann_names=getattr(problem.mesh_config, 'neumann_boundaries', []),
        connectivity_method="fem"
    )
    
    # Get boundary values
    neumann_vals = getattr(problem, 'neumann_values_array', None)
    dirichlet_vals = getattr(problem, 'dirichlet_values_array', None)
    
    # Create sample graph to get dimensions
    sample_data, aux = graph_creator.create_graph(
        T_current=problem.initial_condition,
        t_scalar=0.0,
        neumann_values=neumann_vals,
        dirichlet_values=dirichlet_vals
    )
    
    free_node_data, mapping, new_aux = graph_creator.create_free_node_subgraph(
        full_graph=sample_data, aux=aux
    )
    
    input_dim_node = free_node_data.x.shape[1]
    input_dim_edge = free_node_data.edge_attr.shape[1]
    
    # Time window used in training
    time_window = 20  # This should match the training configuration
    
    print(f"Model architecture: Node dim: {input_dim_node}, Edge dim: {input_dim_edge}, Output dim: {time_window}")
    
    # Create model with same architecture as training
    model = MeshGraphNet(
        input_dim_node=input_dim_node,
        input_dim_edge=input_dim_edge,
        hidden_dim=128,
        output_dim=time_window,
        num_layers=12,
    ).to(device)
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Model loaded successfully!")
    return model, device, time_window

def perform_inference(model, problem, device, time_window, n_steps=None):
    """Perform inference (rollout) on the given problem."""
    print("Starting inference...")
    
    if n_steps is None:
        n_steps = len(problem.time_config.time_steps)
    
    time_steps = problem.time_config.time_steps
    time_steps_bundled = np.array_split(time_steps, len(time_steps) // time_window)
    
    # Start with initial condition
    T_current = problem.initial_condition.copy()
    predictions = [T_current]
    
    # Create graph creator
    graph_creator = GraphCreator(
        mesh=problem.mesh,
        n_neighbors=2,
        dirichlet_names=problem.mesh_config.dirichlet_boundaries,
        neumann_names=getattr(problem.mesh_config, 'neumann_boundaries', []),
        connectivity_method="fem",
    )
    
    # Get boundary values
    neumann_vals = getattr(problem, 'neumann_values_array', None)
    dirichlet_vals = getattr(problem, 'dirichlet_values_array', None)
    material_field = getattr(problem, 'material_fraction_field', None)
    
    print(f"Running inference for {n_steps} time steps with temporal bundling (window={time_window})")
    
    with torch.no_grad():   
        step_idx = 0
        for batch_idx, batch_times in enumerate(time_steps_bundled):
            starting_time_step = 0 if step_idx == 0 else batch_times[0]
            # Build graph
            data, aux = graph_creator.create_graph(
                T_current=T_current,
                t_scalar=batch_times[0],
                material_node_field=material_field,
                neumann_values=neumann_vals,
                dirichlet_values=dirichlet_vals
            )
            free_graph, node_mapping, free_aux = (
                graph_creator.create_free_node_subgraph(data, aux)
            )
            free_data = free_graph.to(device)

            # Predict next n time steps
            predictions_bundled = model.forward(free_data)
            # Shape: (n_free_nodes, time_window)

            # Extract predictions for each time step
            free_idx = node_mapping["free_to_original"].detach().cpu().numpy()
            dirichlet_mask = aux["dirichlet_mask"].cpu().numpy()
            # Add each predicted time step
            for time_idx, current_time in enumerate(batch_times):

                next_full = np.zeros(problem.n_nodes, dtype=np.float32)
                pred_t = (
                    predictions_bundled[:, time_idx].squeeze().detach().cpu().numpy()
                )
                next_full[free_idx] = pred_t
                next_full[dirichlet_mask] = dirichlet_vals[dirichlet_mask]

                predictions.append(next_full)

            # For next iteration, use the last predicted state
            if len(predictions) > 1:
                T_to_use = predictions[-1]
                # enforce Dirichlet BCs
                dirichlet_mask = aux["dirichlet_mask"].cpu().numpy()
                T_to_use[dirichlet_mask] = dirichlet_vals[dirichlet_mask]
                T_current = T_to_use

            # Move forward by time_window steps
            step_idx += time_window
    
    print(f"Inference completed! Generated {len(predictions)} time steps.")
    return predictions[:n_steps]

if __name__ == "__main__":
    print("="*60)
    print("PHYSICS-INFORMED MESHGRAPHNET (PIMGN) INFERENCE")
    print("="*60)
    
    # Create test problem (same as used in training)
    problem, time_config = create_test_problem(maxh=0.2, alpha=3)
    save_path = Path(__file__).parent.parent / "results" / "inference"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    model_path = Path("results/physics_informed/pimgn_trained_model.pth")
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure you have trained the model first by running test_pimgn.py")
        exit(1)
    
    # Load the trained model
    model, device, time_window = load_trained_model(model_path, problem)
    
    # Perform inference
    predictions = perform_inference(model, problem, device, time_window)
    
    fem_solver = FEMSolver(problem.mesh, problem=problem)
    ground_truth = fem_solver.solve_transient_problem(problem)
    min_length = min(len(predictions), len(time_config.time_steps_export))
    
    # Create a dummy ground truth for export (just use predictions)
    fem_solver.export_to_vtk(
        array_true=ground_truth[:min_length],
        array_pred=predictions[:min_length],
        time_steps=time_config.time_steps_export[:min_length],
        filename=f"{save_path}/inference_results"
    )
    print(f"VTK file saved: {save_path}/inference_results")

    print("="*60)
    print("INFERENCE COMPLETED SUCCESSFULLY!")
    print("="*60)
    