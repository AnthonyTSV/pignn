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
from train_problems import create_test_problem, create_industrial_heating_problem
from test_pimgn import PIMGNTrainer

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
    robin_vals = getattr(problem, 'robin_values_array', None)
    
    # Create sample graph to get dimensions
    sample_data, aux = graph_creator.create_graph(
        T_current=problem.initial_condition,
        t_scalar=0.0,
        neumann_values=neumann_vals,
        dirichlet_values=dirichlet_vals,
        robin_values=robin_vals,
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
    # PyTorch 2.6 changed the default `weights_only` of torch.load from False -> True.
    # For inference, we *prefer* a safe weights-only load when possible.
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch versions don't support `weights_only`.
        state_dict = torch.load(model_path, map_location=device)
    except Exception as e:
        print(
            "Safe weights-only load failed; retrying with weights_only=False. "
            "Only do this if you trust the checkpoint file."
        )
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Model loaded successfully!")
    return model, device, time_window

def rollout(model, problem: MeshProblem, device, n_steps=None):
    """Perform rollout prediction with temporal bundling for a specific problem."""
    model.eval()
    time_window = 20

    time_steps = problem.time_config.time_steps
    time_steps_bundled = np.array_split(time_steps, len(time_steps) // time_window)

    # Start with initial condition
    T_current = problem.initial_condition.copy()
    predictions = [T_current]

    graph_creator = GraphCreator(
        mesh=problem.mesh,
        n_neighbors=2,
        dirichlet_names=problem.mesh_config.dirichlet_boundaries,
        neumann_names=problem.mesh_config.neumann_boundaries,
        robin_names=problem.mesh_config.robin_boundaries,
        connectivity_method="fem",
    )

    # Get the Neumann values for this problem
    neumann_vals = getattr(problem, 'neumann_values_array', None)
    dirichlet_vals = getattr(problem, 'dirichlet_values_array', None)
    robin_vals = getattr(problem, 'robin_values_array', None)
    material_field = getattr(problem, 'material_field', None)

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
                dirichlet_values=dirichlet_vals,
                robin_values=robin_vals,
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

            # Move forward to the next batch
            step_idx += 1

    return predictions[:n_steps]

if __name__ == "__main__":
    print("="*60)
    print("PHYSICS-INFORMED MESHGRAPHNET (PIMGN) INFERENCE")
    print("="*60)
    
    # Create test problem (same as used in training)
    problem, time_config = create_industrial_heating_problem(maxh=0.002)
    save_path = Path("results/inference/inference_test_problem_4_maxh_0.002/")
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    model_path = Path("results/physics_informed/verification_test_problem_4_maxh_0.002/pimgn_trained_model.pth")
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure you have trained the model first by running test_pimgn.py")
        exit(1)
    
    # Perform inference
    config = {
        'epochs': 2000,
        'lr': 1e-3,
        'time_window': 20,
        'generate_ground_truth_for_validation': False,
        'save_dir': str(save_path),
    }
    trainer = PIMGNTrainer([problem], config)
    # Load the trained model
    model, device, time_window = load_trained_model(model_path, problem)
    trainer.model = model
    trainer.device = device
    predictions, ground_truth, errors = trainer.evaluate_with_ground_truth(problem_indices=[0])

    # Save predictions and ground truth
    # last_residuals = trainer.last_residuals
    # trainer.logger.log_evaluation(last_residuals.tolist(), "residuals_per_time_step")
    pos_data = trainer.problems[0].graph_data.pos.numpy()
    try:
        predictions, ground_truth, errors = trainer.evaluate_with_ground_truth(problem_indices=[0])
        # plot_results(errors, trainer.losses, [], last_residuals, pos_data, save_path=save_path)

        print("Exporting results...")
        min_length = min(len(ground_truth[0]), len(predictions[0]), len(time_config.time_steps_export))
        trainer.all_fem_solvers[0].export_to_vtk(
            ground_truth[0][:min_length],
            predictions[0][:min_length],
            time_config.time_steps_export[:min_length],
            filename=f"{save_path}/vtk/result",
            material_field=getattr(trainer.problems[0], 'material_field', None)
        )
    except Exception as e:
        print(f"Ground truth evaluation failed: {e}")

    print("="*60)
    print("INFERENCE COMPLETED SUCCESSFULLY!")
    print("="*60)
    