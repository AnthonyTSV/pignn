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
from logger import TrainingLogger
from meshgraphnet import MeshGraphNet
from fem import FEMSolver
from mesh_utils import create_dirichlet_values, create_rectangular_mesh, create_free_node_subgraph, create_gaussian_initial_condition
from graph_creator import GraphCreator
from containers import TimeConfig, MeshConfig, MeshProblem
from torch_geometric.data import Data
from train_problems import (
    create_test_problem,
    generate_multiple_problems,
    create_lshaped_problem,
    create_nonlinear_rectangular_problem,
)

class PIMGNTrainer:
    """Trainer for Physics-Informed MeshGraphNet (PIMGN)."""
    
    def __init__(self, problems: List[MeshProblem], config: dict):
        self.problems = problems
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Time bundling configuration
        self.time_window = config.get('time_window', 20)
        
        # Create FEM solvers for physics-informed loss computation for all problems
        self.all_fem_solvers: List[FEMSolver] = []
        for problem in problems:
            fem_solver = FEMSolver(problem.mesh, problem=problem)
            self.all_fem_solvers.append(fem_solver)
        
        # Prepare sample data to determine input/output dimensions using first problem
        first_problem = problems[0]
        graph_creator = GraphCreator(
            mesh=first_problem.mesh,
            n_neighbors=2,
            dirichlet_names=first_problem.mesh_config.dirichlet_boundaries,
            neumann_names=getattr(first_problem.mesh_config, 'neumann_boundaries', []),
            connectivity_method="fem"
        )
        
        # Create sample graph to get dimensions
        material_field = getattr(first_problem, 'material_field', None)
        neumann_vals = getattr(first_problem, 'neumann_values_array', None)
        dirichlet_vals = getattr(first_problem, 'dirichlet_values_array', None)
        sample_data, aux = graph_creator.create_graph(
            T_current=first_problem.initial_condition,
            t_scalar=0.0,
            material_node_field=material_field,
            neumann_values=neumann_vals,
            dirichlet_values=dirichlet_vals
        )

        free_node_data, mapping, new_aux = graph_creator.create_free_node_subgraph(
            full_graph=sample_data, aux=aux
        )
        
        input_dim_node = free_node_data.x.shape[1]
        input_dim_edge = free_node_data.edge_attr.shape[1]
        output_dim = self.time_window  # Predict multiple time steps
        
        print(f"Input dimensions - Node: {input_dim_node}, Edge: {input_dim_edge}, Output: {output_dim}")
        print(f"Time bundling window: {self.time_window}")
        print(f"Training on {len(problems)} problems")
        
        # Create MeshGraphNet model with temporal bundling
        self.model = MeshGraphNet(
            input_dim_node=input_dim_node,
            input_dim_edge=input_dim_edge,
            hidden_dim=128,
            output_dim=output_dim,
            num_layers=12,
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)
        
        # Training history
        self.losses = []
        self.val_losses = []

        # Initialize logger
        save_dir = config.get("save_dir", "results")
        log_filename = config.get("log_filename", "training_log.json")
        save_interval = config.get("save_interval", None)
        save_epoch_interval = config.get("save_epoch_interval", None)
        
        self.logger = TrainingLogger(save_dir=save_dir, filename=log_filename, save_interval=save_interval, save_epoch_interval=save_epoch_interval)
        self.logger.log_config(config)
        self.logger.set_device(self.device)
        self.logger.log_problems(problems)

        # Generate ground truth for validation
        if config.get('generate_ground_truth_for_validation', False):
            print("Generating ground truth for validation...")
            self.all_ground_truth = []
            for i, problem in enumerate(problems):
                print(f"Solving problem {i+1}/{len(problems)} for validation...")
                ground_truth = self.all_fem_solvers[i].solve_transient_problem(problem)
                self.all_ground_truth.append(ground_truth)
        else:
            self.all_ground_truth = None
    
    
    def compute_physics_informed_loss(self, predictions_bundled_free, t_current, dt, problem_idx, aux=None, node_mapping=None, start_time: float = 0.0):
        """
        Compute FEM loss following the paper's methodology.
        
        Loss = MSE over (N_TB x N_test_functions) errors
        """
        problem: MeshProblem = self.problems[problem_idx]
        fem_solver: FEMSolver = self.all_fem_solvers[problem_idx]
        
        t_current_tensor = t_current.to(dtype=torch.float64, device=self.device) if isinstance(t_current, torch.Tensor) else torch.tensor(t_current, dtype=torch.float64, device=self.device)
        
        free_to_original = node_mapping["free_to_original"]
        all_residuals = []  # Store all element-wise errors
        
        # Compute residual for each time step in bundle
        for t_idx in range(self.time_window):
            # Reconstruct full prediction
            prediction_free_t = predictions_bundled_free[:, t_idx]
            prediction_full_t = torch.zeros(problem.n_nodes, dtype=torch.float64, device=self.device)
            prediction_full_t[free_to_original] = prediction_free_t.to(dtype=torch.float64)
            
            # Get previous state
            if t_idx == 0:
                t_prev = t_current_tensor
            else:
                prev_prediction_free = predictions_bundled_free[:, t_idx - 1]
                t_prev = torch.zeros(problem.n_nodes, dtype=torch.float64, device=self.device)
                t_prev[free_to_original] = prev_prediction_free.to(dtype=torch.float64)
            
            # Compute FEM residual (element-wise errors)
            time_for_step = start_time + t_idx * dt
            residual = fem_solver.compute_residual(
                t_pred_next=prediction_full_t,
                t_prev=t_prev,
                problem=problem,
                time_scalar=float(time_for_step)
            )
            
            all_residuals.append(residual)
        
        # Stack all residuals: [N_TB, N_free_dofs]
        all_residuals_stacked = torch.stack(all_residuals)  # [N_TB, N_free_dofs]
        
        # Compute MSE over all (N_TB × N_free_dofs) values
        # Note: N_free_dofs corresponds to N_test_functions in the paper
        fem_loss = torch.mean(all_residuals_stacked)

        self.last_residuals = all_residuals_stacked.detach().cpu().numpy()
        
        return fem_loss
    
    def train_step(self, t_current, current_time, problem_idx):
        """
        Compute physics loss for temporal bundling WITHOUT backpropagation.
        Returns the loss tensor (with gradients) and bundled predictions.
        """
        problem: MeshProblem = self.problems[problem_idx]
        
        # Create graph creator for this specific problem
        graph_creator = GraphCreator(
            mesh=problem.mesh,
            n_neighbors=2,
            dirichlet_names=problem.mesh_config.dirichlet_boundaries,
            neumann_names=getattr(problem.mesh_config, 'neumann_boundaries', []),
            connectivity_method="fem"
        )
        
        # Get the Neumann values for this problem
        neumann_vals = getattr(problem, 'neumann_values_array', None)
        dirichlet_vals = getattr(problem, 'dirichlet_values_array', None)
        material_field = getattr(problem, 'material_field', None)
        
        self.optimizer.zero_grad()
        # Build full graph from current state
        data, aux = graph_creator.create_graph(
            T_current=t_current,
            t_scalar=current_time,
            material_node_field=material_field,
            neumann_values=neumann_vals,
            dirichlet_values=dirichlet_vals
        )
        
        # Create free node subgraph (only non-Dirichlet nodes)
        free_data, node_mapping, free_aux = graph_creator.create_free_node_subgraph(
            full_graph=data, aux=aux
        )
        free_data = free_data.to(self.device)

        # Forward pass - get bundled predictions for FREE nodes [N_free_nodes, time_window]
        predictions_bundled_free = self.model.forward(free_data)
        
        # Convert current state to tensor
        t_current_tensor = torch.tensor(t_current, dtype=torch.float32, device=self.device) if not torch.is_tensor(t_current) else t_current

        # Compute physics-informed loss for temporal bundling with free nodes
        dt = problem.time_config.dt
        physics_loss = self.compute_physics_informed_loss(
            predictions_bundled_free,  # FREE node predictions [N_free_nodes, time_window]
            t_current_tensor,
            dt,
            problem_idx,
            aux,
            node_mapping,
            start_time=float(current_time)
        )
        physics_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Convert bundled predictions to numpy for next iteration
        # Reconstruct full state from free node predictions
        predictions_bundled_np = predictions_bundled_free.detach().cpu().numpy()
        next_state_free = predictions_bundled_np[:, -1]  # Last time step prediction for free nodes
        
        # Reconstruct full state
        next_state_full = np.zeros(problem.n_nodes, dtype=np.float32)
        free_to_original = node_mapping["free_to_original"].cpu().numpy()
        next_state_full[free_to_original] = next_state_free
        if dirichlet_vals is not None:
            dirichlet_mask = aux["dirichlet_mask"].cpu().numpy()
            next_state_full[dirichlet_mask] = dirichlet_vals[dirichlet_mask]

        return physics_loss.item(), next_state_full, predictions_bundled_np

    def train(self, train_problems_indices, val_problems_indices=None):
        """Main training loop following paper's methodology with multiple problems."""
        print(f"Starting PIMGN training on {self.device}")
        print(f"Training on problems: {train_problems_indices}")
        if val_problems_indices:
            print(f"Validation on problems: {val_problems_indices}")
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            epoch_losses = []
            
            # Shuffle training problems for each epoch
            shuffled_train_indices = train_problems_indices.copy()
            np.random.shuffle(shuffled_train_indices)
            
            # Train on each problem in the training set
            for problem_idx in shuffled_train_indices:
                problem = self.problems[problem_idx]
                
                # Optimization loop from initial condition for this problem
                t_current = problem.initial_condition.copy()
                time_steps = problem.time_config.time_steps
                time_steps_batched = np.array_split(time_steps, len(time_steps) // self.time_window)
                
                problem_losses = []
                for batch_times in time_steps_batched:
                    current_time = batch_times[0]
                    physics_loss, t_next, _ = self.train_step(t_current, current_time, problem_idx)
                    problem_losses.append(physics_loss)
                    t_current = t_next  # Update current state for next batch
                
                # Average loss over all time steps for this problem
                if problem_losses:
                    avg_problem_loss = np.mean(problem_losses)
                    epoch_losses.append(avg_problem_loss)

            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            self.losses.append(avg_epoch_loss)
            
            # Validation (if validation problems are provided and ground truth is available)
            val_loss = None
            if val_problems_indices and self.all_ground_truth is not None:
                val_loss = self.validate(val_problems_indices)
                self.val_losses.append(val_loss)
            
            elapsed = time.time() - epoch_start
            self.logger.log_epoch(epoch, avg_epoch_loss, val_loss, elapsed)
            
            if epoch % 10 == 0:
                elapsed = time.time() - epoch_start
                val_str = f" | Val Loss: {val_loss:.3e}" if val_loss is not None else ""
                print(f"Epoch {epoch+1:4d} | Train Loss: {avg_epoch_loss:.3e}{val_str} | Time: {elapsed:.2f}s")
            
            self.scheduler.step()

        print("Physics-Informed MeshGraphNet training with multiple problems completed!")
    
    def validate(self, val_problems_indices):
        """Validate model on held-out problems using physics-informed loss."""
        if self.all_ground_truth is None:
            print("Warning: No ground truth available for validation. Skipping validation.")
            return None
            
        self.model.eval()
        total_loss = 0.0
        num_validations = 0
        
        with torch.no_grad():
            for problem_idx in val_problems_indices:
                problem: MeshProblem = self.problems[problem_idx]
                
                # Use a few time steps from the ground truth for validation
                ground_truth = self.all_ground_truth[problem_idx]
                validation_steps = min(5, len(ground_truth) - self.time_window)  # Validate on first few steps
                
                for step_idx in range(0, validation_steps, self.time_window):
                    t_current = ground_truth[step_idx]
                    current_time = problem.time_config.time_steps[step_idx]
                    
                    # Create graph creator for this specific problem
                    graph_creator = GraphCreator(
                        mesh=problem.mesh,
                        n_neighbors=2,
                        dirichlet_names=problem.mesh_config.dirichlet_boundaries,
                        neumann_names=[],
                        connectivity_method="fem"
                    )
                    
                    # Build graph and make prediction
                    data, aux = graph_creator.create_graph(
                        T_current=t_current,
                        t_scalar=current_time,
                        material_node_field=getattr(problem, 'material_field', None),
                        neumann_values=getattr(problem, 'neumann_values_array', None),
                        dirichlet_values=getattr(problem, 'dirichlet_values_array', None)
                    )
                    
                    free_data, node_mapping, free_aux = graph_creator.create_free_node_subgraph(
                        full_graph=data, aux=aux
                    )
                    free_data = free_data.to(self.device)
                    
                    predictions_bundled_free = self.model.forward(free_data)
                    t_current_tensor = torch.tensor(t_current, dtype=torch.float32, device=self.device)
                    
                    # Compute physics loss for validation
                    dt = problem.time_config.dt
                    physics_loss = self.compute_physics_informed_loss(
                        predictions_bundled_free,
                        t_current_tensor,
                        dt,
                        problem_idx,
                        aux,
                        node_mapping,
                        start_time=float(current_time)
                    )
                    
                    total_loss += physics_loss.item()
                    num_validations += 1
        
        self.model.train()
        return total_loss / num_validations if num_validations > 0 else 0.0
    
    def rollout(self, problem_idx=0, n_steps=None):
        """Perform rollout prediction with temporal bundling for a specific problem."""
        self.model.eval()
        
        problem = self.problems[problem_idx]

        if n_steps is None:
            n_steps = len(problem.time_config.time_steps)

        time_steps = problem.time_config.time_steps
        time_steps_bundled = np.array_split(time_steps, len(time_steps) // self.time_window)

        # Start with initial condition
        T_current = problem.initial_condition.copy()
        predictions = [T_current]

        graph_creator = GraphCreator(
            mesh=problem.mesh,
            n_neighbors=2,
            dirichlet_names=problem.mesh_config.dirichlet_boundaries,
            neumann_names=problem.mesh_config.neumann_boundaries,
            connectivity_method="fem",
        )

        # Get the Neumann values for this problem
        neumann_vals = getattr(problem, 'neumann_values_array', None)
        dirichlet_vals = getattr(problem, 'dirichlet_values_array', None)
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
                    dirichlet_values=dirichlet_vals
                )
                free_graph, node_mapping, free_aux = (
                    graph_creator.create_free_node_subgraph(data, aux)
                )
                free_data = free_graph.to(self.device)

                # Predict next n time steps
                predictions_bundled = self.model.forward(free_data)
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
    
    def evaluate_with_ground_truth(self, problem_indices=None):
        """
        Evaluate the trained model against ground truth FEM solution.
        
        This generates ground truth data and compares with PIMGN predictions.
        """
        if problem_indices is None:
            problem_indices = range(len(self.problems))
            
        print("Evaluating PIMGN on multiple problems...")
        
        all_errors = []
        all_predictions = []
        all_ground_truth = []
        
        for i, problem_idx in enumerate(problem_indices):
            print(f"Evaluating problem {problem_idx + 1}...")
            
            problem = self.problems[problem_idx]
            fem_solver = self.all_fem_solvers[problem_idx]
            
            # Generate ground truth using FEM
            ground_truth = fem_solver.solve_transient_problem(problem)
            
            # Get predictions from trained model
            predictions = self.rollout(problem_idx=problem_idx)
            
            # Compute errors
            errors = []
            for j, (pred, true) in enumerate(zip(predictions, ground_truth)):
                if len(pred) == len(true):
                    l2_error = np.linalg.norm(pred - true) / np.linalg.norm(true)
                    errors.append(l2_error)
                else:
                    print(f"Warning: Size mismatch at step {j}: pred={len(pred)}, true={len(true)}")
            
            if errors:
                print(f"Problem {problem_idx + 1} - Average L2 error: {np.mean(errors):.6f}, Final L2 error: {errors[-1]:.6f}")
                all_errors.append(errors)
                all_predictions.append(predictions)
                all_ground_truth.append(ground_truth)
            else:
                print(f"Problem {problem_idx + 1} - No valid errors computed")
        
        # Overall statistics
        if all_errors:
            avg_errors = np.mean([np.mean(errors) for errors in all_errors])
            final_errors = np.mean([errors[-1] for errors in all_errors])
            print(f"\nOverall - Average L2 error: {avg_errors:.6f}, Final L2 error: {final_errors:.6f}")
            
            self.logger.log_evaluation(all_errors, "l2_errors_per_problem")
            self.logger.log_evaluation(avg_errors, "mean_l2_error")
            self.logger.log_evaluation(final_errors, "mean_final_l2_error")
        
        return all_predictions, all_ground_truth, all_errors
    
    def evaluate(self, problem_indices=None):
        """Evaluate the trained model on specified problems."""
        if problem_indices is None:
            problem_indices = range(len(self.problems))
            
        print("Evaluating model...")
        
        all_errors = []
        for prob_idx in problem_indices:
            print(f"Evaluating problem {prob_idx + 1}...")
            
            # Get predictions for this problem
            predictions = self.rollout(problem_idx=prob_idx)
            
            # If we have ground truth, compute errors
            if self.all_ground_truth is not None:
                ground_truth = self.all_ground_truth[prob_idx]
                
                # Compute errors
                errors = []
                for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
                    l2_error = np.linalg.norm(pred - true) / np.linalg.norm(true)
                    errors.append(l2_error)
                
                all_errors.append(errors)
                print(f"Problem {prob_idx + 1} - Average L2 error: {np.mean(errors):.6f}, Final L2 error: {errors[-1]:.6f}")
            else:
                print(f"Problem {prob_idx + 1} - Rollout completed (no ground truth for comparison)")

        # Overall statistics
        if all_errors:
            avg_errors = np.mean([np.mean(errors) for errors in all_errors])
            final_errors = np.mean([errors[-1] for errors in all_errors])
            print(f"\nOverall - Average L2 error: {avg_errors:.6f}, Final L2 error: {final_errors:.6f}")

            self.logger.log_evaluation(all_errors, "l2_errors_per_problem")
            self.logger.log_evaluation(avg_errors, "mean_l2_error")
            self.logger.log_evaluation(final_errors, "mean_final_l2_error")

        return all_errors

    def save_logs(self, filename="training_log.json"):
        self.logger.save(filename)

def plot_results(errors, losses, val_losses, last_residuals, pos_data, save_path="results"):
    """Plot training results."""
    Path(save_path).mkdir(exist_ok=True)

    # Plot results
    plt.figure(figsize=(8, 6))

    # Training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(losses, label='Training')
    if val_losses:
        plt.plot(val_losses, label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PIMGN Training/Validation Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)

    # L2 error over time for first problem (if errors exist)
    plt.subplot(2, 2, 2)
    if errors and len(errors) > 0:
        plt.plot(errors[0])
        plt.xlabel("Time Step")
        plt.ylabel("L2 Error")
        plt.title("L2 Error over Time (Problem 1)")
        plt.yscale("log")
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No error data available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=plt.gca().transAxes)
        plt.title("L2 Error over Time")

    # Average L2 error across all test problems (if errors exist)
    plt.subplot(2, 2, 3)
    if errors and len(errors) > 0:
        avg_errors = [np.mean(error_list) for error_list in errors]
        plt.bar(range(1, len(avg_errors) + 1), avg_errors)
        plt.xlabel("Problem Index")
        plt.ylabel("Average L2 Error")
        plt.title("Average L2 Error per Problem")
        plt.yscale("log")
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No error data available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=plt.gca().transAxes)
        plt.title("Average L2 Error per Problem")

    # Last residuals heatmap
    plt.subplot(2, 2, 4)
    if last_residuals is not None and last_residuals.size > 0:
        from scipy.interpolate import griddata
        
        # last_residuals has shape [N_TB, N_free_dofs]
        # We need to get the positions of the free nodes only
        # For now, let's take the last time step residuals and use absolute values
        time_step_idx = min(1, last_residuals.shape[0] - 1)  # Use second time step if available, else first
        residuals_to_plot = np.abs(last_residuals[time_step_idx])  # [N_free_dofs]
        
        # For this to work properly, we would need to know which nodes are free
        # Since we don't have that mapping here, let's check if dimensions match
        if len(residuals_to_plot) == len(pos_data):
            # Dimensions match - use all positions
            pos_to_use = pos_data
            residuals_final = residuals_to_plot
        elif len(residuals_to_plot) < len(pos_data):
            # More positions than residuals - assume first N positions are free nodes
            pos_to_use = pos_data[:len(residuals_to_plot)]
            residuals_final = residuals_to_plot
        else:
            # More residuals than positions - shouldn't happen, but handle gracefully
            print(f"Warning: Residuals shape {residuals_to_plot.shape} doesn't match pos_data shape {pos_data.shape}")
            pos_to_use = pos_data
            residuals_final = residuals_to_plot[:len(pos_data)]
        
        x_min, x_max = pos_to_use[:, 0].min(), pos_to_use[:, 0].max()
        y_min, y_max = pos_to_use[:, 1].min(), pos_to_use[:, 1].max()
        
        # Add small padding to avoid edge effects
        padding = 0.05
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range
        
        # Create grid
        grid_resolution = 100
        xi = np.linspace(x_min, x_max, grid_resolution)
        yi = np.linspace(y_min, y_max, grid_resolution)
        XI, YI = np.meshgrid(xi, yi)
        
        try:
            ZI = griddata((pos_to_use[:, 0], pos_to_use[:, 1]), residuals_final, 
                            (XI, YI), method='cubic', fill_value=0)
            
            # Create heatmap
            im = plt.imshow(ZI, extent=[x_min, x_max, y_min, y_max], 
                            origin='lower', cmap='viridis', aspect='equal')
            plt.colorbar(im, label='|Residual|')
            
            # Overlay scatter points to show actual node locations
            plt.scatter(pos_to_use[:, 0], pos_to_use[:, 1], 
                        c='white', s=2, alpha=0.3, edgecolors='black', linewidths=0.1)
            
            plt.title(f"FEM Residuals (t_step={time_step_idx})")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.axis('equal')
            
        except Exception as e:
            print(f"Error creating residual heatmap: {e}")
            plt.text(0.5, 0.5, f'Error plotting residuals:\n{str(e)}', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=plt.gca().transAxes)
            plt.title("FEM Residuals (Error)")
            
    elif last_residuals is not None:
        plt.text(0.5, 0.5, f'Empty residual data\nShape: {last_residuals.shape}', 
                horizontalalignment='center', verticalalignment='center', 
                transform=plt.gca().transAxes)
        plt.title("FEM Residuals (Empty)")
    else:
        plt.text(0.5, 0.5, 'No residual data available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=plt.gca().transAxes)
        plt.title("FEM Residuals (None)")

    plt.tight_layout()
    plt.savefig(f"{save_path}/pimgn_results.png", dpi=150)
    # plt.show()

def train_pimgn_on_multiple_problems():
    """Train Physics-Informed MeshGraphNet on multiple problems."""
    print("=" * 60)
    print("PHYSICS-INFORMED MESHGRAPHNET (PIMGN) TRAINING ON MULTIPLE PROBLEMS")
    print("=" * 60)
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    Path("results/physics_informed").mkdir(exist_ok=True)
    
    print("=" * 40)
    
    # Generate multiple problems
    n_problems = 5
    all_problems, time_config = generate_multiple_problems(n_problems=n_problems, seed=42)
    
    # Training configuration
    config = {
        'epochs': 10,
        'lr': 1e-3,
        'time_window': 20,
        'generate_ground_truth_for_validation': True,
        'save_interval': 300,
        'save_epoch_interval': 100,
        'log_filename': "pimgn_multiple_problems_log.json"
    }
    
    print(f"Training configuration:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Time window: {config['time_window']}")
    print(f"  Time steps: {time_config.num_steps}")
    print(f"  dt: {time_config.dt}")
    print(f"  Physics-informed loss: FEM residual with temporal bundling")
    
    # Create PIMGN trainer
    trainer = PIMGNTrainer(all_problems, config)
    
    # Train model with physics-informed loss
    print("\nStarting physics-informed training...")
    train_indices = list(range(n_problems - 1))  # Last for validation
    val_indices = list(range(n_problems - 1, n_problems))
    trainer.train(train_problems_indices=train_indices, val_problems_indices=val_indices)
    
    # Evaluate model
    print("\nEvaluating trained PIMGN...")
    last_residuals = trainer.last_residuals
    pos_data = trainer.problems[0].graph_data.pos.numpy()
    predictions, ground_truth, errors = trainer.evaluate_with_ground_truth(problem_indices=val_indices)
    
    # Plot results with ground truth comparison
    plot_results(errors, trainer.losses, trainer.val_losses, last_residuals, pos_data, save_path="results/physics_informed")
    trainer.save_logs()
    min_length = min(len(ground_truth[0]), len(predictions[0]), len(time_config.time_steps_export))
    trainer.all_fem_solvers[-1].export_to_vtk(
        ground_truth[0][:min_length], 
        predictions[0][:min_length], 
        time_config.time_steps_export[:min_length], 
        filename="results/physics_informed/pimgn_single_comparison.vtk",
        material_field=getattr(trainer.problems[-1], 'material_field', None)
    )

    print("maxh for last problem:", trainer.problems[-1].mesh_config.maxh)

    model_path = "results/physics_informed/pimgn_trained_model.pth"
    torch.save(trainer.model.state_dict(), model_path)
    print(f"Trained model saved to: {model_path}")

def _run_single_problem_experiment(problem, time_config, config, experiment_name: str):
    print("=" * 60)
    print(f"PIMGN TEST - {experiment_name.upper()}")
    print("=" * 60)

    Path("results").mkdir(exist_ok=True)
    Path("results/physics_informed").mkdir(exist_ok=True)

    print("=" * 40)
    print("Training configuration:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Time window: {config['time_window']}")
    print(f"  Time steps: {time_config.num_steps}")
    print(f"  dt: {time_config.dt}")
    print(f"  Physics-informed loss: FEM residual with temporal bundling")

    config['log_filename'] = f"pimgn_{experiment_name.replace(' ', '_').lower()}_log.json"
    config['save_interval'] = 300
    config['save_epoch_interval'] = 100

    trainer = PIMGNTrainer([problem], config)

    print("\nStarting physics-informed training...")
    trainer.train(train_problems_indices=[0])

    print("\nEvaluating trained PIMGN...")
    last_residuals = trainer.last_residuals
    pos_data = trainer.problems[0].graph_data.pos.numpy()
    try:
        predictions, ground_truth, errors = trainer.evaluate_with_ground_truth(problem_indices=[0])
        plot_results(errors, trainer.losses, [], last_residuals, pos_data, save_path="results/physics_informed")

        print("Exporting results...")
        min_length = min(len(ground_truth[0]), len(predictions[0]), len(time_config.time_steps_export))
        trainer.all_fem_solvers[0].export_to_vtk(
            ground_truth[0][:min_length],
            predictions[0][:min_length],
            time_config.time_steps_export[:min_length],
            filename="results/physics_informed/pimgn_single_comparison.vtk",
            material_field=getattr(trainer.problems[0], 'material_field', None)
        )
    except Exception as e:
        print(f"Ground truth evaluation failed: {e}")

    print("Physics-Informed MeshGraphNet test completed!")
    print("Results saved to: results/physics_informed/")
    trainer.save_logs()

    model_path = "results/physics_informed/pimgn_trained_model.pth"
    torch.save(trainer.model.state_dict(), model_path)
    print(f"Trained model saved to: {model_path}")

def _run_multiple_problem_experiment(problems, time_config, config, experiment_name: str):
    print("=" * 60)
    print(f"PIMGN TEST - {experiment_name.upper()}")
    print("=" * 60)

    Path("results").mkdir(exist_ok=True)
    Path("results/physics_informed").mkdir(exist_ok=True)

    print("=" * 40)
    print("Training configuration:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Time window: {config['time_window']}")
    print(f"  Time steps: {time_config.num_steps}")
    print(f"  dt: {time_config.dt}")
    print(f"  Physics-informed loss: FEM residual with temporal bundling")

    config['log_filename'] = f"pimgn_{experiment_name.replace(' ', '_').lower()}_log.json"
    config['save_interval'] = 300
    config['save_epoch_interval'] = 100

    trainer = PIMGNTrainer(problems, config)

    print("\nStarting physics-informed training...")
    train_indices = list(range(len(problems) - 1))  # Last for validation
    val_indices = list(range(len(problems) - 1, len(problems)))
    trainer.train(train_problems_indices=train_indices, val_problems_indices=val_indices)

    print("\nEvaluating trained PIMGN...")
    last_residuals = trainer.last_residuals
    pos_data = trainer.problems[0].graph_data.pos.numpy()
    predictions, ground_truth, errors = trainer.evaluate_with_ground_truth(problem_indices=val_indices)

    plot_results(errors, trainer.losses, trainer.val_losses, last_residuals, pos_data, save_path="results/physics_informed")

    print("Physics-Informed MeshGraphNet test completed!")
    print("Results saved to: results/physics_informed/")
    trainer.save_logs()

    min_length = min(len(ground_truth[0]), len(predictions[0]), len(time_config.time_steps_export))
    trainer.all_fem_solvers[0].export_to_vtk(
        ground_truth[0][:min_length],
        predictions[0][:min_length],
        time_config.time_steps_export[:min_length],
        filename="results/physics_informed/pimgn_single_comparison.vtk",
        material_field=getattr(trainer.problems[0], 'material_field', None)
    )

    model_path = "results/physics_informed/pimgn_trained_model.pth"
    torch.save(trainer.model.state_dict(), model_path)
    print(f"Trained model saved to: {model_path}")

def train_pimgn_on_single_problem():
    problem, time_config = create_test_problem(maxh=0.2, alpha=3)
    config = {
        'epochs': 500,
        'lr': 1e-3,
        'time_window': 20,
        'generate_ground_truth_for_validation': False,
    }
    _run_single_problem_experiment(problem, time_config, config, "Test problem")

def train_test_multiple_problems():
    problems = []
    time_config = None
    for i in range(3):
        problem, time_config = create_test_problem(maxh=0.1, alpha=np.random.uniform(0.1, 5.0))
        problems.append(problem)
    config = {
        'epochs': 100,
        'lr': 1e-3,
        'time_window': 20,
        'generate_ground_truth_for_validation': False,
    }
    _run_multiple_problem_experiment(problems, time_config, config, "Multiple test problems")

def train_pimgn_on_nonlinear_rectangular_problem():
    problems = []
    time_config = None
    for _ in range(5):
        problem, time_config = create_nonlinear_rectangular_problem(maxh=0.1, seed=np.random.randint(0, 10000))
        problems.append(problem)
    config = {
        'epochs': 500,
        'lr': 1e-3,
        'time_window': 20,
        'generate_ground_truth_for_validation': False,
    }
    _run_multiple_problem_experiment(problems, time_config, config, "Nonlinear rectangular heating")

def main():
    """Main function to run Physics-Informed MeshGraphNet training and evaluation."""
    # Uncomment one of the following lines to run the desired test
    train_pimgn_on_single_problem()
    # train_test_multiple_problems()
    # train_pimgn_on_multiple_problems()
    # train_pimgn_on_nonlinear_rectangular_problem()

if __name__ == "__main__":
    main()