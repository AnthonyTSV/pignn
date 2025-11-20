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
from mesh_utils import (
    create_rectangular_mesh,
    build_graph_from_mesh,
    create_free_node_subgraph,
    create_gaussian_initial_condition,
    create_dirichlet_values,
)
from graph_creator import GraphCreator
from containers import TimeConfig, MeshConfig, MeshProblem
from torch_geometric.data import Data
from new_pignn.train_problems import create_test_problem, generate_multiple_problems


class DataDrivenMGNTrainer:
    """Trainer for data-driven MeshGraphNet baseline."""

    def __init__(self, problems: List[MeshProblem], config: dict):
        self.problems = problems
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.time_window = config.get("time_window", 5)

        # Generate ground truth data using FEM for all problems
        self.all_ground_truth = []
        self.all_fem_solvers: List[FEMSolver] = []
        
        print(f"Generating ground truth data for {len(problems)} problems...")
        for i, problem in enumerate(problems):
            print(f"Solving problem {i+1}/{len(problems)}...")
            fem_solver = FEMSolver(problem.mesh, problem=problem)
            ground_truth = fem_solver.solve_transient_problem(problem)
            self.all_ground_truth.append(ground_truth)
            self.all_fem_solvers.append(fem_solver)
            print(f"Problem {i+1}: Generated {len(ground_truth)} time steps")

        # Prepare training data from all problems
        self.training_data = self.prepare_training_data_all_problems()
        print(f"Total training data: {len(self.training_data)} time step pairs")

        if len(self.training_data) == 0:
            raise ValueError("No training data generated! Check time steps.")

        # Get dimensions from the first training sample
        sample_data = self.training_data[0]["graph_data"]
        input_dim_node = sample_data.x.shape[1]
        input_dim_edge = sample_data.edge_attr.shape[1]
        output_dim = self.time_window  # Predict multiple time steps

        print(
            f"Input dimensions - Node: {input_dim_node}, Edge: {input_dim_edge}, Output: {output_dim}"
        )

        self.model = MeshGraphNet(
            input_dim_node=input_dim_node,
            input_dim_edge=input_dim_edge,
            hidden_dim=128,
            output_dim=output_dim,
            num_layers=12,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)

        # Training history
        self.losses = []
        self.val_losses = []

    def create_dirichlet_values_for_problem(self, problem, graph_creator):
        """Create Dirichlet values array based on problem's boundary conditions."""
        # Create a temporary graph to get positions and auxiliary data
        temp_data, temp_aux = graph_creator.create_graph()
        
        # Define a custom function based on problem's Dirichlet boundary values
        def dirichlet_function(x, y):
            # Get the boundary values from the problem
            boundary_values = getattr(problem, 'boundary_values', {})
            
            # Determine which boundary the point (x, y) is on based on position
            # For a rectangular domain, check which boundary is closest
            pos_array = temp_data.pos.numpy()
            x_min, x_max = pos_array[:, 0].min(), pos_array[:, 0].max()
            y_min, y_max = pos_array[:, 1].min(), pos_array[:, 1].max()
            
            tolerance = 1e-6
            
            # Check boundaries in order of priority (left, right, bottom, top)
            if abs(x - x_min) < tolerance:  # Left boundary
                return boundary_values.get("left", 0.0)
            elif abs(x - x_max) < tolerance:  # Right boundary  
                return boundary_values.get("right", 0.0)
            elif abs(y - y_min) < tolerance:  # Bottom boundary
                return boundary_values.get("bottom", 0.0)
            elif abs(y - y_max) < tolerance:  # Top boundary
                return boundary_values.get("top", 0.0)
            else:
                # Interior point - should not have Dirichlet BC, but default to 0
                return 0.0
        
        # Create Dirichlet values using the custom function
        dirichlet_vals = create_dirichlet_values(
            pos=temp_data.pos,
            aux_data=temp_aux,
            dirichlet_names=problem.mesh_config.dirichlet_boundaries,
            temperature_function=dirichlet_function
        )
        
        return dirichlet_vals

    def prepare_training_data_all_problems(self):
        """Prepare training data for temporal bundling prediction from all problems."""
        all_training_data = []
        
        for prob_idx, (problem, ground_truth) in enumerate(zip(self.problems, self.all_ground_truth)):
            time_steps = self.config["time_config"].time_steps

            graph_creator = GraphCreator(
                mesh=problem.mesh,
                n_neighbors=2,
                dirichlet_names=problem.mesh_config.dirichlet_boundaries,
                neumann_names=problem.mesh_config.neumann_boundaries,
                connectivity_method="fem",
            )

            # Get the Neumann values for this problem
            neumann_vals = getattr(problem, 'neumann_values_array', None)

            
            # Get the Dirichlet values for this problem
            dirichlet_vals = getattr(problem, 'dirichlet_values_array', None)

            # Create training pairs for temporal bundling for this problem
            for idx in range(len(time_steps) - self.time_window):  # Need n future steps
                current_time = time_steps[idx]
                input_state = ground_truth[idx]

                # Collect next n time steps as targets
                future_states = []
                for future_idx in range(1, self.time_window + 1):
                    if idx + future_idx < len(ground_truth):
                        future_states.append(ground_truth[idx + future_idx])
                    else:
                        # Pad with last available state if needed
                        future_states.append(ground_truth[-1])

                # Stack future states: shape (n_nodes, time_window)
                bundled_target = np.stack(future_states, axis=1)

                # Build graph from current state
                data, aux = graph_creator.create_graph(
                    T_current=input_state, t_scalar=current_time, neumann_values=neumann_vals, dirichlet_values=dirichlet_vals
                )

                free_graph, node_mapping, free_aux = (
                    graph_creator.create_free_node_subgraph(data, aux)
                )
                free_idx = node_mapping["free_to_original"]

                # Extract free node targets: shape (n_free_nodes, time_window)
                free_target = bundled_target[free_idx.cpu().numpy()]

                all_training_data.append(
                    {
                        "graph_data": free_graph,
                        "target": free_target,  # Now shape (n_free_nodes, time_window)
                        "time": current_time,
                        "aux_full": aux,
                        "node_mapping": node_mapping,
                        "problem_id": prob_idx,
                    }
                )

        return all_training_data

    def prepare_training_data(self):
        """Prepare training data for temporal bundling prediction."""
        training_data = []
        time_steps = self.config["time_config"].time_steps

        graph_creator = GraphCreator(
            mesh=self.problem.mesh,
            n_neighbors=2,
            dirichlet_names=self.problem.mesh_config.dirichlet_boundaries,
            neumann_names=self.problem.mesh_config.neumann_boundaries,
            connectivity_method="fem",
        )

        # Get the Neumann values for this problem
        neumann_vals = getattr(self.problem, 'neumann_values_array', None)
        dirichlet_vals = getattr(self.problem, 'dirichlet_values_array', None)
    

        # Create training pairs for temporal bundling
        for idx in range(len(time_steps) - self.time_window):  # Need n future steps
            current_time = time_steps[idx]
            input_state = self.ground_truth[idx]

            # Collect next n time steps as targets
            future_states = []
            for future_idx in range(1, self.time_window + 1):
                if idx + future_idx < len(self.ground_truth):
                    future_states.append(self.ground_truth[idx + future_idx])
                else:
                    # Pad with last available state if needed
                    future_states.append(self.ground_truth[-1])

            # Stack future states: shape (n_nodes, time_window)
            bundled_target = np.stack(future_states, axis=1)

            # Build graph from current state
            data, aux = graph_creator.create_graph(
                T_current=input_state, t_scalar=current_time, neumann_values=neumann_vals, dirichlet_values=dirichlet_vals
            )

            free_graph, node_mapping, free_aux = (
                graph_creator.create_free_node_subgraph(data, aux)
            )
            free_idx = node_mapping["free_to_original"]

            # Extract free node targets: shape (n_free_nodes, time_window)
            free_target = bundled_target[free_idx.cpu().numpy()]

            training_data.append(
                {
                    "graph_data": free_graph,
                    "target": free_target,  # Now shape (n_free_nodes, time_window)
                    "time": current_time,
                    "aux_full": aux,
                    "node_mapping": node_mapping,
                }
            )

        return training_data

    def split_train_validation(self, train_problems_indices, val_problems_indices):
        """Split training data into train and validation sets based on problem indices."""
        train_data = []
        val_data = []
        
        for data_point in self.training_data:
            problem_id = data_point["problem_id"]
            if problem_id in train_problems_indices:
                train_data.append(data_point)
            elif problem_id in val_problems_indices:
                val_data.append(data_point)
        
        return train_data, val_data

    def validate(self, val_data):
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data_point in val_data:
                graph_data = data_point["graph_data"].to(self.device)
                target_tensor = torch.tensor(data_point["target"], dtype=torch.float32, device=self.device)
                
                predictions = self.model.forward(graph_data)
                loss = nn.MSELoss()(predictions, target_tensor)
                total_loss += loss.item()
        
        return total_loss / len(val_data)

    def train_step(self, graph_data, target):
        """Single training step with temporal bundling."""
        self.optimizer.zero_grad()

        # Move data to device
        graph_data = graph_data.to(self.device)
        # target shape: (n_free_nodes, time_window)
        target_tensor = torch.tensor(target, dtype=torch.float32, device=self.device)

        # predictions shape: (n_free_nodes, time_window)
        predictions = self.model.forward(graph_data)

        # Compute MSE loss across all time steps
        loss = nn.MSELoss()(predictions, target_tensor)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, train_problems_indices, val_problems_indices):
        """Main training loop with validation."""
        print(f"Starting data-driven MeshGraphNet training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        # Split data
        train_data, val_data = self.split_train_validation(train_problems_indices, val_problems_indices)
        print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

        self.model.train()

        for epoch in range(self.config["epochs"]):
            epoch_loss = 0.0
            epoch_start = time.time()

            # Shuffle training data
            np.random.shuffle(train_data)

            # Training
            self.model.train()
            for data_point in train_data:
                loss = self.train_step(
                    graph_data=data_point["graph_data"], target=data_point["target"]
                )
                epoch_loss += loss

            # Average loss over all training steps
            epoch_loss /= len(train_data)
            self.losses.append(epoch_loss)

            # Validation
            val_loss = self.validate(val_data)
            self.val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step()

            # Log progress
            if epoch % 10 == 0:
                elapsed = time.time() - epoch_start
                print(
                    f"Epoch {epoch:4d} | Train Loss: {epoch_loss:.3e} | Val Loss: {val_loss:.3e} | Time: {elapsed:.2f}s | LR: {self.scheduler.get_last_lr()[0]:.3e}"
                )

        print("Data-driven MeshGraphNet training completed!")

    def rollout(self, problem_idx=0, n_steps=None):
        """Perform rollout prediction with temporal bundling for a specific problem."""
        self.model.eval()
        
        problem = self.problems[problem_idx]
        ground_truth = self.all_ground_truth[problem_idx]

        if n_steps is None:
            n_steps = len(ground_truth)

        time_steps = self.config["time_config"].time_steps[:n_steps]

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
        
        # Get the Dirichlet values for this problem
        dirichlet_vals = getattr(problem, 'dirichlet_values_array', None)

        with torch.no_grad():
            step_idx = 0
            while step_idx < len(time_steps) - 1:
                current_time = time_steps[step_idx]

                # Build graph
                data, aux = graph_creator.create_graph(
                    T_current=T_current, t_scalar=current_time, neumann_values=neumann_vals, dirichlet_values=dirichlet_vals
                )
                free_graph, node_mapping, free_aux = (
                    graph_creator.create_free_node_subgraph(data, aux)
                )
                free_data = free_graph.to(self.device)

                # Predict next n time steps
                predictions_bundled = self.model.forward(free_data) # Shape: (n_free_nodes, time_window)

                # Extract predictions for each time step
                free_idx = node_mapping["free_to_original"].detach().cpu().numpy()

                # Add each predicted time step
                for t_idx in range(self.time_window):
                    if step_idx + 1 + t_idx >= n_steps:
                        break

                    next_full = np.zeros(problem.n_nodes, dtype=np.float32)
                    pred_t = (
                        predictions_bundled[:, t_idx].squeeze().detach().cpu().numpy()
                    )
                    next_full[free_idx] = pred_t
                    # Apply Dirichlet boundary conditions with actual values
                    dirichlet_mask = aux["dirichlet_mask"].cpu().numpy()
                    next_full[dirichlet_mask] = dirichlet_vals[dirichlet_mask]

                    predictions.append(next_full)

                # For next iteration, use the last predicted state
                # Or use the first predicted state for overlapping windows
                if len(predictions) > 1:
                    T_to_use = predictions[-1]
                    # enforce Dirichlet BCs
                    dirichlet_mask = aux["dirichlet_mask"].cpu().numpy()
                    T_to_use[dirichlet_mask] = dirichlet_vals[dirichlet_mask]
                    T_current = T_to_use

                # Move forward by time_window steps (non-overlapping)
                # Or by 1 step for overlapping windows
                step_idx += self.time_window  # Non-overlapping bundling
                # step_idx += 1  # Uncomment for overlapping bundling

        return predictions[:n_steps]

    def evaluate(self, problem_indices=None):
        """Evaluate the trained model on specified problems."""
        print("Evaluating model...")
        
        if problem_indices is None:
            problem_indices = range(len(self.problems))

        all_errors = []
        for prob_idx in problem_indices:
            print(f"Evaluating problem {prob_idx + 1}...")
            
            # Get predictions for this problem
            predictions = self.rollout(problem_idx=prob_idx)
            ground_truth = self.all_ground_truth[prob_idx]

            # Compute errors
            errors = []
            for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
                l2_error = np.linalg.norm(pred - true) / np.linalg.norm(true)
                errors.append(l2_error)
            
            all_errors.append(errors)
            print(f"Problem {prob_idx + 1} - Average L2 error: {np.mean(errors):.6f}, Final L2 error: {errors[-1]:.6f}")

        # Overall statistics
        avg_errors = np.mean([np.mean(errors) for errors in all_errors])
        final_errors = np.mean([errors[-1] for errors in all_errors])
        print(f"\nOverall - Average L2 error: {avg_errors:.6f}, Final L2 error: {final_errors:.6f}")

        return all_errors


def plot_results(predictions, ground_truth, errors, losses, val_losses, save_path="results"):
    """Plot training results."""
    Path(save_path).mkdir(exist_ok=True)

    # Plot results
    plt.figure(figsize=(20, 4))

    # Training and validation loss
    plt.subplot(1, 4, 1)
    plt.plot(losses, label='Training')
    plt.plot(val_losses, label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MeshGraphNet Training/Validation Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)

    # L2 error over time for first problem
    plt.subplot(1, 4, 2)
    plt.plot(errors[0])
    plt.xlabel("Time Step")
    plt.ylabel("L2 Error")
    plt.title("L2 Error over Time (Problem 1)")
    plt.yscale("log")
    plt.grid(True)

    # Average L2 error across all test problems
    plt.subplot(1, 4, 3)
    avg_errors = [np.mean(error_list) for error_list in errors]
    plt.bar(range(1, len(avg_errors) + 1), avg_errors)
    plt.xlabel("Problem Index")
    plt.ylabel("Average L2 Error")
    plt.title("Average L2 Error per Problem")
    plt.yscale("log")
    plt.grid(True)

    # Predictions vs ground truth for first problem, first few time steps
    plt.subplot(1, 4, 4)
    num_plots = min(4, len(predictions))
    for i in range(1, num_plots):
        plt.scatter(ground_truth[i], predictions[i], alpha=0.5, label=f"Time Step {i}")
    max_overall = max(np.max(ground_truth[num_plots-1]), np.max(predictions[num_plots-1]))
    plt.plot([0, max_overall], [0, max_overall], "k--", label="Ideal")
    plt.title("Predictions vs Ground Truth (Problem 1)")
    plt.xlabel("Ground Truth Temperature")
    plt.ylabel("Predicted Temperature")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_path}/meshgraphnet_multiproblem_results.png", dpi=150)
    plt.show()


def main():
    """Main function to run data-driven MeshGraphNet test with multiple problems."""
    print("=" * 60)
    print("DATA-DRIVEN MESHGRAPHNET MULTI-PROBLEM TEST")
    print("=" * 60)

    # Create results directory
    Path("results").mkdir(exist_ok=True)
    Path("results/data_driven").mkdir(exist_ok=True)

    # all_problems, time_config = generate_multiple_problems(n_problems=2, seed=42)
    mesh_problem, time_config = create_test_problem()
    all_problems = [mesh_problem, mesh_problem]
    
    # Split problems into train and validation
    train_indices = list(range(1))
    val_indices = list(range(1, 2))

    print(f"Training problems: {train_indices}")
    print(f"Validation problems: {val_indices}")

    # Training configuration
    config = {
        "epochs": 500,  # Reduced for testing
        "lr": 1e-3,
        "time_config": time_config,
        "time_window": 20,
    }

    print(f"Training configuration:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Time steps: {time_config.num_steps}")
    print(f"  Time window: {config['time_window']}")
    print(f"  dt: {time_config.dt}")

    # Create trainer with all problems
    trainer = DataDrivenMGNTrainer(all_problems, config)

    # Train model
    trainer.train(train_indices, val_indices)

    # Evaluate model on validation problems
    print("\nEvaluating on validation problems:")
    val_errors = trainer.evaluate(problem_indices=val_indices)

    # Evaluate model on training problems (to check for overfitting)
    print("\nEvaluating on training problems:")
    train_errors = trainer.evaluate(problem_indices=train_indices[:3])  # Just first 3 for efficiency

    # Get predictions for first validation problem for visualization
    predictions_val = trainer.rollout(problem_idx=val_indices[0])
    ground_truth_val = trainer.all_ground_truth[val_indices[0]]

    # Plot results
    plot_results(
        predictions_val, 
        ground_truth_val, 
        val_errors, 
        trainer.losses, 
        trainer.val_losses
    )

    # Export results for first validation problem
    print("Exporting results for first validation problem...")
    fem_solver = trainer.all_fem_solvers[val_indices[0]]
    
    # Ensure all arrays have the same length for export
    min_length = min(
        len(ground_truth_val), 
        len(predictions_val), 
        len(time_config.time_steps_export)
    )
    
    fem_solver.export_to_vtk(
        ground_truth_val[:min_length],
        predictions_val[:min_length],
        time_config.time_steps_export[:min_length],
        filename="results/data_driven/meshgraphnet_multiproblem_comparison.vtk",
    )

    # Save training summary
    summary = {
        "total_problems": len(all_problems),
        "train_problems": len(train_indices),
        "val_problems": len(val_indices),
        "final_train_loss": trainer.losses[-1],
        "final_val_loss": trainer.val_losses[-1],
        "val_avg_errors": [np.mean(errors) for errors in val_errors],
        "train_avg_errors": [np.mean(errors) for errors in train_errors],
    }
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total problems: {summary['total_problems']}")
    print(f"Training problems: {summary['train_problems']}")
    print(f"Validation problems: {summary['val_problems']}")
    print(f"Final training loss: {summary['final_train_loss']:.6f}")
    print(f"Final validation loss: {summary['final_val_loss']:.6f}")
    print(f"Validation average L2 errors: {summary['val_avg_errors']}")
    print(f"Training average L2 errors (sample): {summary['train_avg_errors']}")

    print("Data-driven MeshGraphNet multi-problem test completed!")
    print(f"Results saved to: results/data_driven/")


if __name__ == "__main__":
    main()
