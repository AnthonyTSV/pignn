import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import time
from argparse import Namespace

# Import our modules
from meshgraphnet import MeshGraphNet
from fem import FEMSolver
from mesh_utils import create_rectangular_mesh, create_free_node_subgraph, create_gaussian_initial_condition
from graph_creator import GraphCreator
from containers import TimeConfig, MeshConfig, MeshProblem
from torch_geometric.data import Data


class PIMGNTrainer:
    """Trainer for Physics-Informed MeshGraphNet (PIMGN)."""
    
    def __init__(self, problem: MeshProblem, config: dict):
        self.problem = problem
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Time bundling configuration
        self.time_window = config.get('time_window', 5)
        
        # Create FEM solver for physics-informed loss computation
        self.fem_solver = FEMSolver(problem.mesh, problem=problem)
        
        # Prepare sample data to determine input/output dimensions
        graph_creator = GraphCreator(
            mesh=problem.mesh,
            n_neighbors=2,
            dirichlet_names=problem.mesh_config.dirichlet_boundaries,
            neumann_names=[],
            connectivity_method="fem"
        )
        
        # Create sample graph to get dimensions
        sample_data, aux = graph_creator.create_graph(
            T_current=problem.initial_condition,
            t_scalar=0.0
        )

        free_node_data, mapping, new_aux = graph_creator.create_free_node_subgraph(
            full_graph=sample_data, aux=aux
        )
        
        input_dim_node = free_node_data.x.shape[1]
        input_dim_edge = free_node_data.edge_attr.shape[1]
        output_dim = self.time_window  # Predict multiple time steps
        
        print(f"Input dimensions - Node: {input_dim_node}, Edge: {input_dim_edge}, Output: {output_dim}")
        print(f"Time bundling window: {self.time_window}")
        
        # Create MeshGraphNet model with temporal bundling
        args = Namespace(num_layers=12, time_window=self.time_window)  # 12 message passing layers
        self.model = MeshGraphNet(
            input_dim_node=input_dim_node,
            input_dim_edge=input_dim_edge,
            hidden_dim=128,
            output_dim=output_dim,
            args=args
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)
        
        # Store graph creator for training
        self.graph_creator = graph_creator
        
        # Training history
        self.losses = []

        self.ground_truth = self.fem_solver.solve_transient_problem(self.problem)
    
    
    def compute_physics_informed_loss(self, predictions_bundled_free, t_current, dt, aux=None, node_mapping=None):
        """
        Compute FEM loss following the paper's methodology.
        
        Loss = MSE over (N_TB × N_test_functions) errors
        """
        t_current_tensor = t_current.to(dtype=torch.float64, device=self.device) if isinstance(t_current, torch.Tensor) else torch.tensor(t_current, dtype=torch.float64, device=self.device)
        
        free_to_original = node_mapping["free_to_original"]
        all_residuals = []  # Store all element-wise errors
        
        # Compute residual for each time step in bundle
        for t_idx in range(self.time_window):
            # Reconstruct full prediction
            prediction_free_t = predictions_bundled_free[:, t_idx]
            prediction_full_t = torch.zeros(self.problem.n_nodes, dtype=torch.float64, device=self.device)
            prediction_full_t[free_to_original] = prediction_free_t.to(dtype=torch.float64)
            
            # Get previous state
            if t_idx == 0:
                t_prev = t_current_tensor
            else:
                prev_prediction_free = predictions_bundled_free[:, t_idx - 1]
                t_prev = torch.zeros(self.problem.n_nodes, dtype=torch.float64, device=self.device)
                t_prev[free_to_original] = prev_prediction_free.to(dtype=torch.float64)
            
            # Compute FEM residual (element-wise errors)
            residual = self.fem_solver.compute_residual(
                t_pred_next=prediction_full_t,
                t_prev=t_prev,
                problem=self.problem
            )
            
            all_residuals.append(residual)
        
        # Stack all residuals: [N_TB, N_free_dofs]
        all_residuals_stacked = torch.stack(all_residuals)  # [N_TB, N_free_dofs]
        
        # Compute MSE over all (N_TB × N_free_dofs) values
        # Note: N_free_dofs corresponds to N_test_functions in the paper
        fem_loss = torch.mean(all_residuals_stacked ** 2)
        
        return fem_loss
    
    def train_step(self, t_current, current_time):
        """
        Compute physics loss for temporal bundling WITHOUT backpropagation.
        Returns the loss tensor (with gradients) and bundled predictions.
        """
        self.optimizer.zero_grad()
        # Build full graph from current state
        data, aux = self.graph_creator.create_graph(
            T_current=t_current,
            t_scalar=current_time
        )
        
        # Create free node subgraph (only non-Dirichlet nodes)
        free_data, node_mapping, free_aux = self.graph_creator.create_free_node_subgraph(
            full_graph=data, aux=aux
        )
        free_data = free_data.to(self.device)

        # Forward pass - get bundled predictions for FREE nodes [N_free_nodes, time_window]
        predictions_bundled_free = self.model.forward(free_data)
        
        # Convert current state to tensor
        t_current_tensor = torch.tensor(t_current, dtype=torch.float32, device=self.device) if not torch.is_tensor(t_current) else t_current

        # Compute physics-informed loss for temporal bundling with free nodes
        dt = self.problem.time_config.dt
        physics_loss = self.compute_physics_informed_loss(
            predictions_bundled_free,  # FREE node predictions [N_free_nodes, time_window]
            t_current_tensor,
            dt,
            aux,
            node_mapping
        )
        physics_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Convert bundled predictions to numpy for next iteration
        # Reconstruct full state from free node predictions
        predictions_bundled_np = predictions_bundled_free.detach().cpu().numpy()
        next_state_free = predictions_bundled_np[:, -1]  # Last time step prediction for free nodes
        
        # Reconstruct full state
        next_state_full = np.zeros(self.problem.n_nodes, dtype=np.float32)
        free_to_original = node_mapping["free_to_original"].cpu().numpy()
        next_state_full[free_to_original] = next_state_free
        # Dirichlet nodes remain zero

        return physics_loss.item(), next_state_full, predictions_bundled_np

    def train(self):
        """Main training loop following paper's methodology."""
        print(f"Starting PIMGN training on {self.device}")
        
        time_steps = self.problem.time_config.time_steps
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            epoch_losses = []
            
            # Optimization loop from initial condition
            t_current = self.problem.initial_condition.copy()
            time_steps = self.problem.time_config.time_steps
            time_steps_batched = np.array_split(time_steps, len(time_steps) // self.time_window)
            for batch_times in time_steps_batched:
                current_time = batch_times[0]
                physics_loss, t_next, _ = self.train_step(t_current, current_time)
                t_next += np.random.normal(0, 1e-4, size=t_next.shape)  # Small noise for stability?
                epoch_losses.append(physics_loss)
                t_current = t_next  # Update current state for next batch

            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            self.losses.append(avg_epoch_loss)
            
            if epoch % 10 == 0:
                elapsed = time.time() - epoch_start
                print(f"Epoch {epoch+1:4d} | Loss: {avg_epoch_loss:.3e} | Time: {elapsed:.2f}s")
            
            self.scheduler.step()

        print("Physics-Informed MeshGraphNet training with time bundling completed!")
    
    def rollout(self, n_steps=None):
        """Perform rollout prediction with temporal bundling for a specific problem."""
        self.model.eval()
        
        problem = self.problem
        ground_truth = self.fem_solver.solve_transient_problem(self.problem)

        if n_steps is None:
            n_steps = len(ground_truth)

        time_steps = self.problem.time_config.time_steps

        # Start with initial condition
        T_current = problem.initial_condition.copy()
        predictions = [T_current]

        graph_creator = GraphCreator(
            mesh=problem.mesh,
            n_neighbors=2,
            dirichlet_names=problem.mesh_config.dirichlet_boundaries,
            neumann_names=[],
            connectivity_method="fem",
        )

        with torch.no_grad():
            step_idx = 0
            while step_idx < len(time_steps) - 1:
                current_time = time_steps[step_idx]

                # Build graph
                data, aux = graph_creator.create_graph(
                    T_current=T_current, t_scalar=current_time
                )
                free_graph, node_mapping, free_aux = (
                    graph_creator.create_free_node_subgraph(data, aux)
                )
                free_data = free_graph.to(self.device)

                # Predict next 5 time steps
                predictions_bundled = self.model.forward(free_data)
                # Shape: (n_free_nodes, time_window)

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
                    next_full[aux["dirichlet_mask"].cpu().numpy()] = 0.0

                    predictions.append(next_full)

                # For next iteration, use the last predicted state
                # Or use the first predicted state for overlapping windows
                if len(predictions) > 1:
                    T_current = predictions[-1]  # Use last prediction

                # Move forward by time_window steps (non-overlapping)
                # Or by 1 step for overlapping windows
                step_idx += self.time_window  # Non-overlapping bundling
                # step_idx += 1  # Uncomment for overlapping bundling

        return predictions[:n_steps]
    
    def evaluate_with_ground_truth(self):
        """
        Evaluate the trained model against ground truth FEM solution.
        
        This generates ground truth data and compares with PIMGN predictions.
        """
        print("Generating ground truth for evaluation...")
        
        # Generate ground truth using FEM
        ground_truth = self.fem_solver.solve_transient_problem(self.problem)
        
        print("Computing PIMGN predictions...")
        # Get predictions from trained model
        predictions = self.rollout()
        
        # Compute errors
        errors = []
        for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
            if len(pred) == len(true):
                l2_error = np.linalg.norm(pred - true) / np.linalg.norm(true)
                errors.append(l2_error)
            else:
                print(f"Warning: Size mismatch at step {i}: pred={len(pred)}, true={len(true)}")
        
        if errors:
            print(f"Average L2 error: {np.mean(errors):.6f}")
            print(f"Final L2 error: {errors[-1]:.6f}")
        
        return predictions, ground_truth, errors

def create_test_problem():
    """Create a simple test problem for PIMGN training."""
    print("Creating test problem for Physics-Informed training...")
    
    # Time configuration
    time_config = TimeConfig(
        dt=0.01,
        t_final=1.0
    )
    maxh = 0.1  # Mesh element size
    # Create rectangular mesh
    mesh = create_rectangular_mesh(width=1.0, height=1.0, maxh=maxh)

    # Mesh configuration
    mesh_config = MeshConfig(
        maxh=maxh,
        order=1,
        dim=2,
        dirichlet_boundaries=["left", "right", "top", "bottom"],
        mesh_type="rectangle"
    )
    
    # Create graph to get node positions
    graph_creator = GraphCreator(
        mesh=mesh,
        n_neighbors=2,
        dirichlet_names=["left", "right", "top", "bottom"],
        neumann_names=[],
        connectivity_method="fem"
    )
    temp_data, _ = graph_creator.create_graph()
    
    # Create Gaussian initial condition
    initial_condition = create_gaussian_initial_condition(
        pos=temp_data['pos'],
        num_gaussians=1,
        amplitude_range=(10.0, 10.0),
        sigma_fraction_range=(0.2, 0.2),
        seed=42,
        centered=True,
        enforce_boundary_conditions=True
    )
    
    # Create problem
    problem = MeshProblem(
        mesh=mesh,
        graph_data=temp_data,
        initial_condition=initial_condition,
        alpha=1.0,  # Thermal diffusivity
        time_config=time_config,
        mesh_config=mesh_config,
        problem_id=0
    )
    
    # Set boundary conditions (homogeneous Dirichlet)
    problem.boundary_values = {
        "left": 0.0, 
        "right": 0.0, 
        "top": 0.0, 
        "bottom": 0.0
    }
    
    problem.source_function = None
    
    print(f"Problem created with {problem.n_nodes} nodes and {problem.n_edges} edges")
    print(f"Time steps: {len(time_config.time_steps)}, dt: {time_config.dt}")
    
    return problem, time_config


def plot_results(predictions, ground_truth, errors, losses, val_losses, save_path="results"):
    """Plot training results."""
    Path(save_path).mkdir(exist_ok=True)

    # Plot results
    plt.figure(figsize=(20, 4))

    # Training and validation loss
    plt.subplot(1, 4, 1)
    plt.plot(losses, label='Training')
    if val_losses:
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
    """Main function to run Physics-Informed MeshGraphNet training and evaluation."""
    print("=" * 60)
    print("PHYSICS-INFORMED MESHGRAPHNET (PIMGN) TEST")
    print("=" * 60)
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Create test problem
    problem, time_config = create_test_problem()
    
    # Training configuration
    config = {
        'epochs': 300,  # Physics-informed training epochs (reduced for faster testing)
        'lr': 1e-3,     # Learning rate for stable physics-informed training
        'time_window': 20,  # Time bundling window
    }
    
    print(f"Training configuration:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Time window: {config['time_window']}")
    print(f"  Time steps: {time_config.num_steps}")
    print(f"  dt: {time_config.dt}")
    print(f"  Physics-informed loss: FEM residual with temporal bundling")
    
    # Create PIMGN trainer
    trainer = PIMGNTrainer(problem, config)
    
    # Train model with physics-informed loss
    print("\nStarting physics-informed training...")
    trainer.train()
    
    # Evaluate model
    print("\nEvaluating trained PIMGN...")
    try:
        predictions, ground_truth, errors = trainer.evaluate_with_ground_truth()
        val_losses = []  # No validation in physics-informed training
        # Plot results with ground truth comparison
        plot_results(predictions, ground_truth, errors, trainer.losses, val_losses, save_path="results")
        
        # Export results for visualization
        print("Exporting results...")
        min_length = min(len(ground_truth), len(predictions), len(time_config.time_steps_export))
        trainer.fem_solver.export_to_vtk(
            ground_truth[:min_length], 
            predictions[:min_length], 
            time_config.time_steps_export[:min_length], 
            filename="results/physics_informed/pimgn_comparison.vtk"
        )
        
    except Exception as e:
        print(f"Ground truth evaluation failed: {e}")
        print("Performing rollout evaluation without ground truth...")
        
        # Just get predictions without ground truth comparison
        predictions = trainer.rollout()
        plot_results(predictions, None, None, trainer.losses, None, save_path="results")
    
    print("Physics-Informed MeshGraphNet test completed!")
    print(f"Results saved to: results/")


if __name__ == "__main__":
    main()