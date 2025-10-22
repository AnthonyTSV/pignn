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

class PIMGNTrainer:
    """Trainer for Physics-Informed MeshGraphNet (PIMGN)."""
    
    def __init__(self, problems: List[MeshProblem], config: dict):
        self.problems = problems
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Time bundling configuration
        self.time_window = config.get('time_window', 5)
        
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
        sample_data, aux = graph_creator.create_graph(
            T_current=first_problem.initial_condition,
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
    
    
    def compute_physics_informed_loss(self, predictions_bundled_free, t_current, dt, problem_idx, aux=None, node_mapping=None):
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
            residual = fem_solver.compute_residual(
                t_pred_next=prediction_full_t,
                t_prev=t_prev,
                problem=problem
            )
            
            all_residuals.append(residual)
        
        # Stack all residuals: [N_TB, N_free_dofs]
        all_residuals_stacked = torch.stack(all_residuals)  # [N_TB, N_free_dofs]
        
        # Compute MSE over all (N_TB × N_free_dofs) values
        # Note: N_free_dofs corresponds to N_test_functions in the paper
        fem_loss = torch.mean(all_residuals_stacked ** 2)
        
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
        
        self.optimizer.zero_grad()
        # Build full graph from current state
        data, aux = graph_creator.create_graph(
            T_current=t_current,
            t_scalar=current_time,
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
        next_state_full = np.zeros(problem.n_nodes, dtype=np.float32)
        free_to_original = node_mapping["free_to_original"].cpu().numpy()
        next_state_full[free_to_original] = next_state_free
        # Dirichlet nodes remain zero

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
                        t_scalar=current_time
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
                        node_mapping
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
                predictions_bundled = self.model.forward(free_data)
                # Shape: (n_free_nodes, time_window)

                # Extract predictions for each time step
                free_idx = node_mapping["free_to_original"].detach().cpu().numpy()
                dirichlet_mask = aux["dirichlet_mask"].cpu().numpy()
                # Add each predicted time step
                for t_idx in range(self.time_window):
                    if step_idx + 1 + t_idx >= n_steps:
                        print(f"Warning: Exceeded rollout steps at {step_idx + 1 + t_idx}")
                        break

                    next_full = np.zeros(problem.n_nodes, dtype=np.float32)
                    pred_t = (
                        predictions_bundled[:, t_idx].squeeze().detach().cpu().numpy()
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
                step_idx += self.time_window

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

        return all_errors

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

    dirichlet_boundaries = ["bottom", "right", "top", "left"]
    neumann_boundaries = []
    dirichlet_boundaries_dict = {"bottom": 0, "right": 0, "top": 0, "left": 0}
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
        num_gaussians=1,
        amplitude_range=(10.0, 10.0),
        sigma_fraction_range=(0.2, 0.2),
        seed=42,
        centered=True,
        enforce_boundary_conditions=True,
    )
    # initial_condition = np.zeros_like(initial_condition)
    
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
    
    # Store the Neumann values array for later use
    problem.set_neumann_values_array(neumann_vals)
    problem.set_dirichlet_values_array(dirichlet_vals)
    
    # Set boundary conditions
    problem.set_neumann_values(neumann_boundaries_dict)
    problem.set_dirichlet_values(dirichlet_boundaries_dict)
    problem.source_function = None
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


def generate_multiple_problems(n_problems=15, seed=42):
    """Generate multiple problems with varying mesh sizes and Gaussian initial conditions."""
    print(f"Generating {n_problems} problems with varying parameters...")
    
    np.random.seed(seed)
    problems = []
    
    # Define parameter ranges
    mesh_sizes = np.linspace(0.05, 0.15, 5)  # 5 different mesh sizes
    num_gaussians_options = [1, 2, 3]  # 1-3 Gaussians
    amplitude_ranges = [(5, 15), (1.0, 2.0), (0.8, 1.2)]  # Different amplitude ranges
    sigma_fractions = [(0.1, 0.3), (0.15, 0.25), (0.2, 0.4)]  # Different width ranges
    
    # Time configuration (same for all problems)
    time_config = TimeConfig(dt=0.01, t_final=1.0)
    
    for i in range(n_problems):
        # Vary mesh size
        maxh = np.random.choice(mesh_sizes)
        
        # Create rectangular mesh
        mesh = create_rectangular_mesh(width=1.0, height=1.0, maxh=maxh)
        
        # Mesh configuration
        mesh_config = MeshConfig(
            maxh=maxh,
            order=1,
            dim=2,
            dirichlet_boundaries=["left", "right", "bottom"],
            neumann_boundaries=["top"],
            mesh_type="rectangle",
        )
        
        # Create graph to get node positions
        graph_creator = GraphCreator(
            mesh=mesh,
            n_neighbors=2,
            dirichlet_names=["left", "right", "bottom"],
            neumann_names=["top"],
            connectivity_method="fem",
        )
        # First create a temporary graph to get positions and aux data
        temp_data, temp_aux = graph_creator.create_graph()
        
        # Create Neumann values based on the temporary data
        neumann_vals = graph_creator.create_neumann_values(
            pos=temp_data.pos,
            aux_data=temp_aux,
            neumann_names=["top"],
            flux_magnitude=1.0,
            seed=42
        )
        
        # Create the final graph with Neumann values
        temp_data, _ = graph_creator.create_graph(neumann_values=neumann_vals)
        
        # Store Neumann values array for later use
        problem_neumann_vals = neumann_vals
        
        # Vary Gaussian initial condition parameters
        num_gaussians = np.random.choice(num_gaussians_options)
        amplitude_range = amplitude_ranges[i % len(amplitude_ranges)]
        sigma_fraction_range = sigma_fractions[i % len(sigma_fractions)]
        
        # Create varied Gaussian initial condition
        initial_condition = create_gaussian_initial_condition(
            pos=temp_data.pos,
            num_gaussians=num_gaussians,
            amplitude_range=amplitude_range,
            sigma_fraction_range=sigma_fraction_range,
            seed=seed + i,  # Different seed for each problem
            centered=np.random.choice([True, False]),  # Sometimes centered, sometimes not
            enforce_boundary_conditions=True,
        )
        # initial_condition = np.zeros_like(initial_condition)
        
        # Create problem
        problem = MeshProblem(
            mesh=mesh,
            graph_data=temp_data,
            initial_condition=initial_condition,
            alpha=1.0,  # Thermal diffusivity
            time_config=time_config,
            mesh_config=mesh_config,
            problem_id=i,
        )
        
        # Store the Neumann values array for use in training
        problem.set_neumann_values_array(problem_neumann_vals)
        
        # Set boundary conditions
        problem.set_neumann_values({"top": 1})
        problem.set_dirichlet_values({"left": 0.0, "right": 0.0, "bottom": 0.0})
        problem.source_function = None
        
        problems.append(problem)
        
        print(f"Problem {i+1}: {problem.n_nodes} nodes, {problem.n_edges} edges, "
              f"mesh_size={maxh:.3f}, num_gaussians={num_gaussians}")
    
    print(f"Generated {len(problems)} problems successfully!")
    return problems, time_config


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
    plt.title("PIMGN Training/Validation Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)

    # L2 error over time for first problem (if errors exist)
    plt.subplot(1, 4, 2)
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
    plt.subplot(1, 4, 3)
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

    # Predictions vs ground truth for first problem, first few time steps (if available)
    plt.subplot(1, 4, 4)
    if predictions and ground_truth:
        # Handle case where predictions/ground_truth might be lists
        pred_data = predictions[0] if isinstance(predictions[0], list) else predictions
        gt_data = ground_truth[0] if isinstance(ground_truth[0], list) else ground_truth
        
        num_plots = min(4, len(pred_data))
        for i in range(1, num_plots):
            plt.scatter(gt_data[i], pred_data[i], alpha=0.5, label=f"Time Step {i}")
        
        if num_plots > 1:
            max_overall = max(np.max(gt_data[num_plots-1]), np.max(pred_data[num_plots-1]))
            plt.plot([0, max_overall], [0, max_overall], "k--", label="Ideal")
        
        plt.title("Predictions vs Ground Truth (Problem 1)")
        plt.xlabel("Ground Truth Temperature")
        plt.ylabel("Predicted Temperature")
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No prediction/ground truth data', 
                horizontalalignment='center', verticalalignment='center', 
                transform=plt.gca().transAxes)
        plt.title("Predictions vs Ground Truth")

    plt.tight_layout()
    plt.savefig(f"{save_path}/pimgn_results.png", dpi=150)
    plt.show()


def main():
    """Main function to run Physics-Informed MeshGraphNet training and evaluation."""
    print("=" * 60)
    print("PHYSICS-INFORMED MESHGRAPHNET (PIMGN) TEST")
    print("=" * 60)
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    Path("results/physics_informed").mkdir(exist_ok=True)
    
    # Configuration for training mode
    use_multiple_problems = False  # Set to True for multi-problem training, False for single problem
    
    if use_multiple_problems:
        print("MULTI-PROBLEM PHYSICS-INFORMED TRAINING")
        print("=" * 40)
        
        # Generate multiple problems (12 training + 3 validation)
        all_problems, time_config = generate_multiple_problems(n_problems=15, seed=42)
        
        # Split problems into train and validation
        train_indices = list(range(1))  # First 12 problems for training
        val_indices = list(range(1, 2))  # Last 3 problems for validation
        
        print(f"Training problems: {train_indices}")
        print(f"Validation problems: {val_indices}")
        
        # Training configuration
        config = {
            'epochs': 1,  # Physics-informed training epochs (reduced for faster testing)
            'lr': 1e-3,     # Learning rate for stable physics-informed training
            'time_window': 20,  # Time bundling window
            'generate_ground_truth_for_validation': True,  # Generate ground truth for validation
        }
        
        print(f"Training configuration:")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Learning rate: {config['lr']}")
        print(f"  Time window: {config['time_window']}")
        print(f"  Time steps: {time_config.num_steps}")
        print(f"  dt: {time_config.dt}")
        print(f"  Physics-informed loss: FEM residual with temporal bundling")
        print(f"  Number of problems: {len(all_problems)}")
        
        # Create PIMGN trainer with multiple problems
        trainer = PIMGNTrainer(all_problems, config)
        
        # Train model with physics-informed loss on multiple problems
        print("\nStarting multi-problem physics-informed training...")
        trainer.train(train_indices, val_indices)
        
        # Evaluate model on validation problems
        print("\nEvaluating on validation problems:")
        val_errors = trainer.evaluate(problem_indices=val_indices)
        
        # Evaluate model on training problems (to check for overfitting)
        print("\nEvaluating on sample training problems:")
        train_errors = trainer.evaluate(problem_indices=train_indices[:3])  # Just first 3 for efficiency
        
        # Get predictions for first validation problem for visualization
        predictions_val = trainer.rollout(problem_idx=val_indices[0])
    
        fem_solver = trainer.all_fem_solvers[val_indices[0]]
        ground_truth = fem_solver.solve_transient_problem(all_problems[val_indices[0]])
        
        # Plot results with validation comparison
        plot_results([predictions_val], [ground_truth], val_errors, trainer.losses, trainer.val_losses, save_path="results/physics_informed")
        
        # Export results for first validation problem
        print("Exporting results for first validation problem...")
        
        
        # Generate ground truth for visualization
        ground_truth_val = fem_solver.solve_transient_problem(all_problems[val_indices[0]])
        
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
            filename="results/physics_informed/pimgn_multiproblem_comparison.vtk",
        )
        
    else:
        print("SINGLE-PROBLEM PHYSICS-INFORMED TRAINING")
        print("=" * 40)
        
        # Create single test problem
        problem, time_config = create_test_problem()
        all_problems = [problem]
        
        # Training configuration
        config = {
            'epochs': 300,  # Physics-informed training epochs (reduced for faster testing)
            'lr': 1e-3,     # Learning rate for stable physics-informed training
            'time_window': 20,  # Time bundling window
            'generate_ground_truth_for_validation': False,  # Don't need validation for single problem
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
        trainer.train(train_problems_indices=[0])  # Only train on the single problem
        
        # Evaluate model
        print("\nEvaluating trained PIMGN...")
        try:
            predictions, ground_truth, errors = trainer.evaluate_with_ground_truth(problem_indices=[0])
            
            # Plot results with ground truth comparison
            plot_results(predictions[0], ground_truth[0], errors, trainer.losses, [], save_path="results/physics_informed")
            
            # Export results for visualization
            print("Exporting results...")
            min_length = min(len(ground_truth[0]), len(predictions[0]), len(time_config.time_steps_export))
            trainer.all_fem_solvers[0].export_to_vtk(
                ground_truth[0][:min_length], 
                predictions[0][:min_length], 
                time_config.time_steps_export[:min_length], 
                filename="results/physics_informed/pimgn_single_comparison.vtk"
            )
            
        except Exception as e:
            print(f"Ground truth evaluation failed: {e}")
            print("Performing rollout evaluation without ground truth...")
            
            # Just get predictions without ground truth comparison
            predictions = trainer.rollout(problem_idx=0)
            plot_results([predictions], None, None, trainer.losses, None, save_path="results/physics_informed")
    
    print("Physics-Informed MeshGraphNet test completed!")
    print(f"Results saved to: results/physics_informed/")


if __name__ == "__main__":
    main()