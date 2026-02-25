import numpy as np
import os
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
        self.time_window = config.get("time_window", 20)

        # Create FEM solvers for physics-informed loss computation for all problems
        self.all_fem_solvers: List[FEMSolver] = []
        for problem in problems:
            fem_solver = FEMSolver(
                problem.mesh, order=problem.mesh_config.order, problem=problem
            )
            self.all_fem_solvers.append(fem_solver)

        # Prepare sample data to determine input/output dimensions using first problem
        first_problem = problems[0]
        graph_creator = GraphCreator(
            mesh=first_problem.mesh,
            n_neighbors=2,
            dirichlet_names=first_problem.mesh_config.dirichlet_boundaries,
            neumann_names=getattr(first_problem.mesh_config, "neumann_boundaries", []),
            robin_names=getattr(first_problem.mesh_config, "robin_boundaries", []),
            connectivity_method="fem",
            fes=fem_solver.fes,
        )

        # Create sample graph to get dimensions
        material_field = getattr(first_problem, "material_field", None)
        neumann_vals = getattr(first_problem, "neumann_values_array", None)
        dirichlet_vals = getattr(first_problem, "dirichlet_values_array", None)
        robin_vals = getattr(first_problem, "robin_values_array", None)
        source_vals = getattr(first_problem, "source_function", None)
        sample_data, aux = graph_creator.create_graph(
            T_current=first_problem.initial_condition,
            t_scalar=0.0,
            material_node_field=material_field,
            neumann_values=neumann_vals,
            dirichlet_values=dirichlet_vals,
            robin_values=robin_vals,
            source_values=source_vals,
        )

        # Use workpiece subgraph if applicable, otherwise free-node subgraph
        wp_mask = getattr(first_problem, "wp_node_mask", None)
        if wp_mask is not None:
            free_node_data, mapping, new_aux = graph_creator.create_workpiece_subgraph(
                data=sample_data, aux=aux, wp_node_mask=wp_mask
            )
        else:
            free_node_data, mapping, new_aux = graph_creator.create_free_node_subgraph(
                data=sample_data, aux=aux
            )

        input_dim_node = free_node_data.x.shape[1]
        input_dim_edge = free_node_data.edge_attr.shape[1]
        output_dim = self.time_window  # Predict multiple time steps

        print(
            f"Input dimensions - Node: {input_dim_node}, Edge: {input_dim_edge}, Output: {output_dim}"
        )
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

        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9995)

        # Training history
        self.losses = []
        self.val_losses = []
        self.last_residuals = None

        # Initialize logger
        save_dir = config.get("save_dir", "results/physics_informed")
        log_filename = config.get("log_filename", "training_log.json")
        save_interval = config.get("save_interval", None)
        save_epoch_interval = config.get("save_epoch_interval", None)

        self.logger = TrainingLogger(
            save_dir=save_dir,
            filename=log_filename,
            save_interval=save_interval,
            save_epoch_interval=save_epoch_interval,
        )
        self.logger.log_config(config)
        self.logger.set_device(self.device)
        self.logger.log_problems(problems)

        # Load checkpoint if resuming training
        self.start_epoch = 0
        resume_from = config.get("resume_from", None)
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming training from checkpoint: {resume_from}")
            # PyTorch 2.6 changed the default `weights_only` of torch.load from False -> True.
            # We store a full training checkpoint (optimizer/scheduler/etc.), so we must load
            # with `weights_only=False`. Only do this for trusted checkpoint files.
            try:
                checkpoint = torch.load(
                    resume_from, map_location=self.device, weights_only=False
                )
            except TypeError:
                # Backward compatibility with older PyTorch versions.
                checkpoint = torch.load(resume_from, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Full checkpoint with optimizer state
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                self.start_epoch = checkpoint.get("epoch", 0) + 1
                self.losses = checkpoint.get("losses", [])
                self.val_losses = checkpoint.get("val_losses", [])
                print(f"Resumed from epoch {self.start_epoch}")
            else:
                # Just model weights (backward compatibility)
                self.model.load_state_dict(checkpoint)
                print("Loaded model weights only (no optimizer state)")

        # Generate ground truth for validation
        if config.get("generate_ground_truth_for_validation", False):
            print("Generating ground truth for validation...")
            self.all_ground_truth = []
            for i, problem in enumerate(problems):
                print(f"Solving problem {i+1}/{len(problems)} for validation...")
                ground_truth = self.all_fem_solvers[i].solve_transient_problem(problem)
                self.all_ground_truth.append(ground_truth)
        else:
            self.all_ground_truth = None

    def _is_workpiece_problem(self, problem_idx: int) -> bool:
        """Check if this problem uses workpiece-only domain."""
        problem = self.problems[problem_idx]
        return getattr(problem, "wp_node_mask", None) is not None

    def _create_subgraph(self, graph_creator, data, aux, problem_idx):
        """Create the appropriate subgraph (workpiece or free-node) based on problem type."""
        problem = self.problems[problem_idx]
        wp_mask = getattr(problem, "wp_node_mask", None)
        if wp_mask is not None:
            wp_data, node_mapping, wp_aux = graph_creator.create_workpiece_subgraph(
                data=data, aux=aux, wp_node_mask=wp_mask
            )
            # Unify mapping keys so downstream code works the same way
            node_mapping["free_to_original"] = node_mapping["wp_to_original"]
            node_mapping["n_free"] = node_mapping["n_wp"]
            return wp_data, node_mapping, wp_aux
        else:
            return graph_creator.create_free_node_subgraph(data=data, aux=aux)

    def compute_physics_informed_loss(
        self,
        predictions_bundled_free,
        t_current,
        dt,
        problem_idx,
        aux=None,
        node_mapping=None,
        start_time: float = 0.0,
    ):
        """
        Compute FEM loss following the paper's methodology.

        Loss = MSE over (N_TB x N_test_functions) errors
        """
        problem: MeshProblem = self.problems[problem_idx]
        fem_solver: FEMSolver = self.all_fem_solvers[problem_idx]

        t_current_tensor = (
            t_current.to(dtype=torch.float64, device=self.device)
            if isinstance(t_current, torch.Tensor)
            else torch.tensor(t_current, dtype=torch.float64, device=self.device)
        )

        free_to_original = node_mapping["free_to_original"]
        all_residuals = []  # Store all element-wise errors

        # Get Dirichlet mask and values for enforcing boundary conditions
        dirichlet_mask = aux["dirichlet_mask"].to(self.device)
        dirichlet_vals = getattr(problem, "dirichlet_values_array", None)
        if dirichlet_vals is not None:
            dirichlet_vals_tensor = torch.tensor(
                dirichlet_vals, dtype=torch.float64, device=self.device
            )
        else:
            dirichlet_vals_tensor = None

        # Compute residual for each time step in bundle
        for t_idx in range(self.time_window):
            # Reconstruct full prediction
            prediction_free_t = predictions_bundled_free[:, t_idx]
            prediction_full_t = torch.zeros(
                problem.n_nodes, dtype=torch.float64, device=self.device
            )
            prediction_full_t[free_to_original] = prediction_free_t.to(
                dtype=torch.float64
            )
            # Enforce Dirichlet BCs on prediction
            if dirichlet_vals_tensor is not None:
                prediction_full_t[dirichlet_mask] = dirichlet_vals_tensor[
                    dirichlet_mask
                ]

            # Get previous state
            if t_idx == 0:
                t_prev = t_current_tensor.clone()
                # Ensure Dirichlet BCs are set on initial state
                if dirichlet_vals_tensor is not None:
                    t_prev[dirichlet_mask] = dirichlet_vals_tensor[dirichlet_mask]
            else:
                prev_prediction_free = predictions_bundled_free[:, t_idx - 1]
                t_prev = torch.zeros(
                    problem.n_nodes, dtype=torch.float64, device=self.device
                )
                t_prev[free_to_original] = prev_prediction_free.to(dtype=torch.float64)
                # Enforce Dirichlet BCs on previous state
                if dirichlet_vals_tensor is not None:
                    t_prev[dirichlet_mask] = dirichlet_vals_tensor[dirichlet_mask]

            # Compute FEM residual (element-wise errors)
            time_for_step = start_time + (t_idx + 1) * dt
            residual = fem_solver.compute_residual(
                t_pred_next=prediction_full_t,
                t_prev=t_prev,
                problem=problem,
                time_scalar=float(time_for_step),
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
            neumann_names=getattr(problem.mesh_config, "neumann_boundaries", []),
            robin_names=getattr(problem.mesh_config, "robin_boundaries", []),
            connectivity_method="fem",
            fes=self.all_fem_solvers[problem_idx].fes,
        )

        # Get the Neumann values for this problem
        neumann_vals = getattr(problem, "neumann_values_array", None)
        dirichlet_vals = getattr(problem, "dirichlet_values_array", None)
        robin_vals = getattr(problem, "robin_values_array", None)
        material_field = getattr(problem, "material_field", None)
        source_vals = getattr(problem, "source_function", None)

        self.optimizer.zero_grad()

        # add gaussian noise to current state for regularization
        # t_current = t_current + np.random.normal(0, 1, size=t_current.shape)

        # Build full graph from current state
        data, aux = graph_creator.create_graph(
            T_current=t_current,
            t_scalar=current_time,
            material_node_field=material_field,
            neumann_values=neumann_vals,
            dirichlet_values=dirichlet_vals,
            robin_values=robin_vals,
            source_values=source_vals,
        )

        # Create free node subgraph (workpiece-only or non-Dirichlet nodes)
        free_data, node_mapping, free_aux = self._create_subgraph(
            graph_creator, data, aux, problem_idx
        )
        free_data = free_data.to(self.device)

        # Forward pass - get bundled predictions for FREE nodes [N_free_nodes, time_window]
        predictions_bundled_free = self.model.forward(free_data)

        # Convert current state to tensor
        t_current_tensor = (
            torch.tensor(t_current, dtype=torch.float32, device=self.device)
            if not torch.is_tensor(t_current)
            else t_current
        )

        # Compute physics-informed loss for temporal bundling with free nodes
        dt = problem.time_config.dt
        physics_loss = self.compute_physics_informed_loss(
            predictions_bundled_free,  # FREE node predictions [N_free_nodes, time_window]
            t_current_tensor,
            dt,
            problem_idx,
            aux,
            node_mapping,
            start_time=float(current_time),
        )
        physics_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Convert bundled predictions to numpy for next iteration
        # Reconstruct full state from free node predictions
        predictions_bundled_np = predictions_bundled_free.detach().cpu().numpy()
        next_state_free = predictions_bundled_np[
            :, -1
        ]  # Last time step prediction for free nodes

        # Reconstruct full state
        next_state_full = np.zeros(problem.n_nodes, dtype=np.float32)
        free_to_original = node_mapping["free_to_original"].cpu().numpy()
        next_state_full[free_to_original] = next_state_free
        if dirichlet_vals is not None:
            dirichlet_mask = aux["dirichlet_mask"].cpu().numpy()
            next_state_full[dirichlet_mask] = dirichlet_vals[dirichlet_mask]
        # For workpiece problems, set non-workpiece nodes to initial condition (e.g. T_amb)
        wp_mask = getattr(problem, "wp_node_mask", None)
        if wp_mask is not None:
            next_state_full[~wp_mask] = problem.initial_condition[~wp_mask]

        return physics_loss.item(), next_state_full, predictions_bundled_np

    def train(self, train_problems_indices, val_problems_indices=None):
        """Main training loop following paper's methodology with multiple problems."""
        print(f"Starting PIMGN training on {self.device}")
        print(f"Training on problems: {train_problems_indices}")
        if val_problems_indices:
            print(f"Validation on problems: {val_problems_indices}")

        for epoch in range(self.start_epoch, self.config["epochs"]):
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
                time_steps_batched = np.array_split(
                    time_steps, len(time_steps) // self.time_window
                )

                problem_losses = []
                for batch_times in time_steps_batched:
                    current_time = batch_times[0]
                    physics_loss, t_next, _ = self.train_step(
                        t_current, current_time, problem_idx
                    )
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
                print(
                    f"Epoch {epoch+1:4d} | Train Loss: {avg_epoch_loss:.3e}{val_str} | Time: {elapsed:.2f}s"
                )
            if (epoch + 1) % 100 == 0:
                self.model.eval()
                prob_idx = train_problems_indices[0]
                predictions = self.rollout(problem_idx=prob_idx)

                # If we have ground truth, compute errors
                if self.all_ground_truth is not None:
                    ground_truth = self.all_ground_truth[prob_idx]

                    # Compute errors
                    problem = self.problems[prob_idx]
                    wp_mask = getattr(problem, "wp_node_mask", None)
                    errors = []
                    for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
                        if wp_mask is not None:
                            pred_wp = pred[wp_mask]
                            true_wp = true[wp_mask]
                            norm_true = np.linalg.norm(true_wp)
                            l2_error = np.linalg.norm(pred_wp - true_wp) / norm_true if norm_true > 0 else 0.0
                        else:
                            l2_error = np.linalg.norm(pred - true) / np.linalg.norm(true)
                        errors.append(l2_error)
                mean_l2_error = np.mean(errors)
                print(f"Epoch {epoch+1:4d} | Rollout L2 Error: {mean_l2_error:.3e}")


            self.scheduler.step()

        print(
            "Physics-Informed MeshGraphNet training with multiple problems completed!"
        )

    def validate(self, val_problems_indices):
        """Validate model on held-out problems using physics-informed loss."""
        if self.all_ground_truth is None:
            print(
                "Warning: No ground truth available for validation. Skipping validation."
            )
            return None

        self.model.eval()
        total_loss = 0.0
        num_validations = 0

        with torch.no_grad():
            for problem_idx in val_problems_indices:
                problem: MeshProblem = self.problems[problem_idx]

                # Use a few time steps from the ground truth for validation
                ground_truth = self.all_ground_truth[problem_idx]
                validation_steps = min(
                    5, len(ground_truth) - self.time_window
                )  # Validate on first few steps

                for step_idx in range(0, validation_steps, self.time_window):
                    t_current = ground_truth[step_idx]
                    current_time = problem.time_config.time_steps[step_idx]

                    # Create graph creator for this specific problem
                    graph_creator = GraphCreator(
                        mesh=problem.mesh,
                        n_neighbors=2,
                        dirichlet_names=problem.mesh_config.dirichlet_boundaries,
                        neumann_names=getattr(problem.mesh_config, "neumann_boundaries", []),
                        robin_names=getattr(problem.mesh_config, "robin_boundaries", []),
                        connectivity_method="fem",
                        fes=self.all_fem_solvers[problem_idx].fes,
                    )

                    # Build graph and make prediction
                    data, aux = graph_creator.create_graph(
                        T_current=t_current,
                        t_scalar=current_time,
                        material_node_field=getattr(problem, "material_field", None),
                        neumann_values=getattr(problem, "neumann_values_array", None),
                        dirichlet_values=getattr(
                            problem, "dirichlet_values_array", None
                        ),
                        robin_values=getattr(problem, "robin_values_array", None),
                        source_values=getattr(problem, "source_function", None),
                    )

                    free_data, node_mapping, free_aux = (
                        self._create_subgraph(graph_creator, data, aux, problem_idx)
                    )
                    free_data = free_data.to(self.device)

                    predictions_bundled_free = self.model.forward(free_data)
                    t_current_tensor = torch.tensor(
                        t_current, dtype=torch.float32, device=self.device
                    )

                    # Compute physics loss for validation
                    dt = problem.time_config.dt
                    physics_loss = self.compute_physics_informed_loss(
                        predictions_bundled_free,
                        t_current_tensor,
                        dt,
                        problem_idx,
                        aux,
                        node_mapping,
                        start_time=float(current_time),
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
        time_steps_bundled = np.array_split(
            time_steps, len(time_steps) // self.time_window
        )

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
            fes=self.all_fem_solvers[problem_idx].fes,
        )

        # Get the Neumann values for this problem
        neumann_vals = getattr(problem, "neumann_values_array", None)
        dirichlet_vals = getattr(problem, "dirichlet_values_array", None)
        robin_vals = getattr(problem, "robin_values_array", None)
        material_field = getattr(problem, "material_field", None)
        source_vals = getattr(problem, "source_function", None)

        with torch.no_grad():
            step_idx = 0
            for batch_idx, batch_times in enumerate(time_steps_bundled):
                starting_time_step = 0 if step_idx == 0 else batch_times[0]
                # Build graph
                data, aux = graph_creator.create_graph(
                    T_current=T_current,
                    t_scalar=starting_time_step,
                    material_node_field=material_field,
                    neumann_values=neumann_vals,
                    dirichlet_values=dirichlet_vals,
                    robin_values=robin_vals,
                    source_values=source_vals,
                )
                free_graph, node_mapping, free_aux = (
                    self._create_subgraph(graph_creator, data, aux, problem_idx)
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
                        predictions_bundled[:, time_idx]
                        .squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    next_full[free_idx] = pred_t
                    next_full[dirichlet_mask] = dirichlet_vals[dirichlet_mask]

                    predictions.append(next_full)

                # For next iteration, use the last predicted state
                if len(predictions) > 1:
                    T_to_use = predictions[-1]
                    # enforce Dirichlet BCs
                    if dirichlet_vals is not None:
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
            wp_mask = getattr(problem, "wp_node_mask", None)
            for j, (pred, true) in enumerate(zip(predictions, ground_truth)):
                if len(pred) == len(true):
                    if wp_mask is not None:
                        # Only evaluate on workpiece nodes
                        pred_wp = pred[wp_mask]
                        true_wp = true[wp_mask]
                        norm_true = np.linalg.norm(true_wp)
                        l2_error = np.linalg.norm(pred_wp - true_wp) / norm_true if norm_true > 0 else 0.0
                    else:
                        l2_error = np.linalg.norm(pred - true) / np.linalg.norm(true)
                    errors.append(l2_error)
                else:
                    print(
                        f"Warning: Size mismatch at step {j}: pred={len(pred)}, true={len(true)}"
                    )

            if errors:
                print(
                    f"Problem {problem_idx + 1} - Average L2 error: {np.mean(errors):.6f}, Final L2 error: {errors[-1]:.6f}"
                )
                all_errors.append(errors)
                all_predictions.append(predictions)
                all_ground_truth.append(ground_truth)
            else:
                print(f"Problem {problem_idx + 1} - No valid errors computed")

        # Overall statistics
        if all_errors:
            avg_errors = np.mean([np.mean(errors) for errors in all_errors])
            final_errors = np.mean([errors[-1] for errors in all_errors])
            print(
                f"\nOverall - Average L2 error: {avg_errors:.6f}, Final L2 error: {final_errors:.6f}"
            )

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
                problem = self.problems[prob_idx]
                wp_mask = getattr(problem, "wp_node_mask", None)
                errors = []
                for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
                    if wp_mask is not None:
                        pred_wp = pred[wp_mask]
                        true_wp = true[wp_mask]
                        norm_true = np.linalg.norm(true_wp)
                        l2_error = np.linalg.norm(pred_wp - true_wp) / norm_true if norm_true > 0 else 0.0
                    else:
                        l2_error = np.linalg.norm(pred - true) / np.linalg.norm(true)
                    errors.append(l2_error)

                all_errors.append(errors)
                print(
                    f"Problem {prob_idx + 1} - Average L2 error: {np.mean(errors):.6f}, Final L2 error: {errors[-1]:.6f}"
                )
            else:
                print(
                    f"Problem {prob_idx + 1} - Rollout completed (no ground truth for comparison)"
                )

        # Overall statistics
        if all_errors:
            avg_errors = np.mean([np.mean(errors) for errors in all_errors])
            final_errors = np.mean([errors[-1] for errors in all_errors])
            print(
                f"\nOverall - Average L2 error: {avg_errors:.6f}, Final L2 error: {final_errors:.6f}"
            )

            self.logger.log_evaluation(all_errors, "l2_errors_per_problem")
            self.logger.log_evaluation(avg_errors, "mean_l2_error")
            self.logger.log_evaluation(final_errors, "mean_final_l2_error")

        return all_errors

    def save_logs(self, filename="training_log.json"):
        self.logger.save(filename)

    def save_checkpoint(self, path: str, epoch: int):
        """Save a full checkpoint including model, optimizer, and scheduler state."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "losses": self.losses,
            "val_losses": self.val_losses,
            "time_window": self.time_window,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")
