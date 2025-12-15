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
from fem_em import FEMSolverEM
from graph_creator import GraphCreator
from containers import MeshConfig, MeshProblem
from torch_geometric.data import Data

from train_problems import create_em_problem


class PIMGNTrainerEM:
    """Trainer for Physics-Informed MeshGraphNet for Steady-State Electromagnetic Problems."""

    def __init__(self, problems: List[MeshProblem], config: dict):
        self.problems = problems
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create FEM solvers for physics-informed loss computation for all problems
        self.all_fem_solvers: List[FEMSolverEM] = []
        for problem in problems:
            order = int(getattr(getattr(problem, "mesh_config", None), "order", 1))
            fem_solver = FEMSolverEM(
                problem.mesh, order=order, problem=problem, device=self.device
            )
            self.all_fem_solvers.append(fem_solver)

        # Prepare sample data to determine input/output dimensions using first problem
        first_problem = problems[0]
        first_fes = self.all_fem_solvers[0].fes
        graph_creator = GraphCreator(
            mesh=first_problem.mesh,
            n_neighbors=2,
            dirichlet_names=first_problem.mesh_config.dirichlet_boundaries,
            neumann_names=getattr(first_problem.mesh_config, "neumann_boundaries", []),
            connectivity_method="fem",
            fes=first_fes,
        )

        # Create sample graph to get dimensions
        material_field = getattr(first_problem, "material_field", None)
        neumann_vals = getattr(first_problem, "neumann_values_array", None)
        dirichlet_vals = getattr(first_problem, "dirichlet_values_array", None)
        source_vals = getattr(first_problem, "source_function", None)

        # For steady-state EM, use complex zero initial condition (ensures consistent
        # real/imag feature dimensions in the graph creator).
        initial_guess = getattr(
            first_problem,
            "initial_condition",
            np.zeros(
                int(getattr(first_fes, "ndof", first_problem.n_nodes)),
                dtype=np.complex128,
            ),
        )
        if not np.iscomplexobj(initial_guess):
            initial_guess = np.asarray(initial_guess, dtype=np.float64).astype(
                np.complex128
            )

        sample_data, aux = graph_creator.create_graph(
            T_current=initial_guess,
            t_scalar=0.0,
            material_node_field=material_field,
            neumann_values=neumann_vals,
            dirichlet_values=dirichlet_vals,
            source_values=source_vals,
        )

        free_node_data, mapping, new_aux = graph_creator.create_free_node_subgraph(
            data=sample_data, aux=aux
        )

        input_dim_node = free_node_data.x.shape[1]
        input_dim_edge = free_node_data.edge_attr.shape[1]
        output_dim = 2  # Complex-valued solution: [real, imaginary]

        print(
            f"Input dimensions - Node: {input_dim_node}, Edge: {input_dim_edge}, Output: {output_dim} (complex)"
        )
        print(f"Steady-state electromagnetic problem")
        print(f"Training on {len(problems)} problems")

        # Create MeshGraphNet model for steady-state prediction
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

        # Initialize logger
        save_dir = config.get("save_dir", "results/physics_informed_em")
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
                ground_truth = self.all_fem_solvers[i].solve(problem)
                self.all_ground_truth.append(ground_truth)
        else:
            self.all_ground_truth = None

    def compute_physics_informed_loss(
        self,
        prediction_free,
        problem_idx,
        aux=None,
        node_mapping=None,
    ):
        """
        Compute FEM loss for steady-state EM problem.

        Loss = MSE over N_test_functions errors
        """
        problem: MeshProblem = self.problems[problem_idx]
        fem_solver: FEMSolverEM = self.all_fem_solvers[problem_idx]

        free_to_original = node_mapping["free_to_original"].to(self.device)
        n_total = int(node_mapping.get("n_original"))

        # Get Dirichlet mask and values for enforcing boundary conditions
        dirichlet_mask = aux["dirichlet_mask"].to(self.device)
        dirichlet_vals = getattr(problem, "dirichlet_values_array", None)
        if dirichlet_vals is not None:
            if np.iscomplexobj(dirichlet_vals):
                dirichlet_vals_tensor = torch.tensor(
                    dirichlet_vals, dtype=torch.complex128, device=self.device
                )
            else:
                dirichlet_vals_tensor = torch.tensor(
                    dirichlet_vals, dtype=torch.float64, device=self.device
                )
        else:
            dirichlet_vals_tensor = None

        # Reconstruct full prediction from free nodes
        # prediction_free has shape [N_free_nodes, 2] for [real, imag]
        prediction_full_real = torch.zeros(
            n_total, dtype=torch.float64, device=self.device
        )
        prediction_full_imag = torch.zeros(
            n_total, dtype=torch.float64, device=self.device
        )
        prediction_full_real[free_to_original] = prediction_free[:, 0].to(
            dtype=torch.float64
        )
        prediction_full_imag[free_to_original] = prediction_free[:, 1].to(
            dtype=torch.float64
        )

        # Combine into complex tensor
        prediction_full = torch.complex(prediction_full_real, prediction_full_imag)

        # Enforce Dirichlet BCs on prediction
        if dirichlet_vals_tensor is not None:
            dirichlet_vals_complex = (
                dirichlet_vals_tensor
                if torch.is_complex(dirichlet_vals_tensor)
                else dirichlet_vals_tensor.to(dtype=torch.complex128)
            )
            prediction_full[dirichlet_mask] = dirichlet_vals_complex[dirichlet_mask]

        # Compute FEM residual (element-wise errors) for steady-state problem
        residual = fem_solver.compute_residual(
            prediction_full,
        )

        # Compute MSE over free DOFs only.
        # For complex residuals, use squared magnitude: |r|^2 = real^2 + imag^2
        residual_magnitude_sq = residual.real**2 + residual.imag**2
        residual_free = residual_magnitude_sq[free_to_original]
        fem_loss = torch.mean(residual_free)

        self.last_residuals = residual.detach().cpu().numpy()

        return fem_loss

    def train_step(self, problem_idx, iteration=0):
        """
        Compute physics loss for steady-state problem.
        Returns the loss value and prediction.
        """
        problem: MeshProblem = self.problems[problem_idx]
        fem_solver: FEMSolverEM = self.all_fem_solvers[problem_idx]

        # Create graph creator for this specific problem
        graph_creator = GraphCreator(
            mesh=problem.mesh,
            n_neighbors=2,
            dirichlet_names=problem.mesh_config.dirichlet_boundaries,
            neumann_names=getattr(problem.mesh_config, "neumann_boundaries", []),
            connectivity_method="fem",
            fes=fem_solver.fes,
        )

        # Get the values for this problem
        neumann_vals = getattr(problem, "neumann_values_array", None)
        dirichlet_vals = getattr(problem, "dirichlet_values_array", None)
        material_field = getattr(problem, "material_field", None)
        source_vals = getattr(problem, "source_function", None)

        # Initial guess: use complex zeros for consistent real/imag feature dims
        initial_guess = getattr(
            problem,
            "initial_condition",
            np.zeros(
                int(getattr(fem_solver.fes, "ndof", problem.n_nodes)),
                dtype=np.complex128,
            ),
        )
        if not np.iscomplexobj(initial_guess):
            initial_guess = np.asarray(initial_guess, dtype=np.float64).astype(
                np.complex128
            )

        # Add small random noise to initial guess for regularization
        if iteration > 0 and self.config.get("add_noise", False):
            noise_scale = self.config.get("noise_scale", 0.01)
            initial_guess = initial_guess + np.random.normal(
                0, noise_scale, size=initial_guess.shape
            )

        self.optimizer.zero_grad()

        # Build full graph
        data, aux = graph_creator.create_graph(
            T_current=initial_guess,
            t_scalar=0.0,  # Time is irrelevant for steady-state
            material_node_field=material_field,
            neumann_values=neumann_vals,
            dirichlet_values=dirichlet_vals,
            source_values=source_vals,
        )

        # Create free node subgraph (only non-Dirichlet nodes)
        free_data, node_mapping, free_aux = graph_creator.create_free_node_subgraph(
            data=data, aux=aux
        )
        free_data = free_data.to(self.device)

        # Forward pass - get prediction for FREE nodes [N_free_nodes, 2] (real, imag)
        prediction_free = self.model.forward(free_data)

        # Compute physics-informed loss for steady-state
        physics_loss = self.compute_physics_informed_loss(
            prediction_free,
            problem_idx,
            aux,
            node_mapping,
        )

        physics_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Convert prediction to numpy and reconstruct complex values
        prediction_np = prediction_free.detach().cpu().numpy()  # [N_free, 2]

        # Reconstruct full state as complex array
        n_total = int(node_mapping.get("n_original", problem.n_nodes))
        prediction_full = np.zeros(n_total, dtype=np.complex128)
        free_to_original = node_mapping["free_to_original"].cpu().numpy()
        prediction_full[free_to_original] = (
            prediction_np[:, 0] + 1j * prediction_np[:, 1]
        )
        if dirichlet_vals is not None:
            dirichlet_mask = aux["dirichlet_mask"].cpu().numpy()
            prediction_full[dirichlet_mask] = dirichlet_vals[dirichlet_mask]

        return physics_loss.item(), prediction_full

    def train(self, train_problems_indices, val_problems_indices=None):
        """Main training loop for steady-state EM problems with multiple problems."""
        print(f"Starting PIMGN-EM training on {self.device}")
        print(f"Training on problems: {train_problems_indices}")
        if val_problems_indices:
            print(f"Validation on problems: {val_problems_indices}")

        # Number of iterations per problem per epoch
        iterations_per_problem = self.config.get("iterations_per_problem", 10)

        for epoch in range(self.start_epoch, self.config["epochs"]):
            epoch_start = time.time()
            epoch_losses = []

            # Shuffle training problems for each epoch
            shuffled_train_indices = train_problems_indices.copy()
            np.random.shuffle(shuffled_train_indices)

            # Train on each problem in the training set
            for problem_idx in shuffled_train_indices:
                problem_losses = []

                # Multiple iterations on the same problem to improve convergence
                for iteration in range(iterations_per_problem):
                    physics_loss, prediction = self.train_step(problem_idx, iteration)
                    problem_losses.append(physics_loss)

                # Average loss over all iterations for this problem
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

            self.scheduler.step()

        print("Physics-Informed MeshGraphNet EM training completed!")

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
                fem_solver: FEMSolverEM = self.all_fem_solvers[problem_idx]

                # Use initial guess
                initial_guess = getattr(
                    problem,
                    "initial_condition",
                    np.zeros(
                        int(getattr(fem_solver.fes, "ndof", problem.n_nodes)),
                        dtype=np.complex128,
                    ),
                )
                if not np.iscomplexobj(initial_guess):
                    initial_guess = np.asarray(initial_guess, dtype=np.float64).astype(
                        np.complex128
                    )

                # Create graph creator for this specific problem
                graph_creator = GraphCreator(
                    mesh=problem.mesh,
                    n_neighbors=2,
                    dirichlet_names=problem.mesh_config.dirichlet_boundaries,
                    neumann_names=getattr(
                        problem.mesh_config, "neumann_boundaries", []
                    ),
                    connectivity_method="fem",
                    fes=fem_solver.fes,
                )

                # Build graph and make prediction
                data, aux = graph_creator.create_graph(
                    T_current=initial_guess,
                    t_scalar=0.0,
                    material_node_field=getattr(problem, "material_field", None),
                    neumann_values=getattr(problem, "neumann_values_array", None),
                    dirichlet_values=getattr(problem, "dirichlet_values_array", None),
                    source_values=getattr(problem, "source_function", None),
                )

                free_data, node_mapping, free_aux = (
                    graph_creator.create_free_node_subgraph(data=data, aux=aux)
                )
                free_data = free_data.to(self.device)

                prediction_free = self.model.forward(free_data)

                # Compute physics loss for validation
                physics_loss = self.compute_physics_informed_loss(
                    prediction_free,
                    problem_idx,
                    aux,
                    node_mapping,
                )

                total_loss += physics_loss.item()
                num_validations += 1

        self.model.train()
        return total_loss / num_validations if num_validations > 0 else 0.0

    def predict(self, problem_idx=0):
        """Predict steady-state solution for a specific problem."""
        self.model.eval()

        problem = self.problems[problem_idx]
        fem_solver: FEMSolverEM = self.all_fem_solvers[problem_idx]

        # Initial guess (use complex zeros for consistent graph feature dims)
        initial_guess = getattr(
            problem,
            "initial_condition",
            np.zeros(
                int(getattr(fem_solver.fes, "ndof", problem.n_nodes)),
                dtype=np.complex128,
            ),
        )
        if not np.iscomplexobj(initial_guess):
            initial_guess = np.asarray(initial_guess, dtype=np.float64).astype(
                np.complex128
            )

        graph_creator = GraphCreator(
            mesh=problem.mesh,
            n_neighbors=2,
            dirichlet_names=problem.mesh_config.dirichlet_boundaries,
            neumann_names=getattr(problem.mesh_config, "neumann_boundaries", []),
            connectivity_method="fem",
            fes=fem_solver.fes,
        )

        # Get the values for this problem
        neumann_vals = getattr(problem, "neumann_values_array", None)
        dirichlet_vals = getattr(problem, "dirichlet_values_array", None)
        material_field = getattr(problem, "material_field", None)
        source_vals = getattr(problem, "source_function", None)

        with torch.no_grad():
            # Build graph
            data, aux = graph_creator.create_graph(
                T_current=initial_guess,
                t_scalar=0.0,
                material_node_field=material_field,
                neumann_values=neumann_vals,
                dirichlet_values=dirichlet_vals,
                source_values=source_vals,
            )

            free_graph, node_mapping, free_aux = (
                graph_creator.create_free_node_subgraph(data, aux)
            )
            free_data = free_graph.to(self.device)

            # Predict steady-state solution
            prediction_free = self.model.forward(free_data)

            # Reconstruct full solution as complex array
            free_idx = node_mapping["free_to_original"].detach().cpu().numpy()
            dirichlet_mask = aux["dirichlet_mask"].cpu().numpy()

            n_total = int(node_mapping.get("n_original", problem.n_nodes))
            prediction_full = np.zeros(n_total, dtype=np.complex128)
            pred = prediction_free.detach().cpu().numpy()  # [N_free, 2]
            prediction_full[free_idx] = pred[:, 0] + 1j * pred[:, 1]
            if dirichlet_vals is not None:
                prediction_full[dirichlet_mask] = dirichlet_vals[dirichlet_mask]

        return prediction_full

    def evaluate_with_ground_truth(self, problem_indices=None):
        """
        Evaluate the trained model against ground truth FEM solution.

        This generates ground truth data and compares with PIMGN predictions.
        """
        if problem_indices is None:
            problem_indices = range(len(self.problems))

        print("Evaluating PIMGN-EM on multiple problems...")

        all_errors = []
        all_predictions = []
        all_ground_truth = []

        for i, problem_idx in enumerate(problem_indices):
            print(f"Evaluating problem {problem_idx + 1}...")

            problem = self.problems[problem_idx]
            fem_solver = self.all_fem_solvers[problem_idx]

            # Generate ground truth using FEM
            ground_truth = fem_solver.solve(problem)

            # Get prediction from trained model
            prediction = self.predict(problem_idx=problem_idx)

            # Compute error
            if len(prediction) == len(ground_truth):
                l2_error = np.linalg.norm(prediction - ground_truth) / np.linalg.norm(
                    ground_truth
                )
                max_error = np.max(np.abs(prediction - ground_truth))
                print(
                    f"Problem {problem_idx + 1} - L2 error: {l2_error:.6f}, Max error: {max_error:.6f}"
                )
                all_errors.append(l2_error)
                all_predictions.append(prediction)
                all_ground_truth.append(ground_truth)
            else:
                print(
                    f"Warning: Size mismatch: pred={len(prediction)}, true={len(ground_truth)}"
                )

        # Overall statistics
        if all_errors:
            avg_error = np.mean(all_errors)
            print(f"\nOverall - Average L2 error: {avg_error:.6f}")

            self.logger.log_evaluation(all_errors, "l2_errors_per_problem")
            self.logger.log_evaluation(avg_error, "mean_l2_error")

        return all_predictions, all_ground_truth, all_errors

    def evaluate(self, problem_indices=None):
        """Evaluate the trained model on specified problems."""
        if problem_indices is None:
            problem_indices = range(len(self.problems))

        print("Evaluating model...")

        all_errors = []
        for prob_idx in problem_indices:
            print(f"Evaluating problem {prob_idx + 1}...")

            # Get prediction for this problem
            prediction = self.predict(problem_idx=prob_idx)

            # If we have ground truth, compute errors
            if self.all_ground_truth is not None:
                ground_truth = self.all_ground_truth[prob_idx]

                # Compute error
                l2_error = np.linalg.norm(prediction - ground_truth) / np.linalg.norm(
                    ground_truth
                )
                max_error = np.max(np.abs(prediction - ground_truth))
                all_errors.append(l2_error)
                print(
                    f"Problem {prob_idx + 1} - L2 error: {l2_error:.6f}, Max error: {max_error:.6f}"
                )
            else:
                print(
                    f"Problem {prob_idx + 1} - Prediction completed (no ground truth for comparison)"
                )

        # Overall statistics
        if all_errors:
            avg_error = np.mean(all_errors)
            print(f"\nOverall - Average L2 error: {avg_error:.6f}")

            self.logger.log_evaluation(all_errors, "l2_errors_per_problem")
            self.logger.log_evaluation(avg_error, "mean_l2_error")

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
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")


def _run_single_problem_experiment(problem, config, experiment_name: str):
    print("=" * 60)
    print(f"PIMGN TEST - {experiment_name.upper()}")
    print("=" * 60)

    save_path = config.get("save_dir", "results/physics_informed")
    os.makedirs(save_path, exist_ok=True)

    trainer = PIMGNTrainerEM([problem], config)

    print("\nStarting physics-informed training...")
    trainer.train(train_problems_indices=[0])

    print("\nEvaluating trained PIMGN...")
    last_residuals = trainer.last_residuals
    # last_residuals are complex for EM; JSON can't serialize complex numbers.
    trainer.logger.log_evaluation(
        np.abs(last_residuals).tolist(), "residuals_per_time_step_abs"
    )
    pos_data = trainer.problems[0].graph_data.pos.numpy()
    try:
        predictions, ground_truth, errors = trainer.evaluate_with_ground_truth(
            problem_indices=[0]
        )

        print("Exporting results...")
        trainer.all_fem_solvers[0].export_to_vtk(
            ground_truth[0],
            predictions[0],
            filename=f"{save_path}/vtk/result",
        )
    except Exception as e:
        print(f"Ground truth evaluation failed: {e}")

    print("Physics-Informed MeshGraphNet test completed!")
    print(f"Results saved to: {save_path}")
    trainer.save_logs()

    model_path = f"{save_path}/pimgn_trained_model.pth"
    trainer.save_checkpoint(model_path, epoch=config["epochs"] - 1)
    print(f"Trained model saved to: {model_path}")


def train_pimgn_on_single_problem(resume_from: str = None):
    problem = create_em_problem()
    config = {
        "epochs": 100,
        "lr": 1e-2,
        "generate_ground_truth_for_validation": False,
        "save_dir": "results/physics_informed/test_em_problem",
        "resume_from": resume_from,  # Path to checkpoint to resume from
    }
    _run_single_problem_experiment(problem, config, "Second order EM")


if __name__ == "__main__":
    train_pimgn_on_single_problem()
