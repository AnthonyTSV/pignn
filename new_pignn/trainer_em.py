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
from graph_creator_em import GraphCreatorEM
from containers import MeshConfig, MeshProblem, MeshProblemEM
from torch_geometric.data import Data

from train_problems import create_em_problem, create_em_problem_complex, create_em_mixed


class PIMGNTrainerEM:
    """Trainer for Physics-Informed MeshGraphNet for Steady-State Electromagnetic Problems."""

    def __init__(self, problems: List[MeshProblemEM], config: dict):
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
        self.is_complex = any(getattr(problem, "complex", False) for problem in problems)

        # Prepare sample data to determine input/output dimensions using first problem
        first_problem = problems[0]
        first_fes = self.all_fem_solvers[0].fes
        graph_creator = GraphCreatorEM(
            mesh=first_problem.mesh,
            n_neighbors=2,
            dirichlet_names=first_problem.mesh_config.dirichlet_boundaries,
        )

        # Create sample graph to get dimensions
        material_field = getattr(first_problem, "material_field", None)
        sigma_field = getattr(first_problem, "sigma_field", None)
        dirichlet_vals = getattr(first_problem, "dirichlet_values_array", None)
        current_density = getattr(first_problem, "current_density_field", None)
        coil_node_mask = getattr(first_problem, "coil_node_mask", None)

        sample_data, aux = graph_creator.create_graph(
            A_current=None,
            material_node_field=material_field,
            sigma_field=sigma_field,
            dirichlet_values=dirichlet_vals,
            current_density=current_density,
            coil_node_mask=coil_node_mask,
        )

        free_node_data, mapping, new_aux = graph_creator.create_free_node_subgraph(
            data=sample_data, aux=aux
        )

        input_dim_node = free_node_data.x.shape[1]
        input_dim_edge = free_node_data.edge_attr.shape[1]
        output_dim = 2 if self.is_complex else 1  # Real and Imaginary parts

        print(
            f"Input dimensions - Node: {input_dim_node}, Edge: {input_dim_edge}, Output: {output_dim} (complex)"
        )
        print("Steady-state electromagnetic problem")
        print(f"Training on {len(problems)} problems")

        # Create MeshGraphNet model for steady-state prediction
        self.model = MeshGraphNet(
            input_dim_node=input_dim_node,
            input_dim_edge=input_dim_edge,
            hidden_dim=128,
            output_dim=output_dim,
            # input_dim_global=2,
            num_layers=12,
            complex_em=self.is_complex,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.95
        )

        # Training history
        self.losses = []
        self.val_losses = []

        # Populated during training when physics loss is computed.
        # Can remain None if training loop is skipped (e.g., resume at final epoch).
        self.last_residuals = None
        
        # Component losses for mixed A-phi formulation (for logging)
        self._last_loss_A = 0.0
        self._last_loss_phi = 0.0

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
                if problem.complex:
                    ground_truth = self.all_fem_solvers[i].solve_mixed_em(problem)
                else:
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

        For mixed A-φ formulation:
        - A is predicted for all nodes
        - φ is predicted for all nodes but only used for coil nodes
        
        The FEM DOF structure is [A_dofs | phi_dofs] where:
        - A_dofs: n_dofs_A values (one per mesh vertex)
        - phi_dofs: n_dofs_phi values (one per coil vertex only)
        """
        problem: MeshProblemEM = self.problems[problem_idx]
        fem_solver: FEMSolverEM = self.all_fem_solvers[problem_idx]

        free_to_original = node_mapping["free_to_original"].to(self.device)
        n_total = int(node_mapping.get("n_original"))

        # Get Dirichlet mask and values for enforcing boundary conditions
        dirichlet_mask = aux["dirichlet_mask"].to(self.device)
        dirichlet_vals = getattr(problem, "dirichlet_values_array", None)
        if dirichlet_vals is not None:
            dirichlet_vals_tensor = torch.tensor(
                dirichlet_vals, dtype=torch.float64, device=self.device
            )
        else:
            dirichlet_vals_tensor = None
            
        if self.is_complex:
            # Reconstruct full A prediction from free nodes
            # prediction_free has shape [N_free_nodes, 4] for [Areal, Aimag, phireal, phiimag]
            prediction_full_Areal = torch.zeros(
                n_total, dtype=torch.float64, device=self.device
            )
            prediction_full_Aimag = torch.zeros(
                n_total, dtype=torch.float64, device=self.device
            )
            prediction_full_Areal[free_to_original] = prediction_free[:, 0].to(dtype=torch.float64)
            prediction_full_Aimag[free_to_original] = prediction_free[:, 1].to(dtype=torch.float64)

            # Enforce Dirichlet BCs on A prediction
            if dirichlet_vals_tensor is not None:
                dirichlet_vals_float = dirichlet_vals_tensor.to(dtype=torch.float64)
                prediction_full_Areal[dirichlet_mask] = dirichlet_vals_float[dirichlet_mask].real if dirichlet_vals_float.is_complex() else dirichlet_vals_float[dirichlet_mask]
                prediction_full_Aimag[dirichlet_mask] = 0

            # For phi, we need to extract predictions only for coil nodes
            # First reconstruct full phi predictions (on all graph nodes)
            prediction_full_phireal = torch.zeros(
                n_total, dtype=torch.float64, device=self.device
            )
            prediction_full_phiimag = torch.zeros(
                n_total, dtype=torch.float64, device=self.device
            )
            prediction_full_phireal[free_to_original] = prediction_free[:, 2].to(dtype=torch.float64)
            prediction_full_phiimag[free_to_original] = prediction_free[:, 3].to(dtype=torch.float64)
            
            # Extract phi values only for coil nodes (in the order expected by FEM)
            # fem_solver.coil_node_indices maps phi_dof_idx -> graph_node_idx
            coil_node_indices = torch.tensor(
                fem_solver.coil_node_indices, dtype=torch.long, device=self.device
            )
            phi_real_dofs = prediction_full_phireal[coil_node_indices]  # [n_dofs_phi]
            phi_imag_dofs = prediction_full_phiimag[coil_node_indices]  # [n_dofs_phi]

            # Use balanced loss computation that separates A and phi
            phi_weight = self.config.get("phi_weight", 1.0)
            loss_A, loss_phi, residual = fem_solver.compute_mixed_energy_norm_loss_balanced(
                prediction_full_Areal,   # [n_dofs_A]
                prediction_full_Aimag,   # [n_dofs_A]
                phi_real_dofs,           # [n_dofs_phi]
                phi_imag_dofs,           # [n_dofs_phi]
                phi_weight=phi_weight,
            )
            # Store component losses for logging
            self._last_loss_A = loss_A.item()
            self._last_loss_phi = loss_phi.item()
        else:
            prediction_full_real = torch.zeros(
                n_total, dtype=torch.float64, device=self.device
            )
            prediction_full_real[free_to_original] = prediction_free[:, 0]

            residual = fem_solver.compute_energy_loss(
                prediction_full_real,
            )

        return residual

    def train_step(self, problem_idx, prediction=None):
        """
        Compute physics loss for steady-state problem.
        Returns the loss value and prediction.
        """
        problem: MeshProblemEM = self.problems[problem_idx]
        fem_solver: FEMSolverEM = self.all_fem_solvers[problem_idx]

        # Create graph creator for this specific problem
        graph_creator = GraphCreatorEM(
            mesh=problem.mesh,
            n_neighbors=2,
            dirichlet_names=problem.mesh_config.dirichlet_boundaries,
        )

        # Get the values for this problem
        dirichlet_vals = getattr(problem, "dirichlet_values_array", None)
        material_field = getattr(problem, "material_field", None)
        sigma_field = getattr(problem, "sigma_field", None)
        current_density = getattr(problem, "current_density_field", None)
        coil_node_mask = getattr(problem, "coil_node_mask", None)

        self.optimizer.zero_grad()

        # Build full graph
        data, aux = graph_creator.create_graph(
            A_current=None,
            material_node_field=material_field,
            sigma_field=sigma_field,
            dirichlet_values=dirichlet_vals,
            current_density=current_density,
            coil_node_mask=coil_node_mask,
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
        prediction_np = prediction_free.detach().cpu().numpy()  # [N_free, 4]

        # Reconstruct full state as complex array
        n_total = int(node_mapping.get("n_original", problem.n_nodes))
        prediction_full = np.zeros(n_total, dtype=np.float64)
        free_to_original = node_mapping["free_to_original"].cpu().numpy()
        prediction_full[free_to_original] = prediction_np[:, 0]
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

        problem = self.problems[train_problems_indices[0]]
        # Initialize prediction with zeros instead of random noise
        # Random noise can lead to huge residuals (due to large system matrix entries)
        # which can cause the network to collapse to zero output.
        prediction = np.zeros(problem.n_nodes, dtype=np.float64)

        # Generate ground truth once for periodic evaluation
        print("Generating ground truth for evaluation...")
        eval_ground_truth = {}
        for prob_idx in train_problems_indices:
            if self.problems[prob_idx].complex:
                gfA, gfPhi, r1 = self.all_fem_solvers[prob_idx].solve_mixed_em(
                    self.problems[prob_idx]
                )
                # Extract DOF values from GridFunctions
                A_dofs = np.array(gfA.vec, dtype=np.complex128)
                phi_dofs = np.array(gfPhi.vec, dtype=np.complex128)
                eval_ground_truth[prob_idx] = np.concatenate([A_dofs, phi_dofs])
            else:
                eval_ground_truth[prob_idx] = self.all_fem_solvers[prob_idx].solve(
                    self.problems[prob_idx]
                )
        print("Ground truth generated.")

        for epoch in range(self.start_epoch, self.config["epochs"]):
            epoch_start = time.time()

            physics_loss, prediction_next = self.train_step(0, prediction=prediction)
            prediction = prediction_next

            self.losses.append(physics_loss)

            # Validation (if validation problems are provided and ground truth is available)
            val_loss = None
            if val_problems_indices and self.all_ground_truth is not None:
                val_loss = self.validate(val_problems_indices)
                self.val_losses.append(val_loss)

            elapsed = time.time() - epoch_start
            self.logger.log_epoch(epoch, physics_loss, val_loss, elapsed)

            if epoch % 10 == 0:
                elapsed = time.time() - epoch_start
                val_str = f" | Val Loss: {val_loss:.3e}" if val_loss is not None else ""
                # Show component losses for complex/mixed EM problems
                if self.is_complex and hasattr(self, '_last_loss_A'):
                    loss_A_str = f" | L_A: {self._last_loss_A:.3e} | L_φ: {self._last_loss_phi:.3e}"
                else:
                    loss_A_str = ""
                print(
                    f"Epoch {epoch+1:4d} | Train Loss: {physics_loss:.3e}{loss_A_str}{val_str} | Time: {elapsed:.2f}s"
                )

            # Evaluate L2 error every 100 epochs
            if (epoch + 1) % 100 == 0:
                self.model.eval()
                for prob_idx in train_problems_indices:
                    pred = self.predict(problem_idx=prob_idx)
                    gt = eval_ground_truth[prob_idx]
                    if self.is_complex:
                        # pred and gt are both complex arrays: [A_complex | phi_complex]
                        # Compare them directly (no real/imag split needed)
                        fem_s = self.all_fem_solvers[prob_idx]
                        n_A = fem_s.n_dofs_A
                        l2_A = np.linalg.norm(pred[:n_A] - gt[:n_A]) / (np.linalg.norm(gt[:n_A]) + 1e-30)
                        l2_phi = np.linalg.norm(pred[n_A:] - gt[n_A:]) / (np.linalg.norm(gt[n_A:]) + 1e-30)
                        l2_total = np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-30)
                        print(
                            f"  [Eval] Epoch {epoch+1} | Problem {prob_idx+1} "
                            f"| L2 total: {l2_total:.6e} | L2 A: {l2_A:.6e} | L2 phi: {l2_phi:.6e}"
                        )
                    else:
                        if len(pred) == len(gt):
                            l2_error = np.linalg.norm(pred - gt) / np.linalg.norm(gt)
                            print(
                                f"  [Eval] Epoch {epoch+1} | Problem {prob_idx+1} | L2 Error: {l2_error:.6e}"
                            )
                self.model.train()

            self.scheduler.step()

        print("Physics-Informed MeshGraphNet EM training completed!")

    def validate(self, val_problems_indices):
        raise NotImplementedError("Validation not implemented")

    def predict(self, problem_idx=0):
        """Predict steady-state solution for a specific problem."""
        self.model.eval()

        problem = self.problems[problem_idx]
        fem_solver: FEMSolverEM = self.all_fem_solvers[problem_idx]

        graph_creator = GraphCreatorEM(
            mesh=problem.mesh,
            n_neighbors=2,
            dirichlet_names=problem.mesh_config.dirichlet_boundaries,
        )

        # Get the values for this problem
        dirichlet_vals = getattr(problem, "dirichlet_values_array", None)
        material_field = getattr(problem, "material_field", None)
        sigma_field = getattr(problem, "sigma_field", None)
        current_density = getattr(problem, "current_density_field", None)
        coil_node_mask = getattr(problem, "coil_node_mask", None)

        with torch.no_grad():
            # Build graph
            data, aux = graph_creator.create_graph(
                A_current=None,
                material_node_field=material_field,
                sigma_field=sigma_field,
                dirichlet_values=dirichlet_vals,
                current_density=current_density,
                coil_node_mask=coil_node_mask,
            )

            free_graph, node_mapping, free_aux = (
                graph_creator.create_free_node_subgraph(data, aux)
            )
            free_data = free_graph.to(self.device)

            # Predict steady-state solution
            prediction_free = self.model.forward(free_data)

            # Reconstruct full solution
            free_idx = node_mapping["free_to_original"].detach().cpu().numpy()
            dirichlet_mask = aux["dirichlet_mask"].cpu().numpy()

            n_total = int(node_mapping.get("n_original", problem.n_nodes))
            
            if problem.complex:
                # Reconstruct A predictions for all nodes
                pred = prediction_free.detach().cpu().numpy()  # [N_free, 4]
                
                prediction_full_Areal = np.zeros(n_total, dtype=np.float64)
                prediction_full_Aimag = np.zeros(n_total, dtype=np.float64)
                prediction_full_Areal[free_idx] = pred[:, 0]
                prediction_full_Aimag[free_idx] = pred[:, 1]
                
                # Enforce Dirichlet BCs on A
                if dirichlet_vals is not None:
                    dirichlet_vals_real = dirichlet_vals[dirichlet_mask].real if np.iscomplexobj(dirichlet_vals) else dirichlet_vals[dirichlet_mask]
                    prediction_full_Areal[dirichlet_mask] = dirichlet_vals_real
                    prediction_full_Aimag[dirichlet_mask] = 0
                
                # Reconstruct phi predictions for all nodes (will extract coil-only later)
                prediction_full_phireal = np.zeros(n_total, dtype=np.float64)
                prediction_full_phiimag = np.zeros(n_total, dtype=np.float64)
                prediction_full_phireal[free_idx] = pred[:, 2]
                prediction_full_phiimag[free_idx] = pred[:, 3]
                
                # Extract phi values only for coil nodes (in FEM DOF order)
                coil_node_indices = fem_solver.coil_node_indices
                phi_real_dofs = prediction_full_phireal[coil_node_indices]  # [n_dofs_phi]
                phi_imag_dofs = prediction_full_phiimag[coil_node_indices]  # [n_dofs_phi]
                
                # Build full DOF vector in FEM format: [A_dofs | phi_dofs]
                # A is complex: A_real + 1j * A_imag
                A_complex = prediction_full_Areal + 1j * prediction_full_Aimag
                phi_complex = phi_real_dofs + 1j * phi_imag_dofs
                
                # Return as concatenated DOF vector matching FEM structure
                prediction_full = np.concatenate([A_complex, phi_complex])
            else:
                prediction_full = np.zeros(n_total, dtype=np.float64)
                pred = prediction_free.detach().cpu().numpy()  # [N_free, 2]
                prediction_full[free_idx] = pred[:, 0]
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
            if problem.complex:
                gfA, gfPhi, r1 = fem_solver.solve_mixed_em(problem)
                # Extract DOF values from GridFunctions to match prediction format
                A_dofs = np.array(gfA.vec, dtype=np.complex128)
                phi_dofs = np.array(gfPhi.vec, dtype=np.complex128)
                ground_truth = np.concatenate([A_dofs, phi_dofs])
            else:
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


def _run_single_problem_experiment(problem: MeshProblemEM, config, experiment_name: str):
    print("=" * 60)
    print(f"PIMGN TEST - {experiment_name.upper()}")
    print("=" * 60)

    save_path = config.get("save_dir", "results/physics_informed")
    os.makedirs(save_path, exist_ok=True)

    trainer = PIMGNTrainerEM([problem], config)

    print("\nStarting physics-informed training...")
    trainer.train(train_problems_indices=[0])

    print("\nEvaluating trained PIMGN...")
    last_residuals = getattr(trainer, "last_residuals", None)
    if last_residuals is not None:
        # last_residuals are complex for EM
        trainer.logger.log_evaluation(
            np.atleast_1d(np.abs(last_residuals)).tolist(),
            "residuals_per_time_step_abs",
        )
    else:
        print(
            "Note: No `last_residuals` available (training may have been skipped due to resume epoch >= configured epochs)."
        )
    pos_data = trainer.problems[0].graph_data.pos.numpy()
    try:
        predictions, ground_truth, errors = trainer.evaluate_with_ground_truth(
            problem_indices=[0]
        )

        print("Exporting results...")
        if not problem.complex:
            trainer.all_fem_solvers[0].export_to_vtk(
                ground_truth[0],
                predictions[0],
                filename=f"{save_path}/vtk/result",
            )
        else:
            trainer.all_fem_solvers[0].export_to_vtk_mixed(
                ground_truth[0],
                predictions[0],
                filename=f"{save_path}/vtk/result_mixed",
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
        "epochs": 2000,
        "lr": 1e-4,
        "generate_ground_truth_for_validation": False,
        "save_dir": "results/physics_informed/test_em_problem",
        "resume_from": resume_from,  # Path to checkpoint to resume from
    }
    _run_single_problem_experiment(problem, config, "First order EM")

def train_pimgn_em_complex(resume_from: str = None):
    problem = create_em_problem_complex()
    config = {
        "epochs": 10000,
        "lr": 1e-4,
        "generate_ground_truth_for_validation": False,
        "save_dir": "results/physics_informed/test_em_problem_complex",
        "resume_from": resume_from,  # Path to checkpoint to resume from
    }
    _run_single_problem_experiment(problem, config, "First order EM Complex")

def train_pimgn_em_mixed(resume_from: str = None):
    problem = create_em_mixed()
    config = {
        "epochs": 10000,
        "lr": 1e-4,
        "generate_ground_truth_for_validation": False,
        "save_dir": "results/physics_informed/test_em_problem_mixed",
        "resume_from": resume_from,  # Path to checkpoint to resume from
        # Phi weight for balanced loss: increase if phi doesn't converge
        # phi_weight=10 means phi loss contributes 10x more to total loss
        "phi_weight": 10.0,
    }
    _run_single_problem_experiment(problem, config, "First order EM Mixed")

if __name__ == "__main__":
    # train_pimgn_on_single_problem()
    # train_pimgn_em_complex()
    train_pimgn_em_mixed()
