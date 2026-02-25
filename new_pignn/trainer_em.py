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
        self.is_complex = any(
            getattr(problem, "complex", False) for problem in problems
        )
        self.is_mixed = any(getattr(problem, "mixed", False) for problem in problems)

        # Pre-create GraphCreatorEM instances for all problems (avoid re-creation in train_step)
        self.all_graph_creators: List[GraphCreatorEM] = []
        for problem in problems:
            gc = GraphCreatorEM(
                mesh=problem.mesh,
                n_neighbors=2,
                dirichlet_names=problem.mesh_config.dirichlet_boundaries,
                r_star=problem.r_star
            )
            self.all_graph_creators.append(gc)

        # Prepare sample data to determine input/output dimensions using first problem
        first_problem = problems[0]
        first_fes = self.all_fem_solvers[0].fes
        graph_creator = self.all_graph_creators[0]

        # Create sample graph to get dimensions
        material_field = getattr(first_problem, "material_field", None)
        sigma_field = getattr(first_problem, "sigma_field", None)
        dirichlet_vals = getattr(first_problem, "dirichlet_values_array", None)
        current = getattr(first_problem, "I_coil", None)
        coil_node_mask = getattr(first_problem, "coil_node_mask", None)

        sample_data, aux = graph_creator.create_graph(
            A_current=None,
            material_node_field=material_field,
            sigma_field=sigma_field,
            dirichlet_values=dirichlet_vals,
            current=current,
            coil_node_mask=coil_node_mask,
        )

        free_node_data, mapping, new_aux = graph_creator.create_free_node_subgraph(
            data=sample_data, aux=aux
        )

        input_dim_node = free_node_data.x.shape[1]
        input_dim_edge = free_node_data.edge_attr.shape[1]
        
        if self.is_mixed:
            output_dim = 4  # [A_real, A_imag, phi_real, phi_imag]
        elif self.is_complex:
            output_dim = 2  # [A_real, A_imag]
        else:
            output_dim = 1  # [A_real only]

        print(
            f"Input dimensions - Node: {input_dim_node}, Edge: {input_dim_edge}, Output: {output_dim} (complex={self.is_complex}, mixed={self.is_mixed})"
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
            mixed_em=self.is_mixed
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
        self._last_data_loss = 0.0
        self._ground_truth_tensors = {}  # For data supervision

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
                    ground_truth = self.all_fem_solvers[i].solve(problem)
                elif problem.mixed:
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

        For complex A formulation:
        - A is predicted for all nodes

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

        if self.is_mixed:
            # Reconstruct full A prediction from free nodes
            # prediction_free has shape [N_free_nodes, 4] for [Areal, Aimag, phireal, phiimag]
            prediction_full_Areal = torch.zeros(
                n_total, dtype=torch.float64, device=self.device
            )
            prediction_full_Aimag = torch.zeros(
                n_total, dtype=torch.float64, device=self.device
            )
            prediction_full_Areal[free_to_original] = prediction_free[:, 0].to(
                dtype=torch.float64
            )
            prediction_full_Aimag[free_to_original] = prediction_free[:, 1].to(
                dtype=torch.float64
            )

            # Enforce Dirichlet BCs on A prediction
            if dirichlet_vals_tensor is not None:
                dirichlet_vals_float = dirichlet_vals_tensor.to(dtype=torch.float64)
                prediction_full_Areal[dirichlet_mask] = (
                    dirichlet_vals_float[dirichlet_mask].real
                    if dirichlet_vals_float.is_complex()
                    else dirichlet_vals_float[dirichlet_mask]
                )
                prediction_full_Aimag[dirichlet_mask] = 0

            # For phi, we need to extract predictions only for coil nodes
            # First reconstruct full phi predictions (on all graph nodes)
            prediction_full_phireal = torch.zeros(
                n_total, dtype=torch.float64, device=self.device
            )
            prediction_full_phiimag = torch.zeros(
                n_total, dtype=torch.float64, device=self.device
            )
            prediction_full_phireal[free_to_original] = prediction_free[:, 2].to(
                dtype=torch.float64
            )
            prediction_full_phiimag[free_to_original] = prediction_free[:, 3].to(
                dtype=torch.float64
            )

            # Extract phi values only for coil nodes (in the order expected by FEM)
            # fem_solver.coil_node_indices maps phi_dof_idx -> graph_node_idx
            coil_node_indices = torch.tensor(
                fem_solver.coil_node_indices, dtype=torch.long, device=self.device
            )
            phi_real_dofs = prediction_full_phireal[coil_node_indices]  # [n_dofs_phi]
            phi_imag_dofs = prediction_full_phiimag[coil_node_indices]  # [n_dofs_phi]

            # Use balanced loss computation that separates A and phi
            phi_weight = self.config.get("phi_weight", 1.0)
            loss_A, loss_phi, residual = (
                fem_solver.compute_mixed_energy_norm_loss_balanced(
                    prediction_full_Areal,  # [n_dofs_A]
                    prediction_full_Aimag,  # [n_dofs_A]
                    phi_real_dofs,  # [n_dofs_phi]
                    phi_imag_dofs,  # [n_dofs_phi]
                    phi_weight=phi_weight,
                )
            )
            # Store component losses for logging
            self._last_loss_A = loss_A.item()
            self._last_loss_phi = loss_phi.item()
        elif self.is_complex:
            prediction_full_Areal = torch.zeros(
                n_total, dtype=torch.float64, device=self.device
            )
            prediction_full_Aimag = torch.zeros(
                n_total, dtype=torch.float64, device=self.device
            )
            prediction_full_Areal[free_to_original] = prediction_free[:, 0].to(
                dtype=torch.float64
            )
            prediction_full_Aimag[free_to_original] = prediction_free[:, 1].to(
                dtype=torch.float64
            )
            # Enforce Dirichlet BCs on A prediction
            if dirichlet_vals_tensor is not None:
                dirichlet_vals_float = dirichlet_vals_tensor.to(dtype=torch.float64)
                prediction_full_Areal[dirichlet_mask] = (
                    dirichlet_vals_float[dirichlet_mask].real
                    if dirichlet_vals_float.is_complex()
                    else dirichlet_vals_float[dirichlet_mask]
                )
                prediction_full_Aimag[dirichlet_mask] = 0
            residual = fem_solver.compute_complex_energy_norm_loss(
                prediction_full_Areal,
                prediction_full_Aimag,
                normalize="rhs",
            )
        else:
            prediction_full_real = torch.zeros(
                n_total, dtype=torch.float64, device=self.device
            )
            prediction_full_real[free_to_original] = prediction_free[:, 0]

            residual = fem_solver.compute_energy_loss(
                prediction_full_real,
            )

        return residual

    def train_step(
        self, problem_idx, prediction=None, ground_truth=None, data_weight=0.0
    ):
        """
        Compute physics loss for steady-state problem with optional data supervision.

        Args:
            problem_idx: Index of the problem to train on
            prediction: Previous prediction (unused, kept for compatibility)
            ground_truth: Dict with 'A_real', 'A_imag', 'phi_real', 'phi_imag' tensors
            data_weight: Weight for data supervision loss (0 = pure physics)

        Returns the total loss value and prediction.
        """
        problem: MeshProblemEM = self.problems[problem_idx]
        fem_solver: FEMSolverEM = self.all_fem_solvers[problem_idx]

        # Use pre-created graph creator for this problem
        graph_creator = self.all_graph_creators[problem_idx]

        # Get the values for this problem
        dirichlet_vals = getattr(problem, "dirichlet_values_array", None)
        material_field = getattr(problem, "material_field", None)
        sigma_field = getattr(problem, "sigma_field", None)
        current = getattr(problem, "I_coil", None)
        coil_node_mask = getattr(problem, "coil_node_mask", None)

        self.optimizer.zero_grad()

        # Build full graph
        data, aux = graph_creator.create_graph(
            A_current=None,
            material_node_field=material_field,
            sigma_field=sigma_field,
            dirichlet_values=dirichlet_vals,
            current=current,
            coil_node_mask=coil_node_mask,
        )

        # Create free node subgraph (only non-Dirichlet nodes)
        free_data, node_mapping, free_aux = graph_creator.create_free_node_subgraph(
            data=data, aux=aux
        )
        free_data = free_data.to(self.device)

        # Forward pass - get prediction for FREE nodes [N_free_nodes, 4] (A_real, A_imag, phi_real, phi_imag)
        prediction_free = self.model.forward(free_data)

        # Compute physics-informed loss for steady-state
        physics_loss = self.compute_physics_informed_loss(
            prediction_free,
            problem_idx,
            aux,
            node_mapping,
        )

        # Compute data supervision loss if ground truth is provided
        data_loss = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        if ground_truth is not None and data_weight > 0:
            free_to_original = node_mapping["free_to_original"].to(self.device)
            n_total = int(node_mapping.get("n_original"))

            # Reconstruct full predictions
            pred_A_real = torch.zeros(n_total, dtype=torch.float64, device=self.device)
            pred_A_imag = torch.zeros(n_total, dtype=torch.float64, device=self.device)
            pred_A_real[free_to_original] = prediction_free[:, 0].to(
                dtype=torch.float64
            )
            pred_A_imag[free_to_original] = prediction_free[:, 1].to(
                dtype=torch.float64
            )

            # A supervision: compare on all nodes
            gt_A_real = ground_truth["A_real"]
            gt_A_imag = ground_truth["A_imag"]

            # Normalize by ground truth norm to make loss scale-invariant
            gt_A_norm = torch.sqrt((gt_A_real**2 + gt_A_imag**2).sum()).clamp_min(1e-10)
            data_loss_A = (
                (pred_A_real - gt_A_real) ** 2 + (pred_A_imag - gt_A_imag) ** 2
            ).mean() / (gt_A_norm**2 / n_total)

            # Phi supervision: compare on coil nodes only
            pred_phi_real = torch.zeros(
                n_total, dtype=torch.float64, device=self.device
            )
            pred_phi_imag = torch.zeros(
                n_total, dtype=torch.float64, device=self.device
            )
            pred_phi_real[free_to_original] = prediction_free[:, 2].to(
                dtype=torch.float64
            )
            pred_phi_imag[free_to_original] = prediction_free[:, 3].to(
                dtype=torch.float64
            )

            coil_idx = ground_truth["coil_node_indices"]
            gt_phi_real = ground_truth["phi_real"]
            gt_phi_imag = ground_truth["phi_imag"]

            # Extract predicted phi at coil nodes
            pred_phi_real_coil = pred_phi_real[coil_idx]
            pred_phi_imag_coil = pred_phi_imag[coil_idx]

            gt_phi_norm = torch.sqrt(
                (gt_phi_real**2 + gt_phi_imag**2).sum()
            ).clamp_min(1e-10)
            n_phi = len(coil_idx)
            data_loss_phi = (
                (pred_phi_real_coil - gt_phi_real) ** 2
                + (pred_phi_imag_coil - gt_phi_imag) ** 2
            ).mean() / (gt_phi_norm**2 / n_phi)

            data_loss = data_loss_A + data_loss_phi
            self._last_data_loss = data_loss.item()

        # Combined loss
        total_loss = physics_loss + data_weight * data_loss

        total_loss.backward()
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

        return total_loss.item(), prediction_full

    def train(self, train_problems_indices, val_problems_indices=None):
        """Main training loop for steady-state EM problems with multiple problems."""
        print(f"Starting PIMGN-EM training on {self.device}")
        print(
            f"Training on {len(train_problems_indices)} problems: {train_problems_indices}"
        )
        if val_problems_indices:
            print(
                f"Validation on {len(val_problems_indices)} problems: {val_problems_indices}"
            )

        # Initialize per-problem predictions with zeros
        predictions = {}
        for prob_idx in train_problems_indices:
            problem = self.problems[prob_idx]
            predictions[prob_idx] = np.zeros(problem.n_nodes, dtype=np.float64)

        # Generate ground truth once for periodic evaluation AND data supervision
        print("Generating ground truth for evaluation and data supervision...")
        eval_ground_truth = {}
        self._ground_truth_tensors = {}  # For data supervision during training
        for prob_idx in train_problems_indices:
            print(f"  Solving problem {prob_idx} ...")
            if self.problems[prob_idx].mixed:
                gfA, gfPhi, r1 = self.all_fem_solvers[prob_idx].solve_mixed_em(
                    self.problems[prob_idx]
                )
                # Extract DOF values from GridFunctions
                A_dofs = np.array(gfA.vec, dtype=np.complex128)
                phi_dofs = np.array(gfPhi.vec, dtype=np.complex128)
                eval_ground_truth[prob_idx] = np.concatenate([A_dofs, phi_dofs])

                # Store as tensors for data supervision (on graph nodes, not DOFs)
                # A is defined on all nodes, phi only on coil nodes
                fem_s = self.all_fem_solvers[prob_idx]
                self._ground_truth_tensors[prob_idx] = {
                    "A_real": torch.tensor(
                        A_dofs.real, dtype=torch.float64, device=self.device
                    ),
                    "A_imag": torch.tensor(
                        A_dofs.imag, dtype=torch.float64, device=self.device
                    ),
                    "phi_real": torch.tensor(
                        phi_dofs.real, dtype=torch.float64, device=self.device
                    ),
                    "phi_imag": torch.tensor(
                        phi_dofs.imag, dtype=torch.float64, device=self.device
                    ),
                    "coil_node_indices": torch.tensor(
                        fem_s.coil_node_indices, dtype=torch.long, device=self.device
                    ),
                }
            elif self.problems[prob_idx].complex:
                gfA = self.all_fem_solvers[prob_idx].solve(
                    self.problems[prob_idx]
                )
                A_dofs = np.array(gfA, dtype=np.complex128)
                eval_ground_truth[prob_idx] = A_dofs

                self._ground_truth_tensors[prob_idx] = {
                    "A_real": torch.tensor(
                        A_dofs.real, dtype=torch.float64, device=self.device
                    ),
                    "A_imag": torch.tensor(
                        A_dofs.imag, dtype=torch.float64, device=self.device
                    ),
                }
            else:
                eval_ground_truth[prob_idx] = self.all_fem_solvers[prob_idx].solve(
                    self.problems[prob_idx]
                )
        print(f"Ground truth generated for {len(eval_ground_truth)} problems.")

        # Data supervision weight - starts high and decays
        data_weight_init = self.config.get("data_weight", 0.1)
        data_weight_decay = self.config.get(
            "data_weight_decay", 0.9995
        )  # Decay per epoch

        n_train = len(train_problems_indices)

        for epoch in range(self.start_epoch, self.config["epochs"]):
            epoch_start = time.time()

            # Compute current data weight (decay schedule)
            data_weight = data_weight_init * (data_weight_decay**epoch)

            # --- Train on ALL problems each epoch ---
            epoch_loss = 0.0
            epoch_loss_A = 0.0
            epoch_loss_phi = 0.0
            epoch_loss_data = 0.0
            for prob_idx in train_problems_indices:
                loss, prediction_next = self.train_step(
                    prob_idx,
                    prediction=predictions[prob_idx],
                    ground_truth=self._ground_truth_tensors.get(prob_idx),
                    data_weight=data_weight,
                )
                predictions[prob_idx] = prediction_next
                epoch_loss += loss
                epoch_loss_A += self._last_loss_A
                epoch_loss_phi += self._last_loss_phi
                epoch_loss_data += self._last_data_loss

            # Average loss across problems
            epoch_loss /= n_train
            epoch_loss_A /= n_train
            epoch_loss_phi /= n_train
            epoch_loss_data /= n_train

            self.losses.append(epoch_loss)

            # Validation (if validation problems are provided and ground truth is available)
            val_loss = None
            if val_problems_indices and self.all_ground_truth is not None:
                val_loss = self.validate(val_problems_indices)
                self.val_losses.append(val_loss)

            elapsed = time.time() - epoch_start
            self.logger.log_epoch(epoch, epoch_loss, val_loss, elapsed)

            if epoch % 10 == 0:
                elapsed = time.time() - epoch_start
                val_str = f" | Val Loss: {val_loss:.3e}" if val_loss is not None else ""
                # Show component losses for mixed EM problems
                if self.is_mixed:
                    loss_A_str = (
                        f" | L_A: {epoch_loss_A:.3e} | L_φ: {epoch_loss_phi:.3e}"
                    )
                    if data_weight > 0:
                        loss_A_str += (
                            f" | L_data: {epoch_loss_data:.3e} (w={data_weight:.2e})"
                        )
                else:
                    loss_A_str = ""
                prob_str = f" ({n_train} probs)" if n_train > 1 else ""
                print(
                    f"Epoch {epoch+1:4d} | Train Loss: {epoch_loss:.3e}{loss_A_str}{val_str}{prob_str} | Time: {elapsed:.2f}s"
                )

            # Evaluate L2 error every 100 epochs
            if (epoch + 1) % 100 == 0:
                self.model.eval()
                for prob_idx in train_problems_indices:
                    pred = self.predict(problem_idx=prob_idx)
                    gt = eval_ground_truth[prob_idx]
                    if self.is_mixed:
                        # pred and gt are both complex arrays: [A_complex | phi_complex]
                        # Compare them directly (no real/imag split needed)
                        fem_solver = self.all_fem_solvers[prob_idx]
                        n_A = fem_solver.n_dofs_A
                        n_phi = fem_solver.n_dofs_phi
                        a_pred = pred[:n_A]
                        a_true = gt[:n_A]
                        free_mask = fem_solver._free_dofs_mask()
                        free_mask_phi = free_mask[n_A:].cpu().numpy()
                        phi_pred = pred[n_A : n_A + n_phi][free_mask_phi]
                        phi_true = gt[n_A : n_A + n_phi][free_mask_phi]
                        l2_A = np.linalg.norm(a_pred - a_true) / (
                            np.linalg.norm(a_true) + 1e-30
                        )
                        l2_phi = np.linalg.norm(phi_pred - phi_true) / (
                            np.linalg.norm(phi_true) + 1e-30
                        )
                        l2_total = np.linalg.norm(pred - gt) / (
                            np.linalg.norm(gt) + 1e-30
                        )
                        self.logger.log_evaluation(l2_A, f"l2_A_p{prob_idx}")
                        self.logger.log_evaluation(l2_phi, f"l2_phi_p{prob_idx}")
                        print(
                            f"  [Eval] Epoch {epoch+1} | Problem {prob_idx+1} "
                            f"| L2 total: {l2_total:.6e} | L2 A: {l2_A:.6e} | L2 phi: {l2_phi:.6e}"
                        )
                    elif self.is_complex:
                        # pred and gt are complex arrays: [A_complex]
                        l2_error = np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-30)
                        self.logger.log_evaluation(l2_error, f"l2_A_p{prob_idx}")
                        print(
                            f"  [Eval] Epoch {epoch+1} | Problem {prob_idx+1} | L2 Error: {l2_error:.6e}"
                        )
                    else:
                        if len(pred) == len(gt):
                            l2_error = np.linalg.norm(pred - gt) / np.linalg.norm(gt)
                            print(
                                f"  [Eval] Epoch {epoch+1} | Problem {prob_idx+1} | L2 Error: {l2_error:.6e}"
                            )
                self.model.train()
            if epoch % self.config.get("save_interval", 1000) == 0:
                path_to_checkpoint = self.config.get(
                    "save_dir", "results/physics_informed_em"
                )
                model_path = f"{path_to_checkpoint}/pimgn_trained_model.pth"
                self.save_checkpoint(model_path, epoch)

            self.scheduler.step()

        print("Physics-Informed MeshGraphNet EM training completed!")

    def validate(self, val_problems_indices):
        raise NotImplementedError("Validation not implemented")

    def predict(self, problem_idx=0):
        """Predict steady-state solution for a specific problem."""
        self.model.eval()

        problem = self.problems[problem_idx]
        fem_solver: FEMSolverEM = self.all_fem_solvers[problem_idx]

        # Use pre-created graph creator for this problem
        graph_creator = self.all_graph_creators[problem_idx]

        # Get the values for this problem
        dirichlet_vals = getattr(problem, "dirichlet_values_array", None)
        material_field = getattr(problem, "material_field", None)
        sigma_field = getattr(problem, "sigma_field", None)
        current = getattr(problem, "I_coil", None)
        coil_node_mask = getattr(problem, "coil_node_mask", None)

        with torch.no_grad():
            # Build graph
            data, aux = graph_creator.create_graph(
                A_current=None,
                material_node_field=material_field,
                sigma_field=sigma_field,
                dirichlet_values=dirichlet_vals,
                current=current,
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

            if problem.mixed:
                # Reconstruct A predictions for all nodes
                pred = prediction_free.detach().cpu().numpy()  # [N_free, 4]

                prediction_full_Areal = np.zeros(n_total, dtype=np.float64)
                prediction_full_Aimag = np.zeros(n_total, dtype=np.float64)
                prediction_full_Areal[free_idx] = pred[:, 0]
                prediction_full_Aimag[free_idx] = pred[:, 1]

                # Enforce Dirichlet BCs on A
                if dirichlet_vals is not None:
                    dirichlet_vals_real = (
                        dirichlet_vals[dirichlet_mask].real
                        if np.iscomplexobj(dirichlet_vals)
                        else dirichlet_vals[dirichlet_mask]
                    )
                    prediction_full_Areal[dirichlet_mask] = dirichlet_vals_real
                    prediction_full_Aimag[dirichlet_mask] = 0

                # Reconstruct phi predictions for all nodes (will extract coil-only later)
                prediction_full_phireal = np.zeros(n_total, dtype=np.float64)
                prediction_full_phiimag = np.zeros(n_total, dtype=np.float64)
                prediction_full_phireal[free_idx] = pred[:, 2]
                prediction_full_phiimag[free_idx] = pred[:, 3]

                # Extract phi values only for coil nodes (in FEM DOF order)
                coil_node_indices = fem_solver.coil_node_indices
                phi_real_dofs = prediction_full_phireal[
                    coil_node_indices
                ]  # [n_dofs_phi]
                phi_imag_dofs = prediction_full_phiimag[
                    coil_node_indices
                ]  # [n_dofs_phi]

                # Build full DOF vector in FEM format: [A_dofs | phi_dofs]
                # A is complex: A_real + 1j * A_imag
                A_complex = prediction_full_Areal + 1j * prediction_full_Aimag
                phi_complex = phi_real_dofs + 1j * phi_imag_dofs

                # Return as concatenated DOF vector matching FEM structure
                prediction_full = np.concatenate([A_complex, phi_complex])
            elif problem.complex:
                pred = prediction_free.detach().cpu().numpy()  # [N_free, 2]
                prediction_full_Areal = np.zeros(n_total, dtype=np.float64)
                prediction_full_Aimag = np.zeros(n_total, dtype=np.float64)
                prediction_full_Areal[free_idx] = pred[:, 0]
                prediction_full_Aimag[free_idx] = pred[:, 1]

                # Enforce Dirichlet BCs on A
                if dirichlet_vals is not None:
                    dirichlet_vals_real = (
                        dirichlet_vals[dirichlet_mask].real
                        if np.iscomplexobj(dirichlet_vals)
                        else dirichlet_vals[dirichlet_mask]
                    )
                    prediction_full_Areal[dirichlet_mask] = dirichlet_vals_real
                    prediction_full_Aimag[dirichlet_mask] = 0

                prediction_full = prediction_full_Areal + 1j * prediction_full_Aimag
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
            if problem.mixed:
                gfA, gfPhi, r1 = fem_solver.solve_mixed_em(problem)
                # Extract DOF values from GridFunctions to match prediction format
                A_dofs = np.array(gfA.vec, dtype=np.complex128)
                phi_dofs = np.array(gfPhi.vec, dtype=np.complex128)
                ground_truth = np.concatenate([A_dofs, phi_dofs])
            elif problem.complex:
                gfA = fem_solver.solve(problem)
                ground_truth = np.array(gfA, dtype=np.complex128)
            else:
                ground_truth = fem_solver.solve(problem)

            # Get prediction from trained model
            prediction = self.predict(problem_idx=problem_idx)

            # Compute error
            if len(prediction) == len(ground_truth):
                if self.is_mixed:
                    n_A = fem_solver.n_dofs_A
                    n_phi = fem_solver.n_dofs_phi
                    a_pred = prediction[:n_A]
                    a_true = ground_truth[:n_A]
                    free_mask = fem_solver._free_dofs_mask()
                    free_mask_phi = free_mask[n_A:].cpu().numpy()
                    phi_pred = prediction[n_A : n_A + n_phi][free_mask_phi]
                    phi_true = ground_truth[n_A : n_A + n_phi][free_mask_phi]
                    l2_A = np.linalg.norm(a_pred - a_true) / (
                        np.linalg.norm(a_true) + 1e-30
                    )
                    l2_phi = np.linalg.norm(phi_pred - phi_true) / (
                        np.linalg.norm(phi_true) + 1e-30
                    )
                    l2_error = np.linalg.norm(prediction - ground_truth) / (
                        np.linalg.norm(ground_truth) + 1e-30
                    )
                    # max_error = np.max(np.abs(prediction - ground_truth))
                    print(
                        f"Problem {problem_idx + 1} - L2 total: {l2_error:.6f} | L2 A: {l2_A:.6f} | L2 phi: {l2_phi:.6f}"
                    )
                elif self.is_complex:
                    l2_error = np.linalg.norm(prediction - ground_truth) / (
                        np.linalg.norm(ground_truth) + 1e-30
                    )
                    max_error = np.max(np.abs(prediction - ground_truth))
                    print(
                        f"Problem {problem_idx + 1} - L2 error: {l2_error:.6f}, Max error: {max_error:.6f}"
                    )
                else:
                    l2_error = np.linalg.norm(
                        prediction - ground_truth
                    ) / np.linalg.norm(ground_truth)
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


def _run_experiment(
    problems: List[MeshProblemEM],
    config,
    experiment_name: str,
    train_indices=None,
    val_indices=None,
):
    """Run a PIMGN-EM experiment on one or more problems.

    Args:
        problems: List of MeshProblemEM instances to train on.
        config: Training configuration dict.
        experiment_name: Human-readable name for logging.
        train_indices: Indices into ``problems`` to use for training.
                       Defaults to all problems.
        val_indices: Optional indices for validation problems.
    """
    print("=" * 60)
    print(f"PIMGN TEST - {experiment_name.upper()}")
    print("=" * 60)

    save_path = config.get("save_dir", "results/physics_informed")
    os.makedirs(save_path, exist_ok=True)

    if train_indices is None:
        train_indices = list(range(len(problems)))

    trainer = PIMGNTrainerEM(problems, config)

    print(f"\nStarting physics-informed training on {len(train_indices)} problem(s)...")
    trainer.train(
        train_problems_indices=train_indices,
        val_problems_indices=val_indices,
    )

    print("\nEvaluating trained PIMGN...")
    last_residuals = getattr(trainer, "last_residuals", None)
    if last_residuals is not None:
        trainer.logger.log_evaluation(
            np.atleast_1d(np.abs(last_residuals)).tolist(),
            "residuals_per_time_step_abs",
        )
    else:
        print(
            "Note: No `last_residuals` available (training may have been skipped due to resume epoch >= configured epochs)."
        )

    # Evaluate and export results for every training problem
    try:
        predictions, ground_truth, errors = trainer.evaluate_with_ground_truth(
            problem_indices=train_indices
        )

        print("Exporting results...")
        for i, prob_idx in enumerate(train_indices):
            problem = problems[prob_idx]
            vtk_suffix = f"_p{prob_idx}" if len(train_indices) > 1 else ""
            if problem.complex:
                trainer.all_fem_solvers[prob_idx].export_to_vtk_complex(
                    ground_truth[i],
                    predictions[i],
                    filename=f"{save_path}/vtk/result_complex{vtk_suffix}",
                )
            elif problem.mixed:
                trainer.all_fem_solvers[prob_idx].export_to_vtk_mixed(
                    ground_truth[i],
                    predictions[i],
                    filename=f"{save_path}/vtk/result_mixed{vtk_suffix}",
                )
            else:
                trainer.all_fem_solvers[prob_idx].export_to_vtk(
                    ground_truth[i],
                    predictions[i],
                    filename=f"{save_path}/vtk/result{vtk_suffix}",
                )
    except Exception as e:
        print(f"Ground truth evaluation failed: {e}")

    print("Physics-Informed MeshGraphNet test completed!")
    print(f"Results saved to: {save_path}")
    trainer.save_logs()

    model_path = f"{save_path}/pimgn_trained_model.pth"
    trainer.save_checkpoint(model_path, epoch=config["epochs"] - 1)
    print(f"Trained model saved to: {model_path}")

    return trainer


# Backward-compatible alias
def _run_single_problem_experiment(
    problem: MeshProblemEM, config, experiment_name: str
):
    """Convenience wrapper for a single-problem experiment."""
    return _run_experiment([problem], config, experiment_name, train_indices=[0])


def train_pimgn_on_single_problem(resume_from: str = None):
    problem = create_em_problem()
    config = {
        "epochs": 2000,
        "lr": 1e-4,
        "generate_ground_truth_for_validation": False,
        "save_dir": "results/physics_informed/test_em_problem",
        "resume_from": resume_from,  # Path to checkpoint to resume from
        "data_weight": 0.0,
    }
    _run_single_problem_experiment(problem, config, "First order EM")


def train_pimgn_em_complex(resume_from: str = None):
    problem = create_em_problem_complex()
    config = {
        "epochs": 5000,
        "lr": 1e-3,
        "generate_ground_truth_for_validation": False,
        "save_dir": "results/physics_informed/test_em_problem_complex",
        "data_weight": 0.0,
        "resume_from": resume_from,  # Path to checkpoint to resume from
    }
    _run_single_problem_experiment(problem, config, "First order EM Complex")


# def train_pimgn_em_mixed(resume_from: str = None):
#     problem = create_em_mixed()
#     config = {
#         "epochs": 1010,
#         "lr": 1e-3,
#         "generate_ground_truth_for_validation": False,
#         "save_dir": "results/physics_informed/test_em_problem_mixed",
#         "resume_from": resume_from,  # Path to checkpoint to resume from
#         "save_interval": 1000,  # Save checkpoint every N epochs
#         # Phi weight for balanced loss: increase if phi doesn't converge
#         # phi_weight=10 means phi loss contributes 10x more to total loss
#         "phi_weight": 0.1,
#         # Data supervision to avoid trivial solutions (u approx 0)
#         # The A equation is homogeneous - physics loss alone sometimes allows u=0
#         # Data supervision guides to correct solution, then decays
#         "data_weight": 0.0,  # Initial data supervision weight
#         "data_weight_decay": 0.9995,  # Decay per epoch (0.9995^1000 approx 0.6)
#     }
#     _run_single_problem_experiment(problem, config, "First order EM Mixed")


def train_pimgn_em_multi(resume_from: str = None):
    """Train PIMGN-EM on multiple problems simultaneously."""
    # Create a list of problems with different configurations
    problems = [
        create_em_mixed(i_coil=1000, h_workpiece=8e-3, h_air=0.3, h_coil=3e-3),
        create_em_mixed(i_coil=2000, h_workpiece=8e-3, h_air=0.3, h_coil=3e-3),
        create_em_mixed(i_coil=3000, h_workpiece=8e-3, h_air=0.3, h_coil=3e-3),
    ]
    # Assign unique problem IDs
    for i, p in enumerate(problems):
        p.problem_id = i

    config = {
        "epochs": 20000,
        "lr": 1e-3,
        "generate_ground_truth_for_validation": False,
        "save_dir": "results/physics_informed/test_em_multi",
        "resume_from": resume_from,
        "save_interval": 1000,
        "phi_weight": 0.1,
        "data_weight": 0.0,
        "data_weight_decay": 0.9995,
    }
    train_indices = list(range(len(problems)))
    _run_experiment(
        problems, config, "Multi-problem EM Mixed", train_indices=train_indices
    )


if __name__ == "__main__":
    # train_pimgn_on_single_problem()
    train_pimgn_em_complex()
    # train_pimgn_em_mixed()
    # train_pimgn_em_multi()
