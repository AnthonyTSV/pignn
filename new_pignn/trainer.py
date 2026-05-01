import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import time
from argparse import Namespace
from typing import List

try:
    from .logger import TrainingLogger
    from .meshgraphnet import MeshGraphNet
    from .fem import FEMSolver
    from .graph_creator import GraphCreator
    from .containers import TimeConfig, MeshConfig, MeshProblem
except ImportError:
    from logger import TrainingLogger
    from meshgraphnet import MeshGraphNet
    from fem import FEMSolver
    from graph_creator import GraphCreator
    from containers import TimeConfig, MeshConfig, MeshProblem
from torch_geometric.data import Data, Batch


def _load_checkpoint(path: str | Path, device: torch.device):
    """Load trusted training checkpoint and raise actionable error if file is corrupt."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        try:
            return torch.load(path, map_location=device)
        except RuntimeError as exc:
            if "failed finding central directory" in str(exc):
                raise RuntimeError(
                    f"Checkpoint at {path} is corrupted or incomplete. "
                    "PyTorch could not read zip central directory. "
                    "Replace file with valid checkpoint or retrain model to regenerate it."
                ) from exc
            raise
    except RuntimeError as exc:
        if "failed finding central directory" in str(exc):
            raise RuntimeError(
                f"Checkpoint at {path} is corrupted or incomplete. "
                "PyTorch could not read zip central directory. "
                "Replace file with valid checkpoint or retrain model to regenerate it."
            ) from exc
        raise


def _atomic_torch_save(obj, path: str | Path) -> None:
    """Write checkpoint atomically so interrupted saves do not leave partial zip files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)

        torch.save(obj, temp_path)
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


class PIMGNTrainer:
    """Trainer for Physics-Informed MeshGraphNet (PIMGN)."""

    def __init__(self, problems: List[MeshProblem] | MeshProblem, config: dict):
        if isinstance(problems, MeshProblem):
            problems = [problems]

        self.problems = problems
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Time bundling configuration
        self.time_window = config.get("time_window", 20)

        self.noise_sigma = config.get("noise_sigma", 1e-2)

        # Create FEM solvers for physics-informed loss computation for all problems
        self.all_fem_solvers: List[FEMSolver] = []
        for problem in problems:
            fem_solver = FEMSolver(
                problem.mesh, order=problem.mesh_config.order, problem=problem
            )
            self.all_fem_solvers.append(fem_solver)

        # Pre-create GraphCreator instances for all problems (avoid re-creation each step)
        self.all_graph_creators: List[GraphCreator] = []
        for i, problem in enumerate(problems):
            gc = GraphCreator(
                mesh=problem.mesh,
                n_neighbors=2,
                dirichlet_names=problem.mesh_config.dirichlet_boundaries,
                neumann_names=getattr(problem.mesh_config, "neumann_boundaries", []),
                robin_names=getattr(problem.mesh_config, "robin_boundaries", []),
                connectivity_method="fem",
                fes=self.all_fem_solvers[i].fes,
            )
            self.all_graph_creators.append(gc)

        # Prepare sample data to determine input/output dimensions using first problem
        first_problem = problems[0]
        graph_creator = self.all_graph_creators[0]

        # Create sample graph to get dimensions
        material_field = getattr(first_problem, "material_field", None)
        neumann_vals = getattr(first_problem, "neumann_values_array", None)
        dirichlet_vals = getattr(first_problem, "dirichlet_values_array", None)
        robin_vals = getattr(first_problem, "robin_values_array", None)
        source_vals = getattr(first_problem, "source_function", None)
        k_table_ref = getattr(first_problem, "k_table_ref_values", None)
        # For temp-dependent k, evaluate k at initial T for the sample graph
        k_table = getattr(first_problem, "k_table", None)
        if k_table is not None:
            material_field = k_table.evaluate_array(first_problem.initial_condition)
        sample_data, aux = graph_creator.create_graph(
            T_current=first_problem.initial_condition,
            t_scalar=0.0,
            material_node_field=material_field,
            neumann_values=neumann_vals,
            dirichlet_values=dirichlet_vals,
            robin_values=robin_vals,
            source_values=source_vals,
            k_table_ref_values=k_table_ref,
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
        # Use a slow exponential decay so training does not hit the LR floor
        # within the first few hundred epochs on harder multi-mesh runs.
        initial_lr = float(config["lr"])
        min_lr = float(config.get("min_lr", 1e-5))
        lr_decay = float(config.get("lr_decay", 0.9995))
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(min_lr / initial_lr, lr_decay ** epoch)
        )


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
            checkpoint = _load_checkpoint(resume_from, self.device)
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
                print(
                    "Loaded model weights only; epoch, optimizer, scheduler, and losses "
                    "were not restored. Training will start from epoch 0."
                )
        elif resume_from:
            print(
                f"Resume checkpoint not found: {resume_from}. "
                "Training will start from epoch 0."
            )

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

        # Use pre-created graph creator for this problem
        graph_creator = self.all_graph_creators[problem_idx]

        # Get the Neumann values for this problem
        neumann_vals = getattr(problem, "neumann_values_array", None)
        dirichlet_vals = getattr(problem, "dirichlet_values_array", None)
        robin_vals = getattr(problem, "robin_values_array", None)
        material_field = getattr(problem, "material_field", None)
        source_vals = getattr(problem, "source_function", None)
        k_table_ref = getattr(problem, "k_table_ref_values", None)
        k_table = getattr(problem, "k_table", None)

        self.optimizer.zero_grad()

        # Noise injection (paper Sec. 2.4): add eps ~ N(0, Isigma) to the prediction
        # to construct the input graph, while the noiseless state is used in the
        # FEM loss.  Noise is only added during training.
        t_current_graph = t_current + np.random.normal(
            0, self.noise_sigma, size=t_current.shape
        )

        # For temperature-dependent k, evaluate k at the (noisy) current temperature
        if k_table is not None:
            material_field = k_table.evaluate_array(t_current_graph)

        # Build full graph from current state
        data, aux = graph_creator.create_graph(
            T_current=t_current_graph,
            t_scalar=current_time,
            material_node_field=material_field,
            neumann_values=neumann_vals,
            dirichlet_values=dirichlet_vals,
            robin_values=robin_vals,
            source_values=source_vals,
            k_table_ref_values=k_table_ref,
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

    def train_step_batched(self, problem_data: List[dict]):
        """
        Batched training step: build graphs for multiple problems, run a single
        forward pass through the model, then compute per-problem physics losses.

        Args:
            problem_data: List of dicts with keys:
                - problem_idx: Index of the problem
                - t_current: Current temperature state (np.array)
                - current_time: Current time scalar (float)

        Returns:
            Tuple of (avg_loss_value, dict of {prob_idx: (next_state_full, predictions_bundled_np)})
        """
        self.optimizer.zero_grad()

        # ---- Build per-problem free-node graphs ----
        free_data_list: List[Data] = []
        per_problem_meta: List[dict] = []

        for pd in problem_data:
            problem_idx = pd["problem_idx"]
            t_current = pd["t_current"]
            current_time = pd["current_time"]
            problem = self.problems[problem_idx]
            graph_creator = self.all_graph_creators[problem_idx]

            neumann_vals = getattr(problem, "neumann_values_array", None)
            dirichlet_vals = getattr(problem, "dirichlet_values_array", None)
            robin_vals = getattr(problem, "robin_values_array", None)
            material_field = getattr(problem, "material_field", None)
            source_vals = getattr(problem, "source_function", None)
            k_table_ref = getattr(problem, "k_table_ref_values", None)
            k_table = getattr(problem, "k_table", None)

            # Noise injection
            t_current_graph = t_current + np.random.normal(
                0, self.noise_sigma, size=t_current.shape
            )

            if k_table is not None:
                material_field = k_table.evaluate_array(t_current_graph)

            data, aux = graph_creator.create_graph(
                T_current=t_current_graph,
                t_scalar=current_time,
                material_node_field=material_field,
                neumann_values=neumann_vals,
                dirichlet_values=dirichlet_vals,
                robin_values=robin_vals,
                source_values=source_vals,
                k_table_ref_values=k_table_ref,
            )

            free_data, node_mapping, free_aux = self._create_subgraph(
                graph_creator, data, aux, problem_idx
            )
            free_data_list.append(free_data)
            per_problem_meta.append({
                "problem_idx": problem_idx,
                "t_current": t_current,
                "current_time": current_time,
                "aux": aux,
                "node_mapping": node_mapping,
            })

        # ---- Batch all free-node graphs into one ----
        batched_data = Batch.from_data_list(free_data_list)
        batched_data = batched_data.to(self.device)

        # ---- Single forward pass ----
        batched_prediction = self.model.forward(batched_data)

        # ---- Split predictions back per-problem ----
        batch_tensor = batched_data.batch
        total_loss = torch.tensor(0.0, device=self.device)
        next_states = {}

        for i, meta in enumerate(per_problem_meta):
            problem_idx = meta["problem_idx"]
            problem = self.problems[problem_idx]
            aux = meta["aux"]
            node_mapping = meta["node_mapping"]
            t_current = meta["t_current"]
            current_time = meta["current_time"]

            # Extract this problem's nodes from the batched prediction
            node_mask = batch_tensor == i
            predictions_bundled_free = batched_prediction[node_mask]

            # Compute physics loss
            t_current_tensor = (
                torch.tensor(t_current, dtype=torch.float32, device=self.device)
                if not torch.is_tensor(t_current)
                else t_current
            )
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
            total_loss = total_loss + physics_loss

            # Reconstruct full state
            predictions_bundled_np = predictions_bundled_free.detach().cpu().numpy()
            next_state_free = predictions_bundled_np[:, -1]

            next_state_full = np.zeros(problem.n_nodes, dtype=np.float32)
            free_to_original = node_mapping["free_to_original"].cpu().numpy()
            next_state_full[free_to_original] = next_state_free
            dirichlet_vals = getattr(problem, "dirichlet_values_array", None)
            if dirichlet_vals is not None:
                dirichlet_mask = aux["dirichlet_mask"].cpu().numpy()
                next_state_full[dirichlet_mask] = dirichlet_vals[dirichlet_mask]
            wp_mask = getattr(problem, "wp_node_mask", None)
            if wp_mask is not None:
                next_state_full[~wp_mask] = problem.initial_condition[~wp_mask]

            next_states[problem_idx] = (next_state_full, predictions_bundled_np)

        # Average the loss across problems in the batch
        n_batch = len(problem_data)
        total_loss = total_loss / n_batch

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item(), next_states

    def train(self, train_problems_indices, val_problems_indices=None):
        """Main training loop following paper's methodology with multiple problems."""
        print(f"Starting PIMGN training on {self.device}")
        print(f"Training on problems: {train_problems_indices}")
        if val_problems_indices:
            print(f"Validation on problems: {val_problems_indices}")

        batch_size = self.config.get("batch_size", 1)

        for epoch in range(self.start_epoch, self.config["epochs"]):
            epoch_start = time.time()
            epoch_losses = []

            # Shuffle training problems for each epoch
            shuffled_train_indices = train_problems_indices.copy()
            np.random.shuffle(shuffled_train_indices)

            if batch_size > 1 and len(shuffled_train_indices) > 1:
                # Batched training: iterate temporal windows, batch across problems
                first_problem = self.problems[shuffled_train_indices[0]]
                time_steps = first_problem.time_config.time_steps
                time_steps_windowed = np.array_split(
                    time_steps, len(time_steps) // self.time_window
                )
                dt = first_problem.time_config.dt

                # Initialize states for all training problems
                problem_states = {}
                for idx in shuffled_train_indices:
                    problem_states[idx] = self.problems[idx].initial_condition.copy()

                for window_times in time_steps_windowed:
                    current_time = window_times[0] - dt

                    for b_start in range(0, len(shuffled_train_indices), batch_size):
                        batch_indices = shuffled_train_indices[
                            b_start : b_start + batch_size
                        ]

                        if len(batch_indices) > 1:
                            pdata = [
                                {
                                    "problem_idx": idx,
                                    "t_current": problem_states[idx],
                                    "current_time": current_time,
                                }
                                for idx in batch_indices
                            ]
                            loss, next_states = self.train_step_batched(pdata)
                            for idx, (state, _) in next_states.items():
                                problem_states[idx] = state
                        else:
                            idx = batch_indices[0]
                            loss, t_next, _ = self.train_step(
                                problem_states[idx], current_time, idx
                            )
                            problem_states[idx] = t_next

                        epoch_losses.append(loss)
            else:
                # Original sequential training (batch_size=1 or single problem)
                for problem_idx in shuffled_train_indices:
                    problem = self.problems[problem_idx]

                    t_current = problem.initial_condition.copy()
                    time_steps = problem.time_config.time_steps
                    time_steps_windowed = np.array_split(
                        time_steps, len(time_steps) // self.time_window
                    )

                    problem_losses = []
                    for window_times in time_steps_windowed:
                        current_time = window_times[0] - problem.time_config.dt
                        physics_loss, t_next, _ = self.train_step(
                            t_current, current_time, problem_idx
                        )
                        problem_losses.append(physics_loss)
                        t_current = t_next

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
            checkpoint_epoch_interval = self.config.get(
                "save_epoch_interval", self.config.get("save_interval", 100)
            )
            if (
                checkpoint_epoch_interval
                and (epoch + 1) % checkpoint_epoch_interval == 0
            ):
                path_to_checkpoint = self.config.get(
                    "save_dir", "results/physics_informed_em"
                )
                model_path = f"{path_to_checkpoint}/pimgn_trained_model.pth"
                self.save_checkpoint(model_path, epoch)

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
                    # ground_truth[i] is at time i*dt; use export grid for correct alignment
                    current_time = problem.time_config.time_steps_export[step_idx]

                    # Use pre-created graph creator
                    graph_creator = self.all_graph_creators[problem_idx]

                    # Build graph and make prediction
                    data, aux = graph_creator.create_graph(
                        T_current=t_current,
                        t_scalar=current_time,
                        material_node_field=(
                            k_table.evaluate_array(t_current)
                            if (k_table := getattr(problem, "k_table", None)) is not None
                            else getattr(problem, "material_field", None)
                        ),
                        neumann_values=getattr(problem, "neumann_values_array", None),
                        dirichlet_values=getattr(
                            problem, "dirichlet_values_array", None
                        ),
                        robin_values=getattr(problem, "robin_values_array", None),
                        source_values=getattr(problem, "source_function", None),
                        k_table_ref_values=getattr(problem, "k_table_ref_values", None),
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
            n_steps = len(problem.time_config.time_steps_export)

        time_steps = problem.time_config.time_steps
        time_steps_bundled = np.array_split(
            time_steps, len(time_steps) // self.time_window
        )

        # Start with initial condition
        T_current = problem.initial_condition.copy()
        predictions = [T_current]

        graph_creator = self.all_graph_creators[problem_idx]

        # Get the Neumann values for this problem
        neumann_vals = getattr(problem, "neumann_values_array", None)
        dirichlet_vals = getattr(problem, "dirichlet_values_array", None)
        robin_vals = getattr(problem, "robin_values_array", None)
        material_field = getattr(problem, "material_field", None)
        source_vals = getattr(problem, "source_function", None)
        k_table_ref = getattr(problem, "k_table_ref_values", None)
        k_table = getattr(problem, "k_table", None)

        with torch.no_grad():
            step_idx = 0
            for batch_idx, batch_times in enumerate(time_steps_bundled):
                # batch_times are target times; current state is one dt earlier
                starting_time_step = batch_times[0] - problem.time_config.dt
                # For temperature-dependent k, evaluate k at current T
                mat_field_dynamic = (
                    k_table.evaluate_array(T_current)
                    if k_table is not None
                    else material_field
                )
                # Build graph
                data, aux = graph_creator.create_graph(
                    T_current=T_current,
                    t_scalar=starting_time_step,
                    material_node_field=mat_field_dynamic,
                    neumann_values=neumann_vals,
                    dirichlet_values=dirichlet_vals,
                    robin_values=robin_vals,
                    source_values=source_vals,
                    k_table_ref_values=k_table_ref,
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
        _atomic_torch_save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")
