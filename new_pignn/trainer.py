"""
Training module for Physics-Informed Graph Neural Networks (PI-GNN).
Based on the training methodology from the paper with multi-mesh support.
"""

from matplotlib.pyplot import step
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import time
import random
from abc import ABC, abstractmethod

from graph_creator import GraphCreator
from pignn import PossionNet
from fem import FEMSolver
from mesh_utils import build_graph_from_mesh, create_free_node_subgraph
from containers import MeshProblem, TrainingConfig


class PIGNNTrainer:
    """
    Trainer for Physics-Informed Graph Neural Networks with multi-mesh support.

    Implements the optimization loop from the paper for diverse mesh geometries.
    """

    def __init__(
        self,
        training_problems: List[MeshProblem],
        validation_problems: List[MeshProblem],
        config: TrainingConfig,
    ):
        """
        Initialize trainer.

        Args:
            training_problems: List of training problems with diverse meshes
            validation_problems: List of validation problems
            config: Training configuration
        """
        self.training_problems = training_problems
        self.validation_problems = validation_problems
        self.config = config

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model with strict BC enforcement
        self.model = PossionNet(
            nci=2,  # input channels
            nco=1,  # output channels
            kk=10   # Chebyshev polynomial order
        ).to(self.device)

        # Store free node mappings for each problem
        self._free_node_mappings: Dict[int, Dict] = {}

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0)

        # Store free node mappings for each problem
        self._free_node_mappings: Dict[int, Dict] = {}

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)

        # Training history
        self.training_history = {"epoch": [], "loss_pde": [], "val_loss": [], "ic_loss": []}

        self._fem_solvers: Dict[int, FEMSolver] = {}

        # Problem statistics
        self._log_problem_statistics()

    def _log_problem_statistics(self):
        """Log statistics about the problems."""
        print("=" * 60)
        print("MULTI-MESH TRAINING SETUP")
        print("=" * 60)

        print(f"Training problems: {len(self.training_problems)}")
        print(f"Validation problems: {len(self.validation_problems)}")

        # Group by mesh type
        mesh_type_counts = {}
        node_stats = {"min": float("inf"), "max": 0, "total": 0}

        for problem in self.training_problems:
            mesh_type = problem.mesh_config.mesh_type
            mesh_type_counts[mesh_type] = mesh_type_counts.get(mesh_type, 0) + 1

            node_stats["min"] = min(node_stats["min"], problem.n_nodes)
            node_stats["max"] = max(node_stats["max"], problem.n_nodes)
            node_stats["total"] += problem.n_nodes

        print("\nMesh type distribution:")
        for mesh_type, count in mesh_type_counts.items():
            print(f"  {mesh_type}: {count} problems")

        print(f"\nMesh size statistics:")
        print(f"  Min nodes: {node_stats['min']}")
        print(f"  Max nodes: {node_stats['max']}")
        print(f"  Avg nodes: {node_stats['total'] / len(self.training_problems):.1f}")
        print("=" * 60)

    def compute_loss(
        self, problem: MeshProblem, T_current, predictions_next, T_initial=None, epoch=0
    ):
        # Create FEM solver for this problem if needed
        if not hasattr(self, '_fem_solvers'):
            self._fem_solvers = {}
        
        if problem.problem_id not in self._fem_solvers:
            self._fem_solvers[problem.problem_id] = FEMSolver(problem.mesh, problem=problem)
        
        fem_solver = self._fem_solvers[problem.problem_id]

        # Convert to appropriate precision and device
        if isinstance(T_current, torch.Tensor):
            T_current_tensor = T_current.to(dtype=torch.float64, device=self.device)
        else:
            T_current_tensor = torch.tensor(T_current, dtype=torch.float64, device=self.device)
        
        predictions_tensor = predictions_next.to(dtype=torch.float64, device=self.device)
        
        # Physics-informed loss (FEM residual)
        fem_losses = []
        for t in range(self.model.time_window):
            n_prediction = predictions_tensor[:, t]
            fem_loss = fem_solver.compute_residual(n_prediction, T_current_tensor, problem)
            fem_losses.append(torch.sum(fem_loss)**2) # Sum over nodes

        fem_loss = torch.sum(torch.stack(fem_losses)) / self.model.time_window
        fem_loss = fem_loss / fem_solver.fes.ndof # Normalize by number of test functions
        
        total_loss = fem_loss
        
        return total_loss

    def training_step(self, problem: MeshProblem, T_current, t_current, t_prev, apply_ic_loss=False, epoch=0):
        self.optimizer.zero_grad()

        data, aux = build_graph_from_mesh(
            mesh=problem.mesh,
            dirichlet_names=problem.mesh_config.dirichlet_boundaries,
            T_current=T_current.detach().cpu().numpy(),  # <-- fixed prev state for the whole bundle
            t_scalar=t_current,
        )
        data = data.to(self.device)

        free_data, free_node_mapping = create_free_node_subgraph(data, aux)
        free_data = free_data.to(self.device)

        self._free_node_mappings[problem.problem_id] = {
            'mapping': free_node_mapping,
            'free_mask': aux["free_mask"],
            'n_total_nodes': len(aux["free_mask"]),
            'n_free_nodes': len(free_node_mapping),
        }

        # 3) Forward once → bundle predictions for FREE nodes
        predictions_free = self.model(free_data)  # [N_free, time_window]

        # 4) Lift back to FULL vector space (Dirichlet entries stay zero)
        n_total_nodes = len(aux["free_mask"])
        predictions_full = torch.zeros(
            n_total_nodes, self.model.time_window,
            dtype=predictions_free.dtype, device=self.device
        )
        free_indices = torch.tensor(
            [i for i, is_free in enumerate(aux["free_mask"]) if is_free],
            device=self.device
        )
        predictions_full[free_indices] = predictions_free

        # 5) Compute loss - include IC loss for first time step
        T_prev_fixed = T_current.detach()  # keep full previous state fixed for the whole bundle
        
        T_initial = torch.tensor(problem.initial_condition, dtype=torch.float32, device=self.device)
        
        step_loss = self.compute_loss(
            problem=problem,
            T_current=T_prev_fixed,
            predictions_next=predictions_full,
            T_initial=T_initial,
            epoch=epoch
        )
        
        # 6) Backprop + step
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {"loss": {"total": step_loss}, "predictions": predictions_full}

    def train_epoch(self, epoch=0):
        """Train for one epoch using diverse mesh problems."""
        self.model.train()

        # Calculate number of time steps
        num_timesteps = self.config.time_config.num_steps

        # Shuffle training problems for this epoch
        shuffled_problems = self.training_problems.copy()
        random.shuffle(shuffled_problems)

        epoch_losses = {"total": 0.0, "pde": 0.0}
        step_count = 0

        # Process problems in batches
        for batch_start in range(0, len(shuffled_problems), self.config.batch_size):
            batch_problems = shuffled_problems[
                batch_start : batch_start + self.config.batch_size
            ]

            for problem in batch_problems:
                # Start with initial condition for this problem
                T0_tensor = torch.tensor(
                    problem.initial_condition, dtype=torch.float32, device=self.device
                )
                T_current = T0_tensor.clone()
                time_steps = self.config.time_config.time_steps
                time_steps_batched = np.array_split(time_steps, num_timesteps // self.model.time_window)
                # Optimization loop over time steps for this problem
                for time_step_batch in time_steps_batched:
                    dt = self.config.time_config.dt
                    t_current = time_step_batch[0] * dt
                    t_prev = max(0, (time_step_batch[0] - 1) * dt)

                    result = self.training_step(problem, T_current, t_current, t_prev, epoch=epoch)
                    loss = result["loss"]["total"]
                    predictions = result["predictions"]  # All timesteps

                    T_current = predictions[:, -1].detach()  # Use last prediction
                    epoch_losses["total"] += loss.item()
                    epoch_losses["pde"] += loss.item()
                    step_count += 1

        # Average losses over all steps
        for key in epoch_losses:
            epoch_losses[key] /= step_count

        return epoch_losses

    def validate_epoch(self):
        """Validate on validation problems."""
        # Skip validation for now to focus on getting training working
        return {"total": 0.0, "pde": 0.0}

    def train(self):
        """Main training loop."""
        print(f"Starting multi-mesh training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Train one epoch
            epoch_losses = self.train_epoch(epoch=epoch)

            # Validate
            # val_losses = self.validate_epoch()
            val_losses = {"total": 0.0}  # Skip validation for speed

            # Update learning rate
            self.scheduler.step()

            # Log progress
            if epoch % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch:4d} | "
                    f"Train Loss: {epoch_losses['total']:.3e} | "
                    f"Val Loss: {val_losses['total']:.3e} | "
                    f"PDE: {epoch_losses['pde']:.3e} | "
                    f"Time: {elapsed:.2f}s"
                )

            # Save training history
            self.training_history["epoch"].append(epoch)
            self.training_history["loss_pde"].append(epoch_losses["pde"])
            self.training_history["val_loss"].append(val_losses["total"])

            # Save model checkpoint
            if epoch > 0 and epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoints/checkpoint_epoch_{epoch}.pth")
                rollout = self.rollout(self.validation_problems[0], T_start=self.validation_problems[0].initial_condition)  # Test rollout on first validation problem
                time_steps = self.config.time_config.time_steps_export
                rollout = np.array(rollout)
                self._fem_solvers[0].export_to_vtk(rollout, rollout, time_steps, f"checkpoints/rollout_epoch_{epoch}.vtk")
            # TODO: Add early stopping based on validation loss

        print("Multi-mesh training completed!")

    def rollout(self, problem: MeshProblem, T_start=None):
        """
        Perform rollout simulation using trained model.

        Args:
            problem: The mesh problem to simulate
            T_start: Starting temperature field (uses problem's T0 if None)

        Returns:
            List of temperature states over time
        """
        self.model.eval()

        if T_start is None:
            T_current = torch.tensor(
                problem.initial_condition, dtype=torch.float32, device=self.device
            )
        else:
            T_current = torch.tensor(T_start, dtype=torch.float32, device=self.device)

        # Calculate number of time steps
        num_timesteps = int(self.config.time_config.t_final / self.config.time_config.dt)
        time_steps = self.config.time_config.time_steps
        time_steps_batched = np.array_split(time_steps, num_timesteps // self.model.time_window)

        states = [T_current.detach().cpu().numpy()]
        time_window = self.model.time_window

        graph_creator = GraphCreator(
            mesh=problem.mesh,
            n_neighbors=2,
            dirichlet_names=problem.mesh_config.dirichlet_boundaries,
            neumann_names=[],
            connectivity_method="fem"
        )
        
        with torch.no_grad():
            for batch in time_steps_batched:
                # Build full graph features from the current state
                data, aux = graph_creator.create_graph(
                    T_current=T_current.detach().cpu().numpy(),
                    t_scalar=batch[0]
                )

                data = data.to(self.device)
                
                # Create free-node subgraph
                free_data, free_node_mapping = create_free_node_subgraph(data, aux)
                free_data = free_data.to(self.device)
                
                # Get predictions for time_window steps (free nodes only)
                predictions_free = self.model(free_data)  # Shape: [N_free_nodes, time_window]

                # Reconstruct full predictions
                n_total_nodes = len(aux["free_mask"])
                predictions_full = torch.zeros(n_total_nodes, time_window, 
                                             dtype=predictions_free.dtype, device=self.device)
                
                # Fill in free node predictions  
                free_indices = torch.tensor([i for i, is_free in enumerate(aux["free_mask"]) if is_free], 
                                           device=self.device)
                predictions_full[free_indices] = predictions_free
                
                # Dirichlet nodes remain zero (homogeneous BC)
                
                # Add states to results
                states.extend(predictions_full.detach().cpu().numpy().T)

                # Update current state to last prediction
                T_current = predictions_full[:, -1]  # Last prediction becomes current state
                
        return states

    def save_checkpoint(self, filepath):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "training_history": self.training_history,
            "num_training_problems": len(self.training_problems),
            "num_validation_problems": len(self.validation_problems),
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.training_history = checkpoint["training_history"]
        print(f"Checkpoint loaded: {filepath}")
