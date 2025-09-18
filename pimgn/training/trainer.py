"""
Training module for Physics-Informed Graph Neural Networks (PI-GNN).
Based on the training methodology from the paper with multi-mesh support.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import time
import random
from abc import ABC, abstractmethod

from ..model.pimgn import PIMGN, build_features
from ..utils.fem_utils import (
    FEMSolver, compute_fem_residual, compute_fem_loss, apply_dirichlet_bc,
    compute_fem_residual_strict_bc, reconstruct_full_solution, extract_free_dofs,
    get_boundary_conditions, get_boundary_values_tensor
)
from ..utils.mesh_utils import (
    create_rectangular_mesh, create_lshape_mesh, create_polygon_mesh, 
    create_circle_mesh, create_hollow_cylinder_mesh, create_hollow_circle_mesh,
    create_mesh, build_graph_from_mesh, create_gaussian_initial_condition
)


@dataclass
class MeshConfig:
    """Configuration for a single mesh type."""
    mesh_type: str  # 'rectangular', 'lshape', 'polygon', 'circle', 'hollow_cylinder', 'hollow_circle'
    
    # Common parameters
    maxh_range: Tuple[float, float] = (0.03, 0.08)
    
    # Rectangular mesh parameters
    width_range: Tuple[float, float] = (0.5, 1.5)
    height_range: Tuple[float, float] = (0.5, 1.5)
    
    # Polygon mesh parameters
    num_points_range: Tuple[int, int] = (5, 9)
    domain_size_range: Tuple[float, float] = (0.8, 1.2)
    
    # Circle mesh parameters
    radius_range: Tuple[float, float] = (0.3, 0.7)
    center_x_range: Tuple[float, float] = (0.3, 0.7)
    center_y_range: Tuple[float, float] = (0.3, 0.7)
    
    # Hollow cylinder/circle parameters
    outer_radius_range: Tuple[float, float] = (0.4, 0.8)
    inner_radius_ratio_range: Tuple[float, float] = (0.3, 0.7)  # ratio of inner to outer
    length_range: Tuple[float, float] = (0.8, 1.5)  # for 3D cylinder
    
    # Initial condition parameters
    num_gaussians_range: Tuple[int, int] = (3, 8)
    amplitude_range: Tuple[float, float] = (0.4, 1.0)
    sigma_fraction_range: Tuple[float, float] = (0.05, 0.15)
    
    # Physics parameters
    alpha_range: Tuple[float, float] = (0.5, 2.0)


@dataclass
class TrainingConfig:
    """Configuration for PI-GNN training."""
    
    # Time parameters
    t_final: float = 1.0
    dt: float = 0.01
    
    # Model parameters
    hidden_size: int = 128
    num_layers: int = 12
    
    # Training parameters
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 2  # Number of problems per batch
    
    # Multi-mesh training parameters
    num_problems: int = 100  # Total problems per mesh type
    train_split: float = 0.75  # Fraction for training
    mesh_configs: List[MeshConfig] = None  # List of mesh configurations
    
    # Loss weights
    lambda_pde: float = 1.0
    # NOTE: lambda_bc removed due to strict BC enforcement
    # lambda_bc: float = 1.0
    lambda_ic: float = 1.0
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    
    def __post_init__(self):
        """Set default mesh configs if not provided."""
        if self.mesh_configs is None:
            self.mesh_configs = [
                MeshConfig(
                    mesh_type='rectangular',
                    maxh_range=(0.03, 0.08),
                    width_range=(0.5, 1.5),
                    height_range=(0.5, 1.5)
                ),
                MeshConfig(
                    mesh_type='lshape',
                    maxh_range=(0.03, 0.08)
                )
            ]


class MeshProblem:
    """Container for a single training problem with mesh, initial condition, and physics parameters."""
    
    def __init__(self, mesh, graph_data, initial_condition, alpha, mesh_config, problem_id, 
                 boundary_condition_type='homogeneous_dirichlet'):
        self.mesh = mesh
        self.graph_data = graph_data
        self.initial_condition = initial_condition
        self.alpha = alpha
        self.mesh_config = mesh_config
        self.problem_id = problem_id
        self.boundary_condition_type = boundary_condition_type
        
        # Statistics for logging
        self.n_nodes = graph_data['pos'].shape[0]
        self.n_edges = graph_data['edge_index'].shape[1]
        self.n_boundary = graph_data['boundary_mask'].sum()
        
        # Store boundary condition specification
        # For this implementation, we use homogeneous Dirichlet BC: T = 0 on boundary
        # This can be extended to support time-varying or non-homogeneous cases
        self.boundary_values = np.zeros(self.n_nodes)  # Default: homogeneous Dirichlet


class MeshProblemGenerator:
    """Generates diverse mesh problems for training."""
    
    def __init__(self, mesh_configs: List[MeshConfig], num_problems_per_config: int, seed: int = 42):
        self.mesh_configs = mesh_configs
        self.num_problems_per_config = num_problems_per_config
        self.seed = seed
        
    def generate_all_problems(self) -> List[MeshProblem]:
        """Generate all training problems."""
        problems = []
        problem_id = 0
        
        for config in self.mesh_configs:
            for i in range(self.num_problems_per_config):
                # Use different seeds for each problem to ensure diversity
                problem_seed = self.seed + problem_id
                problem = self._generate_problem(config, problem_seed, problem_id)
                problems.append(problem)
                problem_id += 1
        
        return problems
    
    def _generate_problem(self, config: MeshConfig, seed: int, problem_id: int) -> MeshProblem:
        """Generate a single mesh problem."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Sample mesh parameters
        maxh = random.uniform(*config.maxh_range)
        alpha = random.uniform(*config.alpha_range)
        
        # Create mesh based on type
        if config.mesh_type == 'rectangular':
            width = random.uniform(*config.width_range)
            height = random.uniform(*config.height_range)
            mesh = create_rectangular_mesh(width=width, height=height, maxh=maxh)
            
        elif config.mesh_type == 'lshape':
            mesh = create_lshape_mesh(maxh=maxh, seed=seed)
            
        elif config.mesh_type == 'polygon':
            num_points = random.randint(*config.num_points_range)
            domain_size = random.uniform(*config.domain_size_range)
            mesh = create_polygon_mesh(maxh=maxh, num_points=num_points, 
                                     domain_size=domain_size, seed=seed)
            
        elif config.mesh_type == 'circle':
            radius = random.uniform(*config.radius_range)
            center_x = random.uniform(*config.center_x_range)
            center_y = random.uniform(*config.center_y_range)
            mesh = create_circle_mesh(radius=radius, center=(center_x, center_y), 
                                    maxh=maxh, seed=seed)
            
        elif config.mesh_type == 'hollow_cylinder':
            outer_radius = random.uniform(*config.outer_radius_range)
            inner_ratio = random.uniform(*config.inner_radius_ratio_range)
            inner_radius = outer_radius * inner_ratio
            length = random.uniform(*config.length_range)
            mesh = create_hollow_cylinder_mesh(length=length, outer_radius=outer_radius,
                                             inner_radius=inner_radius, maxh=maxh, seed=seed)
            
        elif config.mesh_type == 'hollow_circle':
            outer_radius = random.uniform(*config.outer_radius_range)
            inner_ratio = random.uniform(*config.inner_radius_ratio_range)
            inner_radius = outer_radius * inner_ratio
            mesh = create_hollow_circle_mesh(outer_radius=outer_radius, 
                                           inner_radius=inner_radius, maxh=maxh, seed=seed)
        else:
            raise ValueError(f"Unknown mesh type: {config.mesh_type}")
        
        # Build graph from mesh
        graph_data = build_graph_from_mesh(mesh)
        
        # Generate initial condition
        num_gaussians = random.randint(*config.num_gaussians_range)
        x = graph_data['pos'].T
        initial_condition = np.exp(-alpha * (x[0]**2 + x[1]**2))
        
        return MeshProblem(mesh, graph_data, initial_condition, alpha, config, problem_id)


class PIGNNTrainer:
    """
    Trainer for Physics-Informed Graph Neural Networks with multi-mesh support.
    
    Implements the optimization loop from the paper for diverse mesh geometries.
    """
    
    def __init__(self, training_problems: List[MeshProblem], validation_problems: List[MeshProblem], 
                 config: TrainingConfig):
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
        self.device = torch.device(config.device)
        
        # Initialize model with strict BC enforcement
        self.model = PIMGN(
            node_input_size=6,  # [x, y, T_prev, node_type_one_hot(3)]
            edge_input_size=4,  # [dx, dy, dist, dT]
            global_input_size=2,  # [alpha, dt]
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=1,  # Single timestep prediction
            device=config.device,
            predict_free_dofs_only=True  # Enable strict BC enforcement
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.995
        )
        
        # Training history
        self.training_history = {
            'epoch': [],
            'loss_total': [],
            'loss_pde': [],
            'loss_bc': [],
            'loss_ic': [],
            'val_loss': []
        }
        
        # Problem statistics
        self._log_problem_statistics()
    
    def _log_problem_statistics(self):
        """Log statistics about the problems."""
        print("="*60)
        print("MULTI-MESH TRAINING SETUP")
        print("="*60)
        
        print(f"Training problems: {len(self.training_problems)}")
        print(f"Validation problems: {len(self.validation_problems)}")
        
        # Group by mesh type
        mesh_type_counts = {}
        node_stats = {'min': float('inf'), 'max': 0, 'total': 0}
        
        for problem in self.training_problems:
            mesh_type = problem.mesh_config.mesh_type
            mesh_type_counts[mesh_type] = mesh_type_counts.get(mesh_type, 0) + 1
            
            node_stats['min'] = min(node_stats['min'], problem.n_nodes)
            node_stats['max'] = max(node_stats['max'], problem.n_nodes)
            node_stats['total'] += problem.n_nodes
        
        print("\nMesh type distribution:")
        for mesh_type, count in mesh_type_counts.items():
            print(f"  {mesh_type}: {count} problems")
        
        print(f"\nMesh size statistics:")
        print(f"  Min nodes: {node_stats['min']}")
        print(f"  Max nodes: {node_stats['max']}")
        print(f"  Avg nodes: {node_stats['total'] / len(self.training_problems):.1f}")
        print("="*60)
    
    def compute_losses(self, problem: MeshProblem, T_current, predictions_free, t_current, t_prev):
        """
        Compute all loss components for a specific problem using strict BC enforcement.
        
        Following the paper methodology: "This directly avoids competing training losses 
        and the resulting training difficulties" by only computing FEM residuals for 
        free degrees of freedom.
        
        Args:
            problem: The mesh problem containing geometry and physics info
            T_current: Current temperature state (ALL DOFs)
            predictions_free: Model predictions for next timestep (FREE DOFs only)
            t_current: Current time
            t_prev: Previous time
        
        Returns:
            Dictionary of losses
        """
        # Create FEM solver for this problem
        fem_solver = FEMSolver(problem.mesh, alpha=problem.alpha)
        
        # PDE loss: FEM residual for free DOFs only (strict BC enforcement)
        residuals = compute_fem_residual_strict_bc(
            predictions_free, T_current, problem, t_current, t_prev, fem_solver, self.config.dt
        )
        loss_pde = compute_fem_loss(residuals)
        
        # Debug: Check if residuals make sense
        if torch.isnan(loss_pde) or torch.isinf(loss_pde):
            print(f"WARNING: Invalid PDE loss at t={t_current}: {loss_pde}")
            print(f"Residuals stats: min={residuals.min():.3e}, max={residuals.max():.3e}, mean={residuals.mean():.3e}")
            print(f"Predictions stats: min={predictions_free.min():.3e}, max={predictions_free.max():.3e}, mean={predictions_free.mean():.3e}")
            
        # NO boundary condition loss needed! BCs are strictly enforced through matrix structure
        # This eliminates competing training losses as mentioned in the paper
        loss_bc = torch.tensor(0.0, device=self.device)
        
        # Initial condition loss only for free DOFs (applied only at first step)
        loss_ic = torch.tensor(0.0, device=self.device)
        if abs(t_current) < 1e-6:  # Only at initial time (use tolerance for floating point)
            # CRITICAL FIX: Compare model predictions with initial condition, not input state!
            T0_free = extract_free_dofs(
                torch.tensor(problem.initial_condition, dtype=torch.float32, device=self.device),
                fem_solver
            )
            # The model's prediction should match the initial condition
            loss_ic = torch.nn.functional.mse_loss(predictions_free, T0_free)
            print(f"IC Loss applied: {loss_ic:.6f} (T0_free mean: {T0_free.mean():.3f}, predictions_free mean: {predictions_free.mean():.3f})")
        
        return {
            'pde': loss_pde,
            'bc': loss_bc,  # Always zero with strict enforcement
            'ic': loss_ic
        }
    
    def training_step(self, problem: MeshProblem, T_current, t_current, t_prev, apply_ic_loss=False):
        """
        Perform one training step on a specific problem with strict BC enforcement.
        
        Following the paper methodology: model only predicts free DOFs, boundary 
        conditions are strictly enforced through matrix structure.
        
        Args:
            problem: The mesh problem to train on
            T_current: Current temperature state (ALL DOFs)
            t_current: Current time
            t_prev: Previous time
            apply_ic_loss: Whether to apply initial condition loss
        
        Returns:
            Dictionary of losses and full solution
        """
        self.optimizer.zero_grad()
        
        # Create FEM solver to get free DOF structure
        fem_solver = FEMSolver(problem.mesh, alpha=problem.alpha)
        
        # Build graph features using current state - ONLY for free DOFs
        data = build_features(
            problem.graph_data, T_current.detach().cpu().numpy(),
            problem.alpha, self.config.dt, self.device, 
            fem_solver=fem_solver, free_dofs_only=True
        )
        
        # Forward pass - model predicts FREE DOFs only for NEXT timestep
        predictions_free = self.model(data).squeeze()
        
        # Compute losses using only free DOF predictions
        # The key insight: we're predicting T^{n+1} from T^n
        losses = self.compute_losses(problem, T_current.detach(), predictions_free, t_current, t_prev)
        
        # Apply strong initial condition loss for the first few steps
        # This ensures the model learns the correct initial state
        total_loss = self.config.lambda_pde * losses['pde']
        
        if apply_ic_loss:
            # Use very strong IC weight for first step to ensure proper initialization
            ic_weight = self.config.lambda_ic  # Make IC loss dominant
            total_loss += ic_weight * losses['ic']
            
        # Debug: Print loss components occasionally
        if apply_ic_loss:  # Only print at first step of each problem
            print(f"Problem {problem.problem_id}: PDE={losses['pde']:.3e}, IC={losses['ic']:.3e}, Total={total_loss:.3e}")
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Reconstruct full solution for next time step using proper boundary conditions
        full_predictions = reconstruct_full_solution(
            predictions_free, problem, t_current + self.config.dt, fem_solver, self.device
        )
        
        # Add total loss to losses dict
        losses['total'] = total_loss
        
        return {
            'losses': losses,
            'predictions': full_predictions,  # Full solution for next time step
            'predictions_free': predictions_free  # Free DOFs only for analysis
        }
    
    def train_epoch(self):
        """Train for one epoch using diverse mesh problems."""
        self.model.train()
        
        # Calculate number of time steps
        num_timesteps = int(self.config.t_final / self.config.dt)
        
        # Shuffle training problems for this epoch
        shuffled_problems = self.training_problems.copy()
        random.shuffle(shuffled_problems)
        
        epoch_losses = {'total': 0.0, 'pde': 0.0, 'bc': 0.0, 'ic': 0.0}
        step_count = 0
        
        # Process problems in batches
        for batch_start in range(0, len(shuffled_problems), self.config.batch_size):
            batch_problems = shuffled_problems[batch_start:batch_start + self.config.batch_size]
            
            for problem in batch_problems:
                # Start with initial condition for this problem
                T0_tensor = torch.tensor(
                    problem.initial_condition, dtype=torch.float32, device=self.device
                )
                T_current = T0_tensor.clone()
                
                # Optimization loop over time steps for this problem
                for step in range(num_timesteps):
                    t_current = step * self.config.dt
                    t_prev = max(0, (step - 1) * self.config.dt)
                    apply_ic_loss = (step == 0)  # Only apply IC loss at first step
                    
                    result = self.training_step(problem, T_current, t_current, t_prev, apply_ic_loss)
                    losses = result['losses']
                    T_next = result['predictions']
                    
                    # Accumulate losses
                    for key in epoch_losses:
                        epoch_losses[key] += losses[key].item()
                    
                    step_count += 1
                    
                    # Update current state for next step - detach to avoid gradient graph reuse
                    T_current = T_next.detach()
        
        # Average losses over all steps
        for key in epoch_losses:
            epoch_losses[key] /= step_count
        
        return epoch_losses
    
    def validate_epoch(self):
        """Validate on validation problems."""
        self.model.eval()
        
        val_losses = {'total': 0.0, 'pde': 0.0, 'bc': 0.0, 'ic': 0.0}
        num_timesteps = int(self.config.t_final / self.config.dt)
        step_count = 0
        
        with torch.no_grad():
            for problem in self.validation_problems:
                # Create FEM solver for this problem
                fem_solver = FEMSolver(problem.mesh, alpha=problem.alpha)
                
                T0_tensor = torch.tensor(
                    problem.initial_condition, dtype=torch.float32, device=self.device
                )
                T_current = T0_tensor.clone()
                
                for step in range(num_timesteps):
                    t_current = step * self.config.dt
                    t_prev = max(0, (step - 1) * self.config.dt)
                    
                    # Build graph features using current state - ONLY for free DOFs
                    data = build_features(
                        problem.graph_data, T_current.detach().cpu().numpy(),
                        problem.alpha, self.config.dt, self.device,
                        fem_solver=fem_solver, free_dofs_only=True
                    )
                    
                    # Forward pass - model outputs FREE DOFs only
                    predictions_free = self.model(data).squeeze()
                    
                    # Compute losses using strict BC enforcement
                    losses = self.compute_losses(problem, T_current, predictions_free, t_current, t_prev)
                    
                    # Total loss - only PDE loss needed with strict BC enforcement
                    total_loss = self.config.lambda_pde * losses['pde']
                    if step == 0:
                        total_loss += self.config.lambda_ic * losses['ic']
                    
                    # Accumulate losses
                    for key in ['pde', 'bc', 'ic']:
                        val_losses[key] += losses[key].item()
                    val_losses['total'] += total_loss.item()
                    
                    step_count += 1
                    
                    # Reconstruct full solution for next step using proper boundary conditions
                    T_current = reconstruct_full_solution(
                        predictions_free, problem, t_current + self.config.dt, fem_solver, self.device
                    ).detach()  # Detach for safety even though we're in no_grad context
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= step_count
        
        return val_losses
    
    def train(self):
        """Main training loop."""
        print(f"Starting multi-mesh training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # Train one epoch
            epoch_losses = self.train_epoch()
            
            # Validate
            # val_losses = self.validate_epoch()
            val_losses = {'total': 0.0, 'pde': 0.0, 'bc': 0.0, 'ic': 0.0}  # Skip validation for speed
            
            # Update learning rate
            self.scheduler.step()
            
            # Log progress
            if epoch % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:4d} | "
                      f"Train Loss: {epoch_losses['total']:.3e} | "
                      f"Val Loss: {val_losses['total']:.3e} | "
                      f"PDE: {epoch_losses['pde']:.3e} | "
                      f"IC: {epoch_losses['ic']:.3e} | "
                      f"Time: {elapsed:.2f}s")
            
            # Save training history
            self.training_history['epoch'].append(epoch)
            for key in ['total', 'pde', 'bc', 'ic']:
                self.training_history[f'loss_{key}'].append(epoch_losses[key])
            self.training_history['val_loss'].append(val_losses['total'])
            
            # Save model checkpoint
            if epoch > 0 and epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoints/checkpoint_epoch_{epoch}.pth")
        
        print("Multi-mesh training completed!")

    def rollout(self, problem: MeshProblem, T_start=None):
        """
        Perform rollout simulation using trained model with strict BC enforcement.
        
        Args:
            problem: The mesh problem to simulate
            T_start: Starting temperature field (uses problem's T0 if None)
        
        Returns:
            List of temperature states over time
        """
        self.model.eval()
        
        # Create FEM solver for this problem
        fem_solver = FEMSolver(problem.mesh, alpha=problem.alpha)
        
        if T_start is None:
            T_current = torch.tensor(
                problem.initial_condition, dtype=torch.float32, device=self.device
            )
        else:
            T_current = torch.tensor(T_start, dtype=torch.float32, device=self.device)
        
        # Calculate number of time steps
        num_timesteps = int(self.config.t_final / self.config.dt)
        
        states = [T_current.detach().cpu().numpy()]
        
        with torch.no_grad():
            for step in range(num_timesteps):
                t_current = step * self.config.dt
                
                # Build graph features from the current state - FREE DOFs only
                data = build_features(
                    problem.graph_data, T_current.detach().cpu().numpy(),
                    problem.alpha, self.config.dt, self.device,
                    fem_solver=fem_solver, free_dofs_only=True
                )
                
                # Forward pass - model outputs FREE DOFs only
                predictions_free = self.model(data).squeeze()
                
                # Reconstruct full solution with strict BC enforcement
                T_next = reconstruct_full_solution(
                    predictions_free, problem, t_current + self.config.dt, fem_solver, self.device
                ).detach()
                
                # Store state
                states.append(T_next.cpu().numpy())

                # Update for next iteration
                T_current = T_next
        
        return states
    
    def compute_ground_truth(self, problem: MeshProblem):
        """
        Compute ground truth solution using FEM for a specific problem.
        
        Args:
            problem: The mesh problem to solve
        
        Returns:
            List of temperature states over time
        """
        num_timesteps = int(self.config.t_final / self.config.dt)
        
        T_current = problem.initial_condition.copy()
        states = [T_current.copy()]
        
        # Create boundary value dict for FEM solver
        boundary_indices = np.where(problem.graph_data['boundary_mask'])[0]
        boundary_values_dict = {idx: problem.initial_condition[idx] for idx in boundary_indices}
        
        # Initialize FEM solver for this specific problem
        fem_solver = FEMSolver(problem.mesh, alpha=problem.alpha)

        states = fem_solver.solve_time_problem(T0=T_current, dt=self.config.dt, nsamples=num_timesteps, t_final=self.config.t_final)

        return states
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'num_training_problems': len(self.training_problems),
            'num_validation_problems': len(self.validation_problems)
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint['training_history']
        print(f"Checkpoint loaded: {filepath}")
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        loss_names = ['total', 'pde', 'bc', 'ic']
        
        for i, loss_name in enumerate(loss_names):
            ax = axes[i]
            epochs = self.training_history['epoch']
            losses = self.training_history[f'loss_{loss_name}']
            ax.plot(epochs, losses, label=f'Train {loss_name.upper()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(f'Loss {loss_name.upper()}')
            ax.set_title(f'{loss_name.upper()} Loss')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.legend()
        
        # Validation loss
        ax = axes[4]
        epochs = self.training_history['epoch']
        train_losses = self.training_history['loss_total']
        val_losses = self.training_history['val_loss']
        ax.plot(epochs, train_losses, label='Train Total')
        ax.plot(epochs, val_losses, label='Validation Total')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Training vs Validation Loss')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.legend()
        
        # Remove empty subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history saved: {save_path}")
        
        # plt.show()


def create_multi_mesh_trainer(config: TrainingConfig, seed: int = 42) -> PIGNNTrainer:
    """
    Create a PI-GNN trainer with diverse mesh problems.
    
    Args:
        config: Training configuration with mesh configs
        seed: Random seed for reproducibility
    
    Returns:
        PIGNNTrainer configured for multi-mesh training
    """
    # Generate all problems
    generator = MeshProblemGenerator(
        mesh_configs=config.mesh_configs,
        num_problems_per_config=config.num_problems,
        seed=seed
    )
    
    all_problems = generator.generate_all_problems()
    
    # Split into training and validation
    random.seed(seed)
    random.shuffle(all_problems)
    
    n_train = int(len(all_problems) * config.train_split)
    training_problems = all_problems[:n_train]
    validation_problems = all_problems[n_train:]
    
    # Create trainer
    trainer = PIGNNTrainer(
        training_problems=training_problems,
        validation_problems=validation_problems,
        config=config
    )
    
    return trainer
