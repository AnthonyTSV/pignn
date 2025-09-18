"""
Main training script for Physics-Informed Graph Neural Network (PI-GNN) 
with multi-mesh support for diverse geometries and parameters.

Based on the methodology from the paper with support for training on:
- Multiple rectangular domains with varying dimensions
- L-shaped domains with random parameters
- Different mesh resolutions (maxh)
- Diverse initial conditions and physics parameters
"""

import torch
import numpy as np
import argparse
import os
from pathlib import Path
import random

# Import our PI-GNN implementation
from pimgn import (
    PIGNNTrainer, TrainingConfig, MeshConfig,
    create_rectangular_mesh, create_lshape_mesh, build_graph_from_mesh,
    create_heatmap_gif
)
from pimgn.training.trainer import create_multi_mesh_trainer
from pimgn.utils.mesh_utils import create_gaussian_initial_condition
from pimgn.utils.visualization import (
    extract_triangles_from_mesh, compute_l2_error_timeseries,
    plot_l2_error_evolution, plot_single_heatmap, export_to_vtk
)


def create_mesh_configs(args):
    """Create mesh configurations based on command line arguments."""
    mesh_configs = []
    
    if args.use_rectangular:
        mesh_configs.append(MeshConfig(
            mesh_type='rectangular',
            maxh_range=(args.min_maxh, args.max_maxh),
            width_range=(args.min_width, args.max_width),
            height_range=(args.min_height, args.max_height),
            num_gaussians_range=(args.min_gaussians, args.max_gaussians),
            amplitude_range=(args.min_amplitude, args.max_amplitude),
            sigma_fraction_range=(args.min_sigma_frac, args.max_sigma_frac),
            alpha_range=(args.min_alpha, args.max_alpha)
        ))
    
    if args.use_lshape:
        mesh_configs.append(MeshConfig(
            mesh_type='lshape',
            maxh_range=(args.min_maxh, args.max_maxh),
            num_gaussians_range=(args.min_gaussians, args.max_gaussians),
            amplitude_range=(args.min_amplitude, args.max_amplitude),
            sigma_fraction_range=(args.min_sigma_frac, args.max_sigma_frac),
            alpha_range=(args.min_alpha, args.max_alpha)
        ))
    
    if args.use_polygon:
        mesh_configs.append(MeshConfig(
            mesh_type='polygon',
            maxh_range=(args.min_maxh, args.max_maxh),
            num_points_range=(args.min_polygon_points, args.max_polygon_points),
            domain_size_range=(args.min_domain_size, args.max_domain_size),
            num_gaussians_range=(args.min_gaussians, args.max_gaussians),
            amplitude_range=(args.min_amplitude, args.max_amplitude),
            sigma_fraction_range=(args.min_sigma_frac, args.max_sigma_frac),
            alpha_range=(args.min_alpha, args.max_alpha)
        ))
    
    if args.use_circle:
        mesh_configs.append(MeshConfig(
            mesh_type='circle',
            maxh_range=(args.min_maxh, args.max_maxh),
            radius_range=(args.min_radius, args.max_radius),
            center_x_range=(args.min_center, args.max_center),
            center_y_range=(args.min_center, args.max_center),
            num_gaussians_range=(args.min_gaussians, args.max_gaussians),
            amplitude_range=(args.min_amplitude, args.max_amplitude),
            sigma_fraction_range=(args.min_sigma_frac, args.max_sigma_frac),
            alpha_range=(args.min_alpha, args.max_alpha)
        ))
    
    if args.use_hollow_circle:
        mesh_configs.append(MeshConfig(
            mesh_type='hollow_circle',
            maxh_range=(args.min_maxh, args.max_maxh),
            outer_radius_range=(args.min_outer_radius, args.max_outer_radius),
            inner_radius_ratio_range=(args.min_inner_ratio, args.max_inner_ratio),
            num_gaussians_range=(args.min_gaussians, args.max_gaussians),
            amplitude_range=(args.min_amplitude, args.max_amplitude),
            sigma_fraction_range=(args.min_sigma_frac, args.max_sigma_frac),
            alpha_range=(args.min_alpha, args.max_alpha)
        ))
    
    if not mesh_configs:
        raise ValueError("At least one mesh type must be enabled!")
    
    return mesh_configs


def main():
    parser = argparse.ArgumentParser(description='Train PI-GNN on multiple mesh types')
    
    # Mesh type selection
    parser.add_argument('--use_rectangular', action='store_true', default=True,
                        help='Include rectangular meshes in training')
    parser.add_argument('--use_lshape', action='store_true', default=False,
                        help='Include L-shaped meshes in training')
    parser.add_argument('--use_polygon', action='store_true', default=False,
                        help='Include random convex polygon meshes in training')
    parser.add_argument('--use_circle', action='store_true', default=False,
                        help='Include circular meshes in training')
    parser.add_argument('--use_hollow_circle', action='store_true', default=False,
                        help='Include hollow circle (annulus) meshes in training')
    
    # Mesh parameter ranges
    parser.add_argument('--min_maxh', type=float, default=0.5, help='Minimum element size')
    parser.add_argument('--max_maxh', type=float, default=0.5, help='Maximum element size')
    
    # Rectangular mesh parameters
    parser.add_argument('--min_width', type=float, default=2, help='Minimum domain width')
    parser.add_argument('--max_width', type=float, default=2, help='Maximum domain width')
    parser.add_argument('--min_height', type=float, default=2, help='Minimum domain height')
    parser.add_argument('--max_height', type=float, default=2, help='Maximum domain height')

    # Polygon mesh parameters
    parser.add_argument('--min_polygon_points', type=int, default=5, help='Minimum polygon points')
    parser.add_argument('--max_polygon_points', type=int, default=9, help='Maximum polygon points')
    parser.add_argument('--min_domain_size', type=float, default=0.8, help='Minimum polygon domain size')
    parser.add_argument('--max_domain_size', type=float, default=1.2, help='Maximum polygon domain size')
    
    # Circle mesh parameters
    parser.add_argument('--min_radius', type=float, default=0.3, help='Minimum circle radius')
    parser.add_argument('--max_radius', type=float, default=0.7, help='Maximum circle radius')
    parser.add_argument('--min_center', type=float, default=0.3, help='Minimum circle center coordinate')
    parser.add_argument('--max_center', type=float, default=0.7, help='Maximum circle center coordinate')
    
    # Hollow circle parameters
    parser.add_argument('--min_outer_radius', type=float, default=0.4, help='Minimum outer radius')
    parser.add_argument('--max_outer_radius', type=float, default=0.8, help='Maximum outer radius')
    parser.add_argument('--min_inner_ratio', type=float, default=0.3, help='Minimum inner/outer radius ratio')
    parser.add_argument('--max_inner_ratio', type=float, default=0.7, help='Maximum inner/outer radius ratio')
    
    # Physics parameter ranges
    parser.add_argument('--min_alpha', type=float, default=1, help='Minimum thermal diffusivity')
    parser.add_argument('--max_alpha', type=float, default=1, help='Maximum thermal diffusivity')
    
    # Time parameters
    parser.add_argument('--t_final', type=float, default=1.0, help='Final time')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step size')
    
    # Initial condition parameter ranges
    parser.add_argument('--min_gaussians', type=int, default=1, help='Minimum number of Gaussian peaks')
    parser.add_argument('--max_gaussians', type=int, default=1, help='Maximum number of Gaussian peaks')
    parser.add_argument('--min_amplitude', type=float, default=1.0, help='Minimum Gaussian amplitude')
    parser.add_argument('--max_amplitude', type=float, default=1.0, help='Maximum Gaussian amplitude')
    parser.add_argument('--min_sigma_frac', type=float, default=0.05, help='Minimum sigma fraction')
    parser.add_argument('--max_sigma_frac', type=float, default=0.15, help='Maximum sigma fraction')
    
    # Training dataset parameters
    parser.add_argument('--num_problems', type=int, default=2, help='Number of problems per mesh type')
    parser.add_argument('--train_split', type=float, default=0.75, help='Fraction of problems for training')

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of message passing layers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (problems per batch)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu/auto)')
    
    # Loss weights
    parser.add_argument('--lambda_pde', type=float, default=1.0, help='PDE loss weight')
    # NOTE: Boundary condition loss removed due to strict BC enforcement
    # parser.add_argument('--lambda_bc', type=float, default=1.0, help='Boundary condition loss weight') 
    parser.add_argument('--lambda_ic', type=float, default=1.0, help='Initial condition loss weight')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--save_animations', action='store_true', default=True, help='Save heatmap animations')
    parser.add_argument('--log_interval', type=int, default=1, help='Logging interval')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 10000), help='Random seed')

    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("PHYSICS-INFORMED MESGGRAPHNET MULTI-MESH TRAINING")
    print("Heat Equation on Diverse Mesh Geometries")
    print("="*80)
    
    # Create mesh configurations
    mesh_configs = create_mesh_configs(args)
    
    print(f"Mesh types to train on:")
    for config in mesh_configs:
        print(f"  - {config.mesh_type}")
        print(f"    maxh range: {config.maxh_range}")
        if config.mesh_type == 'rectangular':
            print(f"    width range: {config.width_range}")
            print(f"    height range: {config.height_range}")
        elif config.mesh_type == 'polygon':
            print(f"    points range: {config.num_points_range}")
            print(f"    domain size range: {config.domain_size_range}")
        elif config.mesh_type == 'circle':
            print(f"    radius range: {config.radius_range}")
            print(f"    center x range: {config.center_x_range}")
            print(f"    center y range: {config.center_y_range}")
        elif config.mesh_type == 'hollow_circle':
            print(f"    outer radius range: {config.outer_radius_range}")
            print(f"    inner ratio range: {config.inner_radius_ratio_range}")
        print(f"    alpha range: {config.alpha_range}")
        print(f"    gaussians range: {config.num_gaussians_range}")
    
    # Configure training
    config = TrainingConfig(
        # Time parameters
        t_final=args.t_final,
        dt=args.dt,
        
        # Model parameters
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        
        # Training parameters
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        
        # Multi-mesh parameters
        num_problems=args.num_problems,
        train_split=args.train_split,
        mesh_configs=mesh_configs,
        
        # Loss weights
        lambda_pde=args.lambda_pde,
        # NOTE: lambda_bc removed due to strict BC enforcement
        # lambda_bc=args.lambda_bc,
        lambda_ic=args.lambda_ic,
        
        # Device and logging
        device=device,
        log_interval=args.log_interval
    )
    
    # Create multi-mesh trainer
    print("Creating multi-mesh trainer...")
    trainer = create_multi_mesh_trainer(config, seed=args.seed)
    
    # Train model
    print("\nStarting multi-mesh training...")
    trainer.train()
    
    # Save final model
    model_path = output_dir / "pignn_multimesh_model.pth"
    trainer.save_checkpoint(str(model_path))
    
    # Plot training history
    history_path = output_dir / "training_history.png"
    trainer.plot_training_history(str(history_path))
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Training history saved to: {history_path}")
    
    # Optionally test on a single problem
    if args.save_animations:
        print("\nGenerating test animation...")
        test_problem = trainer.validation_problems[0]
        train_problem = trainer.training_problems[0]
        for problem in [("train", train_problem), ("test", test_problem)]:
            type_name, test_problem = problem
            print(f"\n{type_name} Problem:")
            # print(f"Test problem mesh type: {test_problem.mesh_type}, num nodes: {test_problem.graph_data['pos'].shape[0]}")
            predicted_states = trainer.rollout(test_problem)
            ground_truth_states = trainer.compute_ground_truth(test_problem)
            
            # Save animation
            vtk_path = output_dir / "vtk" / type_name
            from ngsolve import H1
            time_steps = list(np.arange(0, config.t_final + config.dt, config.dt))
            export_to_vtk(
                test_problem.mesh, H1(test_problem.mesh, order=1), 
                np.array(ground_truth_states), np.array(predicted_states), 
                time_steps,
                filename=str(vtk_path / "result")
            )
            print(f"Test animation saved to: {vtk_path}")
            if type_name == "test" and args.save_animations:
                exact_sequence = np.array(ground_truth_states)
                predicted_sequence = np.array(predicted_states)
                
                # Compute errors
                l2_errors = compute_l2_error_timeseries(exact_sequence, predicted_sequence)

                print("Plotting L2 error evolution...")
                plot_l2_error_evolution(
                    l2_errors, dt=config.dt,
                    title="L2 Error Evolution (PI-GNN vs FEM)",
                    save_path=str(output_dir / "l2_error_evolution.png")
                )


if __name__ == "__main__":
    main()
