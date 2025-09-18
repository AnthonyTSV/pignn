"""
Evaluation script for trained PI-GNN model.
Loads a trained model and evaluates it against FEM ground truth on a single mesh with custom initial conditions.
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from pimgn import PIGNNTrainer, create_rectangular_mesh, build_graph_from_mesh, create_lshape_mesh
from pimgn.utils.mesh_utils import create_gaussian_initial_condition
from pimgn.utils.visualization import (
    extract_triangles_from_mesh, compute_l2_error_timeseries,
    plot_l2_error_evolution, plot_single_heatmap, create_heatmap_gif, export_to_vtk
)

# Import mesh creation functions from pimgn
try:
    from pimgn.utils.mesh_utils import create_circle_mesh, create_polygon_mesh, create_hollow_circle_mesh
except ImportError:
    print("Warning: Some mesh types may not be available")
    create_circle_mesh = None
    create_polygon_mesh = None
    create_hollow_circle_mesh = None


def create_mesh_from_args(args):
    """Create mesh based on command line arguments."""
    print(f"Creating {args.mesh_type} mesh...")
    
    if args.mesh_type == 'rectangular':
        mesh = create_rectangular_mesh(
            width=args.width, 
            height=args.height, 
            maxh=args.maxh
        )
    elif args.mesh_type == 'lshape':
        mesh = create_lshape_mesh(
            maxh=args.maxh, 
            seed=args.mesh_seed
        )
    elif args.mesh_type == 'circle':
        if create_circle_mesh is None:
            raise ValueError("Circle mesh creation not available. Check pimgn implementation.")
        mesh = create_circle_mesh(
            radius=args.radius,
            center_x=args.center_x,
            center_y=args.center_y,
            maxh=args.maxh
        )
    elif args.mesh_type == 'polygon':
        if create_polygon_mesh is None:
            raise ValueError("Polygon mesh creation not available. Check pimgn implementation.")
        mesh = create_polygon_mesh(
            num_points=args.num_points,
            domain_size=args.domain_size,
            maxh=args.maxh,
            seed=args.mesh_seed
        )
    elif args.mesh_type == 'hollow_circle':
        if create_hollow_circle_mesh is None:
            raise ValueError("Hollow circle mesh creation not available. Check pimgn implementation.")
        mesh = create_hollow_circle_mesh(
            outer_radius=args.outer_radius,
            inner_radius_ratio=args.inner_radius_ratio,
            maxh=args.maxh
        )
    else:
        raise ValueError(f"Unknown mesh type: {args.mesh_type}")
    
    return mesh


def create_initial_condition_from_args(graph_data, args):
    """Create initial condition based on command line arguments."""
    print(f"Creating initial condition with {args.num_gaussians} Gaussian peaks...")
    
    T0 = create_gaussian_initial_condition(
        graph_data['pos'], 
        num_gaussians=args.num_gaussians,
        amplitude_range=(args.min_amplitude, args.max_amplitude),
        sigma_fraction_range=(args.min_sigma_frac, args.max_sigma_frac),
        seed=args.ic_seed
    )
    
    return T0


def run_inference(model_path, mesh, graph_data, T0, config_override=None, output_dir="inference_results", save_visualizations=True):
    """
    Run inference with a trained PI-GNN model on a single mesh.
    
    Args:
        model_path: Path to the saved model checkpoint
        mesh: NGSolve mesh object
        graph_data: Graph representation of the mesh
        T0: Initial condition
        config_override: Dictionary to override config parameters
        output_dir: Directory to save results
        save_visualizations: Whether to save visualizations
    """
    
    print("Loading trained PI-GNN model...")
    print(f"Model path: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Override config if provided
    if config_override:
        for key, value in config_override.items():
            setattr(config, key, value)
    
    print(f"Model configuration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    
    print(f"Mesh statistics:")
    print(f"  Nodes: {graph_data['pos'].shape[0]}")
    print(f"  Edges: {graph_data['edge_index'].shape[1]}")
    print(f"  Boundary nodes: {graph_data['boundary_mask'].sum()}")
    
    # Create a single MeshProblem for inference
    from pimgn.training.trainer import MeshProblem, MeshConfig
    
    # Create a dummy mesh config (not used during inference)
    mesh_config = MeshConfig(mesh_type='custom')
    
    # Create problem with alpha from config (or use default)
    alpha = getattr(config, 'alpha', 1.0)
    if hasattr(config, 'alpha_range'):
        alpha = (config.alpha_range[0] + config.alpha_range[1]) / 2  # Use middle value
    
    problem = MeshProblem(
        mesh=mesh,
        graph_data=graph_data,
        initial_condition=T0,
        alpha=alpha,
        mesh_config=mesh_config,
        problem_id=0
    )
    
    # Create trainer with single problem
    trainer = PIGNNTrainer(
        training_problems=[problem],
        validation_problems=[problem],
        config=config
    )
    
    # Load the trained weights
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    if 'training_history' in checkpoint:
        trainer.training_history = checkpoint['training_history']
    
    print(f"Model loaded successfully!")
    if 'training_history' in checkpoint:
        print(f"Training epochs completed: {len(trainer.training_history['epoch'])}")
    
    # Run inference
    print("\nRunning inference...")
    
    # Compute predictions
    print("Computing PI-GNN predictions...")
    predicted_states = trainer.rollout(problem)
    
    # Compute ground truth
    print("Computing FEM ground truth...")
    exact_states = trainer.compute_ground_truth(problem)
    
    # Ensure same number of timesteps
    min_steps = min(len(exact_states), len(predicted_states))
    exact_states = exact_states[:min_steps]
    predicted_states = predicted_states[:min_steps]
    
    if save_visualizations:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export to VTK format
        print("Exporting results to VTK format...")
        time_steps = np.linspace(0, min_steps * config.dt, min_steps)
        
        # Create FEM solver for VTK export
        from pimgn.utils.fem_utils import FEMSolver
        fem_solver = FEMSolver(mesh, alpha=alpha)
        filename = output_path / "vtk/results"
        export_to_vtk(mesh, fem_solver.fes, exact_states, predicted_states, time_steps=time_steps, filename=filename)

    # Convert to numpy arrays
    exact_sequence = np.array(exact_states)
    predicted_sequence = np.array(predicted_states)
    
    # Compute errors
    l2_errors = compute_l2_error_timeseries(exact_sequence, predicted_sequence)
    
    print(f"\nInference Results:")
    print(f"  Number of timesteps: {len(l2_errors)}")
    print(f"  Initial L2 error: {l2_errors[0]:.3e}")
    print(f"  Final L2 error: {l2_errors[-1]:.3e}")
    print(f"  Mean L2 error: {l2_errors.mean():.3e}")
    print(f"  Max L2 error: {l2_errors.max():.3e}")
    print(f"  Std L2 error: {l2_errors.std():.3e}")
    
    if save_visualizations:
        # Plot L2 error evolution
        print("Plotting L2 error evolution...")
        plot_l2_error_evolution(
            l2_errors, dt=config.dt,
            title="L2 Error Evolution (PI-GNN vs FEM)",
            save_path=str(output_path / "l2_error_evolution.png")
        )

        print(f"Results saved in: {output_path}")
    
    return {
        'l2_errors': l2_errors,
        'exact_sequence': exact_sequence,
        'predicted_sequence': predicted_sequence,
        'mesh_info': {
            'n_nodes': graph_data['pos'].shape[0],
            'n_edges': graph_data['edge_index'].shape[1],
            'n_boundary': graph_data['boundary_mask'].sum()
        }
    }


def evaluate_model(model_path, config_override=None):
    """
    Evaluate a trained PI-GNN model (legacy function for backward compatibility).
    """
    # Create default rectangular mesh and initial condition
    mesh = create_rectangular_mesh(maxh=0.1, seed=42)
    graph_data = build_graph_from_mesh(mesh)
    T0 = create_gaussian_initial_condition(graph_data['pos'], num_gaussians=1)
    
    return run_inference(model_path, mesh, graph_data, T0, config_override)


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained PI-GNN model on a single mesh')
    
    # Model and device
    parser.add_argument('--model_path', type=str, default='results/pignn_multimesh_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for evaluation (cuda/cpu/auto)')
    
    # Mesh type and parameters
    parser.add_argument('--mesh_type', type=str, default='lshape', 
                       choices=['rectangular', 'lshape', 'circle', 'polygon', 'hollow_circle'],
                       help='Type of mesh to create')
    parser.add_argument('--maxh', type=float, default=0.3, help='Maximum element size')
    parser.add_argument('--mesh_seed', type=int, default=42, help='Random seed for mesh generation')
    
    # Rectangular mesh parameters
    parser.add_argument('--width', type=float, default=2.0, help='Domain width (rectangular)')
    parser.add_argument('--height', type=float, default=2.0, help='Domain height (rectangular)')
    
    # Circle mesh parameters
    parser.add_argument('--radius', type=float, default=0.5, help='Circle radius')
    parser.add_argument('--center_x', type=float, default=0.5, help='Circle center x-coordinate')
    parser.add_argument('--center_y', type=float, default=0.5, help='Circle center y-coordinate')
    
    # Polygon mesh parameters
    parser.add_argument('--num_points', type=int, default=6, help='Number of polygon vertices')
    parser.add_argument('--domain_size', type=float, default=1.0, help='Polygon domain size')
    
    # Hollow circle parameters
    parser.add_argument('--outer_radius', type=float, default=0.6, help='Outer radius (hollow circle)')
    parser.add_argument('--inner_radius_ratio', type=float, default=0.5, help='Inner/outer radius ratio')
    
    # Initial condition parameters
    parser.add_argument('--num_gaussians', type=int, default=3, help='Number of Gaussian peaks')
    parser.add_argument('--min_amplitude', type=float, default=0.5, help='Minimum Gaussian amplitude')
    parser.add_argument('--max_amplitude', type=float, default=1.0, help='Maximum Gaussian amplitude')
    parser.add_argument('--min_sigma_frac', type=float, default=0.08, help='Minimum sigma fraction')
    parser.add_argument('--max_sigma_frac', type=float, default=0.12, help='Maximum sigma fraction')
    parser.add_argument('--ic_seed', type=int, default=123, help='Random seed for initial condition')
    
    # Physics parameters (overrides)
    parser.add_argument('--alpha', type=float, default=None, help='Thermal diffusivity (override)')
    parser.add_argument('--t_final', type=float, default=None, help='Final time (override)')
    parser.add_argument('--dt', type=float, default=None, help='Time step (override)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--no_visualizations', action='store_true', help='Skip saving visualizations')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PI-GNN INFERENCE ON SINGLE MESH")
    print("="*80)
    print(f"Mesh type: {args.mesh_type}")
    print(f"Model: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Create mesh
    mesh = create_rectangular_mesh(width=2, height=2, maxh=0.1)
    graph_data = build_graph_from_mesh(mesh)
    
    # Create initial condition
    # T0 = create_gaussian_initial_condition(graph_data['pos'], num_gaussians=1)
    a = np.random.randint(1, 5)
    x = graph_data['pos'].T
    T0 = np.exp(-a * (x[0]**2 + x[1]**2))
    
    # Set up config overrides
    config_override = {}
    if args.device != 'auto':
        config_override['device'] = args.device
    if args.alpha is not None:
        config_override['alpha'] = args.alpha
    if args.t_final is not None:
        config_override['t_final'] = args.t_final
    if args.dt is not None:
        config_override['dt'] = args.dt
    
    # Run inference
    results = run_inference(
        args.model_path, 
        mesh, 
        graph_data, 
        T0, 
        config_override=config_override,
        output_dir=args.output_dir,
        save_visualizations=not args.no_visualizations
    )
    
    print(f"\nFinal Summary:")
    print(f"  Mesh nodes: {results['mesh_info']['n_nodes']}")
    print(f"  Mesh edges: {results['mesh_info']['n_edges']}")
    print(f"  Boundary nodes: {results['mesh_info']['n_boundary']}")
    print(f"  Mean L2 Error: {results['l2_errors'].mean():.3e}")
    print(f"  Final L2 Error: {results['l2_errors'][-1]:.3e}")
    print(f"  Inference completed successfully!")


if __name__ == "__main__":
    main()
