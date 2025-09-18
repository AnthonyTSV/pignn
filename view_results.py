"""
Results viewer for PI-GNN heat equation simulation.
Displays and analyzes the trained model results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def view_results(results_dir="results"):
    """
    View and analyze PI-GNN results.
    
    Args:
        results_dir: Directory containing results
    """
    results_path = Path(results_dir) / "results.npz"
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run train_pignn.py first to generate results.")
        return
    
    # Load results
    print(f"Loading results from {results_path}")
    data = np.load(str(results_path))
    
    positions = data['positions']
    exact_sequence = data['exact_sequence']
    predicted_sequence = data['predicted_sequence']
    l2_errors = data['l2_errors']
    
    n_timesteps = len(exact_sequence)
    n_nodes = positions.shape[0]
    dt = 0.01  # Default time step
    
    print(f"\nResults Summary:")
    print(f"  Number of nodes: {n_nodes}")
    print(f"  Number of timesteps: {n_timesteps}")
    print(f"  Simulation time: {(n_timesteps-1) * dt:.2f}s")
    
    # Error statistics
    print(f"\nError Statistics:")
    print(f"  Initial L2 error: {l2_errors[0]:.3e}")
    print(f"  Final L2 error: {l2_errors[-1]:.3e}")
    print(f"  Mean L2 error: {l2_errors.mean():.3e}")
    print(f"  Max L2 error: {l2_errors.max():.3e}")
    print(f"  Min L2 error: {l2_errors.min():.3e}")
    print(f"  Std L2 error: {l2_errors.std():.3e}")
    
    # Temperature statistics
    exact_final = exact_sequence[-1]
    pred_final = predicted_sequence[-1]
    
    print(f"\nFinal Temperature Statistics:")
    print(f"  FEM solution - Min: {exact_final.min():.3f}, Max: {exact_final.max():.3f}, Mean: {exact_final.mean():.3f}")
    print(f"  PI-GNN pred - Min: {pred_final.min():.3f}, Max: {pred_final.max():.3f}, Mean: {pred_final.mean():.3f}")
    print(f"  Difference  - Min: {(exact_final-pred_final).min():.3f}, Max: {(exact_final-pred_final).max():.3f}")
    
    # Energy conservation analysis
    exact_energy = np.array([np.sum(state) for state in exact_sequence])
    pred_energy = np.array([np.sum(state) for state in predicted_sequence])
    energy_diff = np.abs(exact_energy - pred_energy)
    
    print(f"\nEnergy Conservation:")
    print(f"  Initial energy diff: {energy_diff[0]:.3e}")
    print(f"  Final energy diff: {energy_diff[-1]:.3e}")
    print(f"  Max energy diff: {energy_diff.max():.3e}")
    print(f"  Energy conservation error: {energy_diff[-1] / exact_energy[0] * 100:.3f}%")
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # L2 error evolution
    times = np.arange(n_timesteps) * dt
    axes[0, 0].plot(times, l2_errors, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('L2 Error')
    axes[0, 0].set_title('L2 Error Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Energy conservation
    axes[0, 1].plot(times, exact_energy, 'b-', label='FEM', linewidth=2)
    axes[0, 1].plot(times, pred_energy, 'r--', label='PI-GNN', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Total Energy')
    axes[0, 1].set_title('Energy Conservation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Temperature evolution at center node
    center_idx = n_nodes // 2  # Approximate center
    axes[1, 0].plot(times, exact_sequence[:, center_idx], 'b-', label='FEM', linewidth=2)
    axes[1, 0].plot(times, predicted_sequence[:, center_idx], 'r--', label='PI-GNN', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Temperature')
    axes[1, 0].set_title(f'Temperature at Node {center_idx}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error distribution at final time
    final_errors = np.abs(exact_final - pred_final)
    axes[1, 1].hist(final_errors, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Final Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save analysis plot
    analysis_path = Path(results_dir) / "results_analysis.png"
    plt.savefig(str(analysis_path), dpi=150, bbox_inches='tight')
    print(f"\nAnalysis plot saved: {analysis_path}")
    
    plt.show()
    
    # List available visualization files
    results_path_obj = Path(results_dir)
    viz_files = []
    for ext in ['.png', '.gif']:
        viz_files.extend(list(results_path_obj.glob(f"*{ext}")))
    
    if viz_files:
        print(f"\nAvailable visualization files in {results_dir}:")
        for f in sorted(viz_files):
            print(f"  {f.name}")
    
    return {
        'l2_errors': l2_errors,
        'energy_conservation_error': energy_diff[-1] / exact_energy[0],
        'final_mean_error': np.mean(np.abs(exact_final - pred_final)),
        'results_dir': results_dir
    }


def compare_training_configs():
    """Compare results from different training configurations if available."""
    results_dirs = ['results', 'evaluation_results', 'example_results']
    available_dirs = [d for d in results_dirs if Path(d).exists()]
    
    if len(available_dirs) < 2:
        print("Multiple result directories not found for comparison.")
        return
    
    print("\nComparing results from different runs:")
    
    all_results = {}
    for dir_name in available_dirs:
        try:
            results = view_results(dir_name)
            all_results[dir_name] = results
            print(f"\n{dir_name.upper()}:")
            print(f"  Final L2 error: {results['l2_errors'][-1]:.3e}")
            print(f"  Mean final error: {results['final_mean_error']:.3e}")
            print(f"  Energy conservation: {results['energy_conservation_error']*100:.3f}%")
        except Exception as e:
            print(f"Failed to load {dir_name}: {e}")
    
    return all_results


def main():
    """Main results viewing function."""
    print("="*60)
    print("PI-GNN Results Viewer")
    print("="*60)
    
    # View main results
    results = view_results("results")
    
    # Try to compare multiple result sets
    try:
        compare_training_configs()
    except Exception as e:
        print(f"Comparison failed: {e}")
    
    print("\n" + "="*60)
    print("Results viewing completed!")
    print("="*60)


if __name__ == "__main__":
    main()
