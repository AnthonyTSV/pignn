"""
Visualization utilities for creating heatmap animations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import TriMesh
import matplotlib.tri as tri
from PIL import Image
import os
from ngsolve import Mesh, GridFunction, VTKOutput, Parameter


def create_heatmap_gif(positions, triangles, temperature_sequences, 
                      titles=None, filename='heatmap_comparison.gif',
                      fps=10, figsize=(16, 5), vmin=None, vmax=None):
    """
    Create animated GIF comparing temperature solutions.
    
    Args:
        positions: Node positions (N, 2)
        triangles: Triangle connectivity (M, 3)
        temperature_sequences: List of temperature sequences, each (T, N)
                              [exact_sequence, predicted_sequence, difference_sequence]
        titles: List of subplot titles
        filename: Output filename
        fps: Frames per second
        figsize: Figure size
        vmin, vmax: Color scale limits
    
    Returns:
        None (saves GIF file)
    """
    n_sequences = len(temperature_sequences)
    n_timesteps = temperature_sequences[0].shape[0]
    
    if titles is None:
        titles = [f'Sequence {i+1}' for i in range(n_sequences)]
    
    # Determine color scale if not provided
    if vmin is None or vmax is None:
        all_temps = np.concatenate([seq.flatten() for seq in temperature_sequences])
        if vmin is None:
            vmin = np.min(all_temps)
        if vmax is None:
            vmax = np.max(all_temps)
    
    # Create triangulation
    triang = tri.Triangulation(positions[:, 0], positions[:, 1], triangles)
    
    # Set up figure and subplots
    fig, axes = plt.subplots(1, n_sequences, figsize=figsize)
    if n_sequences == 1:
        axes = [axes]
    
    # Initialize plots
    plots = []
    for i, ax in enumerate(axes):
        plot = ax.tripcolor(triang, temperature_sequences[i][0], 
                           vmin=vmin, vmax=vmax, cmap='coolwarm', shading='flat')
        ax.set_title(f'{titles[i]} - t=0.00')
        ax.set_aspect('equal')
        ax.axis('off')
        plots.append(plot)
    
    # Add colorbar
    plt.tight_layout()
    cbar = plt.colorbar(plots[0], ax=axes, orientation='horizontal', 
                       fraction=0.05, pad=0.1)
    cbar.set_label('Temperature')
    
    def animate(frame):
        for i, (plot, ax) in enumerate(zip(plots, axes)):
            # Clear previous plot
            plot.remove()
            
            # Create new plot for this frame
            new_plot = ax.tripcolor(triang, temperature_sequences[i][frame], 
                                   vmin=vmin, vmax=vmax, cmap='coolwarm', shading='flat')
            plots[i] = new_plot
            
            # Update title with time
            time = frame * 0.01  # Assuming dt = 0.01
            ax.set_title(f'{titles[i]} - t={time:.2f}')
        
        return plots
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_timesteps, 
                                 interval=1000//fps, blit=False, repeat=True)
    
    # Save as GIF
    anim.save(filename, writer='pillow', fps=fps, dpi=100)
    plt.close(fig)
    
    print(f"Animation saved as {filename}")


def plot_single_heatmap(positions, triangles, temperature, title="Temperature", 
                       figsize=(8, 6), vmin=None, vmax=None, save_path=None):
    """
    Plot a single temperature heatmap.
    
    Args:
        positions: Node positions (N, 2)
        triangles: Triangle connectivity (M, 3)
        temperature: Temperature values (N,)
        title: Plot title
        figsize: Figure size
        vmin, vmax: Color scale limits
        save_path: Path to save figure (optional)
    """
    # Create triangulation
    triang = tri.Triangulation(positions[:, 0], positions[:, 1], triangles)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if vmin is None:
        vmin = np.min(temperature)
    if vmax is None:
        vmax = np.max(temperature)
    
    plot = ax.tripcolor(triang, temperature, vmin=vmin, vmax=vmax, 
                       cmap='coolwarm', shading='flat')
    
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add colorbar
    plt.colorbar(plot, ax=ax, label='Temperature')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved as {save_path}")
    
    plt.show()


def extract_triangles_from_mesh(mesh):
    """
    Extract triangle connectivity from NGSolve mesh.
    
    Args:
        mesh: NGSolve mesh object
    
    Returns:
        triangles: Triangle connectivity array (M, 3)
    """
    triangles = []
    
    for el in mesh.Elements():
        if len(el.vertices) == 3:  # Triangle
            verts = [v.nr for v in el.vertices]
            triangles.append(verts)
    
    return np.array(triangles)


def export_to_vtk(mesh, fes, array_true, array_pred, time_steps, filename="results/vtk/results.vtk"):
    """
    Export solutions to VTK file for visualization in Paraview.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    gfu_true = GridFunction(fes)
    gfu_pred = GridFunction(fes)
    gfu_diff = GridFunction(fes)
    gfu_true.vec.FV().NumPy()[:] = array_true[0]
    gfu_pred.vec.FV().NumPy()[:] = array_pred[0]
    # relative error, %
    gfu_diff.vec.FV().NumPy()[:] = (array_true[0] - array_pred[0]) / (np.max(np.abs(array_true[0])) - np.min(np.abs(array_true[0]))) * 100
    vtk_out = VTKOutput(mesh, coefs=[gfu_true, gfu_pred, gfu_diff], names=["ExactSolution", "PredictedSolution", "Difference"], filename=str(filename))
    for idx, time in enumerate(time_steps):
        gfu_true.vec.FV().NumPy()[:] = array_true[idx]
        gfu_pred.vec.FV().NumPy()[:] = array_pred[idx]
        gfu_diff.vec.FV().NumPy()[:] = (array_true[idx] - array_pred[idx]) / (np.max(np.abs(array_true[idx])) - np.min(np.abs(array_true[idx]))) * 100
        vtk_out.Do(time=time)
    print(f"VTK file saved as {filename}")


def compute_l2_error_timeseries(exact_sequence, predicted_sequence):
    """
    Compute L2 error at each timestep.
    
    Args:
        exact_sequence: Exact solution sequence (T, N)
        predicted_sequence: Predicted solution sequence (T, N)
    
    Returns:
        l2_errors: L2 error at each timestep (T,)
    """
    n_timesteps = exact_sequence.shape[0]
    l2_errors = np.zeros(n_timesteps)
    
    for t in range(n_timesteps):
        error = exact_sequence[t] - predicted_sequence[t]
        l2_errors[t] = np.sqrt(np.mean(error**2))
    
    return l2_errors


def plot_l2_error_evolution(l2_errors, dt=0.01, title="L2 Error Evolution", 
                          save_path=None):
    """
    Plot L2 error evolution over time.
    
    Args:
        l2_errors: L2 error at each timestep
        dt: Time step size
        title: Plot title
        save_path: Path to save figure (optional)
    """
    times = np.arange(len(l2_errors)) * dt
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, l2_errors, 'b-', linewidth=2, label='L2 Error')
    plt.xlabel('Time')
    plt.ylabel('L2 Error')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved as {save_path}")
    
    # plt.show()
