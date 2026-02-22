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
from fem import FEMSolver
from graph_creator import GraphCreator
from containers import TimeConfig, MeshConfig, MeshProblem
from torch_geometric.data import Data
from train_problems import (
    create_test_problem,
    generate_multiple_problems,
    create_lshaped_problem,
    create_mms_problem,
    create_source_test_problem,
    create_industrial_heating_problem,
)
from trainer import PIMGNTrainer


def plot_results(
    errors, losses, val_losses, last_residuals, pos_data, save_path="results"
):
    """Plot training results."""
    Path(save_path).mkdir(exist_ok=True)

    # Plot results
    plt.figure(figsize=(8, 6))

    # Training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(losses, label="Training")
    if val_losses:
        plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PIMGN Training/Validation Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)

    # L2 error over time for first problem (if errors exist)
    plt.subplot(2, 2, 2)
    if errors and len(errors) > 0:
        plt.plot(errors[0])
        plt.xlabel("Time Step")
        plt.ylabel("L2 Error")
        plt.title("L2 Error over Time (Problem 1)")
        plt.yscale("log")
        plt.grid(True)
    else:
        plt.text(
            0.5,
            0.5,
            "No error data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
        plt.title("L2 Error over Time")

    # Average L2 error across all test problems (if errors exist)
    plt.subplot(2, 2, 3)
    if errors and len(errors) > 0:
        avg_errors = [np.mean(error_list) for error_list in errors]
        plt.bar(range(1, len(avg_errors) + 1), avg_errors)
        plt.xlabel("Problem Index")
        plt.ylabel("Average L2 Error")
        plt.title("Average L2 Error per Problem")
        plt.yscale("log")
        plt.grid(True)
    else:
        plt.text(
            0.5,
            0.5,
            "No error data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
        plt.title("Average L2 Error per Problem")

    # Last residuals heatmap
    plt.subplot(2, 2, 4)
    if last_residuals is not None and last_residuals.size > 0:
        from scipy.interpolate import griddata

        # last_residuals has shape [N_TB, N_free_dofs]
        # We need to get the positions of the free nodes only
        # For now, let's take the last time step residuals and use absolute values
        time_step_idx = min(
            1, last_residuals.shape[0] - 1
        )  # Use second time step if available, else first
        residuals_to_plot = np.abs(last_residuals[time_step_idx])  # [N_free_dofs]

        # For this to work properly, we would need to know which nodes are free
        # Since we don't have that mapping here, let's check if dimensions match
        if len(residuals_to_plot) == len(pos_data):
            # Dimensions match - use all positions
            pos_to_use = pos_data
            residuals_final = residuals_to_plot
        elif len(residuals_to_plot) < len(pos_data):
            # More positions than residuals - assume first N positions are free nodes
            pos_to_use = pos_data[: len(residuals_to_plot)]
            residuals_final = residuals_to_plot
        else:
            # More residuals than positions - shouldn't happen, but handle gracefully
            print(
                f"Warning: Residuals shape {residuals_to_plot.shape} doesn't match pos_data shape {pos_data.shape}"
            )
            pos_to_use = pos_data
            residuals_final = residuals_to_plot[: len(pos_data)]

        x_min, x_max = pos_to_use[:, 0].min(), pos_to_use[:, 0].max()
        y_min, y_max = pos_to_use[:, 1].min(), pos_to_use[:, 1].max()

        # Add small padding to avoid edge effects
        padding = 0.05
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range

        # Create grid
        grid_resolution = 100
        xi = np.linspace(x_min, x_max, grid_resolution)
        yi = np.linspace(y_min, y_max, grid_resolution)
        XI, YI = np.meshgrid(xi, yi)

        try:
            ZI = griddata(
                (pos_to_use[:, 0], pos_to_use[:, 1]),
                residuals_final,
                (XI, YI),
                method="cubic",
                fill_value=0,
            )

            # Create heatmap
            im = plt.imshow(
                ZI,
                extent=[x_min, x_max, y_min, y_max],
                origin="lower",
                cmap="viridis",
                aspect="equal",
            )
            plt.colorbar(im, label="|Residual|")

            # Overlay scatter points to show actual node locations
            plt.scatter(
                pos_to_use[:, 0],
                pos_to_use[:, 1],
                c="white",
                s=2,
                alpha=0.3,
                edgecolors="black",
                linewidths=0.1,
            )

            plt.title(f"FEM Residuals (t_step={time_step_idx})")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.axis("equal")

        except Exception as e:
            print(f"Error creating residual heatmap: {e}")
            plt.text(
                0.5,
                0.5,
                f"Error plotting residuals:\n{str(e)}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )
            plt.title("FEM Residuals (Error)")

    elif last_residuals is not None:
        plt.text(
            0.5,
            0.5,
            f"Empty residual data\nShape: {last_residuals.shape}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
        plt.title("FEM Residuals (Empty)")
    else:
        plt.text(
            0.5,
            0.5,
            "No residual data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )
        plt.title("FEM Residuals (None)")

    plt.tight_layout()
    plt.savefig(f"{save_path}/pimgn_results.png", dpi=150)
    # plt.show()


def train_pimgn_on_multiple_problems():
    """Train Physics-Informed MeshGraphNet on multiple problems."""
    print("=" * 60)
    print("PHYSICS-INFORMED MESHGRAPHNET (PIMGN) TRAINING ON MULTIPLE PROBLEMS")
    print("=" * 60)

    # Create results directory
    Path("results").mkdir(exist_ok=True)
    Path("results/physics_informed").mkdir(exist_ok=True)

    print("=" * 40)

    # Generate multiple problems
    n_problems = 5
    all_problems, time_config = generate_multiple_problems(
        n_problems=n_problems, seed=42
    )

    # Training configuration
    config = {
        "epochs": 10,
        "lr": 1e-3,
        "time_window": 20,
        "generate_ground_truth_for_validation": True,
        "save_interval": 300,
        "save_epoch_interval": 100,
        "log_filename": "pimgn_multiple_problems_log.json",
    }

    print(f"Training configuration:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Time window: {config['time_window']}")
    print(f"  Time steps: {time_config.num_steps}")
    print(f"  dt: {time_config.dt}")
    print(f"  Physics-informed loss: FEM residual with temporal bundling")

    # Create PIMGN trainer
    trainer = PIMGNTrainer(all_problems, config)

    # Train model with physics-informed loss
    print("\nStarting physics-informed training...")
    train_indices = list(range(n_problems - 1))  # Last for validation
    val_indices = list(range(n_problems - 1, n_problems))
    trainer.train(
        train_problems_indices=train_indices, val_problems_indices=val_indices
    )

    # Evaluate model
    print("\nEvaluating trained PIMGN...")
    last_residuals = trainer.last_residuals
    pos_data = trainer.problems[0].graph_data.pos.numpy()
    predictions, ground_truth, errors = trainer.evaluate_with_ground_truth(
        problem_indices=val_indices
    )

    # Plot results with ground truth comparison
    plot_results(
        errors,
        trainer.losses,
        trainer.val_losses,
        last_residuals,
        pos_data,
        save_path="results/physics_informed",
    )
    trainer.save_logs()
    min_length = min(
        len(ground_truth[0]), len(predictions[0]), len(time_config.time_steps_export)
    )
    trainer.all_fem_solvers[-1].export_to_vtk(
        ground_truth[0][:min_length],
        predictions[0][:min_length],
        time_config.time_steps_export[:min_length],
        filename="results/physics_informed/pimgn_single_comparison.vtk",
        material_field=getattr(trainer.problems[-1], "material_field", None),
    )

    print("maxh for last problem:", trainer.problems[-1].mesh_config.maxh)

    model_path = "results/physics_informed/pimgn_trained_model.pth"
    torch.save(trainer.model.state_dict(), model_path)
    print(f"Trained model saved to: {model_path}")


def _run_single_problem_experiment(problem, time_config, config, experiment_name: str):
    print("=" * 60)
    print(f"PIMGN TEST - {experiment_name.upper()}")
    print("=" * 60)

    save_path = config.get("save_dir", "results/physics_informed")
    os.makedirs(save_path, exist_ok=True)

    print("Training configuration:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Time window: {config['time_window']}")
    print(f"  Time steps: {time_config.num_steps}")
    print(f"  dt: {time_config.dt}")
    print(f"  Physics-informed loss: FEM residual with temporal bundling")

    config["log_filename"] = (
        f"pimgn_{experiment_name.replace(' ', '_').lower()}_log.json"
    )
    config["save_interval"] = 300  # Save every 300 seconds
    config["save_epoch_interval"] = 100

    trainer = PIMGNTrainer([problem], config)

    print("\nStarting physics-informed training...")
    trainer.train(train_problems_indices=[0])

    print("\nEvaluating trained PIMGN...")
    last_residuals = trainer.last_residuals
    trainer.logger.log_evaluation(last_residuals.tolist(), "residuals_per_time_step")
    pos_data = trainer.problems[0].graph_data.pos.numpy()
    try:
        predictions, ground_truth, errors = trainer.evaluate_with_ground_truth(
            problem_indices=[0]
        )
        plot_results(
            errors, trainer.losses, [], last_residuals, pos_data, save_path=save_path
        )

        print("Exporting results...")
        min_length = min(
            len(ground_truth[0]),
            len(predictions[0]),
            len(time_config.time_steps_export),
        )
        trainer.all_fem_solvers[0].export_to_vtk(
            ground_truth[0][:min_length],
            predictions[0][:min_length],
            time_config.time_steps_export[:min_length],
            filename=f"{save_path}/vtk/result",
            material_field=getattr(trainer.problems[0], "material_field", None),
        )
    except Exception as e:
        print(f"Ground truth evaluation failed: {e}")

    print("Physics-Informed MeshGraphNet test completed!")
    print(f"Results saved to: {save_path}")
    trainer.save_logs()

    model_path = f"{save_path}/pimgn_trained_model.pth"
    trainer.save_checkpoint(model_path, epoch=config["epochs"] - 1)
    print(f"Trained model saved to: {model_path}")


def _run_multiple_problem_experiment(
    problems, time_config, config, experiment_name: str
):
    print("=" * 60)
    print(f"PIMGN TEST - {experiment_name.upper()}")
    print("=" * 60)

    Path("results").mkdir(exist_ok=True)
    Path("results/physics_informed").mkdir(exist_ok=True)

    print("=" * 40)
    print("Training configuration:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Time window: {config['time_window']}")
    print(f"  Time steps: {time_config.num_steps}")
    print(f"  dt: {time_config.dt}")
    print(f"  Physics-informed loss: FEM residual with temporal bundling")

    config["log_filename"] = (
        f"pimgn_{experiment_name.replace(' ', '_').lower()}_log.json"
    )
    config["save_interval"] = 300
    config["save_epoch_interval"] = 100

    trainer = PIMGNTrainer(problems, config)

    print("\nStarting physics-informed training...")
    train_indices = list(range(len(problems) - 1))  # Last for validation
    val_indices = list(range(len(problems) - 1, len(problems)))
    trainer.train(
        train_problems_indices=train_indices, val_problems_indices=val_indices
    )

    print("\nEvaluating trained PIMGN...")
    last_residuals = trainer.last_residuals
    pos_data = trainer.problems[0].graph_data.pos.numpy()
    predictions, ground_truth, errors = trainer.evaluate_with_ground_truth(
        problem_indices=val_indices
    )

    plot_results(
        errors,
        trainer.losses,
        trainer.val_losses,
        last_residuals,
        pos_data,
        save_path="results/physics_informed",
    )

    print("Physics-Informed MeshGraphNet test completed!")
    print("Results saved to: results/physics_informed/")
    trainer.save_logs()

    min_length = min(
        len(ground_truth[0]), len(predictions[0]), len(time_config.time_steps_export)
    )
    trainer.all_fem_solvers[0].export_to_vtk(
        ground_truth[0][:min_length],
        predictions[0][:min_length],
        time_config.time_steps_export[:min_length],
        filename="results/physics_informed/vtk/result",
        material_field=getattr(trainer.problems[0], "material_field", None),
    )

    model_path = "results/physics_informed/pimgn_trained_model.pth"
    torch.save(trainer.model.state_dict(), model_path)
    print(f"Trained model saved to: {model_path}")


def train_pimgn_on_single_problem(resume_from: str = None):
    problem, time_config = create_mms_problem(maxh=0.2)
    config = {
        "epochs": 500,
        "lr": 1e-3,
        "time_window": 20,
        "generate_ground_truth_for_validation": False,
        "save_dir": "results/physics_informed/test_first_order_mms_maxh_0.2",
        "resume_from": resume_from,  # Path to checkpoint to resume from
    }
    _run_single_problem_experiment(problem, time_config, config, "First order MMS")


def train_multiple_problems():
    problems = []
    time_config = None
    for i in range(5):
        problem, time_config = create_test_problem(
            maxh=0.1, alpha=np.random.uniform(0.1, 5.0)
        )
        problems.append(problem)
    config = {
        "epochs": 100,
        "lr": 1e-3,
        "time_window": 20,
        "generate_ground_truth_for_validation": True,
    }
    _run_multiple_problem_experiment(
        problems, time_config, config, "Multiple test problems"
    )


def train_pimgn_on_industrial_problem():
    problem, time_config = create_industrial_heating_problem(maxh=3e-3)
    config = {
        "epochs": 100,
        "lr": 1e-3,
        "time_window": 20,
        "generate_ground_truth_for_validation": False,
        "save_dir": "results/physics_informed/verification_test_problem_4_maxh_0.003",
    }
    _run_single_problem_experiment(
        problem, time_config, config, "Industrial heating problem"
    )

def em_to_thermal():
    from train_problems import create_em_to_thermal
    problem, time_config = create_em_to_thermal()
    config = {
        "epochs": 5000,
        "lr": 1e-3,
        "time_window": 5,
        "generate_ground_truth_for_validation": True,
        "save_dir": "results/physics_informed/em_to_thermal",
    }
    _run_single_problem_experiment(
        problem, time_config, config, "EM to thermal problem"
    )

def main():
    """Main function to run Physics-Informed MeshGraphNet training and evaluation."""
    # Uncomment one of the following lines to run the desired test
    # train_pimgn_on_single_problem("results/physics_informed/verification_test_problem_3_maxh_0.1/pimgn_trained_model.pth")
    # train_pimgn_on_single_problem()
    # train_multiple_problems()
    # train_pimgn_on_multiple_problems()
    # train_pimgn_on_industrial_problem()
    em_to_thermal()


if __name__ == "__main__":
    main()
