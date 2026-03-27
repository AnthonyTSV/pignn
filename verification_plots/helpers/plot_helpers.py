import math
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import xml.etree.ElementTree as ET
from vtk.util import numpy_support
import vtk
import ngsolve as ng
import netgen
import scienceplots
from helpers.vtk_extractor import VTKToPlotConverter
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
from new_pignn.plotter import load_log


def _get_run_label(log_data: dict, log_path: Path) -> str:
    problems = log_data.get("problems", [])
    if problems:
        mesh_config = problems[0].get("mesh_config", {})
        maxh_value = mesh_config.get("maxh")
        if maxh_value is not None:
            return f"maxh={maxh_value}"
    return log_path.stem

def plot_l2(paths: list[Path] = None, save_dir: Path = Path("verification_plots/problem2")):
    log_paths = paths
    l2_errors = {}
    train_losses = {}
    for log_path in log_paths:
        log_data = load_log(log_path)
        eval_data = log_data.get("evaluation", {})
        l2_error = eval_data.get("l2_errors_per_problem", [])[0]
        maxh_value = log_data["problems"][0]["mesh_config"]["maxh"]
        l2_errors[maxh_value] = l2_error
        train_loss = log_data["training_history"]["train_loss"]
        train_losses[maxh_value] = train_loss
    plt.figure(figsize=(6, 4))
    for key, l2_error in l2_errors.items():
        plt.plot(l2_error, label=f"maxh={key}", linewidth=1)
    plt.yscale("log")
    plt.xlabel("Time $t$ [s]")
    plt.ylabel("L2 Error")
    plt.legend(fancybox=True, frameon=True)
    plt.savefig(save_dir / "l2_error_comparison.pdf", dpi=300)

    plt.figure(figsize=(6, 4))
    for key, train_loss in train_losses.items():
        plt.plot(train_loss, label=f"maxh={key}", linewidth=1)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend(fancybox=True, frameon=True)
    plt.savefig(save_dir / "train_loss_comparison.pdf", dpi=300)

def epoch_vs_l2(paths: list[Path] = None, save_dir: Path = Path("verification_plots/problem2")):
    log_paths = paths
    l2_errors = {}
    for log_path in log_paths:
        log_data = load_log(log_path)
        eval_data = log_data.get("evaluation", {})
        l2_error = eval_data.get("l2_errors_per_problem", [])[0]
        maxh_value = log_data["problems"][0]["mesh_config"]["maxh"]
        l2_errors[maxh_value] = l2_error
    plt.figure(figsize=(6, 4))
    for key, l2_error in l2_errors.items():
        plt.plot(l2_error, label=f"maxh={key}", linewidth=1)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("L2 Error")
    plt.legend(fancybox=True, frameon=True)
    plt.savefig(save_dir / "epoch_vs_l2_error.pdf", dpi=300)

def epoch_vs_train_loss(paths: list[Path] = None, save_dir: Path = Path("verification_plots/problem2"), apply_smoothing: bool = True, is_error_negative: bool = False):
    """
    Plot training loss vs epoch for multiple runs. If apply_smoothing is True, applies a moving average filter to smooth the curves
    and keeps the original curves as well only with opacity 0.5.
    """
    log_paths = paths
    train_losses = {}
    for log_path in log_paths:
        log_data = load_log(log_path)
        train_loss = log_data["training_history"]["train_loss"]
        if is_error_negative:
            train_loss = np.abs(train_loss)
        maxh_value = log_data["problems"][0]["mesh_config"]["maxh"]
        train_losses[maxh_value] = train_loss
    plt.figure(figsize=(6, 4))
    for key, train_loss in train_losses.items():
        if apply_smoothing:
            window_size = max(1, len(train_loss) // 100)  # Adjust window size based on length of train_loss
            smoothed_loss = np.convolve(train_loss, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_loss, label=f"maxh={key} (smoothed)", linewidth=2)
            plt.plot(train_loss, label=f"maxh={key} (original)", linewidth=1, alpha=0.5)
        else:
            plt.plot(train_loss, label=f"maxh={key}", linewidth=1)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend(fancybox=True, frameon=True)
    plt.savefig(save_dir / "epoch_vs_train_loss.pdf", dpi=300)


def epoch_vs_training_l2(
    paths: list[Path] = None,
    save_dir: Path = Path("verification_plots/problem2"),
    metric_name: str = "l2_A",
    problem_idx: int = 0,
    apply_smoothing: bool = False,
):
    """
    Plot training-time L2 metrics logged in training_history["l2_error"] vs epoch.
    """
    log_paths = paths or []
    save_dir.mkdir(parents=True, exist_ok=True)

    curves = {}
    for log_path in log_paths:
        log_data = load_log(log_path)
        l2_history = log_data.get("training_history", {}).get("l2_error", [])
        filtered = [
            entry for entry in l2_history
            if entry.get("problem_idx") == problem_idx and metric_name in entry
        ]
        if not filtered:
            continue

        epochs = [entry["epoch"] for entry in filtered]
        values = [entry[metric_name] for entry in filtered]
        curves[_get_run_label(log_data, log_path)] = (epochs, values)

    if not curves:
        raise ValueError(
            f"No training-time L2 history found for metric '{metric_name}' and problem_idx={problem_idx}."
        )

    plt.figure(figsize=(6, 4))
    for label, (epochs, values) in curves.items():
        if apply_smoothing and len(values) > 1:
            window_size = max(1, len(values) // 20)
            smoothed_values = np.convolve(
                values, np.ones(window_size) / window_size, mode="valid"
            )
            smoothed_epochs = epochs[window_size - 1 :]
            plt.plot(smoothed_epochs, smoothed_values, label=f"{label} (smoothed)", linewidth=2)
            plt.plot(epochs, values, label=f"{label} (original)", linewidth=1, alpha=0.5)
        else:
            plt.plot(epochs, values, label=label, linewidth=1)

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.replace("_", " ").upper())
    plt.legend(fancybox=True, frameon=True)
    filename = f"epoch_vs_training_{metric_name}_p{problem_idx}.pdf"
    plt.savefig(save_dir / filename, dpi=300)