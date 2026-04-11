from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import xml.etree.ElementTree as ET
import sys
import vtk
from vtk.util import numpy_support

import scienceplots

from helpers.vtk_extractor import VTKToPlotConverter

plt.style.use(["science", "grid"])

from helpers.mpl_style import apply_mpl_style

apply_mpl_style()

workspace_root = Path(__file__).resolve().parents[1]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from new_pignn.plotter import load_log
from helpers.error_metrics import compute_relative_error
from helpers.plot_helpers import epoch_vs_train_loss, time_vs_l2
from helpers.line_data_extractor import ExtractDataOverLine

def _compute_analytical_solution(x, y, t, alpha=1.0):
    """Compute the analytical solution for the 2D heat equation with homogeneous Dirichlet BCs."""
    return (
        np.exp(-2 * alpha * np.pi**2 * t) * 100 * np.sin(np.pi * x) * np.sin(np.pi * y)
    )

def read_logs_and_plot_l2():
    log_paths = [
        Path("results/physics_informed/thermal_mms/training_log.json"),
    ]
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
        plt.plot(l2_error, linewidth=1)
    plt.yscale("log")
    plt.xlabel("Time $t$ [s]")
    plt.ylabel("L2 Error")
    plt.legend(fancybox=True, frameon=True)
    plt.savefig("verification_plots/thermal_mms/l2_error_comparison.pdf", dpi=300)

    plt.figure(figsize=(6, 4))
    for key, train_loss in train_losses.items():
        plt.plot(train_loss, linewidth=1)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend(fancybox=True, frameon=True)
    plt.savefig("verification_plots/thermal_mms/train_loss_comparison.pdf", dpi=300)

def plot_pred_vs_fem():
    vtk_file = Path("results/physics_informed/thermal_mms/vtk/result.vtu")
    plot_converter = VTKToPlotConverter(vtk_file, last_time_step=100, val_range=(0, 14))
    plot_converter.plot_pred_and_fem(save_path=Path("verification_plots/thermal_mms/pred_vs_fem_subplot.pdf"), figsize=(9, 7))

def probe_at_center_vs_time():
    vtk_file = Path("results/physics_informed/thermal_mms/vtk/result.vtu")
    extract_data = ExtractDataOverLine(vtk_file, has_time_data=True)
    extract_data.set_points((0, 0.5, 0), (1, 0.5, 0))
    time_values = np.arange(0, 1.01, 0.01).round(2)
    predicted_solution = []
    analytical = []
    fem_solution = []
    for time_value in time_values:
        extract_data.set_time_value(time_value)
        result_data = extract_data.get_data(["ExactSolution", "PredictedSolution"])
        predicted_solution.append(np.array(result_data.results["PredictedSolution"][50]))
        fem_solution.append(np.array(result_data.results["ExactSolution"][50]))
        analytical.append(_compute_analytical_solution(0.5, 0.5, time_value, alpha=0.1))
    predicted_solution = np.array(predicted_solution)
    analytical = np.array(analytical)
    fem_solution = np.array(fem_solution)
    rel_error_fem = compute_relative_error(predicted_solution, fem_solution)
    rel_error_analytical = compute_relative_error(predicted_solution, analytical)

    print("Max relative error vs FEM: {:.2f}%".format(np.max(rel_error_fem)))
    print("Max relative error vs Analytical: {:.2f}%".format(np.max(rel_error_analytical)))

    fig, (ax, ax_err) = plt.subplots(
        2, 1,
        figsize=(6.4, 5.0),
        sharex=True,
        height_ratios=[3.0, 1.4],
        constrained_layout=True
    )
    
    ax.plot(time_values, fem_solution, label="FEM", linestyle="--")
    ax.plot(time_values, analytical, label="Analytical", linestyle=":")
    ax.plot(
        time_values,
        predicted_solution,
        linewidth=1,
        marker="o",
        markevery=max(len(predicted_solution) // 15, 1), # fewer markers
        markersize=4,
        label="PI-GNN"
    )
    ax.set_ylabel(r"$T$ at center [$^\circ$C$]$")
    ax.grid(True)
    ax.legend(frameon=True, ncols=1)
    ax_err.clear()
    ax_err.plot(time_values, rel_error_fem, linewidth=1, color=ax.get_lines()[0].get_color())
    ax_err.plot(time_values, rel_error_analytical, linewidth=1, color=ax.get_lines()[1].get_color())
    ax_err.set_xlabel("Time $t$ [s]")
    ax_err.set_ylabel(r"$\epsilon_{\mathrm{rel}} [\%]$")
    ax_err.legend(["vs FEM", "vs Analytical"], frameon=True, ncols=2, fontsize="small")
    ax_err.grid(True)

    plt.savefig("verification_plots/thermal_mms/center_probe_vs_time.pdf", dpi=300)

if __name__ == "__main__":
    epoch_vs_train_loss(
        paths=[
            Path("results/physics_informed/thermal_mms/training_log.json"),
        ],
        save_dir=Path("verification_plots/thermal_mms"),
        need_maxh_in_label=False
    )
    time_vs_l2(
        paths=[
            Path("results/physics_informed/thermal_mms/training_log.json"),
        ],
        save_dir=Path("verification_plots/thermal_mms"),
        need_maxh_in_label=False,
        time_range=np.arange(0, 1.01, 0.01)
    )
    plot_pred_vs_fem()
    probe_at_center_vs_time()