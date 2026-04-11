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
from helpers.error_metrics import compute_relative_error
from helpers.vtk_extractor import VTKToPlotConverter
from helpers.line_data_extractor import ExtractDataOverLine
from helpers.plot_helpers import (
    plot_l2,
    epoch_vs_train_loss,
    epoch_vs_training_l2,
    _get_run_label,
)
from new_pignn.plotter import load_log

from helpers.mpl_style import apply_mpl_style

apply_mpl_style()

# mesh_filename = "results/physics_informed/test_magnetostatics/results_data/mesh.vol"
# ngmesh = netgen.meshing.Mesh(dim=2)
# ngmesh.Load(mesh_filename)
# mesh = ng.Mesh(ngmesh)
# space = ng.H1(mesh, order=1, dirichlet="left|right|top|bottom")
# gf = ng.GridFunction(space)
# npz_filename = "results/physics_informed/test_magnetostatics/results_data/results.npz"
# data = np.load(npz_filename)

# epoch_vs_train_loss(
#     paths=[Path("results/physics_informed/magnetostatics_billet_1_rect_coil/training_log.json")],
#     save_dir=Path("verification_plots/magnetostatics"),
#     apply_smoothing=False,
#     is_error_negative=True
# )


def get_train_loss(log_path: Path) -> np.ndarray:
    log_data = load_log(log_path)
    train_loss = log_data["training_history"]["train_loss"]
    return np.abs(train_loss)


save_dir = Path("verification_plots/magnetostatics")
save_dir.mkdir(exist_ok=True, parents=True)
log_path_1_coil = Path(
    "results/physics_informed/magnetostatics_billet_1_rect_coil/training_log.json"
)
log_path_2_coil = Path(
    "results/physics_informed/magnetostatics_billet_2_rect_coil/training_log.json"
)

def ema(y, alpha=0.08):
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out

target_1 = 0.002053
target_2 = 0.004144

loss_1 = get_train_loss(log_path_1_coil)
loss_2 = get_train_loss(log_path_2_coil)

epochs_1 = np.arange(1, len(loss_1) + 1)
epochs_2 = np.arange(1, len(loss_2) + 1)

fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)

line1, = ax.plot(epochs_1, loss_1, lw=0.8, alpha=0.25)
line2, = ax.plot(epochs_2, loss_2, lw=0.8, alpha=0.25)

ax.plot(epochs_1, ema(loss_1), lw=2.0, color=line1.get_color(), label="1 coil")
ax.plot(epochs_2, ema(loss_2), lw=2.0, color=line2.get_color(), label="2 coils")

ax.axhline(target_1, color=line1.get_color(), ls="--", lw=1.2, alpha=0.9)
ax.axhline(target_2, color=line2.get_color(), ls="--", lw=1.2, alpha=0.9)

import matplotlib.transforms as mtransforms
trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
ax.text(1.01, target_1, "1 coil target", transform=trans,
        va="center", ha="left", color=line1.get_color(), fontsize=12)
ax.text(1.01, target_2, "2 coils target", transform=trans,
        va="center", ha="left", color=line2.get_color(), fontsize=12)

ax.set_ylim(1e-4, 1e-1)
ax.set_yscale("log")
ax.set_ylabel(r"$|\text{Training Loss}|$")

ax.set_xlabel("Epoch")

ax.legend(frameon=False, loc="upper right")
plt.legend(fancybox=True, frameon=True)
plt.savefig(save_dir / "epoch_vs_train_loss.pdf", dpi=300)

curves = {}
metric_name = "l2_A"
for log_path in [log_path_1_coil, log_path_2_coil]:
    log_data = load_log(log_path)
    l2_history = log_data.get("training_history", {}).get("l2_error", [])
    filtered = [
        entry
        for entry in l2_history
        if entry.get("problem_idx") == 0 and metric_name in entry
    ]
    if not filtered:
        continue

    epochs = [entry["epoch"] for entry in filtered]
    values = [entry[metric_name] for entry in filtered]
    curves[log_path._parts[-2]] = (epochs, values)

plt.figure(figsize=(6, 4))
labels = {
    "magnetostatics_billet_1_rect_coil": "1 coil",
    "magnetostatics_billet_2_rect_coil": "2 coils",
}
for label, (epochs, values) in curves.items():
    plt.plot(epochs, values, label=labels.get(label, label), linewidth=1)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("L2 Error of A")
plt.legend(fancybox=True, frameon=True)
plt.savefig(save_dir / "epoch_vs_training_l2.pdf", dpi=300)

vtk_file_1_coil = Path(
    "results/physics_informed/magnetostatics_billet_1_rect_coil/vtk/result.vtu"
)
vtk_file_2_coil = Path(
    "results/physics_informed/magnetostatics_billet_2_rect_coil/vtk/result.vtu"
)


def get_line_data(vtk_file: Path, point1: tuple, point2: tuple) -> tuple:
    extract_data = ExtractDataOverLine(vtk_file, has_time_data=False)
    extract_data.set_points(point1, point2)
    result_data = extract_data.get_data(
        ["PredictedSolution_real", "ExactSolution_real"]
    )
    predicted_solution = np.array(result_data.results["PredictedSolution_real"])
    fem_solution = np.array(result_data.results["ExactSolution_real"])
    x_line = np.asarray(result_data.point_data)[:, 0]
    return x_line, predicted_solution, fem_solution


point1 = (0.0004562261719171998, 0.10469228023861457, 0)
point2 = (0.1198514935894682, 0.10469228023861457, 0)

coil_1_data = get_line_data(vtk_file_1_coil, point1, point2)
x_line1, predicted_solution1, fem_solution1 = coil_1_data
coil_2_data = get_line_data(vtk_file_2_coil, point1, point2)
x_line2, predicted_solution2, fem_solution2 = coil_2_data

rel_err_1 = compute_relative_error(fem_solution1, predicted_solution1)
rel_err_2 = compute_relative_error(fem_solution2, predicted_solution2)


fig, (ax, ax_err) = plt.subplots(
    2, 1,
    figsize=(6.4, 5.0),
    sharex=True,
    height_ratios=[3.0, 1.4],
    constrained_layout=True
)

x_mm = 1e3 * x_line1

# Main comparison
ax.plot(x_mm, fem_solution1, linewidth=1, label="FEM (1 coil)", linestyle="--")
ax.plot(x_mm, fem_solution2, linewidth=1, label="FEM (2 coils)", linestyle="--")
ax.plot(
    x_mm,
    predicted_solution1,
    linewidth=1,
    marker="o",
    markevery=max(len(x_mm) // 15, 1), # fewer markers
    markersize=4,
    label="PI-GNN (1 coil)"
)
ax.plot(
    x_mm,
    predicted_solution2,
    linewidth=1,
    marker="x",
    markevery=max(len(x_mm) // 15, 1), # fewer markers
    markersize=4,
    label="PI-GNN (2 coils)"
)

ax.set_ylabel(r"$|J|$ [A/m$^2$]")
# ax.set_yscale("log")
ax.grid(True)
# ax.legend(frameon=True, ncols=1)
# make legend smaller and put it inside the plot
ax.legend(frameon=True, ncols=1, fontsize="small")

ax_err.clear()
ax_err.plot(x_mm, rel_err_1, linewidth=1, label="1 coil", color=ax.get_lines()[2].get_color())
ax_err.plot(x_mm, rel_err_2, linewidth=1, label="2 coils", color=ax.get_lines()[3].get_color())
ax_err.set_xlabel("x [mm]")
ax_err.set_ylabel(r"$\epsilon_{\mathrm{rel}} [\%]$")
ax_err.grid(True)
ax_err.legend(frameon=True, ncols=2)

plt.savefig(save_dir / "line_plot_coils.pdf", dpi=300)

plotter1 = VTKToPlotConverter(vtk_file_1_coil, last_time_step=0, val_range=(0, 2.5), has_time_data=False)
plotter1.plot_steady_state(
    save_dir / "steady_state_1_coil.pdf",
    exact_field_name="ExactSolution_real",
    predicted_field_name="PredictedSolution_real",
    label=r"Magnetic vector potential $\tilde{A}$",
    contours=True,
)
plotter1.plot_relative_error(save_dir / "relative_error_1_coil.pdf", field_name="RelError")

plotter2 = VTKToPlotConverter(vtk_file_2_coil, last_time_step=0, val_range=(0, 2.5), has_time_data=False)
plotter2.plot_steady_state(
    save_dir / "steady_state_2_coil.pdf",
    exact_field_name="ExactSolution_real",
    predicted_field_name="PredictedSolution_real",
    label=r"Magnetic vector potential $\tilde{A}$",
    contours=True,
)
plotter2.plot_relative_error(save_dir / "relative_error_2_coil.pdf", field_name="RelError")
