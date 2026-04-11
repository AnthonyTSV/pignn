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
    time_vs_l2,
    epoch_vs_train_loss,
)
from new_pignn.plotter import load_log
from new_pignn.thermal_problems import create_temp_dependent_material_problem

plt.style.use(["science", "grid"])

from helpers.mpl_style import apply_mpl_style

apply_mpl_style()

save_dir = Path("verification_plots/thermal_temp_materials")
save_dir.mkdir(parents=True, exist_ok=True)

vtk_file = Path(
    "results/physics_informed/thermal_temp_dependent_material/vtk/result.vtu"
)


def get_line_data(vtk_file: Path, point1: tuple, point2: tuple, time_value: float) -> tuple:
    extract_data = ExtractDataOverLine(vtk_file)
    extract_data.set_time_value(time_value)
    extract_data.set_points(point1, point2)
    result_data = extract_data.get_data(
        ["PredictedSolution", "ExactSolution"]
    )
    predicted_solution = np.array(result_data.results["PredictedSolution"])
    fem_solution = np.array(result_data.results["ExactSolution"])
    x_line = np.asarray(result_data.point_data)[:, 0]
    return x_line, predicted_solution, fem_solution


point1 = (0, 0.5, 0)
point2 = (1, 0.5, 0)

def get_data_for_time(vtk_file: Path, time_value: float) -> tuple:
    x_line, predicted_solution, fem_solution = get_line_data(vtk_file, point1, point2, time_value)
    rel_err = compute_relative_error(predicted_solution, fem_solution)
    return x_line, predicted_solution, fem_solution, rel_err

t_middle = 0.05
t_end = 1.0

x_line1, predicted_solution1, fem_solution1, rel_err_1 = get_data_for_time(vtk_file, time_value=t_end)
x_line2, predicted_solution2, fem_solution2, rel_err_2 = get_data_for_time(vtk_file, time_value=t_middle)

fig, (ax, ax_err) = plt.subplots(
    2, 1,
    figsize=(6.4, 5.0),
    sharex=True,
    height_ratios=[3.0, 1.4],
    constrained_layout=True
)

x_mm = 1e3 * x_line1

# Main comparison
ax.plot(x_mm, fem_solution1, linewidth=1, label=f"FEM ({t_end} s)", linestyle="--")
ax.plot(x_mm, fem_solution2, linewidth=1, label=f"FEM ({t_middle} s)", linestyle="--")
ax.plot(
    x_mm,
    predicted_solution1,
    linewidth=1,
    marker="o",
    markevery=max(len(x_mm) // 15, 1), # fewer markers
    markersize=4,
    label=f"PI-GNN ({t_end} s)"
)
ax.plot(
    x_mm,
    predicted_solution2,
    linewidth=1,
    marker="o",
    markevery=max(len(x_mm) // 15, 1), # fewer markers
    markersize=4,
    label=f"PI-GNN ({t_middle} s)"
)

ax.set_ylabel(r"$T$ [$^\circ$C$]$")
# ax.set_yscale("log")
ax.grid(True)
# ax.legend(frameon=True, ncols=1)
# make legend smaller and put it inside the plot
ax.legend(frameon=True, ncols=1, fontsize="small")

ax_err.clear()
ax_err.plot(x_mm, rel_err_1, linewidth=1, label=f"{t_end} s", color=ax.get_lines()[1].get_color())
ax_err.plot(x_mm, rel_err_2, linewidth=1, label=f"{t_middle} s", color=ax.get_lines()[3].get_color())
ax_err.set_xlabel("x [mm]")
ax_err.set_ylabel(r"$\epsilon_{\mathrm{rel}} [\%]$")
ax_err.grid(True)
ax_err.legend(frameon=True, ncols=2)

plt.savefig(save_dir / "sol_line_plot.pdf", dpi=300)

# K field plot

def get_k_field(vtk_file: Path, time_value: float) -> tuple:
    extract_data = ExtractDataOverLine(vtk_file)
    extract_data.set_time_value(time_value)
    extract_data.set_points(point1, point2)
    result_data = extract_data.get_data(
        ["k_Surrogate", "k_FEM"]
    )
    predicted_k = np.array(result_data.results["k_Surrogate"])
    fem_k = np.array(result_data.results["k_FEM"])
    x_line = np.asarray(result_data.point_data)[:, 0]
    rel_err = compute_relative_error(predicted_k, fem_k)
    return x_line, predicted_k, fem_k, rel_err

x_line1, predicted_k_1, fem_k_1, rel_err_1 = get_k_field(vtk_file, time_value=t_end)
x_line2, predicted_k_2, fem_k_2, rel_err_2 = get_k_field(vtk_file, time_value=t_middle)

fig, (ax, ax_err) = plt.subplots(
    2, 1,
    figsize=(6.4, 5.0),
    sharex=True,
    height_ratios=[3.0, 1.4],
    constrained_layout=True
)

x_mm = 1e3 * x_line1

# Main comparison
ax.plot(x_mm, fem_k_1, linewidth=1, label=f"FEM ({t_end} s)", linestyle="--")
ax.plot(x_mm, fem_k_2, linewidth=1, label=f"FEM ({t_middle} s)", linestyle="--")
ax.plot(
    x_mm,
    predicted_k_1,
    linewidth=1,
    marker="o",
    markevery=max(len(x_mm) // 15, 1), # fewer markers
    markersize=4,
    label=f"PI-GNN ({t_end} s)"
)
ax.plot(
    x_mm,
    predicted_k_2,
    linewidth=1,
    marker="o",
    markevery=max(len(x_mm) // 15, 1), # fewer markers
    markersize=4,
    label=f"PI-GNN ({t_middle} s)"
)

ax.set_ylabel(r"$k$ [$\mathrm{W/(m \cdot K)}$]")
# ax.set_yscale("log")
ax.grid(True)
# ax.legend(frameon=True, ncols=1)
# make legend smaller and put it inside the plot
ax.legend(frameon=True, ncols=1, fontsize="small")

ax_err.clear()
ax_err.plot(x_mm, rel_err_1, linewidth=1, label=f"{t_end} s", color=ax.get_lines()[1].get_color())
ax_err.plot(x_mm, rel_err_2, linewidth=1, label=f"{t_middle} s", color=ax.get_lines()[3].get_color())
ax_err.set_xlabel("x [mm]")
ax_err.set_ylabel(r"$\epsilon_{\mathrm{rel}} [\%]$")
ax_err.grid(True)
ax_err.legend(frameon=True, ncols=2)

plt.savefig(save_dir / "k_field.pdf", dpi=300)

plotter1 = VTKToPlotConverter(vtk_file, last_time_step=100, val_range=(1, 2.4), has_time_data=True)
plotter1.plot_steady_state(
    save_dir / "k_field_2d.pdf",
    exact_field_name="k_FEM",
    predicted_field_name="k_Surrogate",
    label=r"$k$ [$\mathrm{W/(m \cdot K)}$]",
    time_value=t_end,
    # contours=True,
)
plotter1.plot_relative_error_fields(
    exact_name="k_FEM",
    predicted_name="k_Surrogate",
    save_path=save_dir / "k_field_rel_err.pdf"
)

def plot_pred_vs_fem():
    vtk_file = Path("results/physics_informed/thermal_temp_dependent_material/vtk/result.vtu")
    plot_converter = VTKToPlotConverter(vtk_file, last_time_step=100, val_range=(59, 441))
    plot_converter.plot_pred_and_fem(save_path=Path("verification_plots/thermal_temp_materials/pred_vs_fem_subplot.pdf"), shrink=0.9, figsize=(9, 7))

time_vs_l2(
    paths=[
        Path("results/physics_informed/thermal_temp_dependent_material/training_log.json"),
    ],
    save_dir=Path("verification_plots/thermal_temp_materials"),
    need_maxh_in_label=False,
    time_range=np.arange(0, 1.01, 0.01)
)

plot_pred_vs_fem()