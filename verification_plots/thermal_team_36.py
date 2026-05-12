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
from helpers.error_metrics import compute_relative_error, compute_l2_error
from helpers.vtk_extractor import VTKToPlotConverter
from helpers.line_data_extractor import ExtractDataOverLine
from helpers.plot_helpers import (
    plot_l2,
    epoch_vs_train_loss,
    epoch_vs_training_l2,
    _get_run_label,
)
from new_pignn.containers import MeshProblem
from new_pignn.thermal_problems import ih_team_36_problem
from new_pignn.plotter import load_log

plt.style.use(["science"])

from helpers.mpl_style import apply_mpl_style

apply_mpl_style()


def get_train_loss(log_path: Path) -> np.ndarray:
    log_data = load_log(log_path)
    train_loss = log_data["training_history"]["train_loss"]
    return np.abs(train_loss)


def _as_temperature_coefficient(gfT, thermal_problem: MeshProblem):
    if not isinstance(gfT, (list, tuple, np.ndarray)):
        return gfT

    values = np.asarray(gfT, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected 1D temperature vector, got shape {values.shape}")

    order = getattr(thermal_problem.mesh_config, "order", 1)
    space = ng.H1(thermal_problem.mesh, order=order)
    if values.shape[0] != space.ndof:
        raise ValueError(
            f"Temperature vector has {values.shape[0]} entries, expected {space.ndof}"
        )

    temperature_gf = ng.GridFunction(space)
    temperature_gf.vec.FV().NumPy()[:] = values
    return temperature_gf


def _convective_boundary_terms():
    default_terms = [
        ("bc_workpiece_top", 7.0, 70.0),
        ("bc_workpiece_bottom", 7.0, 70.0),
        ("bc_workpiece_right", 7.0, 25.0),
    ]
    return default_terms

def _radiative_boundary_terms():
    default_terms = [
        ("bc_workpiece_top", 0.8, 70.0),
        ("bc_workpiece_bottom", 0.8, 70.0),
        ("bc_workpiece_right", 0.8, 25.0),
    ]
    return default_terms

def compute_convective_power_loss(gfT, thermal_problem: MeshProblem) -> float:
    # GetDP:
    # Integral { h[{T}, $Time] * ({T} - tref[]) * GeomCoeff[] ;
    # In Region[{ Sur_CONVECTIVE_T }]; Jacobian JSur; Integration I1; }
    temperature = _as_temperature_coefficient(gfT, thermal_problem)
    geom_coeff = 2.0 * math.pi * ng.x

    power_loss = 0.0
    for boundary_name, h_val, tref in _convective_boundary_terms():
        integrand = h_val * (temperature - tref) * geom_coeff
        power_loss += ng.Integrate(
            integrand,
            thermal_problem.mesh,
            definedon=thermal_problem.mesh.Boundaries(boundary_name),
        )

    return float(np.real_if_close(power_loss))

def compute_radiative_power_loss(gfT, thermal_problem: MeshProblem) -> float:
    # GetDP:
    # Integral { [hr[{T}, $Time] * sb_constant * ( ( ( { T } +  273 ) ) ^ 4 - ( ( tref[] + 273 ) ) ^ 4 ) *  GeomCoeff[] ];
    # In Region[{ Sur_RADIATION_T }]; Jacobian JSur; Integration I1;  }
    temperature = _as_temperature_coefficient(gfT, thermal_problem)
    sigma = 5.670374419e-8  # Stefan-Boltzmann constant
    geom_coeff = 2.0 * math.pi * ng.x

    power_loss = 0.0
    for boundary_name, h_val, tref in _radiative_boundary_terms():
        integrand = h_val * sigma * ((temperature + 273.0) ** 4 - (tref + 273.0) ** 4) * geom_coeff
        power_loss += ng.Integrate(
            integrand,
            thermal_problem.mesh,
            definedon=thermal_problem.mesh.Boundaries(boundary_name),
        )

    return float(np.real_if_close(power_loss))

save_dir = Path("verification_plots/team_36_problem")
save_dir.mkdir(exist_ok=True, parents=True)

path_to_thermal_results = Path("results/physics_informed/thermal_team_36_problem")
thermal_npz = np.load(path_to_thermal_results / "results_data/results_thermal.npz")
thermal_mesh_filename = path_to_thermal_results / "results_data/mesh.vol"
ngmesh = netgen.meshing.Mesh(dim=2)
ngmesh.Load(str(thermal_mesh_filename))
thermal_mesh = ng.Mesh(ngmesh)

thermal_problem = ih_team_36_problem()
thermal_problem.mesh = thermal_mesh

thermal_predicted_solution = thermal_npz["predicted"]
thermal_exact_solution = thermal_npz["exact"]

pred_abs = np.abs(thermal_predicted_solution)
exact_abs = np.abs(thermal_exact_solution)

l2_error = compute_l2_error(pred_abs, exact_abs)

print(f"Relative L2 error for Thermal Team 36 problem: {l2_error:.4e}")


def convective_power_loss():
    n_steps = min(
        thermal_exact_solution.shape[0],
        thermal_predicted_solution.shape[0],
        len(thermal_problem.time_config.time_steps_export),
    )

    fem_q_loss = []
    pignn_q_loss = []
    for time_idx in range(n_steps):
        fem_q_loss.append(
            compute_convective_power_loss(
                thermal_exact_solution[time_idx],
                thermal_problem,
            )
        )
        pignn_q_loss.append(
            compute_convective_power_loss(
                thermal_predicted_solution[time_idx],
                thermal_problem,
            )
        )

    fem_q_loss = np.asarray(fem_q_loss)
    pignn_q_loss = np.asarray(pignn_q_loss)
    rel_err = compute_relative_error(fem_q_loss, pignn_q_loss)

    print(f"Convective power loss (FEM): {fem_q_loss[-1]:.4f} W")
    print(f"Convective power loss (PI-GNN): {pignn_q_loss[-1]:.4f} W")
    print(f"Relative error: {rel_err[-1]:.2f}%")

    return fem_q_loss, pignn_q_loss, rel_err

def plot_convective_power_loss_over_time(fem_q_loss, pignn_q_loss, rel_err, time_steps):
    fig, (ax, ax_err) = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.0),
        sharex=True,
        height_ratios=[3.0, 1.4],
        constrained_layout=True,
    )

    ax.plot(fem_q_loss, linewidth=1, label="FEM", linestyle="--")
    ax.plot(
        pignn_q_loss,
        linewidth=1,
        marker="o",
        markevery=max(len(pignn_q_loss) // 15, 1),  # fewer markers
        markersize=4,
        label="PI-GNN",
    )
    ax.legend(ncols=2)
    ax_err.clear()
    ax_err.plot(
        rel_err, linewidth=1, color=ax.get_lines()[1].get_color()
    )
    ax_err.set_xlabel("Time step")
    ax_err.set_ylabel(r"$\epsilon_{\mathrm{rel}} [\%]$")

    ax.set_ylabel(r"Convective Power Loss [W]",)

    fig.savefig(save_dir / "convective_power_loss_over_time.pdf")

def last_max_temp_error():

    max_temp_gnn = np.max(thermal_predicted_solution[-1])
    max_temp_fem = np.max(thermal_exact_solution[-1])

    abs_error = np.abs(max_temp_gnn - max_temp_fem)
    rel_error = abs_error / (np.abs(max_temp_fem) + 1e-8) * 100.0

    print(f"Max temperature (PI-GNN): {max_temp_gnn}")
    print(f"Max temperature (FEM): {max_temp_fem}")
    print(f"Absolute error: {abs_error}")
    print(f"Relative error: {rel_error:.2f}%")

    workpiece_mask = np.where(thermal_exact_solution[0] != 0)
    temp_gnn_masked = thermal_predicted_solution[:, workpiece_mask]
    temp_fem_masked = thermal_exact_solution[:, workpiece_mask]

    l2_error = compute_l2_error(predicted_field=temp_gnn_masked[-1], exact_field=temp_fem_masked[-1])
    print(f"L2 error for temperature field over final time: {l2_error:.4e}")

def plot_temperature_over_time():

    workpiece_mask = np.where(thermal_exact_solution[0] != 0)
    temp_gnn_masked = thermal_predicted_solution[:, workpiece_mask]
    temp_fem_masked = thermal_exact_solution[:, workpiece_mask]

    l2_error = compute_l2_error(predicted_field=temp_gnn_masked[-1], exact_field=temp_fem_masked[-1])
    print(f"L2 error for temperature field over final time: {l2_error:.4e}")

    time_steps = thermal_predicted_solution.shape[0]
    max_temp_gnn = np.max(thermal_predicted_solution, axis=1)
    max_temp_fem = np.max(thermal_exact_solution, axis=1)

    rel_err = compute_relative_error(max_temp_fem, max_temp_gnn)

    fig, (ax, ax_err) = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.0),
        sharex=True,
        height_ratios=[3.0, 1.4],
        constrained_layout=True,
    )

    ax.plot(max_temp_fem, linewidth=1, label="FEM", color="tab:orange")
    # make sure to include last point if time_steps is not divisible by 15
    ax.plot(
        np.arange(time_steps),  # fewer markers
        max_temp_gnn,
        '--',
        label="PI-GNN",
        color="tab:green",
        linewidth=1,
    )
    ax.set_ylabel(r"Temperature [C]")
    ax.legend(ncols=1)

    ax_err.plot(rel_err, linewidth=1, color="tab:green")
    ax_err.set_xlabel("Time step")
    ax_err.set_ylabel(r"$\epsilon_{\mathrm{rel}} [\%]$")
    plt.savefig(save_dir / "max_temperature_over_time.pdf", dpi=300)

def radiative_power_loss():
    n_steps = min(
        thermal_exact_solution.shape[0],
        thermal_predicted_solution.shape[0],
        len(thermal_problem.time_config.time_steps_export),
    )

    fem_q_loss = []
    pignn_q_loss = []
    for time_idx in range(n_steps):
        fem_q_loss.append(
            compute_radiative_power_loss(
                thermal_exact_solution[time_idx],
                thermal_problem,
            )
        )
        pignn_q_loss.append(
            compute_radiative_power_loss(
                thermal_predicted_solution[time_idx],
                thermal_problem,
            )
        )

    fem_q_loss = np.asarray(fem_q_loss)
    pignn_q_loss = np.asarray(pignn_q_loss)
    rel_err = compute_relative_error(fem_q_loss, pignn_q_loss)

    print(f"Radiative power loss (FEM): {fem_q_loss[-1]:.4f} W")
    print(f"Radiative power loss (PI-GNN): {pignn_q_loss[-1]:.4f} W")
    print(f"Relative error: {rel_err[-1]:.2f}%")

    return fem_q_loss, pignn_q_loss, rel_err

def plot_radiative_power_loss_over_time(fem_q_loss, pignn_q_loss, rel_err, time_steps):
    fig, (ax, ax_err) = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.0),
        sharex=True,
        height_ratios=[3.0, 1.4],
        constrained_layout=True,
    )

    # convert to kiloWatts for better readability
    fem_q_loss = fem_q_loss * 1e-3
    pignn_q_loss = pignn_q_loss * 1e-3

    ax.plot(fem_q_loss, linewidth=1, label="FEM", linestyle="--")
    ax.plot(
        pignn_q_loss,
        linewidth=1,
        marker="o",
        markevery=max(len(pignn_q_loss) // 15, 1),  # fewer markers
        markersize=4,
        label="PI-GNN",
    )
    ax.legend(ncols=2)
    ax_err.clear()
    ax_err.plot(
        rel_err, linewidth=1, color=ax.get_lines()[1].get_color()
    )
    ax_err.set_xlabel("Time step")
    ax_err.set_ylabel(r"$\epsilon_{\mathrm{rel}} [\%]$")

    ax.set_ylabel(r"Radiative Power Loss [kW]",)

    fig.savefig(save_dir / "radiative_power_loss_over_time.pdf")

if __name__ == "__main__":
    fem_q_loss, pignn_q_loss, rel_err = convective_power_loss()
    plot_convective_power_loss_over_time(fem_q_loss, pignn_q_loss, rel_err, thermal_problem.time_config.time_steps_export)
    fem_q_loss, pignn_q_loss, rel_err = radiative_power_loss()
    plot_radiative_power_loss_over_time(fem_q_loss, pignn_q_loss, rel_err, thermal_problem.time_config.time_steps_export)
    last_max_temp_error()
    plot_temperature_over_time()