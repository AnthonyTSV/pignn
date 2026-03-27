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
from helpers.line_data_extractor import ExtractDataOverLine
from helpers.plot_helpers import (
    plot_l2,
    epoch_vs_l2,
    epoch_vs_train_loss,
    epoch_vs_training_l2,
    _get_run_label,
)
from new_pignn.containers import MeshProblemEM
from new_pignn.em_eddy_problems import eddy_current_problem_1, eddy_current_problem_2
from new_pignn.plotter import load_log

plt.style.use(["science", "grid"])

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "font.sans-serif": ["cmss10"],
        "font.monospace": ["cmtt10"],
        "axes.formatter.use_mathtext": True,
        "font.size": 14,
    }
)


def get_train_loss(log_path: Path) -> np.ndarray:
    log_data = load_log(log_path)
    train_loss = log_data["training_history"]["train_loss"]
    return np.abs(train_loss)


save_dir = Path("verification_plots/eddy_current")
save_dir.mkdir(exist_ok=True, parents=True)

problem_1 = eddy_current_problem_1()
problem_2 = eddy_current_problem_2()

# skin depth
def compute_skin_depth(sigma, freq, mu_r):
    return math.sqrt(1 / (math.pi * freq * mu_r * sigma))

skin_depth_1 = compute_skin_depth(problem_1.sigma_workpiece * problem_1.sigma_star, problem_1.frequency, problem_1.mu_r_workpiece * problem_1.mu_star)
skin_depth_2 = compute_skin_depth(problem_2.sigma_workpiece * problem_2.sigma_star, problem_2.frequency, problem_2.mu_r_workpiece * problem_2.mu_star)
print(f"Skin depth for problem 1: {skin_depth_1*1e3:.2f} mm")
print(f"Skin depth for problem 2: {skin_depth_2*1e3:.2f} mm")


def get_unscaled_gfa(path_to_results: Path, problem: MeshProblemEM):
    mesh_filename = path_to_results / "results_data/mesh.vol"
    ngmesh = netgen.meshing.Mesh(dim=2)
    ngmesh.Load(str(mesh_filename))
    mesh = ng.Mesh(ngmesh)
    space = ng.H1(
        mesh, order=1, dirichlet=problem.mesh_config.dirichlet_pipe, complex=True
    )
    gfA_fem = ng.GridFunction(space)
    gfA_gnn = ng.GridFunction(space)

    npz_filename = path_to_results / "results_data/results.npz"
    data = np.load(npz_filename)
    exact = data["exact"]
    predicted = data["predicted"]

    gfA_fem.vec[:].FV().NumPy()[:] = exact * problem.A_star
    gfA_gnn.vec[:].FV().NumPy()[:] = predicted * problem.A_star

    gfA_unscaled_fem = gfA_fem
    gfA_unscaled_gnn = gfA_gnn

    return gfA_unscaled_fem, gfA_unscaled_gnn, mesh, data


def get_fields(mesh, gfA, problem: MeshProblemEM):

    sigma = mesh.MaterialCF(
        {
            "mat_workpiece": problem.sigma_workpiece,
            "mat_air": problem.sigma_air,
            "mat_coil": problem.sigma_coil,
        },
        default=0.0,
    )

    E_phi = -1j * problem.omega * gfA
    J_phi = sigma * problem.sigma_star * E_phi

    Q = 0.5 * sigma * problem.sigma_star * ng.Norm(E_phi) ** 2
    current_density = ng.Norm(J_phi)

    return E_phi, J_phi, Q, current_density


gfA_fem1, gfA_gnn1, mesh1, data1 = get_unscaled_gfa(
    Path("results/physics_informed/eddy_current_problem_1_rect_coil"), problem_1
)
gfA_fem2, gfA_gnn2, mesh2, data2 = get_unscaled_gfa(
    Path("results/physics_informed/eddy_current_problem_2_rect_coil"), problem_2
)

E_fem1, J_fem1, Q_fem1, current_density_fem1 = get_fields(mesh1, gfA_fem1, problem_1)
E_gnn1, J_gnn1, Q_gnn1, current_density_gnn1 = get_fields(mesh1, gfA_gnn1, problem_1)

E_fem2, J_fem2, Q_fem2, current_density_fem2 = get_fields(mesh2, gfA_fem2, problem_2)
E_gnn2, J_gnn2, Q_gnn2, current_density_gnn2 = get_fields(mesh2, gfA_gnn2, problem_2)

# export relative error fields for plotting
space = ng.H1(mesh1, order=1, dirichlet=problem_1.mesh_config.dirichlet_pipe)
diff_e_real1 = ng.GridFunction(space)
diff_e_imag1 = ng.GridFunction(space)
diff_e_abs1 = ng.GridFunction(space)

space2 = ng.H1(mesh2, order=1, dirichlet=problem_2.mesh_config.dirichlet_pipe)
diff_e_real2 = ng.GridFunction(space2)
diff_e_imag2 = ng.GridFunction(space2)
diff_e_abs2 = ng.GridFunction(space2)

diff_e_real1.vec[:].FV().NumPy()[:] = np.abs(data1["exact"].real - data1["predicted"].real) / (
    np.abs(data1["exact"].real) + 1e-10
)
diff_e_imag1.vec[:].FV().NumPy()[:] = np.abs(data1["exact"].imag - data1["predicted"].imag) / (
    np.abs(data1["exact"].imag) + 1e-10
)
diff_e_abs1.vec[:].FV().NumPy()[:] = np.abs(data1["exact"] - data1["predicted"]) / (np.abs(data1["exact"]) + 1e-10)

diff_e_real2.vec[:].FV().NumPy()[:] = np.abs(data2["exact"].real - data2["predicted"].real) / (
    np.abs(data2["exact"].real) + 1e-10
)
diff_e_imag2.vec[:].FV().NumPy()[:] = np.abs(data2["exact"].imag - data2["predicted"].imag) / (
    np.abs(data2["exact"].imag) + 1e-10
)
diff_e_abs2.vec[:].FV().NumPy()[:] = np.abs(data2["exact"] - data2["predicted"]) / (np.abs(data2["exact"]) + 1e-10)
# save as vtk

vtk_out = ng.VTKOutput(
    mesh1,
    coefs=[
        E_fem1.real,
        E_fem1.imag,
        ng.Norm(E_fem1),
        current_density_fem1,
        Q_fem1,
        E_gnn1.real,
        E_gnn1.imag,
        ng.Norm(E_gnn1),
        current_density_gnn1,
        Q_gnn1,
        diff_e_real1,
        diff_e_imag1,
        diff_e_abs1,
    ],
    names=[
        "E_real_fem",
        "E_imag_fem",
        "E_abs_fem",
        "J_abs_fem",
        "Q_fem",
        "E_real_gnn",
        "E_imag_gnn",
        "E_abs_gnn",
        "J_abs_gnn",
        "Q_gnn",
        "diff_E_real",
        "diff_E_imag",
        "diff_E_abs",
    ],
    filename="verification_plots/eddy_current/fem_gnn_comparison",
    order=1,
)
vtk_out.Do()

vtk_out_2 = ng.VTKOutput(
    mesh2,
    coefs=[
        E_fem2.real,
        E_fem2.imag,
        ng.Norm(E_fem2),
        current_density_fem2,
        Q_fem2,
        E_gnn2.real,
        E_gnn2.imag,
        ng.Norm(E_gnn2),
        current_density_gnn2,
        Q_gnn2,
        diff_e_real2,
        diff_e_imag2,
        diff_e_abs2,
    ],
    names=[
        "E_real_fem",
        "E_imag_fem",
        "E_abs_fem",
        "J_abs_fem",
        "Q_fem",
        "E_real_gnn",
        "E_imag_gnn",
        "E_abs_gnn",
        "J_abs_gnn",
        "Q_gnn",
        "diff_E_real",
        "diff_E_imag",
        "diff_E_abs",
    ],
    filename="verification_plots/eddy_current/fem_gnn_comparison_2",
    order=1,
)
vtk_out_2.Do()

vtk_file_1_coil = Path("verification_plots/eddy_current/fem_gnn_comparison.vtu")
vtk_file_2_coil = Path("verification_plots/eddy_current/fem_gnn_comparison_2.vtu")

# Current density line plot over the workpiece length

point1 = (0.0149, 0.07019260528546305, 0)
point2 = (0.0149, 0.13970934571534108, 0)


def do_for_one(vtk_file, point1, point2, list_field_names=["J_abs_fem", "J_abs_gnn"], axis=1):
    extract_data = ExtractDataOverLine(vtk_file)
    extract_data.set_points(point1, point2)
    result_data = extract_data.get_data(list_field_names)
    predicted_solution = np.array(result_data.results[list_field_names[1]])
    fem_solution = np.array(result_data.results[list_field_names[0]])
    line = np.asarray(result_data.point_data)[:, axis]

    return line, predicted_solution, fem_solution


line, predicted_solution, fem_solution = do_for_one(vtk_file_1_coil, point1, point2)
# Unit conversion for readability
z_mm = 1e3 * line

_, predicted_solution_2, fem_solution_2 = do_for_one(vtk_file_2_coil, point1, point2)

abs_err = np.abs(predicted_solution - fem_solution)
rel_err = abs_err / np.maximum(np.abs(fem_solution), 1e-12) * 100.0

abs_err_2 = np.abs(predicted_solution_2 - fem_solution_2)
rel_err_2 = abs_err_2 / np.maximum(np.abs(fem_solution_2), 1e-12) * 100.0

fig, (ax, ax_err) = plt.subplots(
    2,
    1,
    figsize=(6.4, 5.0),
    sharex=True,
    height_ratios=[3.0, 1.4],
    constrained_layout=True,
)

ax.plot(z_mm, fem_solution, linewidth=1, label="FEM (1 coil)", linestyle="--")
ax.plot(z_mm, fem_solution_2, linewidth=1, label="FEM (2 coils)", linestyle="--")
ax.plot(
    z_mm,
    predicted_solution,
    linewidth=1,
    marker="o",
    markevery=max(len(z_mm) // 15, 1),  # fewer markers
    markersize=4,
    label="PI-GNN (1 coil)",
)
ax.plot(
    z_mm,
    predicted_solution_2,
    linewidth=1,
    marker="x",
    markevery=max(len(z_mm) // 15, 1),  # fewer markers
    markersize=4,
    label="PI-GNN (2 coils)",
)

ax.set_ylabel(r"$|J|$ [A/m$^2$]")
# ax.set_yscale("log")
ax.grid(True)
# ax.legend(frameon=True, ncols=1)
# make legend smaller and put it inside the plot
ax.legend(frameon=True, ncols=1, fontsize="small")

ax_err.clear()
ax_err.plot(
    z_mm, rel_err, linewidth=1, label="1 coil", color=ax.get_lines()[2].get_color()
)
ax_err.plot(
    z_mm, rel_err_2, linewidth=1, label="2 coils", color=ax.get_lines()[3].get_color()
)
ax_err.set_xlabel("x [mm]")
ax_err.set_ylabel(r"$\epsilon_{\mathrm{rel}} [\%]$")
ax_err.grid(True)
ax_err.legend(frameon=True, ncols=2)

for a in (ax, ax_err):
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

plt.savefig(save_dir / "curr_dens_line_plot_coils.pdf", dpi=300)

# Q line plot

line, predicted_solution, fem_solution = do_for_one(vtk_file_1_coil, point1, point2, list_field_names=["Q_fem", "Q_gnn"])
z_mm = 1e3 * line

_, predicted_solution_2, fem_solution_2 = do_for_one(vtk_file_2_coil, point1, point2, list_field_names=["Q_fem", "Q_gnn"])

abs_err = np.abs(predicted_solution - fem_solution)
rel_err = abs_err / np.maximum(np.abs(fem_solution), 1e-12) * 100.0

abs_err_2 = np.abs(predicted_solution_2 - fem_solution_2)
rel_err_2 = abs_err_2 / np.maximum(np.abs(fem_solution_2), 1e-12) * 100.0

fig, (ax, ax_err) = plt.subplots(
    2,
    1,
    figsize=(6.4, 5.0),
    sharex=True,
    height_ratios=[3.0, 1.4],
    constrained_layout=True,
)

ax.plot(z_mm, fem_solution, linewidth=1, label="FEM (1 coil)", linestyle="--")
ax.plot(z_mm, fem_solution_2, linewidth=1, label="FEM (2 coils)", linestyle="--")
ax.plot(
    z_mm,
    predicted_solution,
    linewidth=1,
    marker="o",
    markevery=max(len(z_mm) // 15, 1),  # fewer markers
    markersize=4,
    label="PI-GNN (1 coil)",
)
ax.plot(
    z_mm,
    predicted_solution_2,
    linewidth=1,
    marker="x",
    markevery=max(len(z_mm) // 15, 1),  # fewer markers
    markersize=4,
    label="PI-GNN (2 coils)",
)

ax.set_ylabel(r"$Q$ [W/m$^2$]")
# ax.set_yscale("log")
ax.grid(True)
# ax.legend(frameon=True, ncols=1)
# make legend smaller and put it inside the plot
ax.legend(frameon=True, ncols=1, fontsize="small")

ax_err.clear()
ax_err.plot(
    z_mm, rel_err, linewidth=1, label="1 coil", color=ax.get_lines()[2].get_color()
)
ax_err.plot(
    z_mm, rel_err_2, linewidth=1, label="2 coils", color=ax.get_lines()[3].get_color()
)
ax_err.set_xlabel("x [mm]")
ax_err.set_ylabel(r"$\epsilon_{\mathrm{rel}} [\%]$")
ax_err.grid(True)
ax_err.legend(frameon=True, ncols=2)

for a in (ax, ax_err):
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

plt.savefig(save_dir / "q_line_plot_coils.pdf", dpi=300)

# Plot over line for thickness

point1 = (0.009, 0.075, 0)
point2 = (0.0149, 0.075, 0)

line, predicted_solution, fem_solution = do_for_one(vtk_file_1_coil, point1, point2, axis=0)
# Unit conversion for readability
z_mm = 1e3 * line

_, predicted_solution_2, fem_solution_2 = do_for_one(vtk_file_2_coil, point1, point2, axis=0)

abs_err = np.abs(predicted_solution - fem_solution)
rel_err = abs_err / np.maximum(np.abs(fem_solution), 1e-12) * 100.0

abs_err_2 = np.abs(predicted_solution_2 - fem_solution_2)
rel_err_2 = abs_err_2 / np.maximum(np.abs(fem_solution_2), 1e-12) * 100.0

fig, (ax, ax_err) = plt.subplots(
    2,
    1,
    figsize=(6.4, 5.0),
    sharex=True,
    height_ratios=[3.0, 1.4],
    constrained_layout=True,
)

ax.plot(z_mm, fem_solution, linewidth=1, label="FEM (1 coil)", linestyle="--")
ax.plot(z_mm, fem_solution_2, linewidth=1, label="FEM (2 coils)", linestyle="--")
ax.plot(
    z_mm,
    predicted_solution,
    linewidth=1,
    marker="o",
    markevery=max(len(z_mm) // 15, 1),  # fewer markers
    markersize=4,
    label="PI-GNN (1 coil)",
)
ax.plot(
    z_mm,
    predicted_solution_2,
    linewidth=1,
    marker="x",
    markevery=max(len(z_mm) // 15, 1),  # fewer markers
    markersize=4,
    label="PI-GNN (2 coils)",
)

ax.set_ylabel(r"$|J|$ [A/m$^2$]")
# ax.set_yscale("log")
ax.grid(True)
# ax.legend(frameon=True, ncols=1)
# make legend smaller and put it inside the plot
ax.legend(frameon=True, ncols=1, fontsize="small")

ax_err.clear()
ax_err.plot(
    z_mm, rel_err, linewidth=1, label="1 coil", color=ax.get_lines()[2].get_color()
)
ax_err.plot(
    z_mm, rel_err_2, linewidth=1, label="2 coils", color=ax.get_lines()[3].get_color()
)
ax_err.set_xlabel("x [mm]")
ax_err.set_ylabel(r"$\epsilon_{\mathrm{rel}} [\%]$")
ax_err.grid(True)
ax_err.legend(frameon=True, ncols=2)

for a in (ax, ax_err):
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

plt.savefig(save_dir / "curr_dens_line_plot_coils_thickness.pdf", dpi=300)

# Q line plot over thickness

line, predicted_solution, fem_solution = do_for_one(vtk_file_1_coil, point1, point2, axis=0, list_field_names=["Q_fem", "Q_gnn"])
# Unit conversion for readability
z_mm = 1e3 * line

_, predicted_solution_2, fem_solution_2 = do_for_one(vtk_file_2_coil, point1, point2, axis=0, list_field_names=["Q_fem", "Q_gnn"])

abs_err = np.abs(predicted_solution - fem_solution)
rel_err = abs_err / np.maximum(np.abs(fem_solution), 1e-12) * 100.0

abs_err_2 = np.abs(predicted_solution_2 - fem_solution_2)
rel_err_2 = abs_err_2 / np.maximum(np.abs(fem_solution_2), 1e-12) * 100.0

fig, (ax, ax_err) = plt.subplots(
    2,
    1,
    figsize=(6.4, 5.0),
    sharex=True,
    height_ratios=[3.0, 1.4],
    constrained_layout=True,
)

ax.plot(z_mm, fem_solution, linewidth=1, label="FEM (1 coil)", linestyle="--")
ax.plot(z_mm, fem_solution_2, linewidth=1, label="FEM (2 coils)", linestyle="--")
ax.plot(
    z_mm,
    predicted_solution,
    linewidth=1,
    marker="o",
    markevery=max(len(z_mm) // 15, 1),  # fewer markers
    markersize=4,
    label="PI-GNN (1 coil)",
)
ax.plot(
    z_mm,
    predicted_solution_2,
    linewidth=1,
    marker="x",
    markevery=max(len(z_mm) // 15, 1),  # fewer markers
    markersize=4,
    label="PI-GNN (2 coils)",
)

ax.set_ylabel(r"$Q$ [W/m$^2$]")
# ax.set_yscale("log")
ax.grid(True)
# ax.legend(frameon=True, ncols=1)
# make legend smaller and put it inside the plot
ax.legend(frameon=True, ncols=1, fontsize="small")

ax_err.clear()
ax_err.plot(
    z_mm, rel_err, linewidth=1, label="1 coil", color=ax.get_lines()[2].get_color()
)
ax_err.plot(
    z_mm, rel_err_2, linewidth=1, label="2 coils", color=ax.get_lines()[3].get_color()
)
ax_err.set_xlabel("x [mm]")
ax_err.set_ylabel(r"$\epsilon_{\mathrm{rel}} [\%]$")
ax_err.grid(True)
ax_err.legend(frameon=True, ncols=2)

for a in (ax, ax_err):
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

plt.savefig(save_dir / "q_line_plot_coils_thickness.pdf", dpi=300)

plotter1 = VTKToPlotConverter(vtk_file_1_coil, last_time_step=0, val_range=(0, 2.5))
plotter1.plot_steady_state(
    save_dir / "steady_state_1_coil_real.pdf",
    exact_field_name="E_real_fem",
    predicted_field_name="E_real_gnn",
    label=r"$\Re{E}$ [V/m]",
    contours=True,
)
plotter1.plot_relative_error(
    save_dir / "relative_error_1_coil_real.pdf", field_name="diff_E_real"
)
plotter1.plot_steady_state(
    save_dir / "steady_state_1_coil_imag.pdf",
    exact_field_name="E_imag_fem",
    predicted_field_name="E_imag_gnn",
    label=r"$\Im{E}$ [V/m]",
    contours=True,
)
plotter1.plot_relative_error(
    save_dir / "relative_error_1_coil_imag.pdf", field_name="diff_E_imag"
)

plotter1.plot_steady_state(
    save_dir / "steady_state_1_coil_abs.pdf",
    exact_field_name="E_abs_fem",
    predicted_field_name="E_abs_gnn",
    label=r"$|E|$ [V/m]",
    contours=True,
)
plotter1.plot_relative_error(
    save_dir / "relative_error_1_coil_abs.pdf", field_name="diff_E_abs"
)

plotter1.plot_steady_state(
    save_dir / "curr_dens_1_coil_abs.pdf",
    exact_field_name="J_abs_fem",
    predicted_field_name="J_abs_gnn",
    label=r"$|J|$ [A/m$^2$]",
    contours=True,
)

plotter2 = VTKToPlotConverter(vtk_file_2_coil, last_time_step=0, val_range=(0, 2.5))
plotter2.plot_steady_state(
    save_dir / "steady_state_2_coil_real.pdf",
    exact_field_name="E_real_fem",
    predicted_field_name="E_real_gnn",
    label=r"$\Re{E}$ [V/m]",
    contours=True,
)
plotter2.plot_relative_error(
    save_dir / "relative_error_2_coil_real.pdf", field_name="diff_E_real"
)

plotter2.plot_steady_state(
    save_dir / "steady_state_2_coil_imag.pdf",
    exact_field_name="E_imag_fem",
    predicted_field_name="E_imag_gnn",
    label=r"$\Im{E}$ [V/m]",
    contours=True,
)
plotter2.plot_relative_error(
    save_dir / "relative_error_2_coil_imag.pdf", field_name="diff_E_imag"
)

plotter2.plot_steady_state(
    save_dir / "steady_state_2_coil_abs.pdf",
    exact_field_name="E_abs_fem",
    predicted_field_name="E_abs_gnn",
    label=r"$|E|$ [V/m]",
    contours=True,
)
plotter2.plot_relative_error(
    save_dir / "relative_error_2_coil_abs.pdf", field_name="diff_E_abs"
)

plotter2.plot_steady_state(
    save_dir / "curr_dens_2_coil_abs.pdf",
    exact_field_name="J_abs_fem",
    predicted_field_name="J_abs_gnn",
    label=r"$|J|$ [A/m$^2$]",
    contours=True,
)