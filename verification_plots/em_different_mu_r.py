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
from new_pignn.containers import MeshProblemEM
from new_pignn.em_eddy_problems import eddy_current_problem_different_mu_r
from new_pignn.plotter import load_log

plt.style.use(["science"])

from helpers.mpl_style import apply_mpl_style

apply_mpl_style()

mu_rs = [1, 10, 50, 100]
save_dir = Path("verification_plots/em_different_mu_r_results")
save_dir.mkdir(exist_ok=True)
result_dir = Path("results/physics_informed/em_different_mu_r")

# calculate current density and power density for each mu_r


def q_current_density(gfA, problem):
    sigma = 6289308
    E_phi = -1j * problem.omega * gfA
    J_phi = sigma * problem.sigma_star * E_phi
    Q = 0.5 * sigma * problem.sigma_star * ng.Norm(E_phi) ** 2
    current_density = ng.Norm(J_phi)
    return current_density, Q


def calculate_current_and_power_density(mu_r):
    problem = eddy_current_problem_different_mu_r(mu_r_workpiece=mu_r)
    mesh_path = result_dir / f"mu_r_{mu_r}" / "results_data" / "mesh.vol"
    results_path = result_dir / f"mu_r_{mu_r}" / "results_data" / "results_em.npz"

    mesh = ng.Mesh(str(mesh_path))
    data = np.load(results_path)
    space = ng.H1(
        mesh, order=1, dirichlet=problem.mesh_config.dirichlet_pipe, complex=True
    )
    gfA_fem = ng.GridFunction(space)
    gfA_gnn = ng.GridFunction(space)

    exact = data["exact"]
    predicted = data["predicted"]

    gfA_fem.vec[:].FV().NumPy()[:] = exact * problem.A_star
    gfA_gnn.vec[:].FV().NumPy()[:] = predicted * problem.A_star

    sigma = 6289308

    current_density_fem, power_density_fem = q_current_density(gfA_fem, problem)
    current_density_gnn, power_density_gnn = q_current_density(gfA_gnn, problem)

    # save as vtk
    vtk_out = ng.VTKOutput(
        mesh,
        coefs=[
            current_density_fem,
            power_density_fem,
            current_density_gnn,
            power_density_gnn,
        ],
        names=[
            "current_density_fem",
            "power_density_fem",
            "current_density_gnn",
            "power_density_gnn",
        ],
        filename=str(save_dir / f"mu_r_{mu_r}"),
        order=1,
    )
    vtk_out.Do()


def do_for_one(
    vtk_file,
    point1,
    point2,
    list_field_names=["current_density_fem", "current_density_gnn"],
    axis=1,
):
    extract_data = ExtractDataOverLine(vtk_file, has_time_data=False)
    extract_data.set_points(point1, point2)
    result_data = extract_data.get_data(list_field_names)
    predicted_solution = np.array(result_data.results[list_field_names[1]])
    fem_solution = np.array(result_data.results[list_field_names[0]])
    line = np.asarray(result_data.point_data)[:, axis]

    return line, predicted_solution, fem_solution


def plot_current_and_power_density(mu_r, ax, ax_error):
    point1 = (0, 0.1, 0)
    point2 = (0.0149, 0.1, 0)

    vtk_file = save_dir / f"mu_r_{mu_r}.vtu"

    line, predicted_solution, fem_solution = do_for_one(
        vtk_file, point1, point2, axis=0
    )

    normalized_predicted_solution = predicted_solution / np.max(predicted_solution)
    normalized_fem_solution = fem_solution / np.max(fem_solution)

    rel_err = compute_relative_error(fem_solution, predicted_solution)

    x_mm = 1e3 * line

    ax.plot(x_mm, normalized_fem_solution, linewidth=1)
    # ax.plot(
    #     x_mm,
    #     normalized_predicted_solution,
    #     label=rf"$\mu_r={mu_r}$",
    #     linewidth=1,
    #     marker="x",
    #     markersize=4,
    #     markevery=max(len(x_mm) // 15, 1),  # fewer markers
    # )
    ax.scatter(
        x_mm[:: max(len(x_mm) // 15, 1)],  # fewer markers
        normalized_predicted_solution[:: max(len(x_mm) // 15, 1)],
        label=rf"$\mu_r={mu_r}$",
        color=ax.get_lines()[-1].get_color(),
        marker="x",
        s=30,
    )
    # ax.set_xlabel("x [mm]")
    ax.set_ylabel(r"Normalised $|J|$ [A/m$^2$]")
    ax.legend(ncols=1, fontsize="small")

    ax_error.plot(x_mm, rel_err, label=rf"$\mu_r={mu_r}$", linewidth=1, color=ax.get_lines()[-1].get_color())
    ax_error.set_xlabel("x [mm]")
    ax_error.set_ylabel(r"$\epsilon_{\mathrm{rel}} [\%]$")


if __name__ == "__main__":
    fig, (ax, ax_err) = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.0),
        sharex=True,
        height_ratios=[3.0, 1.4],
        constrained_layout=True,
    )
    for mu_r in mu_rs:
        calculate_current_and_power_density(mu_r)
        plot_current_and_power_density(mu_r=mu_r, ax=ax, ax_error=ax_err)

    plt.savefig(save_dir / "current_density_comparison.pdf", dpi=300)
