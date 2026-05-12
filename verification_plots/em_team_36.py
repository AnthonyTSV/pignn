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
from new_pignn.containers import MeshProblemEM
from new_pignn.em_eddy_problems import em_team_36_problem
from new_pignn.plotter import load_log

plt.style.use(["science", "grid"])

from helpers.mpl_style import apply_mpl_style

apply_mpl_style()


def get_train_loss(log_path: Path) -> np.ndarray:
    log_data = load_log(log_path)
    train_loss = log_data["training_history"]["train_loss"]
    return np.abs(train_loss)

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

save_dir = Path("verification_plots/team_36_problem")
save_dir.mkdir(exist_ok=True, parents=True)

path_to_em_results = Path("results/physics_informed/em_team_36_problem")
em_npz = np.load(path_to_em_results / "results_data/results_em.npz")
em_mesh_filename = path_to_em_results / "results_data/mesh.vol"
ngmesh = netgen.meshing.Mesh(dim=2)
ngmesh.Load(str(em_mesh_filename))
em_mesh = ng.Mesh(ngmesh)

em_problem = em_team_36_problem(mesh=em_mesh)

em_predicted_solution = em_npz["predicted"] * em_problem.A_star
em_exact_solution = em_npz["exact"] * em_problem.A_star

pred_abs = np.abs(em_predicted_solution)
exact_abs = np.abs(em_exact_solution)

l2_error = compute_l2_error(pred_abs, exact_abs)

print(f"Relative L2 error for EM Team 36 problem: {l2_error:.4e}")

def get_total_deposited_power(a, em_solution, mesh):

    em_solution.vec.data = a

    sigma = 4761904
    frequency = 2000

    omega = 2 * np.pi * frequency
    E_phi = -1j * omega * em_solution
    heat_source_gf = (
        0.5
        * sigma
        * ng.Norm(E_phi) ** 2
    )

    # Integrate over the material region object
    workpiece_region = mesh.Materials("mat_workpiece")
    q_total = ng.Integrate(heat_source_gf, mesh, definedon=workpiece_region) * 2 * math.pi
    return q_total

def total_deposited_power(mesh, a_gnn, a_fem):

    em_fes = ng.H1(mesh, order=1, dirichlet="bc_air|bc_axis|bc_workpiece_left", complex=True)
    em_solution = ng.GridFunction(em_fes)

    q_gnn = get_total_deposited_power(a_gnn, em_solution, mesh)
    q_fem = get_total_deposited_power(a_fem, em_solution, mesh)

    print(f"Total deposited power (PI-GNN): {q_gnn}")
    print(f"Total deposited power (FEM): {q_fem}")

    abs_error = np.abs(q_gnn - q_fem)
    rel_error = abs_error / (np.abs(q_fem)) * 100.0
    print(f"Absolute error: {abs_error}")
    print(f"Relative error: {rel_error:.2f}%")

def export_vtk():
    space = ng.H1(
        em_mesh, order=1, dirichlet=em_problem.mesh_config.dirichlet_pipe, complex=True
    )
    gfA_fem = ng.GridFunction(space)
    gfA_gnn = ng.GridFunction(space)

    gfA_fem.vec[:].FV().NumPy()[:] = em_exact_solution
    gfA_gnn.vec[:].FV().NumPy()[:] = em_predicted_solution

    total_deposited_power(em_mesh, gfA_gnn.vec[:].FV().NumPy(), gfA_fem.vec[:].FV().NumPy())

    E_phi_fem, J_phi_fem, Q_fem, current_density_fem = get_fields(em_mesh, gfA_fem, em_problem)
    E_phi_gnn, J_phi_gnn, Q_gnn, current_density_gnn = get_fields(em_mesh, gfA_gnn, em_problem)

    space = ng.H1(em_mesh, order=1, dirichlet=em_problem.mesh_config.dirichlet_pipe)
    diff_a_real1 = ng.GridFunction(space)
    diff_a_imag1 = ng.GridFunction(space)
    diff_a_abs1 = ng.GridFunction(space)

    q_fem = ng.GridFunction(space)
    q_fem.Set(Q_fem)
    q_gnn = ng.GridFunction(space)
    q_gnn.Set(Q_gnn)
    diff_q_fem_gnn = ng.GridFunction(space)
    diff_q_fem_gnn.vec[:].FV().NumPy()[:] = compute_relative_error(q_fem.vec[:].FV().NumPy(), q_gnn.vec[:].FV().NumPy())

    max_joule_fem = np.max(q_fem.vec[:].FV().NumPy())
    max_joule_gnn = np.max(q_gnn.vec[:].FV().NumPy())
    rel_err_joule = (max_joule_fem - max_joule_gnn) / (max_joule_fem + 1e-8) * 100.0

    print("Max Joule heating Q (FEM):", max_joule_fem)
    print("Max Joule heating Q (GNN):", max_joule_gnn)
    print(f"Relative error for Joule heating Q: {rel_err_joule:.2f}%")

    l2_error_q = compute_l2_error(q_gnn.vec[:].FV().NumPy(), q_fem.vec[:].FV().NumPy())
    print(f"Relative L2 error for Joule heating Q: {l2_error_q:.4e}")

    diff_a_real1.vec[:].FV().NumPy()[:] = compute_relative_error(np.real(em_exact_solution), np.real(em_predicted_solution))
    diff_a_imag1.vec[:].FV().NumPy()[:] = compute_relative_error(np.imag(em_exact_solution), np.imag(em_predicted_solution))
    diff_a_abs1.vec[:].FV().NumPy()[:] = compute_relative_error(np.abs(em_exact_solution), np.abs(em_predicted_solution))


    vtk_out = ng.VTKOutput(
        em_mesh,
        coefs=[
            ng.Norm(gfA_fem),
            gfA_fem.real,
            gfA_fem.imag,
            ng.Norm(gfA_gnn),
            gfA_gnn.real,
            gfA_gnn.imag,
            diff_a_real1,
            diff_a_imag1,
            diff_a_abs1,
            q_fem,
            q_gnn,
            diff_q_fem_gnn,
        ],
        names=[
            "A_fem_abs",
            "A_fem_real",
            "A_fem_imag",
            "A_gnn_abs",
            "A_gnn_real",
            "A_gnn_imag",
            "diff_A_real",
            "diff_A_imag",
            "diff_A_abs",
            "Q_fem",
            "Q_gnn",
            "diff_Q_fem_gnn",
        ],
        filename=str(save_dir / "fields"),
    )
    vtk_out.Do()

def plot_magnetic_vector_potential():
    vtk_file = save_dir / "fields.vtu"
    plotter = VTKToPlotConverter(vtk_file, last_time_step=0, val_range=(0, 2.5), has_time_data=False)
    plotter.plot_steady_state(
        save_dir / "A_field.pdf",
        exact_field_name="A_fem_abs",
        predicted_field_name="A_gnn_abs",
        label=r"$|A| \, [\mathrm{T} \cdot \mathrm{m}]$",
        contours=False,
        use_scientific_ticks=True
    )
    plotter.plot_relative_error(
        save_dir / "A_field_relative_error.pdf", field_name="diff_A_abs"
    )

def do_for_one(vtk_file, point1, point2, list_field_names=["J_abs_fem", "J_abs_gnn"], axis=1):
    extract_data = ExtractDataOverLine(vtk_file, has_time_data=False)
    extract_data.set_points(point1, point2)
    result_data = extract_data.get_data(list_field_names)
    predicted_solution = np.array(result_data.results[list_field_names[1]])
    fem_solution = np.array(result_data.results[list_field_names[0]])
    line = np.asarray(result_data.point_data)[:, axis]

    return line, predicted_solution, fem_solution

def plot_a_over_line():
    vtk_file = save_dir / "fields.vtu"
    point1 = (0.0149, 0.07019260528546305, 0)
    point2 = (0.0149, 0.13970934571534108, 0)

    line, predicted_solution, fem_solution = do_for_one(vtk_file, point1, point2, list_field_names=["A_fem_abs", "A_gnn_abs"], axis=1)
    rel_err = compute_relative_error(fem_solution, predicted_solution)
    z_mm = 1e3 * line

    fig, (ax, ax_err) = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.0),
        sharex=True,
        height_ratios=[3.0, 1.4],
        constrained_layout=True,
    )

    ax.plot(z_mm, fem_solution, linewidth=1, label="FEM", linestyle="--")
    ax.plot(
        z_mm,
        predicted_solution,
        linewidth=1,
        marker="o",
        markevery=max(len(z_mm) // 15, 1),  # fewer markers
        markersize=4,
        label="PI-GNN",
    )
    ax_err.clear()
    ax_err.plot(
        z_mm, rel_err, linewidth=1, color=ax.get_lines()[1].get_color()
    )
    ax_err.set_xlabel("x [mm]")
    ax_err.set_ylabel(r"$\epsilon_{\mathrm{rel}} [\%]$")
    ax_err.grid(True)
    ax_err.legend(frameon=True, ncols=2, fontsize="small")

    ax.set_ylabel(r"$|A| \, [\mathrm{T} \cdot \mathrm{m}]$",)
    ax.grid(True)

    fig.savefig(save_dir / "A_over_line.pdf")
    plt.close(fig)

if __name__ == "__main__":
    export_vtk()
    # plot_magnetic_vector_potential()
    # plot_a_over_line()