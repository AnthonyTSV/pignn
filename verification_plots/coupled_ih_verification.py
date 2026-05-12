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
from new_pignn.em_eddy_problems import eddy_current_problem_different_mu_r
from new_pignn.thermal_problems import create_ih_problem
from new_pignn.plotter import load_log

plt.style.use(["science", "grid"])

from helpers.mpl_style import apply_mpl_style

apply_mpl_style()

save_dir = Path("verification_plots/coupled_ih")
sol_path = Path("results/coupled_physics_informed/test_ih_problem")
em_npz = sol_path / "results_data/results_em.npz"
thermal_npz = sol_path / "results_data/results_thermal.npz"
mesh_path = sol_path / "results_data/mesh.vol"
thermal_vtk = sol_path / "vtk/fem_solution.vtu"

mesh = ng.Mesh(str(mesh_path))
loaded_em = np.load(em_npz)

em_problem = eddy_current_problem_different_mu_r(mu_r_workpiece=1, sigma_workpiece=37037037, a_star=2e-3)

a_gnn = loaded_em["predicted"]
a_fem = loaded_em["exact"]

def get_total_deposited_power(a, em_solution, mesh):

    em_solution.vec.data = a

    sigma = 37037037
    frequency = 3000

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

def total_deposited_power():

    em_fes = ng.H1(mesh, order=1, dirichlet="bc_air|bc_axis|bc_workpiece_left", complex=True)
    em_solution = ng.GridFunction(em_fes)

    q_gnn = get_total_deposited_power(a_gnn, em_solution, mesh)
    q_fem = get_total_deposited_power(a_fem, em_solution, mesh)

    print(f"Total deposited power (PI-GNN): {q_gnn}")
    print(f"Total deposited power (FEM): {q_fem}")

    abs_error = np.abs(q_gnn - q_fem)
    rel_error = abs_error / (np.abs(q_fem) + 1e-8) * 100.0
    print(f"Absolute error: {abs_error}")
    print(f"Relative error: {rel_error:.2f}%")

def last_max_temp_error():

    loaded_thermal = np.load(thermal_npz)
    temp_gnn = loaded_thermal["predicted"]
    temp_fem = loaded_thermal["exact"]

    max_temp_gnn = np.max(temp_gnn[-1])
    max_temp_fem = np.max(temp_fem[-1])

    abs_error = np.abs(max_temp_gnn - max_temp_fem)
    rel_error = abs_error / (np.abs(max_temp_fem) + 1e-8) * 100.0

    print(f"Max temperature (PI-GNN): {max_temp_gnn}")
    print(f"Max temperature (FEM): {max_temp_fem}")
    print(f"Absolute error: {abs_error}")
    print(f"Relative error: {rel_error:.2f}%")

def plot_temperature_over_time():

    loaded_thermal = np.load(thermal_npz)
    temp_gnn = loaded_thermal["predicted"]
    temp_fem = loaded_thermal["exact"]

    workpiece_mask = np.where(temp_fem[0] != 0)
    temp_gnn_masked = temp_gnn[:, workpiece_mask]
    temp_fem_masked = temp_fem[:, workpiece_mask]

    l2_error = compute_l2_error(predicted_field=temp_gnn_masked[-1], exact_field=temp_fem_masked[-1])
    print(f"L2 error for temperature field over final time: {l2_error:.4e}")

    time_steps = temp_gnn.shape[0]
    max_temp_gnn = np.max(temp_gnn, axis=1)
    max_temp_fem = np.max(temp_fem, axis=1)

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
        np.arange(time_steps)[:: max(time_steps // 15, 1)],  # fewer markers
        max_temp_gnn[:: max(time_steps // 15, 1)],
        label="PI-GNN",
        color="tab:green",
        marker="x",
        s=40,
    )
    # ax.set_xlabel("x [mm]")
    ax.set_ylabel(r"Temperature [C]")
    ax.legend(frameon=True, ncols=1, fontsize="small")
    ax.grid(True)

    ax_err.plot(rel_err, linewidth=1, color="tab:green")
    ax_err.set_xlabel("Time step")
    ax_err.set_ylabel(r"$\epsilon_{\mathrm{rel}} [\%]$")
    ax_err.grid(True)
    plt.savefig(save_dir / "max_temperature_over_time.pdf", dpi=300)

def get_fields(mesh, gfA, problem: MeshProblemEM):

    sigma = mesh.MaterialCF(
        {
            "mat_workpiece": problem.sigma_workpiece * problem.sigma_star,
            "mat_air": problem.sigma_air,
            "mat_coil": problem.sigma_coil,
        },
        default=0.0,
    )

    E_phi = -1j * problem.omega * gfA
    J_phi = sigma * E_phi

    Q = 0.5 * sigma * ng.Norm(E_phi) ** 2
    current_density = ng.Norm(J_phi)

    return E_phi, J_phi, Q, current_density

def plot_joule_heat():

    save_dir.mkdir(parents=True, exist_ok=True)

    fes = ng.H1(mesh, order=1, dirichlet="bc_air|bc_axis|bc_workpiece_left", complex=True)
    scalar_fes = ng.H1(mesh, order=1, dirichlet="bc_air|bc_axis|bc_workpiece_left", definedon="mat_workpiece")
    gfA_gnn = ng.GridFunction(fes)
    gfA_fem = ng.GridFunction(fes)

    gfA_gnn.vec.data = a_gnn
    gfA_fem.vec.data = a_fem

    q_gnn = ng.GridFunction(scalar_fes)
    q_fem = ng.GridFunction(scalar_fes)
    diff_q = ng.GridFunction(scalar_fes)

    E_phi_gnn, J_phi_gnn, Q_gnn, current_density_gnn = get_fields(mesh, gfA_gnn, em_problem)
    E_phi_fem, J_phi_fem, Q_fem, current_density_fem = get_fields(mesh, gfA_fem, em_problem)

    q_gnn.Set(Q_gnn)
    q_fem.Set(Q_fem)
    diff_q.vec.FV().NumPy()[:] = compute_relative_error(
        q_fem.vec.FV().NumPy(), q_gnn.vec.FV().NumPy()
    )
    l2_erro_q = compute_l2_error(predicted_field=q_gnn.vec.FV().NumPy(), exact_field=q_fem.vec.FV().NumPy())
    print(f"L2 error for Q: {l2_erro_q:.4e}")

    workpiece_mask = mesh.MaterialCF({"mat_workpiece": 1.0}, default=0.0)
    vtk_file = save_dir / "q_comparison.vtu"
    workpiece_vtk_file = save_dir / "q_comparison_workpiece.vtu"

    vtk_out = ng.VTKOutput(
        mesh,
        coefs=[
            q_fem,
            q_gnn,
            diff_q,
            workpiece_mask,
        ],
        names=[
            "Q_fem",
            "Q_gnn",
            "Q_rel_error",
            "WorkpieceMask",
        ],
        filename=str(save_dir / "q_comparison"),
        order=1,
    )
    vtk_out.Do()

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtk_file))
    reader.Update()

    threshold = vtk.vtkThreshold()
    threshold.SetInputConnection(reader.GetOutputPort())
    threshold.SetInputArrayToProcess(
        0,
        0,
        0,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
        "WorkpieceMask",
    )
    if hasattr(threshold, "SetThresholdFunction"):
        threshold.SetLowerThreshold(0.5)
        threshold.SetUpperThreshold(1.5)
        threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    else:
        threshold.ThresholdBetween(0.5, 1.5)
    threshold.SetAllScalars(True)
    threshold.Update()

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(workpiece_vtk_file))
    writer.SetInputConnection(threshold.GetOutputPort())
    writer.Write()

    plotter = VTKToPlotConverter(workpiece_vtk_file, last_time_step=0, val_range=(0, 2e-3), has_time_data=False)
    plotter.plot_steady_state(
        save_dir / "q_field.pdf",
        exact_field_name="Q_fem",
        predicted_field_name="Q_gnn",
        label=r"$Q$ [W/m$^3$]",
        contours=False,
    )
    plotter.plot_relative_error(
        save_dir / "q_field_relative_error.pdf", field_name="Q_rel_error"
    )

def add_workpiece_mask_to_grid(grid: vtk.vtkUnstructuredGrid):
    if grid.GetPointData().GetArray("WorkpieceMask") is not None:
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    mask_file = save_dir / "workpiece_mask.vtu"
    workpiece_mask = mesh.MaterialCF({"mat_workpiece": 1.0}, default=0.0)
    vtk_out = ng.VTKOutput(
        mesh,
        coefs=[workpiece_mask],
        names=["WorkpieceMask"],
        filename=str(mask_file.with_suffix("")),
        order=1,
    )
    vtk_out.Do()

    mask_reader = vtk.vtkXMLUnstructuredGridReader()
    mask_reader.SetFileName(str(mask_file))
    mask_reader.Update()
    mask_grid = mask_reader.GetOutput()
    if mask_grid.GetNumberOfPoints() != grid.GetNumberOfPoints():
        raise ValueError(
            "Workpiece mask point count does not match thermal VTK point count."
        )

    mask_array = mask_grid.GetPointData().GetArray("WorkpieceMask")
    if mask_array is None:
        raise ValueError("WorkpieceMask was not written to the mask VTK file.")

    copied_mask_array = mask_array.NewInstance()
    copied_mask_array.DeepCopy(mask_array)
    copied_mask_array.SetName("WorkpieceMask")
    grid.GetPointData().AddArray(copied_mask_array)

def write_workpiece_only_vtk(input_vtk_file: Path, output_vtk_file: Path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(input_vtk_file))
    reader.Update()
    thermal_grid = reader.GetOutput()
    add_workpiece_mask_to_grid(thermal_grid)

    threshold = vtk.vtkThreshold()
    threshold.SetInputData(thermal_grid)
    threshold.SetInputArrayToProcess(
        0,
        0,
        0,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
        "WorkpieceMask",
    )
    if hasattr(threshold, "SetThresholdFunction"):
        threshold.SetLowerThreshold(0.5)
        threshold.SetUpperThreshold(1.5)
        threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    else:
        threshold.ThresholdBetween(0.5, 1.5)
    threshold.SetAllScalars(True)
    threshold.Update()

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(output_vtk_file))
    writer.SetInputConnection(threshold.GetOutputPort())
    writer.Write()

def write_workpiece_pvd():
    pvd_path = save_dir / "thermal_workpiece.pvd"
    pvd_path.write_text(
        "\n".join(
            [
                '<?xml version="1.0"?>',
                '<VTKFile type ="Collection" version="1.0" byte_order="LittleEndian">',
                "<Collection>",
                '<DataSet timestep="0" file="thermal_workpiece.vtu"/>',
                '<DataSet timestep="10" file="thermal_workpiece_step00100.vtu"/>',
                "</Collection>",
                "</VTKFile>",
                "",
            ]
        )
    )

def plot_last_state():
    initial_workpiece_vtk = save_dir / "thermal_workpiece.vtu"
    last_workpiece_vtk = save_dir / "thermal_workpiece_step00100.vtu"

    write_workpiece_only_vtk(thermal_vtk, initial_workpiece_vtk)
    write_workpiece_only_vtk(
        thermal_vtk.parent / "fem_solution_step00100.vtu",
        last_workpiece_vtk,
    )
    write_workpiece_pvd()

    plot_converter = VTKToPlotConverter(initial_workpiece_vtk, last_time_step=100, val_range=(70, 87))
    plot_converter.plot_pred_and_fem(save_path=save_dir / "pred_vs_fem_subplot.pdf", figsize=(9, 6))


if __name__ == "__main__":
    total_deposited_power()
    last_max_temp_error()
    plot_temperature_over_time()
    plot_joule_heat()
    plot_last_state()
