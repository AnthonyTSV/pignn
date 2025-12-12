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
from helpers.plot_helpers import plot_l2

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


def compute_convective_flux_top_boundary(
    filename: str,
    field_name: str = "PredictedSolution",
    h: float = 1.0,
    t_amb: float = 0.0,
    y_tol: float = 1e-8,
) -> float:
    """
    Compute the convective heat flux on the top boundary
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    grid: vtk.vtkUnstructuredGrid = reader.GetOutput()
    t_array = grid.GetPointData().GetArray(field_name)
    num_points = grid.GetNumberOfPoints()

    y_top = max(grid.GetPoint(i)[1] for i in range(num_points))

    total_length: float = 0.0
    conv_integral: float = 0.0

    num_cells = grid.GetNumberOfCells()

    for cell_id in range(num_cells):
        cell = grid.GetCell(cell_id)
        p_ids = [cell.GetPointId(i) for i in range(3)]
        edges = [(0, 1), (1, 2), (2, 0)]
        for i_local, j_local in edges:
            pid_i = p_ids[i_local]
            pid_j = p_ids[j_local]
            x_i, y_i, _ = grid.GetPoint(pid_i)
            x_j, y_j, _ = grid.GetPoint(pid_j)
            if abs(y_i - y_top) < y_tol and abs(y_j - y_top) < y_tol:
                length = math.hypot(x_j - x_i, y_j - y_i)
                t_i = t_array.GetTuple1(pid_i)
                t_j = t_array.GetTuple1(pid_j)
                t_avg_edge = 0.5 * (t_i + t_j)
                conv_integral += h * length * (t_amb - t_avg_edge)
                total_length += length
    q_conv = abs(conv_integral / total_length)
    return q_conv

from typing import Tuple

def compute_conductive_flux_left_right(
    filename: str,
    field_name: str = "PredictedSolution",
    k: float = 1.0,
    x_tol: float = 1e-8,
) -> Tuple[float, float]:
    """
    Compute conductive flux on the left and right boundaries:
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    grid: vtk.vtkUnstructuredGrid = reader.GetOutput()
    t_array = grid.GetPointData().GetArray(field_name)
    num_points = grid.GetNumberOfPoints()

    xs = [grid.GetPoint(i)[0] for i in range(num_points)]
    x_min = min(xs)
    x_max = max(xs)

    int_left = 0.0
    int_right = 0.0
    len_left = 0.0
    len_right = 0.0

    num_cells = grid.GetNumberOfCells()

    for cell_id in range(num_cells):
        cell = grid.GetCell(cell_id)

        p_ids = [cell.GetPointId(i) for i in range(3)]

        coords = []
        temps = []
        for pid in p_ids:
            x, y, _ = grid.GetPoint(pid)
            coords.append((x, y))
            temps.append(t_array.GetTuple1(pid))

        (x1, y1), (x2, y2), (x3, y3) = coords
        T1, T2, T3 = temps

        denom = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        if abs(denom) < 1e-14:
            continue  # degenerate triangle, skip

        grad_tx = (
            T1 * (y2 - y3) + T2 * (y3 - y1) + T3 * (y1 - y2)
        ) / denom
        grad_ty = (
            T1 * (x3 - x2) + T2 * (x1 - x3) + T3 * (x2 - x1)
        ) / denom

        # Examine edges (0-1), (1-2), (2-0)
        edges = [(0, 1), (1, 2), (2, 0)]
        for i_local, j_local in edges:
            pid_i = p_ids[i_local]
            pid_j = p_ids[j_local]

            x_i, y_i, _ = grid.GetPoint(pid_i)
            x_j, y_j, _ = grid.GetPoint(pid_j)

            # Edge length
            length = math.hypot(x_j - x_i, y_j - y_i)
            if length == 0.0:
                continue

            # Left boundary
            if abs(x_i - x_min) < x_tol and abs(x_j - x_min) < x_tol:
                n_x, n_y = -1.0, 0.0  # outward normal (left)
                q_n = -k * (grad_tx * n_x + grad_ty * n_y)
                int_left += q_n * length
                len_left += length

            # Right boundary
            elif abs(x_i - x_max) < x_tol and abs(x_j - x_max) < x_tol:
                n_x, n_y = 1.0, 0.0   # outward normal (right)
                q_n = -k * (grad_tx * n_x + grad_ty * n_y)
                int_right += q_n * length
                len_right += length

    if len_left <= 0.0:
        raise RuntimeError("Left boundary length is zero or could not be detected.")
    if len_right <= 0.0:
        raise RuntimeError("Right boundary length is zero or could not be detected.")

    q_cond_left = abs(int_left / len_left)
    q_cond_right = abs(int_right / len_right)

    return q_cond_left, q_cond_right

def plot_cond_flux():
    mesh_filename = "results/physics_informed/verification_test_problem_2_maxh_0.1/results_data/mesh.vol"
    ngmesh = netgen.meshing.Mesh(dim=2)
    ngmesh.Load(mesh_filename)
    mesh = ng.Mesh(ngmesh)
    space = ng.H1(mesh, order=1, dirichlet="left|right|top|bottom")
    gf = ng.GridFunction(space)
    
    npz_filename = "results/physics_informed/verification_test_problem_2_maxh_0.1/results_data/results.npz"
    data = np.load(npz_filename)
    exact = data["exact"]
    predicted = data["predicted"]

    def ngsolve_way(func):

        gf.vec[:].FV().NumPy()[:] = func
        q_conv = ng.Integrate(10 *(20 - gf) * ng.ds("top"), mesh)
        return q_conv
    
    fem_q = []
    pimgn_q = []
    for step in range(exact.shape[0]):
        exact_step = exact[step, :]
        predicted_step = predicted[step, :]

        q_fem = ngsolve_way(exact_step)
        q_pimgn = ngsolve_way(predicted_step)

        fem_q.append(q_fem)
        pimgn_q.append(q_pimgn)
    plt.figure(figsize=(6,4))
    plt.plot(fem_q, label="FEM")
    plt.plot(pimgn_q, label="PIMGN")
    plt.xlabel("Time step")
    plt.ylabel(r"$q_{\text{conv}}$ [W/m$^2$]")
    plt.legend(fancybox=True, frameon=True)
    plt.savefig("verification_plots/problem2/convective_flux_comparison.pdf")
    plt.close()

def plot_pred_vs_analytical():
    vtk_file = Path("results/physics_informed/verification_test_problem_2_maxh_0.1/vtk/result.vtu")
    plot_converter = VTKToPlotConverter(vtk_file, last_time_step=99, val_range=(39, 100))
    plot_converter.plot_pred_and_fem(save_path=Path("verification_plots/problem2/pred_vs_fem_subplot.pdf"))



if __name__ == "__main__":
    plot_pred_vs_analytical()
    plot_cond_flux()
    logs = [
        Path("results/physics_informed/verification_test_problem_2_maxh_0.1/training_log.json"),
        # Path("results/physics_informed/verification_test_problem_2_maxh_0.05/training_log.json"),
    ]
    plot_l2(paths=logs, save_dir=Path("verification_plots/problem2"))