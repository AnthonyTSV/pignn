from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import xml.etree.ElementTree as ET
import scienceplots

plt.style.use(["science", "grid"])

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "font.sans-serif": ["cmss10"],
        "font.monospace": ["cmtt10"],
        "axes.formatter.use_mathtext": True,
        # "font.size": 14,
    }
)

import sys
import vtk
from vtk.util import numpy_support

workspace_root = Path(__file__).resolve().parents[1]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from new_pignn.plotter import load_log
from helpers.line_data_extractor import ExtractDataOverLine


def _compute_analytical_solution(x, y, t, alpha=1.0):
    """Compute the analytical solution for the 2D heat equation with homogeneous Dirichlet BCs."""
    return (
        np.exp(-2 * alpha * np.pi**2 * t) * 100 * np.sin(np.pi * x) * np.sin(np.pi * y)
    )


def _get_time_data_from_pvd(pvd_file: Path):
    """
    Extract time and file information from a ParaView Data (PVD) file.
    """
    tree = ET.parse(pvd_file)
    root = tree.getroot()
    dataset_list = []

    # Iterate through the DataSet elements and extract timestep and file variables
    for idx, dataset in enumerate(
        sorted(root.findall(".//DataSet"), key=lambda x: float(x.get("timestep")))
    ):
        timestep = dataset.get("timestep")
        file = dataset.get("file")
        dataset_dict = {"index": idx, "time_step": timestep, "file": file}
        dataset_list.append(dataset_dict)

    return dataset_list


def _get_all_data(path: Path) -> ExtractDataOverLine:
    ensight_file = path
    point1 = (0, 0.5, 0)
    point2 = (1, 0.5, 0)
    extract_data = ExtractDataOverLine(ensight_file)
    extract_data.set_points(point1, point2)
    extract_data.set_time_value(0.1)

    return extract_data


def plot_middle_line():
    T = 0.99
    paths = [
        Path("results/physics_informed/single_problem_test_maxh_0.1/vtk/result.vtu"),
        Path("results/physics_informed/single_problem_test_maxh_0.2/vtk/result.vtu"),
        Path("results/physics_informed/single_problem_test_maxh_0.05/vtk/result.vtu"),
    ]
    data = {}
    maxh_values = [0.1, 0.2, 0.05]
    for maxh, path in zip(maxh_values, paths):
        extractor = _get_all_data(path)
        extractor.set_time_value(T)
        predicted_solution = extractor.get_data(["PredictedSolution"])
        x_data = np.array(predicted_solution.point_data)[:, 0]  # x-coordinates
        y_data = np.array(predicted_solution.point_data)[:, 1]  # y-coordinates

        pred_sol = predicted_solution.results["PredictedSolution"]
        data[maxh] = (pred_sol, x_data, y_data)

    anal_sol = _compute_analytical_solution(x_data, y_data, t=T, alpha=0.1)

    plt.figure(figsize=(6, 4))
    for key, (pred_sol, x_data, y_data) in data.items():
        plt.plot(x_data, pred_sol, label=f"maxh={key}", linewidth=1)
    plt.plot(x_data, anal_sol, label="Analytical Solution", linestyle="--", linewidth=1)
    plt.xlabel("x [m]")
    plt.ylabel("Temperature [C]")
    plt.legend(fancybox=True, frameon=True)
    plt.savefig("verification_plots/problem1/analytical_comparison.pdf", dpi=300)


def _read_vtk_mesh(vtk_file_name, field_name=None):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtk_file_name)
    reader.Update()
    ugrid = reader.GetOutput()

    if ugrid.GetNumberOfPoints() == 0:
        raise ValueError(f"No points found in {vtk_file_name}")

    points = numpy_support.vtk_to_numpy(ugrid.GetPoints().GetData())
    if points.shape[1] > 2:
        points = points[:, :2]

    triangles = []
    for cell_id in range(ugrid.GetNumberOfCells()):
        cell = ugrid.GetCell(cell_id)
        npts = cell.GetNumberOfPoints()
        if npts < 3:
            continue
        point_ids = [cell.GetPointId(i) for i in range(npts)]
        for i in range(1, npts - 1):
            triangles.append((point_ids[0], point_ids[i], point_ids[i + 1]))

    triangles = np.array(triangles, dtype=int)

    field = None
    if field_name is not None:
        field_array = ugrid.GetPointData().GetArray(field_name)
        if field_array is None:
            raise ValueError(f"Field '{field_name}' not found in {vtk_file_name}")
        field = numpy_support.vtk_to_numpy(field_array)

    return points, triangles, field


def _plot_mesh(ax, points, triangles):
    tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    ax.triplot(tri, color="k", linewidth=0.1)
    ax.set_aspect("equal", "box")
    return tri


def vtk_to_plot(vtk_file_name, field_name):
    points, triangles, field = _read_vtk_mesh(vtk_file_name, field_name)
    fig, ax = plt.subplots(figsize=(8, 6))
    tri = _plot_mesh(ax, points, triangles)

    pcm = ax.tripcolor(tri, field, cmap="coolwarm", shading="flat")
    fig.colorbar(pcm, ax=ax, label=field_name)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{field_name} from {vtk_file_name}")
    fig.tight_layout()
    fig.savefig(f"verification_plots/problem1/{field_name}_triangulated.pdf", dpi=300)


def plot_pred_and_analytical():
    """
    2x2 subplot
    a) Initial condition t=0s
    b) Relative error at t=0.99s
    c) Predicted solution at t=0.99s
    d) Analytical solution at t=0.99s
    """
    vtk_file = "results/physics_informed/single_problem_test_maxh_0.05/vtk/result_step00099.vtu"
    alpha = 0.1
    points, triangles, predicted = _read_vtk_mesh(vtk_file, "PredictedSolution")
    _, _, difference = _read_vtk_mesh(vtk_file, "Difference, %")

    t0 = 0.0
    analytic_initial = _compute_analytical_solution(
        points[:, 0], points[:, 1], t=t0, alpha=alpha
    )
    exact = _compute_analytical_solution(
        points[:, 0], points[:, 1], t=0.99, alpha=alpha
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    panels = [
        (
            analytic_initial,
            "a) Initial condition $t=0$ s",
            "coolwarm",
            "Temperature [C]",
            False,
        ),
        (
            difference,
            "b) Relative error $t=0.99$ s",
            "plasma",
            "Relative error [\%]",
            False,
        ),
        (predicted, "c) PI-GNN $t=0.99$ s", "coolwarm", "Temperature [C]", True),
        (exact, "d) Analytical $t=0.99$ s", "coolwarm", "Temperature [C]", True),
    ]

    for ax, (values, title, cmap, cb_label, fixed_range) in zip(axes, panels):
        tri = _plot_mesh(ax, points, triangles)
        if fixed_range:
            pcm = ax.tripcolor(tri, values, cmap=cmap, shading="flat", vmin=0, vmax=14)
        else:
            pcm = ax.tripcolor(tri, values, cmap=cmap, shading="flat")
        fig.colorbar(pcm, ax=ax, label=cb_label)
        ax.set_title(title, fontsize=16)
        ax.set_axis_off()

    # fig.tight_layout()
    fig.savefig("verification_plots/problem1/pred_vs_analytical_subplot.pdf", dpi=600)


def read_logs_and_plot_l2():
    log_paths = [
        Path("results/physics_informed/single_problem_test_maxh_0.1/training_log.json"),
        Path("results/physics_informed/single_problem_test_maxh_0.2/training_log.json"),
        Path(
            "results/physics_informed/single_problem_test_maxh_0.05/training_log.json"
        ),
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
        plt.plot(l2_error, label=f"maxh={key}", linewidth=1)
    plt.yscale("log")
    plt.xlabel("Time $t$ [s]")
    plt.ylabel("L2 Error")
    plt.legend(fancybox=True, frameon=True)
    plt.savefig("verification_plots/problem1/l2_error_comparison.pdf", dpi=300)

    plt.figure(figsize=(6, 4))
    for key, train_loss in train_losses.items():
        plt.plot(train_loss, label=f"maxh={key}", linewidth=1)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend(fancybox=True, frameon=True)
    plt.savefig("verification_plots/problem1/train_loss_comparison.pdf", dpi=300)


def compute_energy_from_vtu(
    filename: str, field_name: str = "PredictedSolution"
) -> float:
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    grid: vtk.vtkUnstructuredGrid = reader.GetOutput()
    t_array = grid.GetPointData().GetArray(field_name)

    num_cells = grid.GetNumberOfCells()
    energy: float = 0.0

    for cell_id in range(num_cells):
        cell = grid.GetCell(cell_id)
        p_ids = [cell.GetPointId(i) for i in range(3)]
        x1, y1, _ = grid.GetPoint(p_ids[0])
        x2, y2, _ = grid.GetPoint(p_ids[1])
        x3, y3, _ = grid.GetPoint(p_ids[2])
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        t_vals = [t_array.GetTuple1(pid) for pid in p_ids]
        t_avg = sum(t_vals) / 3.0
        energy += area * t_avg

    return energy


def energy_decay_plot():
    """
    Compute E(t) = \int T(x,y,t) dx dy
    """
    time_steps = np.linspace(0, 1, 100, endpoint=False)
    alpha = 0.1
    file_paths = {
        "0.05": "results/physics_informed/single_problem_test_maxh_0.05/vtk/",
        "0.1": "results/physics_informed/single_problem_test_maxh_0.1/vtk/",
        "0.2": "results/physics_informed/single_problem_test_maxh_0.2/vtk/",
    }
    analytical = 400 * np.exp(-2 * np.pi**2 * alpha * time_steps) / (np.pi**2)
    fig, ax = plt.subplots(figsize=(6, 4))
    for maxh, file_path in file_paths.items():
        energies_pred = []
        energies_fem = []
        pvd_file = Path(file_path) / "result.pvd"
        time_steps_data = _get_time_data_from_pvd(pvd_file)
        for next_file in time_steps_data:
            if next_file["index"] > 100:
                continue
            file_name = next_file["file"]
            vtk_file = f"{file_path}/{file_name}"
            energy_pred = compute_energy_from_vtu(vtk_file, "PredictedSolution")
            energy_fem = compute_energy_from_vtu(vtk_file, "ExactSolution")
            energies_pred.append(energy_pred)
            energies_fem.append(energy_fem)
        energies_pred = np.array(energies_pred)
        energies_fem = np.array(energies_fem)
        diff_pred = np.abs(energies_pred - analytical) / analytical
        diff_fem = np.abs(energies_fem - analytical) / analytical
        ax.plot(time_steps, diff_pred, label=f"PI-GNN maxh={maxh}", linewidth=1)
        ax.plot(
            time_steps, diff_fem, label=f"FEM maxh={maxh}", linewidth=1, linestyle="--"
        )

    ax.set_xlabel("Time $t$ [s]")
    ax.set_ylabel("Relative Error of $E(t)$, %")
    ax.set_yscale("log")
    ax.legend(fancybox=True, frameon=True)
    fig.savefig("verification_plots/problem1/energy_decay.pdf", dpi=300)


if __name__ == "__main__":
    plot_middle_line()
    plot_pred_and_analytical()
    read_logs_and_plot_l2()
    energy_decay_plot()
