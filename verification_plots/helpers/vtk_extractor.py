import vtk
import xml.etree.ElementTree as ET
from pathlib import Path
from vtk.util import numpy_support
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from .error_metrics import compute_relative_error


class VtuDataExtractor:
    """Implement this class to extract data from vtu result files.
    need to implement get_unstructured_grid and set_time_value."""

    def __init__(self, base_path: Path, has_time_data: bool = True):
        self.result_file = base_path
        self.reader = vtk.vtkXMLUnstructuredGridReader()
        self.reader.SetFileName(str(base_path))
        self.reader.Update()
        self.has_time_data = has_time_data

        # if base_path.suffix == ".pvd":
        if self.has_time_data:
            self.time_data = self._get_time_data_from_pvd()

    def _get_time_data_from_pvd(self):
        """
        Extract time and file information from a ParaView Data (PVD) file.
        """
        tree = ET.parse(self.result_file.parent / (self.result_file.stem + ".pvd"))
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

    def get_unstructured_grid(self, field_name: str) -> vtk.vtkUnstructuredGrid:
        ug = self.reader.GetOutput()
        if ug.GetPointData().GetArray(field_name) is None:
            raise ValueError(f"Field {field_name} not found in VTU file!")
        return ug

    def set_time_value(self, time_value: float):
        """VTU files do not support time series, so this use time_data."""
        matched_dataset = None
        for dataset in self.time_data:
            if float(dataset["time_step"]) == time_value:
                matched_dataset = dataset
                break
        if matched_dataset is None:
            raise ValueError(f"Time value {time_value} not found in VTU time data!")
        # Update reader to read the matched file
        vtu_file_path = self.result_file.parent / matched_dataset["file"]
        self.reader.SetFileName(str(vtu_file_path))
        self.reader.Update()


class VTKToPlotConverter(VtuDataExtractor):
    def __init__(
        self,
        result_file: Path,
        last_time_step: int = 100,
        val_range: tuple[float, float] = (0.0, 1.0),
        has_time_data: bool = True,
    ):
        super().__init__(result_file, has_time_data=has_time_data)
        self.last_time_step = last_time_step
        self.val_range = val_range

    def _plot_mesh(self, ax, points, triangles, linewidth=0.1):
        tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
        ax.triplot(tri, color="k", linewidth=linewidth)
        ax.set_aspect("equal", "box")
        return tri

    def _get_triangles_point_field(self, field_name: str):
        ugrid = self.get_unstructured_grid(field_name)
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
            field = numpy_support.vtk_to_numpy(field_array)

        return points, triangles, field

    def plot_pred_and_fem(self, save_path: Path, contours: bool = False, shrink: float = 0.8, figsize=(9, 9)):
        """
        2x2 subplot
        a) Initial condition t=0s
        b) Relative error at t=1s
        c) Predicted solution at t=1s
        d) FEM solution at t=1s
        """
        _, _, initial_condition = self._get_triangles_point_field("ExactSolution")
        self.set_time_value(1)
        points, triangles, predicted = self._get_triangles_point_field(
            "PredictedSolution"
        )
        _, _, fem_solution = self._get_triangles_point_field("ExactSolution")
        _, _, difference = self._get_triangles_point_field("Difference, %")

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Top row: initial condition + relative error
        for ax, values, title in zip(
            axes[0],
            [initial_condition, difference],
            [r"a) Initial condition $t=0$ s", r"b) Relative error $t=1$ s"],
        ):
            tri = self._plot_mesh(ax, points, triangles)
            if "Relative error" in title:
                abs_max = np.max(np.abs(difference))
                pcm_err = ax.tripcolor(tri, values, cmap="seismic", vmin=-abs_max, vmax=abs_max)
            else:
                pcm_err = ax.tripcolor(
                    tri, values, cmap="coolwarm"
                )
            ax.set_title(title)
            ax.set_axis_off()
        v = np.linspace(-abs_max, abs_max, 3, endpoint=True).round(1)
        fig.colorbar(
            pcm_err, ax=axes[0].tolist(),
            label=r"Relative error $\epsilon_{\text{rel}}$ [\%]",
            ticks=v,
            shrink=shrink,
        )

        # Bottom row: predicted + FEM
        for ax, values, title in zip(
            axes[1],
            [predicted, fem_solution],
            [r"c) PI-GNN $t=1$ s", r"d) FEM $t=1$ s"],
        ):
            tri = self._plot_mesh(ax, points, triangles)
            if contours:
                pcm_temp = ax.tricontourf(
                    tri, values, cmap="coolwarm", levels=10,
                    vmin=self.val_range[0], vmax=self.val_range[1],
                )
            else:
                pcm_temp = ax.tripcolor(
                    tri, values, cmap="coolwarm",
                    vmin=self.val_range[0], vmax=self.val_range[1],
                )
            ax.set_title(title)
            ax.set_axis_off()
        v = np.linspace(self.val_range[0], self.val_range[1], 5, endpoint=True).round(1)
        fig.colorbar(
            pcm_temp, ax=axes[1].tolist(),
            label=r"Temperature $T$ [$^\circ$C]",
            ticks=v,
            shrink=shrink,
        )

        fig.savefig(save_path, dpi=600)

    def plot_steady_state(
        self,
        save_path: Path,
        exact_field_name: str = "ExactSolution",
        predicted_field_name: str = "PredictedSolution",
        label: str = "Temperature [C]",
        contours: bool = False,
        fraction: float = 0.063,
        time_value: float = 1.0,
    ):
        if self.has_time_data:
            self.set_time_value(time_value)
        points, triangles, predicted = self._get_triangles_point_field(
            predicted_field_name
        )
        _, _, fem_solution = self._get_triangles_point_field(exact_field_name)

        fig, axes = plt.subplots(1, 2, figsize=(6, 4))
        for ax, (values, title) in zip(
            axes, [(predicted, "a) PI-GNN"), (fem_solution, "b) FEM")]
        ):
            tri = self._plot_mesh(ax, points, triangles)
            if contours:
                pcm = ax.tricontourf(tri, values, cmap="coolwarm", levels=10)
            else:
                pcm = ax.tripcolor(tri, values, cmap="coolwarm", shading='gouraud')
            ax.set_title(title)
            ax.set_axis_off()
        fig.colorbar(pcm, ax=axes, label=label, orientation="horizontal", fraction=fraction)
        fig.subplots_adjust(wspace=0, bottom=0.2)
        # fig.tight_layout()
        fig.savefig(save_path, dpi=600)
        # plt.subplot_tool(fig)
        # plt.show()

    def plot_relative_error(self, save_path: Path, field_name: str = "Difference, %"):
        points, triangles, difference = self._get_triangles_point_field(field_name)
        fig, ax = plt.subplots(figsize=(6, 4))
        tri = self._plot_mesh(ax, points, triangles, linewidth=0.05)
        pcm = ax.tripcolor(tri, difference, cmap="Spectral_r", edgecolors='face', antialiased=True, shading='gouraud')
        fig.colorbar(pcm, ax=ax, label=r"Relative error $\epsilon_{\text{err}}$ [\%]", shrink=0.9)
        ax.set_title("c) Relative error")
        ax.set_axis_off()
        fig.savefig(save_path, dpi=300)
    
    def plot_relative_error_fields(self, exact_name: str, predicted_name: str, save_path: Path, **kwargs):
        points, triangles, exact = self._get_triangles_point_field(exact_name)
        _, _, predicted = self._get_triangles_point_field(predicted_name)
        rel_err = compute_relative_error(predicted, exact)
        fig, ax = plt.subplots(figsize=(6, 4))
        tri = self._plot_mesh(ax, points, triangles, linewidth=0.05)
        pcm = ax.tripcolor(tri, rel_err, cmap="Spectral_r", edgecolors='face', antialiased=True, shading='gouraud')
        fig.colorbar(pcm, ax=ax, label=r"Relative error $\epsilon_{\text{err}}$ [\%]", shrink=0.9)
        ax.set_title("c) Relative error")
        ax.set_axis_off()
        fig.savefig(save_path, dpi=300)