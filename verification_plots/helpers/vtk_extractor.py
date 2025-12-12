import vtk
import xml.etree.ElementTree as ET
from pathlib import Path
from vtk.util import numpy_support
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

class VtuDataExtractor:
    """Implement this class to extract data from vtu result files.
    need to implement get_unstructured_grid and set_time_value."""
    def __init__(self, base_path: Path):
        self.result_file = base_path
        self.reader = vtk.vtkXMLUnstructuredGridReader()
        self.reader.SetFileName(str(base_path))
        self.reader.Update()

        self.time_data = self._get_time_data_from_pvd()

    def _get_time_data_from_pvd(self):
        """
        Extract time and file information from a ParaView Data (PVD) file.
        """
        tree = ET.parse(self.result_file.parent / (self.result_file.stem + ".pvd"))
        root = tree.getroot()
        dataset_list = []

        # Iterate through the DataSet elements and extract timestep and file variables
        for idx, dataset in enumerate(sorted(root.findall(".//DataSet"), key=lambda x: float(x.get("timestep")))):
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
            raise ValueError("Time value not found in VTU time data!")
        # Update reader to read the matched file
        vtu_file_path = self.result_file.parent / matched_dataset["file"]
        self.reader.SetFileName(str(vtu_file_path))
        self.reader.Update()

class VTKToPlotConverter(VtuDataExtractor):
    def __init__(self, result_file: Path, last_time_step: int = 100, val_range: tuple[float, float] = (0.0, 1.0)):
        super().__init__(result_file)
        self.last_time_step = last_time_step
        self.val_range = val_range
    
    def _plot_mesh(self, ax, points, triangles):
        tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
        ax.triplot(tri, color="k", linewidth=0.1)
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

    def plot_pred_and_fem(self, save_path: Path):
        """
        2x2 subplot
        a) Initial condition t=0s
        b) Relative error at t=0.99s
        c) Predicted solution at t=0.99s
        d) FEM solution at t=0.99s
        """
        _, _, initial_condition = self._get_triangles_point_field("ExactSolution")
        self.set_time_value(0.99)
        points, triangles, predicted = self._get_triangles_point_field("PredictedSolution")
        _, _, fem_solution = self._get_triangles_point_field("ExactSolution")
        _, _, difference = self._get_triangles_point_field("Difference, %")

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        axes = axes.flatten()
        panels = [
            (
                initial_condition,
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
            (fem_solution, "d) Analytical $t=0.99$ s", "coolwarm", "Temperature [C]", True),
        ]

        for ax, (values, title, cmap, cb_label, fixed_range) in zip(axes, panels):
            tri = self._plot_mesh(ax, points, triangles)
            if fixed_range:
                cmap = plt.get_cmap(cmap)
                pcm = ax.tripcolor(tri, values, cmap=cmap, vmin=self.val_range[0], vmax=self.val_range[1])
            else:
                pcm = ax.tripcolor(tri, values, cmap=cmap)
            fig.colorbar(pcm, ax=ax, label=cb_label)
            ax.set_title(title)
            ax.set_axis_off()

        # fig.tight_layout()
        fig.savefig(save_path, dpi=600)