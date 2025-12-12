"""
Module defines class ExtractDataOverLine that can be used to extract data over line.
"""
import vtk
import xml.etree.ElementTree as ET
from pathlib import Path
from vtk.util import numpy_support
from numpy import ndarray
import datetime
import json

try:
    from vtk_extractor import VtuDataExtractor
except ImportError:
    from .vtk_extractor import VtuDataExtractor
from pydantic import BaseModel


class DataOnLine(BaseModel):
    """Model to store data over line extracted from results"""
    point1: tuple[float, float, float]
    point2: tuple[float, float, float]
    result_time: float
    data_extracted_on: str
    result_file_path: str
    point_data: list[tuple[float, float, float]]
    results: dict[str, list]


class EnsightDataExtractor:
    """Extracts data from ensight format results"""
    def __init__(self, result_file: Path):
        self.reader = vtk.vtkEnSightGoldBinaryReader()
        self.reader.SetCaseFileName(str(result_file))
        self.reader.Update()

    def get_unstructured_grid(self, field_name: str) -> vtk.vtkUnstructuredGrid:
        """Returns unstructured grid combined for domains that has provided field."""
        ugrid_appender = vtk.vtkAppendFilter()
        for i in range(self.reader.GetOutput().GetNumberOfBlocks()):
            block = self.reader.GetOutput().GetBlock(i)
            # append only blocks that have requested field
            if block.GetPointData().GetArray(field_name) is None:
                continue
            ugrid_appender.AddInputData(block)
            ugrid_appender.Update()
        return ugrid_appender.GetOutput()

    def set_time_value(self, time_value: float):
        """Sets time field for active reader results."""
        time_steps = self.reader.GetTimeSets().GetItem(0)
        time_range = time_steps.GetRange()
        if time_value < time_range[0] or time_value > time_range[1]:
            raise ValueError("Time value out of range!")
        self.reader.SetTimeValue(time_value)
        self.reader.Update()

class ExtractDataOverLine:
    """ 
    Class for extracting data over line.

    Usage example:

        ensight_file = Path("path/to/ensight/results/resFile.0.case")
        field_name = "Temperature,_[C]"
        extract_data = ExtractDataOverLine(ensight_file)
        extract_data.set_points((0.0136, 0.0220, 0), (0.0139, 0.0026, 0))
        extract_data.set_time_value(1.0)
        point_data, field_data = extract_data.extract_field(field_name)
        # to save extracted data in DataOnLine format:
        save_data_in = Path("path/to/json/file.json")
        extract_data.save_data(save_data_in, ["Temperature,_[C]"])
    """
    def __init__(self, result_file: Path):
        self.result_file = result_file
        file_extension = result_file.suffix
        self.extractor: EnsightDataExtractor | VtuDataExtractor = None
        if file_extension == ".case":
            self.extractor = EnsightDataExtractor(self.result_file)
        elif file_extension == ".vtu":
            self.extractor = VtuDataExtractor(self.result_file)
        else:
            raise ValueError(f"Unsupported result format {file_extension} in ExtractDataOverLine")
        self.point1: tuple[float] = None
        self.point2: tuple[float] = None
        self.time_value: float = None

    def set_time_value(self, time_value: float):
        """Set time value at which data will be extracted."""
        self.extractor.set_time_value(time_value)
        self.time_value = time_value

    def set_points(self, point1: tuple[float, float, float], point2: tuple[float, float, float]):
        """Set two points which will be used to extract data from results file"""
        if len(point1) != 3 or len(point2) != 3:
            raise ValueError("ExtractDataOverLine.set_points accepts tuples with lenght = 3!")
        self.point1 = point1
        self.point2 = point2

    def extract_field(self, field_name: str, line_resolution: int = 100) -> tuple[ndarray, ndarray]:
        """Extracts data over previously specified line for specified field.
        """
        # Create line source
        if self.point1 is None or self.point2 is None:
            raise ValueError("Points are not set for ExtractDataOverLine")
        line = vtk.vtkLineSource()
        line.SetPoint1(self.point1)
        line.SetPoint2(self.point2)
        line.SetResolution(line_resolution)
        line.Update()

        # Interpolate data over the line
        probe = vtk.vtkProbeFilter()
        probe.SetInputData(line.GetOutput())
        probe.SetSourceData(self.extractor.get_unstructured_grid(field_name))
        probe.Update()

        probe_data = probe.GetOutput()
        point_data = probe_data.GetPointData()
        array = point_data.GetArray(field_name)

        field_data = numpy_support.vtk_to_numpy(array)
        point_data = numpy_support.vtk_to_numpy(probe_data.GetPoints().GetData())
        return point_data, field_data

    def get_data(self, field_names: list[str]) -> DataOnLine:
        """Returns data for requested fields"""
        if not isinstance(field_names, list) or len(field_names) ==0:
            raise ValueError("Incorrect input. Provide valid list of field names!")
        result_data = {
            "point1": self.point1,
            "point2": self.point2,
            "result_time": self.time_value,
            "data_extracted_on": str(datetime.datetime.now()),
            "result_file_path": str(self.result_file),
            "results": {}
        }
        for field_name in field_names:
            point_data, field_data = self.extract_field(field_name)
            result_data["results"][field_name] = field_data.tolist()
        result_data["point_data"] = point_data.tolist()
        return DataOnLine(**result_data)

    def save_data(self, file_name: Path, field_names: list[str]):
        """Saves data for requested fields to specified file. File contents are then readable using DataOnLine class"""
        data_on_line = self.get_data(field_names)
        with open(file_name, "w", encoding="utf-8") as json_file:
            json.dump(data_on_line.model_dump(), json_file)
