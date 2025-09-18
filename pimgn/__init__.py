"""
Physics-Informed MeshGraphNets (PI-MGN) package.

Implementation of PI-MGN for solving heat equation on arbitrary meshes
with multi-mesh training support.
"""

from .model import PIMGN
from .utils import (
    FEMSolver, create_rectangular_mesh, create_lshape_mesh, create_polygon_mesh,
    create_circle_mesh, create_hollow_cylinder_mesh, create_hollow_circle_mesh,
    create_mesh, build_graph_from_mesh, create_heatmap_gif
)
from .training import PIGNNTrainer, TrainingConfig, MeshConfig, create_multi_mesh_trainer

__version__ = "1.0.0"
__author__ = "PI-MGN Implementation"

__all__ = [
    "PIMGN",
    "FEMSolver", 
    "create_rectangular_mesh",
    "create_lshape_mesh",
    "create_polygon_mesh",
    "create_circle_mesh", 
    "create_hollow_cylinder_mesh",
    "create_hollow_circle_mesh",
    "create_mesh",
    "build_graph_from_mesh",
    "create_heatmap_gif",
    "PIGNNTrainer",
    "TrainingConfig",
    "MeshConfig",
    "create_multi_mesh_trainer"
]
