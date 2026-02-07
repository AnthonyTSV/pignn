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
from helpers.plot_helpers import plot_l2, epoch_vs_l2, epoch_vs_train_loss

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

mesh_filename = "results/physics_informed/test_em_problem_mixed/results_data/mesh.vol"
ngmesh = netgen.meshing.Mesh(dim=2)
ngmesh.Load(mesh_filename)
mesh = ng.Mesh(ngmesh)
space = ng.H1(mesh, order=1, dirichlet="left|right|top|bottom")
gf = ng.GridFunction(space)
# npz_filename = "results/physics_informed/test_em_problem_mixed/results_data/results.npz"
# data = np.load(npz_filename)
# epoch_vs_l2(paths=[Path("results/physics_informed/test_em_problem_mixed/training_log.json")], save_dir=Path("verification_plots/em1"))
epoch_vs_train_loss(paths=[Path("results/physics_informed/test_em_problem_mixed/training_log.json")], save_dir=Path("verification_plots/em1"))