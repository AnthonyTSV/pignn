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

def plot_energy_over_time():
    mesh_filename = "results/physics_informed/verification_test_problem_3_maxh_0.1/results_data/mesh.vol"
    ngmesh = netgen.meshing.Mesh(dim=2)
    ngmesh.Load(mesh_filename)
    mesh = ng.Mesh(ngmesh)
    space = ng.H1(mesh, order=1, dirichlet="left|right|top|bottom")
    gf = ng.GridFunction(space)
    
    npz_filename = "results/physics_informed/verification_test_problem_3_maxh_0.1/results_data/results.npz"
    data = np.load(npz_filename)
    exact = data["exact"]
    predicted = data["predicted"]

    def ngsolve_way(func):

        gf.vec[:].FV().NumPy()[:] = func
        q_conv = ng.Integrate(gf, mesh)
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
    fem_q = np.array(fem_q)
    pimgn_q = np.array(pimgn_q)
    diff_q = pimgn_q - fem_q

    fig, ax1 = plt.subplots(figsize=(6, 4))
    line_fem, = ax1.plot(fem_q, label="FEM", color='C0')
    line_pimgn = ax1.scatter(range(pimgn_q.size), pimgn_q, label="PIMGN", color='C1', marker='x')
    ax1.set_xlabel("Time step")
    ax1.set_ylabel(r"$E(t)$ [W/m$^2$]")

    ax2 = ax1.twinx()
    ax2.grid(False)
    line_diff, = ax2.plot(diff_q, label="Diff", linestyle="--", color='C2')
    ax2.set_ylabel(r"$E_{\text{FEM}} - E_{\text{PIMGN}}$ [W/m$^2$]")

    lines = [line_fem, line_pimgn, line_diff]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, fancybox=True, frameon=True, loc=(0.7, 0.6))

    fig.savefig("verification_plots/problem3/energy_over_time.pdf")
    plt.close(fig)

if __name__ == "__main__":
    plot_energy_over_time()