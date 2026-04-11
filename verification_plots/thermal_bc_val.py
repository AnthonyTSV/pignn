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
from helpers.error_metrics import compute_relative_error
from helpers.plot_helpers import epoch_vs_train_loss, time_vs_l2

plt.style.use(["science", "grid"])

from helpers.mpl_style import apply_mpl_style

apply_mpl_style()

def plot_cond_flux():
    mesh_filename = "results/physics_informed/thermal_bc_verification/results_data/mesh.vol"
    ngmesh = netgen.meshing.Mesh(dim=2)
    ngmesh.Load(mesh_filename)
    mesh = ng.Mesh(ngmesh)
    space = ng.H1(mesh, order=1)
    gf = ng.GridFunction(space)
    gf_prev = ng.GridFunction(space)
    
    npz_filename = "results/physics_informed/thermal_bc_verification/results_data/results.npz"
    data = np.load(npz_filename)
    exact = data["exact"]
    predicted = data["predicted"]

    dt = 0.01

    u, v = space.TnT()
    a = ng.BilinearForm(space)
    a += ng.InnerProduct(ng.grad(u), ng.grad(v)) * ng.dx
    a.Assemble()

    m = ng.BilinearForm(space)
    m += u * v * ng.dx
    m.Assemble()

    dofs_l = np.array(space.GetDofs(mesh.Boundaries("left")))
    dofs_r = np.array(space.GetDofs(mesh.Boundaries("right")))
    dofs_t = np.array(space.GetDofs(mesh.Boundaries("top")))
    dofs_top_only = dofs_t & ~dofs_l & ~dofs_r
    dofs_lr = dofs_l | dofs_r

    def compute_fluxes(func, func_prev):
        gf.vec[:].FV().NumPy()[:] = func
        gf_prev.vec[:].FV().NumPy()[:] = func_prev

        Ku = gf.vec.CreateVector()
        Ku.data = a.mat * gf.vec
        r = Ku.FV().NumPy().copy()

        Mdu = gf.vec.CreateVector()
        diff = gf.vec.CreateVector()
        diff.data = gf.vec - gf_prev.vec
        Mdu.data = m.mat * diff
        rdot = Mdu.FV().NumPy().copy() / dt

        q_cond = np.sum(r[dofs_lr]) + np.sum(rdot[dofs_lr])
        q_loss = -(np.sum(r[dofs_top_only]) + np.sum(rdot[dofs_top_only]))
        return q_cond, q_loss

    fem_q_cond = []
    fem_q_loss = []
    PIGNN_q_cond = []
    PIGNN_q_loss = []
    rel_err_cond = []
    rel_err_loss = []
    for step in range(1, exact.shape[0]):
        q_cond_fem, q_loss_fem = compute_fluxes(exact[step], exact[step - 1])
        q_cond_PIGNN, q_loss_PIGNN = compute_fluxes(predicted[step], predicted[step - 1])

        fem_q_cond.append(q_cond_fem)
        fem_q_loss.append(q_loss_fem)
        PIGNN_q_cond.append(q_cond_PIGNN)
        PIGNN_q_loss.append(q_loss_PIGNN)
    
    rel_err_cond = compute_relative_error(np.array(fem_q_cond), np.array(PIGNN_q_cond))
    rel_err_loss = compute_relative_error(np.array(fem_q_loss), np.array(PIGNN_q_loss))
    
    print(f"q_cond (FEM): {fem_q_cond[-1]:.4f} W/m")
    print(f"q_cond (PI-GNN): {PIGNN_q_cond[-1]:.4f} W/m")
    print(f"q_loss (FEM): {fem_q_loss[-1]:.4f} W/m")
    print(f"q_loss (PI-GNN): {PIGNN_q_loss[-1]:.4f} W/m")

    fig, (ax, ax_err) = plt.subplots(
        2, 1,
        figsize=(6.4, 5.0),
        sharex=True,
        height_ratios=[3.0, 1.4],
        constrained_layout=True
    )
    ax.plot(fem_q_cond, label=r"$q_{\mathrm{cond}}$ FEM", linestyle="--")
    ax.plot(fem_q_loss, label=r"$q_{\mathrm{loss}}$ FEM", linestyle="--")
    ax.plot(
        PIGNN_q_cond,
        linewidth=1,
        marker="o",
        markevery=max(len(PIGNN_q_cond) // 15, 1), # fewer markers
        markersize=4,
        label=r"$q_{\mathrm{cond}}$ PI-GNN"
    )
    ax.plot(
        PIGNN_q_loss,
        linewidth=1,
        marker="o",
        markevery=max(len(PIGNN_q_loss) // 15, 1), # fewer markers
        markersize=4,
        label=r"$q_{\mathrm{loss}}$ PI-GNN"
    )
    ax.set_ylabel(r"Heat flux [W/m]")
    ax.grid(True)
    ax.legend(frameon=True, ncols=1, fontsize="small")
    ax_err.clear()
    ax_err.plot(rel_err_cond, linewidth=1, label=r"$\epsilon_{\mathrm{cond}}$", color=ax.get_lines()[1].get_color())
    ax_err.plot(rel_err_loss, linewidth=1, label=r"$\epsilon_{\mathrm{loss}}$", color=ax.get_lines()[3].get_color())
    ax_err.set_xlabel("Time step")
    ax_err.set_ylabel(r"$\epsilon_{\mathrm{rel}} [\%]$")
    ax_err.grid(True)
    ax_err.legend(frameon=True, ncols=2, fontsize="small")

    plt.savefig("verification_plots/thermal_bc_val/total_flux_comparison.pdf")
    plt.close()

def plot_pred_vs_fem():
    vtk_file = Path("results/physics_informed/thermal_bc_verification/vtk/result.vtu")
    plot_converter = VTKToPlotConverter(vtk_file, last_time_step=100, val_range=(39, 100))
    plot_converter.plot_pred_and_fem(save_path=Path("verification_plots/thermal_bc_val/pred_vs_fem_subplot.pdf"), shrink=0.7, figsize=(9, 6))



if __name__ == "__main__":
    plot_pred_vs_fem()
    plot_cond_flux()
    logs = [
        Path("results/physics_informed/thermal_bc_verification/training_log.json"),
    ]
    epoch_vs_train_loss(
        paths=logs,
        save_dir=Path("verification_plots/thermal_bc_val"),
        apply_smoothing=True,
        need_maxh_in_label=False
    )
    time_vs_l2(
        paths=logs,
        save_dir=Path("verification_plots/thermal_bc_val"),
        need_maxh_in_label=False,
        time_range=np.arange(0, 1.01, 0.01)
    )