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
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
from new_pignn.plotter import load_log

def plot_l2(paths: list[Path] = None, save_dir: Path = Path("verification_plots/problem2")):
    log_paths = paths
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
    plt.savefig(save_dir / "l2_error_comparison.pdf", dpi=300)

    plt.figure(figsize=(6, 4))
    for key, train_loss in train_losses.items():
        plt.plot(train_loss, label=f"maxh={key}", linewidth=1)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend(fancybox=True, frameon=True)
    plt.savefig(save_dir / "train_loss_comparison.pdf", dpi=300)