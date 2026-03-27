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
from helpers.line_data_extractor import ExtractDataOverLine
from helpers.plot_helpers import (
    plot_l2,
    epoch_vs_l2,
    epoch_vs_train_loss,
    epoch_vs_training_l2,
    _get_run_label,
)
from new_pignn.containers import MeshProblemEM
from new_pignn.em_eddy_problems import eddy_current_problem_1, eddy_current_problem_2, eddy_current_problem_different_currents
from new_pignn.plotter import load_log
from new_pignn.trainer_em import PIMGNTrainerEM

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


def get_l2(log_path: Path) -> np.ndarray:
    log_data = load_log(log_path)
    l2_error = log_data["evaluation"]["mean_l2_error"]
    return l2_error


save_dir = Path("verification_plots/eddy_current")
save_dir.mkdir(exist_ok=True, parents=True)

config = {
    "epochs": 5000,
    "lr": 1e-3,
    "generate_ground_truth_for_validation": False,
    "save_dir": "results/physics_informed/eddy_current_problem_circ_coil_currents",
    "data_weight": 0.0,
    "resume_from": "results/physics_informed/eddy_current_problem_circ_coil_currents/pimgn_trained_model.pth",
}

trainer = PIMGNTrainerEM(eddy_current_problem_different_currents(current=400), config=config)
predictions, ground_truth, errors = trainer.evaluate_with_ground_truth()
trainer.all_fem_solvers[0].export_to_vtk_complex(
    ground_truth,
    predictions,
    filename="results/physics_informed/eddy_current_problem_circ_coil_currents/vtk/result_complex111",
    suffix="p111",
)