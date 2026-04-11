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
from helpers.error_metrics import compute_relative_error
from helpers.vtk_extractor import VTKToPlotConverter
from helpers.line_data_extractor import ExtractDataOverLine
from helpers.plot_helpers import (
    plot_l2,
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

save_dir = Path("verification_plots/eddy_current")
save_dir.mkdir(exist_ok=True, parents=True)

checkpoint = "results/physics_informed/eddy_current_problem_circ_coil_currents/pimgn_trained_model.pth"

# Training currents
train_currents = np.arange(2000, 7000, 500).tolist()
# Interpolation currents (between training values)
interp_currents = [2250, 2750, 3250, 3750, 4250, 4750, 5250, 5750, 6250, 6750]
# Extrapolation currents (outside training range)
extrap_currents = [1000, 1500, 1750, 7000, 7500, 10000]

all_currents = train_currents + interp_currents + extrap_currents
all_currents_sorted = sorted(all_currents)

# Evaluate each current
results = {}
for current in all_currents_sorted:
    print(f"\n{'='*50}")
    print(f"Evaluating current = {current} A")
    print(f"{'='*50}")
    problem = eddy_current_problem_different_currents(frequency=current)
    config = {
        "epochs": 1,
        "lr": 1e-3,
        "generate_ground_truth_for_validation": False,
        "save_dir": f"results/physics_informed/eddy_current_generalization/I_{current}",
        "enforce_axis_regularity": True,
        "data_weight": 0.0,
        "batch_size": 1,
        "resume_from": checkpoint,
    }
    trainer = PIMGNTrainerEM(problem, config=config)
    predictions, ground_truth, errors = trainer.evaluate_with_ground_truth()
    results[current] = errors[0]
    print(f"Current = {current} A  ->  L2 error = {errors[0]:.6f}")

# Separate results by category
train_errors = [results[c] for c in train_currents]
interp_errors = [results[c] for c in interp_currents]
extrap_errors = [results[c] for c in extrap_currents]

# Print summary table
print(f"\n{'='*60}")
print(f"{'Category':<16} {'Current [A]':>12} {'L2 Error':>12}")
print(f"{'='*60}")
for c in sorted(train_currents):
    print(f"{'Training':<16} {c:>12} {results[c]:>12.6f}")
for c in sorted(interp_currents):
    print(f"{'Interpolation':<16} {c:>12} {results[c]:>12.6f}")
for c in sorted(extrap_currents):
    print(f"{'Extrapolation':<16} {c:>12} {results[c]:>12.6f}")
print(f"{'='*60}")

# ── Plot: L2 error vs current ──
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(
    train_currents, train_errors,
    "o", color="C0", markersize=8, label="Training", zorder=5,
)
ax.plot(
    interp_currents, interp_errors,
    "s", color="C1", markersize=7, label="Interpolation", zorder=5,
)
ax.plot(
    extrap_currents, extrap_errors,
    "D", color="C2", markersize=7, label="Extrapolation", zorder=5,
)

# Connecting line through all sorted points
sorted_currents = sorted(results.keys())
sorted_errors = [results[c] for c in sorted_currents]
ax.plot(sorted_currents, sorted_errors, "-", color="gray", alpha=0.4, zorder=1)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Coil current $I$ [A]")
ax.set_ylabel("Relative $L_2$ error")
ax.set_title("PI-MGN generalisation: eddy-current problem")
ax.legend()
fig.tight_layout()
fig.savefig(save_dir / "generalization_l2_vs_current.pdf", dpi=300)
fig.savefig(save_dir / "generalization_l2_vs_current.png", dpi=300)
plt.show()

print(f"\nPlots saved to {save_dir}")