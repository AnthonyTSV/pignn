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
from helpers.error_metrics import compute_relative_error, compute_l2_error
from helpers.vtk_extractor import VTKToPlotConverter
from helpers.line_data_extractor import ExtractDataOverLine
from helpers.plot_helpers import (
    plot_l2,
    epoch_vs_train_loss,
    epoch_vs_training_l2,
    _get_run_label,
)
from new_pignn.thermal_problems import create_ih_problem
from new_pignn.plotter import load_log
from new_pignn.trainer import PIMGNTrainer

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

save_dir = Path("verification_plots/ih_freq")
save_dir.mkdir(exist_ok=True, parents=True)

checkpoint = "results/physics_informed/pimgn_trained_model.pth"

# Training freqs
train_freqs = np.arange(2000, 7000, 1000).tolist()
# Interpolation freqs (between training values)
interp_freqs = [2500, 3500, 4500, 5500]
# Extrapolation freqs (outside training range)
extrap_freqs = [1500, 6500]

all_freqs = train_freqs + interp_freqs + extrap_freqs
all_freqs_sorted = sorted(all_freqs)

# Evaluate each freq
results = {}
for freq in all_freqs_sorted:
    print(f"\n{'='*50}")
    print(f"Evaluating freq = {freq} A")
    print(f"{'='*50}")
    problem = create_ih_problem(frequency=freq)
    config = {
        "epochs": 1,
        "lr": 1e-3,
        "generate_ground_truth_for_validation": False,
        "save_dir": f"results/physics_informed/thermal_ih_generalization/I_{freq}",
        "enforce_axis_regularity": True,
        "data_weight": 0.0,
        "batch_size": 1,
        "resume_from": checkpoint,
    }
    trainer = PIMGNTrainer(problem, config=config)
    predictions, ground_truth, errors = trainer.evaluate_with_ground_truth()
    # results[freq] = compute_l2_error(predictions, ground_truth)
    results[freq] = np.mean(errors[-1])
    print(f"freq = {freq} A  ->  L2 error = {results[freq]:.6f}")

# Separate results by category
train_errors = [results[c] for c in train_freqs]
interp_errors = [results[c] for c in interp_freqs]
extrap_errors = [results[c] for c in extrap_freqs]

# Print summary table
print(f"\n{'='*60}")
print(f"{'Category':<16} {'freq [A]':>12} {'L2 Error':>12}")
print(f"{'='*60}")
for c in sorted(train_freqs):
    print(f"{'Training':<16} {c:>12} {results[c]:>12.6f}")
for c in sorted(interp_freqs):
    print(f"{'Interpolation':<16} {c:>12} {results[c]:>12.6f}")
for c in sorted(extrap_freqs):
    print(f"{'Extrapolation':<16} {c:>12} {results[c]:>12.6f}")
print(f"{'='*60}")

# ── Plot: L2 error vs freq ──
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(
    train_freqs, train_errors,
    "o", color="C0", markersize=8, label="Training", zorder=5,
)
ax.plot(
    interp_freqs, interp_errors,
    "s", color="C1", markersize=7, label="Interpolation", zorder=5,
)
ax.plot(
    extrap_freqs, extrap_errors,
    "D", color="C2", markersize=7, label="Extrapolation", zorder=5,
)

# Connecting line through all sorted points
sorted_freqs = sorted(results.keys())
sorted_errors = [results[c] for c in sorted_freqs]
ax.plot(sorted_freqs, sorted_errors, "-", color="gray", alpha=0.4, zorder=1)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Coil freq $I$ [A]")
ax.set_ylabel("Relative $L_2$ error")
ax.set_title("PI-MGN generalisation: eddy-freq problem")
ax.legend()
fig.tight_layout()
fig.savefig(save_dir / "generalization_l2_vs_freq.pdf", dpi=300)
plt.show()

print(f"\nPlots saved to {save_dir}")