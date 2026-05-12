import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from sympy import true

from helpers.plot_helpers import (
    plot_l2,
    epoch_vs_train_loss,
    epoch_vs_training_l2,
    _get_run_label,
)

from helpers.error_metrics import compute_relative_error, compute_l2_error
from new_pignn.thermal_problems import create_ih_problem_mu_r_sigma
from new_pignn.trainer import PIMGNTrainer

plt.style.use(["science"])
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

from helpers.mpl_style import apply_mpl_style

apply_mpl_style()

SAVE_DIR = Path("verification_plots/thermal_ih_generalisation_mu_r_sigma_new")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

CHECKPOINT = Path(
    "results/physics_informed/thermal_ih_mu_r_sigma/pimgn_trained_model.pth"
)
TRAIN_MU_R_VALUES = np.array([1, 10, 50, 100], dtype=np.float64)
TRAIN_SIGMA_VALUES = np.array([1.3, 3, 6, 15], dtype=np.float64) * 1e6  # S/m


def _require_checkpoint() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT}. Run the training script first "
            "to generate the model checkpoint."
        )

def _evaluate(mu_r_workpiece: float, sigma_workpiece: float, index: int, export: bool = False) -> float:
    _require_checkpoint()

    problem = create_ih_problem_mu_r_sigma(
        mu_r=mu_r_workpiece,
        sigma=sigma_workpiece,
    )
    config = {
        "epochs": 1,
        "lr": 1e-3,
        "time_window": 10,
        "generate_ground_truth_for_validation": False,
        "save_dir": (
            "results/physics_informed/thermal_ih_mu_r_sigma_generalization_evaluation_new"
        ),
        "enforce_axis_regularity": True,
        "data_weight": 0.0,
        "batch_size": 1,
        "resume_from": CHECKPOINT,
    }
    trainer = PIMGNTrainer([problem], config=config)
    pred, true, errors = trainer.evaluate_with_ground_truth()

    if export:
        trainer.all_fem_solvers[0].export_to_vtk(
            array_true=true[0],
            array_pred=pred[0],
            time_steps=problem.time_config.time_steps_export,
            filename=SAVE_DIR / f"mu_r_sigma_{index}_comparison",
        )
    
    workpiece_mask = problem.wp_node_mask

    pred_arr = np.asarray(pred[0])
    true_arr = np.asarray(true[0])

    temp_gnn_masked = pred_arr[1:, workpiece_mask]
    temp_fem_masked = true_arr[1:, workpiece_mask]

    relative_error = compute_l2_error(temp_gnn_masked, temp_fem_masked)
    print(
        f"Mu_r: {mu_r_workpiece}, sigma: {sigma_workpiece:.6e}, "
        f"Relative Error: {relative_error}"
    )
    return relative_error

def get_data():
    mu_r_values = np.unique(
        np.concatenate(
            (np.linspace(1, 100, 20), TRAIN_MU_R_VALUES, np.linspace(101, 150, 10))
        )
    )
    sigma_values = (
        np.unique(
            np.concatenate(
                ([1.0], np.linspace(1.3, 15, 20), TRAIN_SIGMA_VALUES / 1e6, np.linspace(16.5, 30, 15))
            )
        )
        * 1e6
    )
    relative_errors = np.zeros((len(sigma_values), len(mu_r_values)))
    for i, sigma_value in enumerate(sigma_values):
        for j, mu_r_value in enumerate(mu_r_values):
            idx = i * len(mu_r_values) + j
            relative_errors[i, j] = _evaluate(mu_r_value, sigma_value, idx)
    
    np.savez(
        SAVE_DIR / "relative_errors.npz",
        mu_r_values=mu_r_values,
        sigma_values=sigma_values,
        relative_errors=relative_errors,
    )

def export_one_problem():
    mu_r_value = 5
    sigma_value = 2e6
    _evaluate(mu_r_value, sigma_value, index=0, export=True)

def _mask_values(values: np.ndarray, selected_values: np.ndarray) -> np.ndarray:
    return np.isclose(values[:, None], selected_values[None, :], rtol=1e-12).any(axis=1)

def read_and_plot():
    data = np.load(SAVE_DIR / "relative_errors.npz")
    mu_r_values = data["mu_r_values"]
    sigma_values = data["sigma_values"]
    relative_errors = data["relative_errors"]

    fig, ax = plt.subplots(figsize=(6, 4))
    mu_r_mean = relative_errors.mean(axis=0)
    mu_r_min = relative_errors.min(axis=0)
    mu_r_max = relative_errors.max(axis=0)
    ax.fill_between(mu_r_values, mu_r_min, mu_r_max, color="C0", alpha=0.2, label="Range")
    ax.plot(mu_r_values, mu_r_mean, marker="o", markersize=3, label="Mean")
    train_mask = _mask_values(mu_r_values, TRAIN_MU_R_VALUES)
    ax.plot(mu_r_values[train_mask], mu_r_mean[train_mask], "k*", markersize=8, label="Training")
    ax.set_xlabel(r"Relative Permeability $(\mu_r)$")
    ax.set_ylabel(r"Relative $L_2$ Error")
    ax.legend(ncols=1, fontsize="small")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "thermal_relative_error_vs_mu_r.pdf", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sigma_mean = relative_errors.mean(axis=1)
    sigma_min = relative_errors.min(axis=1)
    sigma_max = relative_errors.max(axis=1)
    ax.fill_between(
        sigma_values / 1e6,
        sigma_min,
        sigma_max,
        color="C0",
        alpha=0.2,
        label="Range",
    )
    ax.plot(sigma_values / 1e6, sigma_mean, marker="o", markersize=3, label="Mean")
    train_mask = _mask_values(sigma_values, TRAIN_SIGMA_VALUES)
    ax.plot(sigma_values[train_mask] / 1e6, sigma_mean[train_mask], "k*", markersize=8, label="Training")
    ax.set_xlabel(r"Electrical Conductivity $\sigma$ [MS/m]")
    ax.set_ylabel(r"Relative $L_2$ Error")
    ax.legend(ncols=1, fontsize="small")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "thermal_relative_error_vs_sigma.pdf", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.pcolormesh(
        mu_r_values,
        sigma_values / 1e6,
        relative_errors,
        cmap="RdYlGn_r",
        shading="nearest",
    )
    fig.colorbar(im, ax=ax, label=r"Relative $L_2$ Error")
    for mu_r_value in TRAIN_MU_R_VALUES:
        for sigma_value in TRAIN_SIGMA_VALUES:
            ax.plot(mu_r_value, sigma_value / 1e6, "k*", markersize=8)
    ax.set_xlabel(r"Relative Permeability $(\mu_r)$")
    ax.set_ylabel(r"Electrical Conductivity $\sigma$ [MS/m]")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "thermal_relative_error_heatmap_mu_r_sigma.pdf", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    # get_data()
    read_and_plot()
    # export_one_problem()
