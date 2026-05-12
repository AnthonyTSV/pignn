import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from helpers.plot_helpers import (
    plot_l2,
    epoch_vs_train_loss,
    epoch_vs_training_l2,
    _get_run_label,
)

from helpers.error_metrics import compute_relative_error, compute_l2_error
from new_pignn.em_eddy_problems import eddy_current_problem_different_mu_r
from new_pignn.trainer_em import PIMGNTrainerEM

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

SAVE_DIR = Path("verification_plots/eddy_current_generalisation_mu_r_sigma")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

CHECKPOINT = Path(
    "results/physics_informed/em_different_mu_r_sigma/pimgn_trained_model.pth"
)
TRAIN_MU_R_VALUES = np.array([1, 10, 50, 100], dtype=np.float64)
TRAIN_SIGMA_VALUES = np.array([1.3, 3, 6, 15], dtype=np.float64) * 1e6  # S/m


def _training_mu_r_plot_values() -> np.ndarray:
    return np.unique(np.concatenate((np.linspace(1, 100, 20), TRAIN_MU_R_VALUES)))


def _training_sigma_plot_values() -> np.ndarray:
    return (
        np.unique(np.concatenate((np.linspace(1.3, 15, 20), TRAIN_SIGMA_VALUES / 1e6)))
        * 1e6
    )


def _mask_values(values: np.ndarray, selected_values: np.ndarray) -> np.ndarray:
    return np.isclose(values[:, None], selected_values[None, :], rtol=1e-12).any(axis=1)


def _rounded_vmax(values: np.ndarray) -> float:
    if np.ma.isMaskedArray(values):
        finite_values = values.compressed()
    else:
        finite_values = values[np.isfinite(values)]
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.01
    return max(np.ceil(float(finite_values.max()) * 100) / 100, 0.01)


def _require_checkpoint() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT}. Run the training script first "
            "to generate the model checkpoint."
        )

def _evaluate(mu_r_workpiece: float, sigma_workpiece: float, index: int, export_vtk: bool = False) -> float:
    _require_checkpoint()

    problem = eddy_current_problem_different_mu_r(
        mu_r_workpiece=mu_r_workpiece,
        sigma_workpiece=sigma_workpiece,
    )
    config = {
        "epochs": 1,
        "lr": 1e-3,
        "generate_ground_truth_for_validation": False,
        "save_dir": (
            "results/physics_informed/eddy_current_problem_mu_r_generalization_evaluation"
        ),
        "enforce_axis_regularity": True,
        "data_weight": 0.0,
        "batch_size": 1,
        "resume_from": CHECKPOINT,
    }
    trainer = PIMGNTrainerEM([problem], config=config)
    pred, true, errors = trainer.evaluate_with_ground_truth()

    if export_vtk:
        trainer.all_fem_solvers[0].export_to_vtk_complex(
            array_true=true,
            array_pred=pred,
            filename=SAVE_DIR / f"mu_r_sigma_{index}_comparison",
        )
    relative_error = compute_l2_error(pred, true)
    print(
        f"Mu_r: {mu_r_workpiece}, sigma: {sigma_workpiece:.6e}, "
        f"Relative Error: {relative_error}"
    )
    return relative_error

def get_data():
    mu_r_values = np.unique(
        np.concatenate(
            (
                [0.99],
                _training_mu_r_plot_values(),
                np.linspace(110, 400, 10),
                [500, 600],
            )
        )
    )
    sigma_values = (
        np.unique(
            np.concatenate(
                (
                    [0.8, 0.9, 1.0, 1.1, 1.2],
                    _training_sigma_plot_values() / 1e6,
                    np.linspace(16, 50, 10),
                    [60, 70],
                )
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

def plot_relative_error_vs_mu_r():
    data = np.load(SAVE_DIR / "relative_errors.npz")
    mu_r_values = data["mu_r_values"]
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
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "relative_error_vs_mu_r.pdf", dpi=300)
    plt.close(fig)

def plot_relative_error_vs_sigma():
    data = np.load(SAVE_DIR / "relative_errors.npz")
    sigma_values = data["sigma_values"]
    relative_errors = data["relative_errors"]

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
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "relative_error_vs_sigma.pdf", dpi=300)
    plt.close(fig)

def _plot_training_points(ax, markersize: float) -> None:
    for mu_r_value in TRAIN_MU_R_VALUES:
        for sigma_value in TRAIN_SIGMA_VALUES:
            ax.plot(mu_r_value, sigma_value / 1e6, "k*", markersize=markersize)


def _plot_training_box(ax) -> None:
    mu_r_min = TRAIN_MU_R_VALUES.min()
    mu_r_max = TRAIN_MU_R_VALUES.max()
    sigma_min = TRAIN_SIGMA_VALUES.min() / 1e6
    sigma_max = TRAIN_SIGMA_VALUES.max() / 1e6
    ax.plot(
        [mu_r_min, mu_r_max, mu_r_max, mu_r_min, mu_r_min],
        [sigma_min, sigma_min, sigma_max, sigma_max, sigma_min],
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )


def plot_relative_error_heatmap_training():
    data = np.load(SAVE_DIR / "relative_errors.npz")
    mu_r_values = data["mu_r_values"]
    sigma_values = data["sigma_values"]
    relative_errors = data["relative_errors"]

    mu_r_train_mask = _mask_values(mu_r_values, _training_mu_r_plot_values())
    sigma_train_mask = _mask_values(sigma_values, _training_sigma_plot_values())
    mu_r_train = mu_r_values[mu_r_train_mask]
    sigma_train = sigma_values[sigma_train_mask]
    errors_train = relative_errors[np.ix_(sigma_train_mask, mu_r_train_mask)]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.pcolormesh(
        mu_r_train,
        sigma_train / 1e6,
        errors_train,
        cmap="RdYlGn_r",
        shading="nearest",
        vmin=0,
        vmax=_rounded_vmax(errors_train),
    )
    fig.colorbar(im, ax=ax, label=r"Relative $L_2$ Error")
    _plot_training_points(ax, markersize=8)
    ax.set_xlabel(r"Relative Permeability $(\mu_r)$")
    ax.set_ylabel(r"Electrical Conductivity $\sigma$ [MS/m]")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "relative_error_heatmap_mu_r_sigma.pdf", dpi=300)
    plt.close(fig)


def plot_relative_error_heatmap_extrapolation():
    data = np.load(SAVE_DIR / "relative_errors.npz")
    mu_r_values = data["mu_r_values"]
    sigma_values = data["sigma_values"]
    relative_errors = data["relative_errors"]

    # mu_r_train_mask = _mask_values(mu_r_values, _training_mu_r_plot_values())
    # sigma_train_mask = _mask_values(sigma_values, _training_sigma_plot_values())
    # training_grid_mask = sigma_train_mask[:, None] & mu_r_train_mask[None, :]
    # errors_extrapolation = np.ma.masked_where(training_grid_mask, relative_errors)

    cmap = plt.get_cmap("RdYlGn_r").copy()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.pcolormesh(
        mu_r_values,
        sigma_values / 1e6,
        relative_errors,
        cmap=cmap,
        shading="gouraud",
        vmin=0,
        vmax=_rounded_vmax(relative_errors),
    )
    fig.colorbar(im, ax=ax, label=r"Relative $L_2$ Error")
    _plot_training_box(ax)
    ax.set_xlabel(r"Relative Permeability $(\mu_r)$")
    ax.set_ylabel(r"Electrical Conductivity $\sigma$ [MS/m]")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "relative_error_heatmap_mu_r_sigma_extrapolation.pdf", dpi=300)
    plt.close(fig)

def read_and_plot():
    plot_relative_error_vs_mu_r()
    plot_relative_error_vs_sigma()
    plot_relative_error_heatmap_training()
    plot_relative_error_heatmap_extrapolation()

if __name__ == "__main__":
    # get_data()
    read_and_plot()
    # _evaluate(mu_r_workpiece=600.0, sigma_workpiece=4761904, index=0, export_vtk=True)
