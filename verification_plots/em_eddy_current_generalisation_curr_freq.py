"""
Generalisation study for the PI-MGN eddy-current model trained on a
(current x frequency) grid.

This script supports three workflows:
1. Generate and save inference data.
2. Load saved data and create plots.
3. Run both steps in sequence.
"""

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

from helpers.error_metrics import compute_l2_error
from new_pignn.em_eddy_problems import eddy_current_problem_different_currents
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

from helpers.mpl_style import apply_mpl_style

apply_mpl_style()

SAVE_DIR = Path("verification_plots/eddy_current_generalisation")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

CHECKPOINT = Path(
    "results/physics_informed/eddy_current_problem_freqs_currents/pimgn_trained_model.pth"
)

FREQ_SWEEP_DATA = SAVE_DIR / "freq_sweep_data.npz"
CURRENT_SWEEP_DATA = SAVE_DIR / "current_sweep_data.npz"
HEATMAP_DATA = SAVE_DIR / "heatmap_data.npz"

LEGACY_FREQ_SWEEP_DATA = SAVE_DIR / "freq_sweep_errors.npy"
LEGACY_CURRENT_SWEEP_DATA = SAVE_DIR / "current_sweep_errors.npy"

# Training grid (must match trainer_em.py)
TRAIN_CURRENTS = np.arange(2000, 6000, 1000).astype(float)
TRAIN_FREQS = np.arange(2000, 6000, 1000).astype(float)


def _require_checkpoint() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT}. Run the training script first "
            "to generate the model checkpoint."
        )


def _require_saved_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved data file not found: {path}. Run this script with "
            "`--mode generate` first."
        )


def _frequency_grid() -> np.ndarray:
    return np.concatenate(
        [
            np.arange(1500, 2000, 250),  # extrapolation
            np.arange(2000, 6000, 250),  # training + interpolation
            np.arange(6000, 8500, 500),  # extrapolation
        ]
    ).astype(float)


def _current_grid() -> np.ndarray:
    return np.concatenate(
        [
            np.arange(1500, 2000, 250),  # extrapolation
            np.arange(2000, 5000, 250),  # training + interpolation
            np.arange(5000, 8500, 500),  # extrapolation
        ]
    ).astype(float)


def _heatmap_grids() -> tuple[np.ndarray, np.ndarray]:
    current_grid = np.arange(1500, 8500, 250).astype(float)
    freq_grid = np.arange(1500, 8500, 250).astype(float)
    return current_grid, freq_grid


def _classify_values(
    values: np.ndarray, training_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    training_values = np.asarray(training_values, dtype=float)

    training_mask = np.array(
        [np.any(np.isclose(value, training_values)) for value in values], dtype=bool
    )
    in_training_range = (values >= training_values.min()) & (
        values <= training_values.max()
    )
    interpolation_mask = in_training_range & ~training_mask
    extrapolation_mask = ~in_training_range
    return training_mask, interpolation_mask, extrapolation_mask


def _evaluate(current: float, frequency: float) -> float:
    """Return relative L2 error for a single (current, frequency) pair."""
    _require_checkpoint()

    problem = eddy_current_problem_different_currents(
        current=current, frequency=frequency
    )
    config = {
        "epochs": 1,
        "lr": 1e-3,
        "generate_ground_truth_for_validation": False,
        "save_dir": (
            "results/physics_informed/eddy_current_generalization/"
            f"I_{int(current)}_f_{int(frequency)}"
        ),
        "enforce_axis_regularity": True,
        "data_weight": 0.0,
        "batch_size": 1,
        "resume_from": CHECKPOINT,
    }
    trainer = PIMGNTrainerEM([problem], config=config)
    pred, true, errors = trainer.evaluate_with_ground_truth()
    del errors
    return compute_l2_error(pred, true)


def generate_frequency_sweep_data(fixed_current: float = 3000.0) -> Path:
    """Run inference for the frequency sweep and save it to disk."""
    frequencies = _frequency_grid()
    errors = np.empty_like(frequencies, dtype=float)

    for idx, frequency in enumerate(frequencies):
        print(f"  freq sweep: I={fixed_current}, f={frequency}")
        errors[idx] = _evaluate(fixed_current, frequency)

    np.savez(
        FREQ_SWEEP_DATA,
        frequencies=frequencies,
        errors=errors,
        fixed_current=float(fixed_current),
        train_freqs=TRAIN_FREQS,
    )
    return FREQ_SWEEP_DATA


def generate_current_sweep_data(fixed_freq: float = 3000.0) -> Path:
    """Run inference for the current sweep and save it to disk."""
    currents = _current_grid()
    errors = np.empty_like(currents, dtype=float)

    for idx, current in enumerate(currents):
        print(f"  current sweep: I={current}, f={fixed_freq}")
        errors[idx] = _evaluate(current, fixed_freq)

    np.savez(
        CURRENT_SWEEP_DATA,
        currents=currents,
        errors=errors,
        fixed_freq=float(fixed_freq),
        train_currents=TRAIN_CURRENTS,
    )
    return CURRENT_SWEEP_DATA


def generate_heatmap_data() -> Path:
    """Run inference on a 2-D grid and save the error map to disk."""
    current_grid, freq_grid = _heatmap_grids()
    error_map = np.full((len(freq_grid), len(current_grid)), np.nan)

    for i, frequency in enumerate(freq_grid):
        for j, current in enumerate(current_grid):
            print(f"  heatmap: I={current}, f={frequency}")
            error_map[i, j] = _evaluate(current, frequency)

    np.savez(
        HEATMAP_DATA,
        current_grid=current_grid,
        freq_grid=freq_grid,
        error_map=error_map,
        train_currents=TRAIN_CURRENTS,
        train_freqs=TRAIN_FREQS,
    )
    return HEATMAP_DATA


def _load_frequency_sweep_data() -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    if FREQ_SWEEP_DATA.is_file():
        data = np.load(FREQ_SWEEP_DATA)
        return (
            data["frequencies"],
            data["errors"],
            float(data["fixed_current"]),
            data["train_freqs"],
        )

    if LEGACY_FREQ_SWEEP_DATA.is_file():
        legacy_data = np.load(LEGACY_FREQ_SWEEP_DATA, allow_pickle=True).item()
        frequencies = np.array(sorted(legacy_data), dtype=float)
        errors = np.array([legacy_data[freq] for freq in frequencies], dtype=float)
        return frequencies, errors, 3000.0, TRAIN_FREQS

    _require_saved_file(FREQ_SWEEP_DATA)
    raise AssertionError("Unreachable")


def _load_current_sweep_data() -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    if CURRENT_SWEEP_DATA.is_file():
        data = np.load(CURRENT_SWEEP_DATA)
        return (
            data["currents"],
            data["errors"],
            float(data["fixed_freq"]),
            data["train_currents"],
        )

    if LEGACY_CURRENT_SWEEP_DATA.is_file():
        legacy_data = np.load(LEGACY_CURRENT_SWEEP_DATA, allow_pickle=True).item()
        currents = np.array(sorted(legacy_data), dtype=float)
        errors = np.array([legacy_data[current] for current in currents], dtype=float)
        return currents, errors, 3000.0, TRAIN_CURRENTS

    _require_saved_file(CURRENT_SWEEP_DATA)
    raise AssertionError("Unreachable")


def _load_heatmap_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _require_saved_file(HEATMAP_DATA)
    data = np.load(HEATMAP_DATA)
    train_currents = data["train_currents"] if "train_currents" in data else TRAIN_CURRENTS
    train_freqs = data["train_freqs"] if "train_freqs" in data else TRAIN_FREQS
    return (
        data["current_grid"],
        data["freq_grid"],
        data["error_map"],
        train_currents,
        train_freqs,
    )


def _plot_categorised_sweep(
    values: np.ndarray,
    errors: np.ndarray,
    training_values: np.ndarray,
    xlabel: str,
    title: str,
    output_stem: str,
) -> None:
    training_mask, interpolation_mask, extrapolation_mask = _classify_values(
        values, training_values
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        values[training_mask],
        errors[training_mask],
        "o",
        color="C0",
        ms=8,
        label="Training",
        zorder=5,
    )
    ax.plot(
        values[interpolation_mask],
        errors[interpolation_mask],
        "s",
        color="C1",
        ms=7,
        label="Interpolation",
        zorder=5,
    )
    ax.plot(
        values[extrapolation_mask],
        errors[extrapolation_mask],
        "D",
        color="C2",
        ms=7,
        label="Extrapolation",
        zorder=5,
    )
    ax.plot(values, errors, "-", color="gray", alpha=0.4, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Relative $L_2$ error")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SAVE_DIR / f"{output_stem}.pdf", dpi=300)
    fig.savefig(SAVE_DIR / f"{output_stem}.png", dpi=300)
    plt.close(fig)


def plot_frequency_sweep() -> None:
    frequencies, errors, fixed_current, train_freqs = _load_frequency_sweep_data()
    _plot_categorised_sweep(
        values=frequencies,
        errors=errors,
        training_values=train_freqs,
        xlabel="Frequency $f$ [Hz]",
        title=f"Frequency generalisation ($I = {int(fixed_current)}$ A)",
        output_stem="generalisation_freq_sweep",
    )


def plot_current_sweep() -> None:
    currents, errors, fixed_freq, train_currents = _load_current_sweep_data()
    _plot_categorised_sweep(
        values=currents,
        errors=errors,
        training_values=train_currents,
        xlabel="Coil current $I$ [A]",
        title=f"Current generalisation ($f = {int(fixed_freq)}$ Hz)",
        output_stem="generalisation_current_sweep",
    )


def plot_heatmap() -> None:
    current_grid, freq_grid, error_map, train_currents, train_freqs = _load_heatmap_data()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.pcolormesh(
        current_grid,
        freq_grid,
        error_map,
        shading="nearest",
        cmap="RdYlGn_r",
    )
    fig.colorbar(im, ax=ax, label=r"Relative $L_2$ error")

    for train_current in train_currents:
        for train_freq in train_freqs:
            ax.plot(train_current, train_freq, "k*", ms=12, zorder=5)
    ax.plot([], [], "k*", ms=12, label="Training points")

    # rect = plt.Rectangle(
    #     (train_currents[0], train_freqs[0]),
    #     train_currents[-1] - train_currents[0],
    #     train_freqs[-1] - train_freqs[0],
    #     linewidth=2,
    #     edgecolor="black",
    #     facecolor="none",
    #     linestyle="--",
    #     label="Training range",
    # )
    # ax.add_patch(rect)

    ax.set_xlabel("Coil current $I$ [A]")
    ax.set_ylabel("Frequency $f$ [Hz]")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "generalisation_heatmap.pdf", dpi=300)
    fig.savefig(SAVE_DIR / "generalisation_heatmap.png", dpi=300)
    plt.close(fig)


def generate_data(
    fixed_current: float = 3000.0, fixed_freq: float = 3000.0
) -> None:
    print("=" * 60)
    print("1-D FREQUENCY SWEEP")
    print("=" * 60)
    generate_frequency_sweep_data(fixed_current=fixed_current)

    print("\n" + "=" * 60)
    print("1-D CURRENT SWEEP")
    print("=" * 60)
    generate_current_sweep_data(fixed_freq=fixed_freq)

    print("\n" + "=" * 60)
    print("2-D HEATMAP")
    print("=" * 60)
    generate_heatmap_data()


def plot_saved_data() -> None:
    print("=" * 60)
    print("PLOTTING SAVED DATA")
    print("=" * 60)
    plot_frequency_sweep()
    plot_current_sweep()
    plot_heatmap()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate inference data and/or plot the saved generalisation results."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("generate", "plot", "all"),
        default="plot",
        help="Choose whether to generate data, plot saved data, or do both.",
    )
    parser.add_argument(
        "--fixed-current",
        type=float,
        default=3000.0,
        help="Current used for the frequency sweep during data generation.",
    )
    parser.add_argument(
        "--fixed-freq",
        type=float,
        default=3000.0,
        help="Frequency used for the current sweep during data generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode in {"generate", "all"}:
        generate_data(
            fixed_current=args.fixed_current,
            fixed_freq=args.fixed_freq,
        )

    if args.mode in {"plot", "all"}:
        plot_saved_data()

    print(f"\nOutputs saved to {SAVE_DIR}")
