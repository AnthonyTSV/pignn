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

from new_pignn.thermal_problems import create_ih_problem_curr_freq
from new_pignn.trainer import PIMGNTrainer
from verification_plots.helpers.error_metrics import compute_l2_error

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

SAVE_DIR = Path("verification_plots/thermal_ih_current_freq")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

CHECKPOINT = Path("results/physics_informed/thermal_ih_current_freq/pimgn_trained_model.pth")
FREQ_SWEEP_DATA = SAVE_DIR / "thermal_generalisation_freq_sweep.npz"
HEATMAP_DATA = SAVE_DIR / "thermal_generalisation_heatmap.npz"

TRAIN_FREQS = np.arange(2000, 6000, 1000).astype(float)
TRAIN_CURRENTS = np.arange(2000, 6000, 1000).astype(float)


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


def _frequency_sweep_grid() -> np.ndarray:
    return np.concatenate(
        [
            np.arange(1500, 2000, 250),
            np.arange(2000, 6000, 250),
            np.arange(6000, 8500, 500),
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


def _evaluate_case(current: float, freq: float) -> float:
    _require_checkpoint()

    print(f"\n{'=' * 50}")
    print(f"Evaluating current = {current}, freq = {freq}")
    print(f"{'=' * 50}")

    problem = create_ih_problem_curr_freq(current=current, frequency=freq)
    config = {
        "epochs": 1,
        "lr": 1e-3,
        "generate_ground_truth_for_validation": False,
        "save_dir": (
            "results/physics_informed/thermal_ih_current_freq/"
            f"I_{int(current)}_f_{int(freq)}"
        ),
        "enforce_axis_regularity": True,
        "data_weight": 0.0,
        "time_window": 10,
        "batch_size": 1,
        "resume_from": CHECKPOINT,
    }
    trainer = PIMGNTrainer(problem, config=config)
    predictions, ground_truth, errors = trainer.evaluate_with_ground_truth()
    workpiece_mask = problem.wp_node_mask

    pred_arr = np.asarray(predictions[0])
    true_arr = np.asarray(ground_truth[0])

    temp_gnn_masked = pred_arr[1:, workpiece_mask]
    temp_fem_masked = true_arr[1:, workpiece_mask]

    relative_error = compute_l2_error(temp_gnn_masked, temp_fem_masked)
    print(f"current = {current}, freq = {freq} -> error = {relative_error:.6f}")
    return relative_error


def generate_frequency_sweep_data(fixed_current: float = 3000.0) -> Path:
    frequencies = _frequency_sweep_grid()
    errors = np.empty_like(frequencies, dtype=float)

    for idx, freq in enumerate(frequencies):
        errors[idx] = _evaluate_case(current=fixed_current, freq=freq)

    np.savez(
        FREQ_SWEEP_DATA,
        frequencies=frequencies,
        errors=errors,
        fixed_current=float(fixed_current),
        train_freqs=TRAIN_FREQS,
    )
    return FREQ_SWEEP_DATA


def generate_heatmap_data() -> Path:
    current_grid, freq_grid = _heatmap_grids()
    error_map = np.full((len(freq_grid), len(current_grid)), np.nan)

    for i, freq in enumerate(freq_grid):
        for j, current in enumerate(current_grid):
            error_map[i, j] = _evaluate_case(current=current, freq=freq)

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
    _require_saved_file(FREQ_SWEEP_DATA)
    data = np.load(FREQ_SWEEP_DATA)
    return (
        data["frequencies"],
        data["errors"],
        float(data["fixed_current"]),
        data["train_freqs"] if "train_freqs" in data else TRAIN_FREQS,
    )


def _load_heatmap_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _require_saved_file(HEATMAP_DATA)
    data = np.load(HEATMAP_DATA)
    return (
        data["current_grid"],
        data["freq_grid"],
        data["error_map"],
        data["train_currents"] if "train_currents" in data else TRAIN_CURRENTS,
        data["train_freqs"] if "train_freqs" in data else TRAIN_FREQS,
    )


def _print_frequency_summary_table(frequencies: np.ndarray, errors: np.ndarray) -> None:
    training_mask, interpolation_mask, extrapolation_mask = _classify_values(
        frequencies, TRAIN_FREQS
    )

    categories = [
        ("Training", frequencies[training_mask], errors[training_mask]),
        ("Interpolation", frequencies[interpolation_mask], errors[interpolation_mask]),
        ("Extrapolation", frequencies[extrapolation_mask], errors[extrapolation_mask]),
    ]

    print(f"\n{'=' * 60}")
    print(f"{'Category':<16} {'Frequency [Hz]':>16} {'Error':>12}")
    print(f"{'=' * 60}")
    for category, category_freqs, category_errors in categories:
        for freq, error in zip(category_freqs, category_errors):
            print(f"{category:<16} {int(freq):>16} {error:>12.6f}")
    print(f"{'=' * 60}")


def plot_frequency_sweep() -> None:
    frequencies, errors, fixed_current, train_freqs = _load_frequency_sweep_data()
    training_mask, interpolation_mask, extrapolation_mask = _classify_values(
        frequencies, train_freqs
    )

    _print_frequency_summary_table(frequencies, errors)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        frequencies[training_mask],
        errors[training_mask],
        "o",
        color="C0",
        markersize=8,
        label="Training",
        zorder=5,
    )
    ax.plot(
        frequencies[interpolation_mask],
        errors[interpolation_mask],
        "s",
        color="C1",
        markersize=7,
        label="Interpolation",
        zorder=5,
    )
    ax.plot(
        frequencies[extrapolation_mask],
        errors[extrapolation_mask],
        "D",
        color="C2",
        markersize=7,
        label="Extrapolation",
        zorder=5,
    )
    ax.plot(frequencies, errors, "-", color="gray", alpha=0.4, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency $f$ [Hz]")
    ax.set_ylabel("Relative $L_2$ error")
    ax.set_title(f"Thermal frequency generalisation ($I = {int(fixed_current)}$ A)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "thermal_generalization_l2_vs_freq.pdf", dpi=300)
    plt.close(fig)


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

    ax.set_xlabel("Coil current $I$ [A]")
    ax.set_ylabel("Frequency $f$ [Hz]")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "thermal_generalization_heatmap_curr_freq.pdf", dpi=300)
    plt.close(fig)


def generate_data(fixed_current: float = 3000.0) -> None:
    print("=" * 60)
    print("THERMAL FREQUENCY SWEEP")
    print("=" * 60)
    generate_frequency_sweep_data(fixed_current=fixed_current)

    print("\n" + "=" * 60)
    print("THERMAL CURRENT-FREQUENCY HEATMAP")
    print("=" * 60)
    generate_heatmap_data()


def plot_saved_data() -> None:
    print("=" * 60)
    print("PLOTTING SAVED DATA")
    print("=" * 60)
    plot_frequency_sweep()
    plot_heatmap()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate inference data and/or plot the saved thermal "
            "generalisation results."
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
        help="Current used for the 1-D frequency sweep during data generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode in {"generate", "all"}:
        generate_data(fixed_current=args.fixed_current)

    if args.mode in {"plot", "all"}:
        plot_saved_data()

    print(f"\nOutputs saved to {SAVE_DIR}")
