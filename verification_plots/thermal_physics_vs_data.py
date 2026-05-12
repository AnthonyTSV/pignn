import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for import_path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

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
from helpers.mpl_style import apply_mpl_style
from new_pignn.thermal_problems import create_ih_problem_mu_r_sigma
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
apply_mpl_style()

SAVE_DIR = PROJECT_ROOT / "verification_plots/thermal_physics_vs_data"
SAVE_DIR.mkdir(exist_ok=True, parents=True)

ORIGINAL_TRAIN_MU_R_VALUES = np.array([1, 10, 50, 100], dtype=np.float64)
ORIGINAL_TRAIN_SIGMA_VALUES = np.array([1.3, 3, 6, 15], dtype=np.float64) * 1e6

NUM_TRAIN_VALUES = (2, 4, 8, 16)
MODEL_KINDS = ("data", "physics")


@dataclass(frozen=True)
class Experiment:
    num_train: int
    model_kind: str

    @property
    def uses_physics(self) -> bool:
        return self.model_kind == "physics"

    @property
    def file_stem(self) -> str:
        return f"N{self.num_train}_{self.model_kind}"

    @property
    def label(self) -> str:
        model_label = "PI-GNN" if self.uses_physics else "Data-only"
        return f"{model_label}, $N_{{train}}={self.num_train}$"


def _training_mu_r_plot_values() -> np.ndarray:
    return np.unique(
        np.concatenate((np.linspace(1, 100, 20), ORIGINAL_TRAIN_MU_R_VALUES))
    )


def _training_sigma_plot_values() -> np.ndarray:
    return (
        np.unique(
            np.concatenate(
                (np.linspace(1.3, 15, 20), ORIGINAL_TRAIN_SIGMA_VALUES / 1e6)
            )
        )
        * 1e6
    )


def _test_grids() -> tuple[np.ndarray, np.ndarray]:
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
    return mu_r_values.astype(float), sigma_values.astype(float)


def _training_points(num_train: int) -> np.ndarray:
    if num_train == 2:
        values = [(1, 1.3e6), (100, 15e6)]
    elif num_train == 4:
        values = [
            (1, 1.3e6),
            (1, 15e6),
            (100, 1.3e6),
            (100, 15e6),
        ]
    elif num_train == 8:
        values = [
            (1, 1.3e6),
            (1, 3e6),
            (1, 6e6),
            (1, 15e6),
            (100, 1.3e6),
            (100, 15e6),
            (10, 1.3e6),
            (50, 1.3e6),
        ]
    elif num_train == 16:
        values = np.array(
            np.meshgrid(ORIGINAL_TRAIN_MU_R_VALUES, ORIGINAL_TRAIN_SIGMA_VALUES)
        ).T.reshape(-1, 2)
    else:
        raise ValueError("num_train must be one of 2, 4, 8, 16.")
    return np.asarray(values, dtype=float)


def _checkpoint_path(experiment: Experiment) -> Path:
    model_dir = (
        f"results/physics_informed/ih_ablation_{experiment.num_train}_pnts_"
        f"{'physics' if experiment.uses_physics else 'data'}"
    )
    return PROJECT_ROOT / model_dir / "pimgn_trained_model.pth"


def _data_path(experiment: Experiment) -> Path:
    return SAVE_DIR / f"relative_errors_{experiment.file_stem}.npz"


def _require_checkpoint(checkpoint: Path) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint}. Run the matching training "
            "experiment first."
        )


def _require_saved_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved data file not found: {path}. Run this script with "
            "`--mode generate` first."
        )


def _is_same_grid(
    data: np.lib.npyio.NpzFile, mu_r_values: np.ndarray, sigma_values: np.ndarray
) -> bool:
    return (
        "mu_r_values" in data
        and "sigma_values" in data
        and np.array_equal(data["mu_r_values"], mu_r_values)
        and np.array_equal(data["sigma_values"], sigma_values)
    )


def _save_error_data(
    path: Path,
    experiment: Experiment,
    mu_r_values: np.ndarray,
    sigma_values: np.ndarray,
    relative_errors: np.ndarray,
) -> None:
    training_points = _training_points(experiment.num_train)
    np.savez_compressed(
        path,
        mu_r_values=mu_r_values,
        sigma_values=sigma_values,
        relative_errors=relative_errors,
        num_train=np.array(experiment.num_train),
        model_kind=np.array(experiment.model_kind),
        checkpoint=np.array(str(_checkpoint_path(experiment))),
        train_mu_r_values=training_points[:, 0],
        train_sigma_values=training_points[:, 1],
    )


def _load_partial_or_empty(
    path: Path, mu_r_values: np.ndarray, sigma_values: np.ndarray
) -> np.ndarray:
    shape = (len(sigma_values), len(mu_r_values))
    if not path.is_file():
        return np.full(shape, np.nan)

    data = np.load(path)
    if not _is_same_grid(data, mu_r_values, sigma_values):
        return np.full(shape, np.nan)
    relative_errors = data["relative_errors"]
    if relative_errors.shape != shape:
        return np.full(shape, np.nan)
    return relative_errors.astype(float)


def _evaluate(
    experiment: Experiment,
    checkpoint: Path,
    mu_r_workpiece: float,
    sigma_workpiece: float,
    index: int,
) -> float:
    _require_checkpoint(checkpoint)

    problem = create_ih_problem_mu_r_sigma(
        mu_r=mu_r_workpiece,
        sigma=sigma_workpiece,
    )
    problem.problem_id = index

    config = {
        "epochs": 1,
        "lr": 1e-3,
        "time_window": 10,
        "noise_sigma": 1e-2,
        "batch_size": 1,
        "training_mode": "physics" if experiment.uses_physics else "data",
        "data_weight": 0.0 if experiment.uses_physics else 1.0,
        "physics_loss": experiment.uses_physics,
        "batch_size": 1,
        "resume_from": checkpoint,
        "save_dir": str(SAVE_DIR / "ih_evaluation_runs" / experiment.file_stem),
    }
    trainer = PIMGNTrainer([problem], config=config)
    pred, true, errors = trainer.evaluate_with_ground_truth()
    del errors
    workpiece_mask = problem.wp_node_mask

    pred_arr = np.asarray(pred[0])
    true_arr = np.asarray(true[0])

    temp_gnn_masked = pred_arr[1:, workpiece_mask]
    temp_fem_masked = true_arr[1:, workpiece_mask]

    relative_error = compute_l2_error(temp_gnn_masked, temp_fem_masked)
    print(
        f"{experiment.file_stem}: mu_r={mu_r_workpiece:.6g}, "
        f"sigma={sigma_workpiece:.6e}, L2={relative_error:.6e}"
    )
    return relative_error


def generate_experiment_data(experiment: Experiment, force: bool = False) -> Path:
    output_path = _data_path(experiment)

    mu_r_values, sigma_values = _test_grids()
    if output_path.is_file() and not force:
        data = np.load(output_path)
        if (
            _is_same_grid(data, mu_r_values, sigma_values)
            and np.isfinite(data["relative_errors"]).all()
        ):
            print(f"{experiment.file_stem}: using cached {output_path}")
            return output_path

    checkpoint = _checkpoint_path(experiment)
    _require_checkpoint(checkpoint)

    relative_errors = (
        np.full((len(sigma_values), len(mu_r_values)), np.nan)
        if force
        else _load_partial_or_empty(output_path, mu_r_values, sigma_values)
    )

    for i, sigma_value in enumerate(sigma_values):
        for j, mu_r_value in enumerate(mu_r_values):
            if np.isfinite(relative_errors[i, j]):
                continue
            idx = i * len(mu_r_values) + j
            relative_errors[i, j] = _evaluate(
                experiment=experiment,
                checkpoint=checkpoint,
                mu_r_workpiece=float(mu_r_value),
                sigma_workpiece=float(sigma_value),
                index=idx,
            )
            _save_error_data(
                output_path,
                experiment,
                mu_r_values,
                sigma_values,
                relative_errors,
            )

    _save_error_data(
        output_path,
        experiment,
        mu_r_values,
        sigma_values,
        relative_errors,
    )
    return output_path


def _data_source_path(experiment: Experiment) -> Path:
    output_path = _data_path(experiment)
    if output_path.is_file():
        return output_path
    _require_saved_file(output_path)
    raise AssertionError("Unreachable")


def _load_experiment_data(
    experiment: Experiment,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(_data_source_path(experiment))
    return data["mu_r_values"], data["sigma_values"], data["relative_errors"]


def _mask_values(values: np.ndarray, selected_values: np.ndarray) -> np.ndarray:
    return np.isclose(values[:, None], selected_values[None, :], rtol=1e-12).any(axis=1)


def _original_training_pair_mask(
    mu_r_values: np.ndarray, sigma_values: np.ndarray
) -> np.ndarray:
    mu_r_mask = _mask_values(mu_r_values, ORIGINAL_TRAIN_MU_R_VALUES)
    sigma_mask = _mask_values(sigma_values, ORIGINAL_TRAIN_SIGMA_VALUES)
    return sigma_mask[:, None] & mu_r_mask[None, :]


def _summary_masks(
    mu_r_values: np.ndarray, sigma_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mu_r_inside = (mu_r_values >= ORIGINAL_TRAIN_MU_R_VALUES.min()) & (
        mu_r_values <= ORIGINAL_TRAIN_MU_R_VALUES.max()
    )
    sigma_inside = (sigma_values >= ORIGINAL_TRAIN_SIGMA_VALUES.min()) & (
        sigma_values <= ORIGINAL_TRAIN_SIGMA_VALUES.max()
    )
    inside_box = sigma_inside[:, None] & mu_r_inside[None, :]
    interpolation_mask = inside_box & ~_original_training_pair_mask(
        mu_r_values, sigma_values
    )
    extrapolation_mask = ~inside_box
    return interpolation_mask, extrapolation_mask


def _mean_masked(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return np.nan
    return float(np.mean(selected))


def summarise_experiment(experiment: Experiment) -> dict[str, float]:
    mu_r_values, sigma_values, relative_errors = _load_experiment_data(experiment)
    interpolation_mask, extrapolation_mask = _summary_masks(mu_r_values, sigma_values)
    return {
        "interp": _mean_masked(relative_errors, interpolation_mask),
        "extrap": _mean_masked(relative_errors, extrapolation_mask),
    }


def _rounded_vmax(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.01
    return max(np.ceil(float(finite_values.max()) * 100) / 100, 0.01)


def _plot_training_points(ax: plt.Axes, num_train: int, markersize: float) -> None:
    points = _training_points(num_train)
    ax.plot(
        points[:, 0],
        points[:, 1] / 1e6,
        "k*",
        markersize=markersize,
        label="Training points",
        zorder=5,
    )


def _plot_training_box(ax: plt.Axes) -> None:
    mu_r_min = ORIGINAL_TRAIN_MU_R_VALUES.min()
    mu_r_max = ORIGINAL_TRAIN_MU_R_VALUES.max()
    sigma_min = ORIGINAL_TRAIN_SIGMA_VALUES.min() / 1e6
    sigma_max = ORIGINAL_TRAIN_SIGMA_VALUES.max() / 1e6
    ax.plot(
        [mu_r_min, mu_r_max, mu_r_max, mu_r_min, mu_r_min],
        [sigma_min, sigma_min, sigma_max, sigma_max, sigma_min],
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )


def _plot_heatmap_on_axis(
    ax: plt.Axes,
    experiment: Experiment,
    mu_r_values: np.ndarray,
    sigma_values: np.ndarray,
    relative_errors: np.ndarray,
    vmax: float,
):
    im = ax.pcolormesh(
        mu_r_values,
        sigma_values / 1e6,
        relative_errors,
        cmap="RdYlGn_r",
        shading="gouraud",
        vmin=0,
        vmax=vmax,
    )
    _plot_training_box(ax)
    _plot_training_points(ax, experiment.num_train, markersize=7)
    ax.set_title(experiment.label)
    ax.set_xlabel(r"Relative Permeability $(\mu_r)$")
    ax.set_ylabel(r"Electrical Conductivity $\sigma$ [MS/m]")
    return im


def _vmax_by_num_train(experiments: list[Experiment]) -> dict[int, float]:
    errors_by_num_train: dict[int, list[np.ndarray]] = {}
    for experiment in experiments:
        _, _, relative_errors = _load_experiment_data(experiment)
        errors_by_num_train.setdefault(experiment.num_train, []).append(
            relative_errors.ravel()
        )

    return {
        num_train: _rounded_vmax(np.concatenate(error_arrays))
        for num_train, error_arrays in errors_by_num_train.items()
    }


def plot_experiment_heatmap(experiment: Experiment, vmax: float | None = None) -> None:
    mu_r_values, sigma_values, relative_errors = _load_experiment_data(experiment)
    if vmax is None:
        vmax = _rounded_vmax(relative_errors)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = _plot_heatmap_on_axis(
        ax,
        experiment,
        mu_r_values,
        sigma_values,
        relative_errors,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label=r"Relative $L_2$ error")
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(SAVE_DIR / f"heatmap_{experiment.file_stem}.pdf", dpi=300)
    plt.close(fig)


def plot_comparison_heatmaps(
    experiments: list[Experiment], vmax_by_num_train: dict[int, float] | None = None
) -> None:
    datasets = [
        (experiment, *_load_experiment_data(experiment)) for experiment in experiments
    ]
    if vmax_by_num_train is None:
        vmax_by_num_train = _vmax_by_num_train(experiments)

    fig, axes = plt.subplots(
        len(NUM_TRAIN_VALUES),
        len(MODEL_KINDS),
        figsize=(9, 16),
        sharex=True,
        sharey=True,
    )

    row_images = {}
    by_key = {(experiment.num_train, experiment.model_kind): data for experiment, *data in datasets}
    for row, num_train in enumerate(NUM_TRAIN_VALUES):
        for col, model_kind in enumerate(MODEL_KINDS):
            ax = axes[row, col]
            experiment = Experiment(num_train=num_train, model_kind=model_kind)
            if (num_train, model_kind) not in by_key:
                ax.axis("off")
                continue
            mu_r_values, sigma_values, relative_errors = by_key[(num_train, model_kind)]
            row_images[row] = _plot_heatmap_on_axis(
                ax,
                experiment,
                mu_r_values,
                sigma_values,
                relative_errors,
                vmax=vmax_by_num_train[num_train],
            )

    for row, image in row_images.items():
        fig.colorbar(
            image,
            ax=axes[row, :].ravel().tolist(),
            label=r"Relative $L_2$ error",
            shrink=0.85,
        )
    fig.savefig(SAVE_DIR / "thermal_heatmaps_physics_vs_data.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_error(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    # in percent with 1 significant figure
    return f"{value * 100:.1f}"


def write_summary_tables(experiments: list[Experiment]) -> None:
    summaries: dict[tuple[int, str], dict[str, float]] = {}
    for experiment in experiments:
        summaries[(experiment.num_train, experiment.model_kind)] = summarise_experiment(
            experiment
        )

    csv_path = SAVE_DIR / "em_physics_loss_ablation_table.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "N_train",
                "data_only_interp",
                "pi_gnn_interp",
                "data_only_extrap",
                "pi_gnn_extrap",
            ]
        )
        for num_train in NUM_TRAIN_VALUES:
            data_summary = summaries.get((num_train, "data"), {})
            physics_summary = summaries.get((num_train, "physics"), {})
            writer.writerow(
                [
                    num_train,
                    _format_error(data_summary.get("interp", np.nan)),
                    _format_error(physics_summary.get("interp", np.nan)),
                    _format_error(data_summary.get("extrap", np.nan)),
                    _format_error(physics_summary.get("extrap", np.nan)),
                ]
            )

    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \begin{tabular}{lcccc}",
        r"        \hline",
        r"        $N_{\mathrm{train}}$",
        r"        & Data-only interp. & PI-GNN interp. & Data-only extrap. & PI-GNN extrap. \\",
        r"        \hline",
    ]
    for num_train in NUM_TRAIN_VALUES:
        data_summary = summaries.get((num_train, "data"), {})
        physics_summary = summaries.get((num_train, "physics"), {})
        lines.append(
            "        "
            f"{num_train}  & "
            f"{_format_error(data_summary.get('interp', np.nan))} & "
            f"{_format_error(physics_summary.get('interp', np.nan))} & "
            f"{_format_error(data_summary.get('extrap', np.nan))} & "
            f"{_format_error(physics_summary.get('extrap', np.nan))} \\\\"
        )
    lines.extend(
        [
            r"        \hline",
            r"    \end{tabular}",
            (
                r"    \caption{Effect of physics loss on electromagnetic surrogate "
                r"data efficiency. Errors are reported as mean relative \(L_2\) "
                r"errors of the predicted magnetic vector potential \(A\). The "
                r"interpolation and extrapolation test sets are identical to those "
                r"used in the \(\mu_r\)-\(\sigma\) generalisation study in "
                r"Section~\ref{section:unseen_em}.}"
            ),
            r"    \label{tab:em_physics_loss_ablation}",
            r"\end{table}",
        ]
    )
    latex_path = SAVE_DIR / "em_physics_loss_ablation_table.tex"
    latex_path.write_text("\n".join(lines) + "\n")

    print(f"CSV table saved to {csv_path}")
    print(f"LaTeX table saved to {latex_path}")


def _selected_experiments(args: argparse.Namespace) -> list[Experiment]:
    return [
        Experiment(num_train=num_train, model_kind=model_kind)
        for num_train in args.num_train
        for model_kind in args.models
    ]


def generate_data(experiments: list[Experiment], force: bool = False) -> None:
    for experiment in experiments:
        generate_experiment_data(experiment, force=force)


def plot_saved_data(experiments: list[Experiment]) -> None:
    shared_vmax = _vmax_by_num_train(experiments)
    shared_vmax_fixed = {num_train: 0.6 for num_train, vmax in shared_vmax.items()}
    for experiment in experiments:
        plot_experiment_heatmap(experiment, vmax=shared_vmax_fixed[experiment.num_train])
    plot_comparison_heatmaps(experiments, vmax_by_num_train=shared_vmax_fixed)
    write_summary_tables(experiments)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and plot thermal physics-loss ablation heatmaps."
    )
    parser.add_argument(
        "--mode",
        choices=("generate", "plot", "table", "all"),
        default="plot",
        help="Generate data, plot saved data, write tables, or run all steps.",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        nargs="+",
        default=list(NUM_TRAIN_VALUES),
        choices=NUM_TRAIN_VALUES,
        help="Training-set sizes to process.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_KINDS),
        choices=MODEL_KINDS,
        help="Model kinds to process: data and/or physics.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate cached error maps instead of resuming/skipping them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_experiments = _selected_experiments(args)

    if args.mode in {"generate", "all"}:
        generate_data(selected_experiments, force=args.force)

    if args.mode in {"plot", "all"}:
        plot_saved_data(selected_experiments)

    if args.mode in {"table", "all"}:
        write_summary_tables(selected_experiments)

    print(f"Outputs saved to {SAVE_DIR}")
