import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for import_path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from helpers.mpl_style import apply_mpl_style
from new_pignn.containers import TimeConfig
from new_pignn.thermal_problems import create_ih_problem, ih_team_36_problem
from new_pignn.trainer import PIMGNTrainer

import scienceplots

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
apply_mpl_style()

SAVE_DIR = SCRIPT_DIR / "rollout_analysis"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "results/physics_informed/thermal_ih_problem/pimgn_trained_model.pth"
)
TEAM_36_CHECKPOINT = (
    PROJECT_ROOT
    / "results/physics_informed/thermal_team_36_problem/pimgn_trained_model.pth"
)


@dataclass(frozen=True)
class RolloutCase:
    key: str
    label: str
    checkpoint: Path
    problem_factory: Callable[[], object]


@dataclass(frozen=True)
class RolloutResult:
    case: RolloutCase
    rows: list[dict[str, float]]
    time: np.ndarray
    normalized_time: np.ndarray
    l2_by_time: np.ndarray


def _require_checkpoint(checkpoint: Path) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint}. Train or copy a thermal model first."
        )


def _checkpoint_time_window(checkpoint: Path, default: int = 10) -> int:
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint_data, dict) and "time_window" in checkpoint_data:
        return int(checkpoint_data["time_window"])
    return default


def _multiplier_label(multiplier: float) -> str:
    if float(multiplier).is_integer():
        return f"x{int(multiplier)}"
    return f"x{multiplier:g}"


def _masked_l2_by_time(predicted: np.ndarray, exact: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(predicted - exact, axis=1)
    denominator = np.linalg.norm(exact, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )


def _masked_space_time_l2(predicted: np.ndarray, exact: np.ndarray) -> float:
    denominator = np.linalg.norm(exact)
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm(predicted - exact) / denominator)


def _build_problem(problem_factory: Callable[[], object], horizon_multiplier: float):
    problem = problem_factory()
    base_time_config = problem.time_config
    problem.time_config = TimeConfig(
        dt=base_time_config.dt,
        t_final=base_time_config.t_final * horizon_multiplier,
    )
    return problem, base_time_config


def _evaluate_extended_rollout(
    case: RolloutCase,
    multipliers: list[float],
    save_dir: Path,
    export_vtk: bool = False,
) -> RolloutResult:
    _require_checkpoint(case.checkpoint)
    save_dir.mkdir(parents=True, exist_ok=True)

    multipliers = sorted(set([1.0, *multipliers]))
    max_multiplier = max(multipliers)
    problem, base_time_config = _build_problem(case.problem_factory, max_multiplier)
    workpiece_mask = np.asarray(problem.wp_node_mask, dtype=bool)
    if workpiece_mask.size == 0 or not workpiece_mask.any():
        raise ValueError("The thermal problem does not provide a non-empty wp_node_mask.")

    time_window = _checkpoint_time_window(case.checkpoint)
    config = {
        "epochs": 1,
        "lr": 1e-3,
        "time_window": time_window,
        "noise_sigma": 1e-2,
        "batch_size": 1,
        "training_mode": "physics",
        "data_weight": 0.0,
        "physics_loss": True,
        "generate_ground_truth_for_validation": False,
        "resume_from": str(case.checkpoint),
        "save_dir": str(save_dir / "evaluation_run"),
    }
    trainer = PIMGNTrainer([problem], config=config)

    print(
        f"Running {case.label} extended rollout "
        f"to {_multiplier_label(max_multiplier)} "
        f"(t_final={problem.time_config.t_final:g}, dt={problem.time_config.dt:g})"
    )
    predictions = np.asarray(trainer.rollout(problem_idx=0), dtype=float)
    ground_truth = np.asarray(
        trainer.all_fem_solvers[0].solve_transient_problem(problem),
        dtype=float,
    )

    if export_vtk:
        trainer.all_fem_solvers[0].export_to_vtk(
            np.array(ground_truth),
            np.array(predictions),
            problem.time_config.time_steps_export,
            filename=save_dir / "evaluation_run/rollout_comparison",
        )

    if predictions.shape != ground_truth.shape:
        raise ValueError(
            "Prediction and FEM arrays must have the same shape. "
            f"Got predicted={predictions.shape}, exact={ground_truth.shape}."
        )

    predicted_wp = predictions[:, workpiece_mask]
    exact_wp = ground_truth[:, workpiece_mask]
    l2_by_time = _masked_l2_by_time(predicted_wp, exact_wp)

    time = problem.time_config.time_steps_export
    normalized_time = time / base_time_config.t_final
    rows = []
    for multiplier in multipliers:
        end_time = base_time_config.t_final * multiplier
        end_idx = int(round(end_time / base_time_config.dt))
        if end_idx >= len(time):
            raise ValueError(
                f"Requested horizon {_multiplier_label(multiplier)} exceeds rollout length."
            )
        horizon_pred = predicted_wp[1 : end_idx + 1]
        horizon_true = exact_wp[1 : end_idx + 1]
        horizon_l2 = l2_by_time[1 : end_idx + 1]
        rows.append(
            {
                "multiplier": multiplier,
                "t_final": float(end_time),
                "n_steps": int(end_idx),
                "final_l2": float(l2_by_time[end_idx]),
                "mean_l2": float(np.mean(horizon_l2)),
                "max_l2": float(np.max(horizon_l2)),
                "space_time_l2": _masked_space_time_l2(horizon_pred, horizon_true),
            }
        )

    baseline_final = rows[0]["final_l2"]
    baseline_space_time = rows[0]["space_time_l2"]
    for row in rows:
        row["final_l2_growth_vs_x1"] = (
            row["final_l2"] / baseline_final if baseline_final > 0.0 else np.nan
        )
        row["space_time_l2_growth_vs_x1"] = (
            row["space_time_l2"] / baseline_space_time
            if baseline_space_time > 0.0
            else np.nan
        )

    np.savez_compressed(
        save_dir / "rollout_l2_by_time.npz",
        time=time,
        normalized_time=normalized_time,
        training_t_final=np.array(base_time_config.t_final, dtype=float),
        l2_by_time=l2_by_time,
        workpiece_mask=workpiece_mask,
        multipliers=np.asarray([row["multiplier"] for row in rows], dtype=float),
        final_l2=np.asarray([row["final_l2"] for row in rows], dtype=float),
        mean_l2=np.asarray([row["mean_l2"] for row in rows], dtype=float),
        max_l2=np.asarray([row["max_l2"] for row in rows], dtype=float),
        space_time_l2=np.asarray([row["space_time_l2"] for row in rows], dtype=float),
    )
    _write_summary_csv(save_dir / "rollout_l2_summary.csv", rows)

    return RolloutResult(
        case=case,
        rows=rows,
        time=time,
        normalized_time=normalized_time,
        l2_by_time=l2_by_time,
    )


def _write_summary_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "multiplier",
        "t_final",
        "n_steps",
        "final_l2",
        "mean_l2",
        "max_l2",
        "space_time_l2",
        "final_l2_growth_vs_x1",
        "space_time_l2_growth_vs_x1",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_summary_rows(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Cached summary not found: {path}. Run without --plot-only first."
        )

    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed_row = {}
            for key, value in row.items():
                if key == "n_steps":
                    parsed_row[key] = int(value)
                else:
                    parsed_row[key] = float(value)
            rows.append(parsed_row)
    if not rows:
        raise ValueError(f"Cached summary is empty: {path}")
    return rows


def _load_rollout_result(case: RolloutCase, save_dir: Path) -> RolloutResult:
    summary_path = save_dir / "rollout_l2_summary.csv"
    data_path = save_dir / "rollout_l2_by_time.npz"
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Cached rollout data not found: {data_path}. Run without --plot-only first."
        )

    rows = _load_summary_rows(summary_path)
    data = np.load(data_path)
    time = np.asarray(data["time"], dtype=float)
    l2_by_time = np.asarray(data["l2_by_time"], dtype=float)

    if "normalized_time" in data.files:
        normalized_time = np.asarray(data["normalized_time"], dtype=float)
    else:
        training_row = min(rows, key=lambda row: abs(row["multiplier"] - 1.0))
        normalized_time = time / training_row["t_final"]

    return RolloutResult(
        case=case,
        rows=rows,
        time=time,
        normalized_time=normalized_time,
        l2_by_time=l2_by_time,
    )


def _write_combined_summary_csv(path: Path, results: list[RolloutResult]) -> None:
    fieldnames = [
        "case",
        "label",
        "multiplier",
        "t_final",
        "n_steps",
        "final_l2",
        "mean_l2",
        "max_l2",
        "space_time_l2",
        "final_l2_growth_vs_x1",
        "space_time_l2_growth_vs_x1",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for row in result.rows:
                writer.writerow(
                    {
                        "case": result.case.key,
                        "label": result.case.label,
                        **row,
                    }
                )


def _write_combined_l2_npz(path: Path, results: list[RolloutResult]) -> None:
    max_len = max(len(result.normalized_time) for result in results)
    n_cases = len(results)
    time = np.full((n_cases, max_len), np.nan, dtype=float)
    normalized_time = np.full((n_cases, max_len), np.nan, dtype=float)
    l2_by_time = np.full((n_cases, max_len), np.nan, dtype=float)
    training_t_final = np.zeros(n_cases, dtype=float)

    for idx, result in enumerate(results):
        n_time = len(result.normalized_time)
        time[idx, :n_time] = result.time
        normalized_time[idx, :n_time] = result.normalized_time
        l2_by_time[idx, :n_time] = result.l2_by_time
        training_row = min(
            result.rows, key=lambda row: abs(row["multiplier"] - 1.0)
        )
        training_t_final[idx] = training_row["t_final"]

    np.savez_compressed(
        path,
        case_keys=np.asarray([result.case.key for result in results]),
        labels=np.asarray([result.case.label for result in results]),
        time=time,
        normalized_time=normalized_time,
        l2_by_time=l2_by_time,
        training_t_final=training_t_final,
        multipliers=np.asarray(
            [row["multiplier"] for row in results[0].rows], dtype=float
        ),
    )


def _plot_l2_growth(
    path: Path,
    results: list[RolloutResult],
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for result in results:
        plot_l2 = np.where(result.l2_by_time > 0.0, result.l2_by_time, np.nan)
        ax.plot(
            result.normalized_time,
            plot_l2,
            linewidth=1.4,
            label=result.case.label,
        )

    rows = results[0].rows
    max_multiplier = max(row["multiplier"] for row in rows)
    ax.set_xlim(0.0, max_multiplier * 1.04)
    blended_transform = blended_transform_factory(ax.transData, ax.transAxes)
    ax.annotate(
        "",
        xy=(1.0, 0.8),
        xytext=(0.0, 0.8),
        xycoords=blended_transform,
        textcoords=blended_transform,
        arrowprops={
            "arrowstyle": "<->",
            "color": "tab:red",
            "linewidth": 1,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
        annotation_clip=False,
    )
    ax.text(
        0.5,
        0.74,
        r"Training",
        transform=blended_transform,
        ha="center",
        va="bottom",
        fontsize=9,
        color="tab:red",
    )
    for row in rows:
        ax.axvline(
            row["multiplier"], color="tab:gray", linewidth=0.5, linestyle="--"
        )
        ax.text(
            row["multiplier"],
            ax.get_ylim()[1],
            _multiplier_label(row["multiplier"]),
            rotation=90,
            va="top",
            ha="right",
            fontsize=12,
        )
    ax.set_xlabel(r"Normalised time $t/t_\mathrm{train}$")
    ax.set_ylabel(r"Relative $L_2$ Error")
    ax.set_yscale("log")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _print_summary(result: RolloutResult) -> None:
    print(f"\n{result.case.label} masked workpiece L2 rollout summary")
    print(
        "horizon,t_final,n_steps,final_l2,space_time_l2,"
        "final_growth_vs_x1,space_time_growth_vs_x1"
    )
    for row in result.rows:
        print(
            f"{_multiplier_label(row['multiplier'])},"
            f"{row['t_final']:.6g},"
            f"{int(row['n_steps'])},"
            f"{row['final_l2']:.6e},"
            f"{row['space_time_l2']:.6e},"
            f"{row['final_l2_growth_vs_x1']:.6g},"
            f"{row['space_time_l2_growth_vs_x1']:.6g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate how a trained thermal PI-GNN rollout error grows when the "
            "rollout horizon is extended. L2 is computed only on wp_node_mask."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("ih", "team36"),
        default=["ih", "team36"],
        help="Problem curves to include in the normalized rollout plot.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint for the baseline induction-heating thermal problem.",
    )
    parser.add_argument(
        "--team36-checkpoint",
        type=Path,
        default=TEAM_36_CHECKPOINT,
        help="Checkpoint for the thermal Team 36 problem.",
    )
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=[2.0, 5.0, 10.0],
        help="Rollout horizon multipliers relative to the trained problem horizon. x1 is added automatically.",
    )
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    parser.add_argument(
        "--export-vtk",
        action="store_true",
        default=False,
        help="Export rollout comparison VTK files for each case.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        default=True,
        help="Regenerate combined CSV/NPZ/PDF from cached per-case rollout data without running rollouts or FEM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_registry = {
        "ih": RolloutCase(
            key="ih",
            label="Aluminium",
            checkpoint=args.checkpoint,
            problem_factory=create_ih_problem,
        ),
        "team36": RolloutCase(
            key="team36",
            label="TEAM 36",
            checkpoint=args.team36_checkpoint,
            problem_factory=ih_team_36_problem,
        ),
    }

    args.save_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case_name in args.cases:
        case = case_registry[case_name]
        case_save_dir = args.save_dir / case.key
        if args.plot_only:
            result = _load_rollout_result(case, case_save_dir)
        else:
            result = _evaluate_extended_rollout(
                case=case,
                multipliers=args.multipliers,
                save_dir=case_save_dir,
                export_vtk=args.export_vtk,
            )
        results.append(result)
        _print_summary(result)

    _write_combined_summary_csv(args.save_dir / "rollout_l2_summary.csv", results)
    _write_combined_l2_npz(args.save_dir / "rollout_l2_by_time.npz", results)
    _plot_l2_growth(args.save_dir / "rollout_l2_growth.pdf", results)
    print(f"\nSaved results to {args.save_dir}")


if __name__ == "__main__":
    main()
