import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc, gaussian_kde
from matplotlib.patches import Rectangle
import scienceplots

from helpers.plot_helpers import (
    plot_l2,
    epoch_vs_train_loss,
    epoch_vs_training_l2,
    _get_run_label,
)

from helpers.error_metrics import compute_l2_error
from helpers.mpl_style import apply_mpl_style
from new_pignn.em_eddy_problems import em_different_geometries
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
apply_mpl_style()

CHECKPOINT = "results/physics_informed/em_different_geometries/pimgn_trained_model.pth"

SAVE_DIR = Path("verification_plots/em_different_geometries")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

GEOMETRY_COLUMNS = (
    "coil_inner_diameter",
    "y_offset",
    "height",
    "diameter",
)

TRAINING_GEOMETRY_BOUNDS = {
    "coil_inner_diameter": (40e-3, 60e-3),
    "y_offset": (-25e-3, 25e-3),
    "height": (50e-3, 90e-3),
    "diameter": (20e-3, 40e-3),
}

def _evaluate(
    coil_inner_diameter: float,
    y_offset: float,
    height: float,
    diameter: float,
) -> float:

    problem = em_different_geometries(
        coil_inner_diameter=coil_inner_diameter,
        y_offset=y_offset,
        height=height,
        diameter=diameter,
    )
    config = {
        "epochs": 1,
        "lr": 1e-3,
        "generate_ground_truth_for_validation": False,
        "save_dir": (
            "results/physics_informed/em_different_geometries_evaluation/"
            f"I_{int(coil_inner_diameter)}_y_{int(y_offset)}_h_{int(height)}_d_{int(diameter)}"
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

def get_training_data_results():
    n_samples = 50
    lhs_seed = 42
    geometry_bounds = np.array(
        [
            [40e-3, 60e-3],  # coil_inner_diameter
            [-25e-3, 25e-3],  # y_offset
            [50e-3, 90e-3],  # height
            [20e-3, 40e-3],  # diameter
        ],
        dtype=np.float64,
    )
    sampler = qmc.LatinHypercube(d=geometry_bounds.shape[0], seed=lhs_seed)
    geometry_samples = qmc.scale(
        sampler.random(n=n_samples),
        geometry_bounds[:, 0],
        geometry_bounds[:, 1],
    )
    training_data = {}
    # read the training .npz and get l2 error for each geometry
    for idx in range(n_samples):
        npz_path = (
            Path("results/physics_informed/em_different_geometries/")
            / f"results_data_p{idx}/results_em_p{idx}.npz"
        )
        if not npz_path.exists():
            print(f"Warning: {npz_path} does not exist. Skipping.")
            continue
        data = np.load(npz_path)
        pred = data["predicted"]
        true = data["exact"]
        l2_error = compute_l2_error(pred, true)
        training_data[idx] = {
            "geometry": geometry_samples[idx],
            "l2_error": l2_error,
        }
    return training_data

def generate_interpolation_data():
    # evaluate to get interpolation
    n_samples = 30
    lhs_seed = 1337
    geometry_bounds = np.array(
        [
            [40e-3, 60e-3],  # coil_inner_diameter
            [-25e-3, 25e-3],  # y_offset
            [50e-3, 90e-3],  # height
            [20e-3, 40e-3],  # diameter
        ],
        dtype=np.float64,
    )
    sampler = qmc.LatinHypercube(d=geometry_bounds.shape[0], seed=lhs_seed)
    geometry_samples = qmc.scale(
        sampler.random(n=n_samples),
        geometry_bounds[:, 0],
        geometry_bounds[:, 1],
    )
    results = []
    for geometry in geometry_samples:
        coil_inner_diameter, y_offset, height, diameter = geometry
        l2_error = _evaluate(
            coil_inner_diameter=coil_inner_diameter,
            y_offset=y_offset,
            height=height,
            diameter=diameter,
        )
        results.append(
            {
                "coil_inner_diameter": coil_inner_diameter,
                "y_offset": y_offset,
                "height": height,
                "diameter": diameter,
                "l2_error": l2_error,
            }
        )
    # save results to csv
    csv_path = SAVE_DIR / "interpolation_results.csv"
    with open(csv_path, mode="w", newline="") as csv_file:
        fieldnames = [
            "coil_inner_diameter",
            "y_offset",
            "height",
            "diameter",
            "l2_error",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def generate_extrapolation_data():
    lhs_seed = 1337
    n_samples = 150
    geometry_bounds = np.array(
        [
            [40e-3, 70e-3],  # coil_inner_diameter
            [-35e-3, 35e-3],  # y_offset
            [30e-3, 150e-3],  # height
            [15e-3, 60e-3],  # diameter
        ],
        dtype=np.float64,
    )
    sampler = qmc.LatinHypercube(d=geometry_bounds.shape[0], seed=lhs_seed)
    geometry_samples = qmc.scale(
        sampler.random(n=n_samples),
        geometry_bounds[:, 0],
        geometry_bounds[:, 1],
    )
    # check for overlap of diameter and coil_inner_diameter bounds in generated samples
    results = []
    for geometry in geometry_samples:
        coil_inner_diameter, y_offset, height, diameter = geometry
        if diameter >= coil_inner_diameter:
            print(
                f"Warning: diameter {diameter:.3e} is greater than or equal to coil_inner_diameter {coil_inner_diameter:.3e}. Skipping this sample."
            )
            continue
        if _is_interpolation_geometry(
            {
                "coil_inner_diameter": coil_inner_diameter,
                "y_offset": y_offset,
                "height": height,
                "diameter": diameter,
            }
        ):
            print(
                "Warning: sampled geometry is inside the training bounds. "
                "Skipping this interpolation sample."
            )
            continue
        l2_error = _evaluate(
            coil_inner_diameter=coil_inner_diameter,
            y_offset=y_offset,
            height=height,
            diameter=diameter,
        )
        results.append(
            {
                "coil_inner_diameter": coil_inner_diameter,
                "y_offset": y_offset,
                "height": height,
                "diameter": diameter,
                "l2_error": l2_error,
            }
        )
    # save results to csv
    csv_path = SAVE_DIR / "extrapolation_results.csv"
    with open(csv_path, mode="w", newline="") as csv_file:
        fieldnames = [
            "coil_inner_diameter",
            "y_offset",
            "height",
            "diameter",
            "l2_error",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def get_data_for_each_geometry_parameter():
    heights = np.linspace(30e-3, 150e-3, 20)
    diameters = np.linspace(15e-3, 60e-3, 20)
    y_offsets = np.linspace(-35e-3, 35e-3, 20)
    coil_inner_diameters = np.linspace(40e-3, 70e-3, 20)
    fixed_height = 70e-3
    fixed_diameter = 30e-3
    fixed_y_offset = 0.0
    fixed_coil_inner_diameter = 50e-3
    results = {
        "heights": {},
        "diameters": {},
        "y_offsets": {},
        "coil_inner_diameters": {},
    }
    for height in heights:
        l2_error = _evaluate(
            coil_inner_diameter=fixed_coil_inner_diameter,
            y_offset=fixed_y_offset,
            height=height,
            diameter=fixed_diameter,
        )
        results["heights"][height] = l2_error
    for diameter in diameters:
        l2_error = _evaluate(
            coil_inner_diameter=fixed_coil_inner_diameter,
            y_offset=fixed_y_offset,
            height=fixed_height,
            diameter=diameter,
        )
        results["diameters"][diameter] = l2_error
    for y_offset in y_offsets:
        l2_error = _evaluate(
            coil_inner_diameter=fixed_coil_inner_diameter,
            y_offset=y_offset,
            height=fixed_height,
            diameter=fixed_diameter,
        )
        results["y_offsets"][y_offset] = l2_error
    for coil_inner_diameter in coil_inner_diameters:
        l2_error = _evaluate(
            coil_inner_diameter=coil_inner_diameter,
            y_offset=fixed_y_offset,
            height=fixed_height,
            diameter=fixed_diameter,
        )
        results["coil_inner_diameters"][coil_inner_diameter] = l2_error
    
    # save results to csv
    csv_path = SAVE_DIR / "geometry_parameter_comparison.csv"
    with open(csv_path, mode="w", newline="") as csv_file:
        fieldnames = ["parameter", "value", "l2_error"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for parameter, values in results.items():
            for value, l2_error in values.items():
                writer.writerow(
                    {
                        "parameter": parameter,
                        "value": value,
                        "l2_error": l2_error,
                    }
                )

def read_geometry_parameter_comparison():
    csv_path = SAVE_DIR / "geometry_parameter_comparison.csv"
    data = {
        "heights": {"values": [], "l2_errors": []},
        "diameters": {"values": [], "l2_errors": []},
        "y_offsets": {"values": [], "l2_errors": []},
        "coil_inner_diameters": {"values": [], "l2_errors": []},
    }
    with open(csv_path, mode="r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            parameter = row["parameter"]
            value = float(row["value"])
            l2_error = float(row["l2_error"])
            if parameter in data:
                data[parameter]["values"].append(value)
                data[parameter]["l2_errors"].append(l2_error)
            else:
                print(f"Warning: unknown parameter '{parameter}' in CSV. Skipping.")
    return data

def plot_aspect_ratio_comparison():
    data = read_geometry_parameter_comparison()
    heights = np.array(data["heights"]["values"])
    height_errors = np.array(data["heights"]["l2_errors"])
    diameters = np.array(data["diameters"]["values"])
    diameter_errors = np.array(data["diameters"]["l2_errors"])

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(heights * 1e3, height_errors, marker="o", label="Height", color="tab:blue")
    # indicate training bounds with vertical lines
    ax.axvline(50, color="tab:grey", linestyle="--", linewidth=0.75, label="Training bounds")
    ax.axvline(90, color="tab:grey", linestyle="--", linewidth=0.75)
    ax.set_xlabel("Billet height [mm]")
    ax.set_ylabel(r"Relative $L_2$ error")
    ax.legend()
    fig.savefig(SAVE_DIR / "height_comparison.pdf", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(diameters * 1e3, diameter_errors, marker="o", label="Diameter", color="tab:orange")
    # indicate training bounds with vertical lines
    ax.axvline(20, color="tab:grey", linestyle="--", linewidth=0.75, label="Training bounds")
    ax.axvline(40, color="tab:grey", linestyle="--", linewidth=0.75)
    ax.set_xlabel("Billet diameter [mm]")
    ax.set_ylabel(r"Relative $L_2$ error")
    ax.legend()
    fig.savefig(SAVE_DIR / "diameter_comparison.pdf", dpi=300)
    plt.close(fig)


def plot_offset_comparison():
    data = read_geometry_parameter_comparison()
    y_offsets = np.array(data["y_offsets"]["values"])
    y_offset_errors = np.array(data["y_offsets"]["l2_errors"])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(y_offsets * 1e3, y_offset_errors, marker="o", label="Y offset", color="tab:green")
    # indicate training bounds with vertical lines
    ax.axvline(-25, color="tab:grey", linestyle="--", linewidth=0.75, label="Training bounds")
    ax.axvline(25, color="tab:grey", linestyle="--", linewidth=0.75)
    ax.set_xlabel("Y offset [mm]")
    ax.set_ylabel(r"Relative $L_2$ error")
    ax.legend()
    fig.savefig(SAVE_DIR / "y_offset_comparison.pdf", dpi=300)
    plt.close(fig)

def plot_air_gap_comparison():
    data = read_geometry_parameter_comparison()
    coil_inner_diameters = np.array(data["coil_inner_diameters"]["values"])
    coil_inner_diameter_errors = np.array(data["coil_inner_diameters"]["l2_errors"])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(coil_inner_diameters * 1e3, coil_inner_diameter_errors, marker="o", label="Coil inner diameter", color="tab:purple")
    # indicate training bounds with vertical lines
    ax.axvline(40, color="tab:grey", linestyle="--", linewidth=0.75, label="Training bounds")
    ax.axvline(60, color="tab:grey", linestyle="--", linewidth=0.75)
    ax.set_xlabel("Coil inner diameter [mm]")
    ax.set_ylabel(r"Relative $L_2$ error")
    ax.legend()
    fig.savefig(SAVE_DIR / "coil_inner_diameter_comparison.pdf", dpi=300)
    plt.close(fig)

def _is_interpolation_geometry(row: dict[str, float]) -> bool:
    return all(
        lower <= row[column] <= upper
        for column, (lower, upper) in TRAINING_GEOMETRY_BOUNDS.items()
    )


def _read_result_rows(csv_path: Path, source: str) -> list[dict[str, float | str]]:
    rows = []
    with open(csv_path, mode="r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            parsed_row = {column: float(row[column]) for column in GEOMETRY_COLUMNS}
            parsed_row["l2_error"] = float(row["l2_error"])
            parsed_row["source"] = source
            rows.append(parsed_row)
    return rows


def _load_separated_results() -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    source_files = (
        (SAVE_DIR / "interpolation_results.csv", "interpolation_results.csv"),
        (SAVE_DIR / "extrapolation_results.csv", "extrapolation_results.csv"),
    )
    interpolation_rows = []
    extrapolation_rows = []

    for csv_path, source in source_files:
        for row in _read_result_rows(csv_path, source):
            if _is_interpolation_geometry(row):
                interpolation_rows.append(row)
            else:
                extrapolation_rows.append(row)

    return interpolation_rows, extrapolation_rows


def _write_separated_results(
    interpolation_rows: list[dict[str, float | str]],
    extrapolation_rows: list[dict[str, float | str]],
) -> None:
    fieldnames = [*GEOMETRY_COLUMNS, "l2_error", "source"]
    output_rows = (
        (SAVE_DIR / "interpolation_results_separated.csv", interpolation_rows),
        (SAVE_DIR / "extrapolation_results_separated.csv", extrapolation_rows),
    )
    for csv_path, rows in output_rows:
        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _format_stat(value: float) -> str:
    return f"{value:.2e}"


def _plot_single_violin(
    rows: list[dict[str, float | str]],
    *,
    title: str,
    subtitle: str,
    color: str,
    point_color: str,
    output_stem: str,
) -> None:
    if not rows:
        raise ValueError(f"No rows available for {title.lower()} violin plot.")

    l2_errors = np.asarray([float(row["l2_error"]) for row in rows], dtype=float)
    quartile1, median, quartile3 = np.percentile(l2_errors, [25, 50, 75])
    mean = float(np.mean(l2_errors))

    print(
        f"\n--- {title} ---\n"
        f"  n      = {l2_errors.size}\n"
        f"  mean   = {_format_stat(mean)}\n"
        f"  median = {_format_stat(median)}\n"
        f"  IQR    = {_format_stat(quartile1)} \u2013 {_format_stat(quartile3)}\n"
        f"  min    = {_format_stat(float(np.min(l2_errors)))}\n"
        f"  max    = {_format_stat(float(np.max(l2_errors)))}\n"
    )

    # KDE computed in log10 space so the violin shape is correct on a log-scale axis
    log_data = np.log10(l2_errors)
    kde = gaussian_kde(log_data, bw_method="scott")
    pad = max((log_data.max() - log_data.min()) * 0.12, 0.15)
    y_log = np.linspace(log_data.min() - pad, log_data.max() + pad, 500)
    density = kde(y_log)
    density = density / density.max() * 0.38  # half-width in x data units
    y_vals = 10.0 ** y_log

    fig, ax = plt.subplots(figsize=(3.2, 4.2))

    # Violin body
    ax.fill_betweenx(y_vals, 1.0 - density, 1.0 + density,
                     color=color, alpha=0.50, linewidth=0, zorder=1)
    ax.plot(1.0 - density, y_vals, color="#2c3e50", linewidth=0.65, alpha=0.75, zorder=1)
    ax.plot(1.0 + density, y_vals, color="#2c3e50", linewidth=0.65, alpha=0.75, zorder=1)

    ax.scatter(np.ones(l2_errors.size), l2_errors, s=8, color=point_color, alpha=0.50, linewidths=0, zorder=2)

    # IQR box (drawn in data coords; log transform places it correctly)
    iqr_hw = 0.052
    rect = Rectangle(
        (1.0 - iqr_hw, quartile1), 2 * iqr_hw, quartile3 - quartile1,
        facecolor="white", edgecolor="#2c3e50", linewidth=1.0, zorder=4,
    )
    ax.add_patch(rect)

    # Median bar
    ax.plot([1.0 - iqr_hw, 1.0 + iqr_hw], [median, median],
            color="#2c3e50", linewidth=2.0, solid_capstyle="butt", zorder=5)

    # Mean marker
    ax.scatter(1.0, mean, s=36, color="tab:red", marker="D",
               edgecolors="white", linewidths=0.5, zorder=6, label="Mean")

    ax.set_yscale("log")
    ax.set_xlim(0.52, 1.48)
    ax.set_xticks([1.0])
    ax.set_xticklabels([subtitle])
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_ylabel(r"Relative $L_2$ error")
    ax.legend(loc="upper right", fontsize="small")

    fig.tight_layout()
    fig.savefig(SAVE_DIR / f"{output_stem}.pdf", dpi=300)
    plt.close(fig)


def plot_violin_charts() -> None:
    interpolation_rows, extrapolation_rows = _load_separated_results()
    _write_separated_results(interpolation_rows, extrapolation_rows)

    _plot_single_violin(
        interpolation_rows,
        title="Interpolation geometry error distribution",
        subtitle="Inside training bounds",
        color="tab:blue",
        point_color="tab:red",
        output_stem="l2_error_violin_plot_interpolation",
    )
    _plot_single_violin(
        extrapolation_rows,
        title="Extrapolation geometry error distribution",
        subtitle="Outside training bounds",
        color="tab:orange",
        point_color="tab:red",
        output_stem="l2_error_violin_plot_extrapolation",
    )


def plot_violin_chart(interpolation: bool = True) -> None:
    interpolation_rows, extrapolation_rows = _load_separated_results()
    rows = interpolation_rows if interpolation else extrapolation_rows
    label = "interpolation" if interpolation else "extrapolation"
    _plot_single_violin(
        rows,
        title=f"{label.capitalize()} geometry error distribution",
        subtitle="Inside training bounds" if interpolation else "Outside training bounds",
        color="tab:blue" if interpolation else "tab:orange",
        point_color="tab:red",
        output_stem=f"l2_error_violin_plot_{label}",
    )

if __name__ == "__main__":
    # generate_interpolation_data()
    # generate_extrapolation_data()
    # get_data_for_each_geometry_parameter()
    plot_aspect_ratio_comparison()
    plot_offset_comparison()
    plot_air_gap_comparison()
    # plot_violin_charts()
    
