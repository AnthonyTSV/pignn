from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from helpers.mpl_style import apply_mpl_style
from helpers.plot_helpers import (
    plot_l2,
    epoch_vs_train_loss,
    epoch_vs_training_l2,
    _get_run_label,
)
from new_pignn.containers import (
    BoundaryCondition,
    ConvectionBC,
    DirichletBC,
    MaterialPropertiesEM,
    MaterialPropertiesHeat,
    SourceProperties,
    TimeConfig,
)
from new_pignn.em_eddy_problems import eddy_current_problem_different_mu_r
from new_pignn.fem import FEMSolver
from new_pignn.ih_nn_solver import IHNNSolver
from new_pignn.thermal_problems import GenericHeatEquationProblem
from new_pignn.trainer import PIMGNTrainer

plt.style.use(["science"])
apply_mpl_style()


DEFAULT_SAVE_DIR = Path("results/coupled_physics_informed/ih_sensitivity_study")
DEFAULT_PLOT_DIR = Path("verification_plots/coupled_ih")


@dataclass(frozen=True)
class ThermalCase:
    key: str
    source_key: str
    thermal_key: str
    label: str
    purpose: str
    dir_name: str


@dataclass
class ThermalCaseResult:
    case: ThermalCase
    temperature: np.ndarray
    source: np.ndarray | None = None


CASES = (
    ThermalCase(
        key="A",
        source_key="q_fem",
        thermal_key="fem",
        label=r"A: FEM thermal, $Q_\mathrm{FEM}$",
        purpose="reference",
        dir_name="case_A_fem_source_fem_thermal",
    ),
    ThermalCase(
        key="B",
        source_key="q_gnn",
        thermal_key="fem",
        label=r"B: FEM thermal, $Q_\mathrm{GNN}$",
        purpose="physical effect of EM source error",
        dir_name="case_B_gnn_source_fem_thermal",
    ),
    ThermalCase(
        key="C",
        source_key="q_fem",
        thermal_key="pignn",
        label=r"C: PI-GNN thermal, $Q_\mathrm{FEM}$",
        purpose="thermal surrogate error",
        dir_name="case_C_fem_source_pignn_thermal",
    ),
    ThermalCase(
        key="D",
        source_key="q_gnn",
        thermal_key="pignn",
        label=r"D: PI-GNN thermal, $Q_\mathrm{GNN}$",
        purpose="full coupled surrogate",
        dir_name="case_D_gnn_source_pignn_thermal",
    ),
)

DECOMPOSITION = (
    (
        "source",
        "B",
        r"$\Delta T_\mathrm{source}$",
        r"$T_\max^\mathrm{FEM}(Q_\mathrm{GNN}) - T_\max^\mathrm{FEM}(Q_\mathrm{FEM})$",
    ),
    (
        "thermal",
        "C",
        r"$\Delta T_\mathrm{thermal}$",
        r"$T_\max^\mathrm{GNN}(Q_\mathrm{FEM}) - T_\max^\mathrm{FEM}(Q_\mathrm{FEM})$",
    ),
    (
        "chain",
        "D",
        r"$\Delta T_\mathrm{chain}$",
        r"$T_\max^\mathrm{GNN}(Q_\mathrm{GNN}) - T_\max^\mathrm{FEM}(Q_\mathrm{FEM})$",
    ),
)


class IHSensitivityStudySolver(IHNNSolver):
    def build_thermal_problem(self, joule_heating: np.ndarray):
        joule_heating = np.asarray(joule_heating, dtype=np.float64)
        return GenericHeatEquationProblem(
            mesh=self.mesh,
            material_properties=self.material_properties,
            initial_condition=self.initial_condition,
            time_config=self.time_config,
            boundary_conditions=self.boundary_conditions_heat,
            source_function=joule_heating,
            thermal_domain_materials=["mat_workpiece"],
            axisymmetric=True,
            mesh_type="ih_mesh",
        ).get_problem()

    def solve_em_sources(self):
        self.em_model = self._set_em_model(self.path_to_em_model)

        a_gnn = np.asarray(self.solve_em())
        a_fem = np.asarray(self.em_model.all_fem_solvers[0].solve(self.em_problem))

        q_gnn = self.compute_joule_heat(a_gnn)
        q_fem = self.compute_joule_heat(a_fem)

        self.em_solution = a_gnn
        return (
            {"q_fem": q_fem, "q_gnn": q_gnn},
            {"a_fem": a_fem, "a_gnn": a_gnn},
        )

    def solve_thermal_fem(self, joule_heating: np.ndarray):
        thermal_problem = self.build_thermal_problem(joule_heating)
        fem_solver = FEMSolver(self.mesh, problem=thermal_problem)
        thermal_solution = fem_solver.solve_transient_problem(thermal_problem)
        return np.asarray(thermal_solution), thermal_problem

    def solve_thermal_pignn(self, joule_heating: np.ndarray):
        self._require_model_checkpoint(self.path_to_thermal_model, "Thermal")
        thermal_problem = self.build_thermal_problem(joule_heating)
        self.thermal_problem = thermal_problem

        config = {
            "epochs": 1,
            "lr": 1e-3,
            "time_window": 10,
            "generate_ground_truth_for_validation": False,
            "save_dir": str(self.save_dir),
            "resume_from": str(self.path_to_thermal_model),
        }
        self.thermal_model = PIMGNTrainer([thermal_problem], config)
        thermal_solution = self.thermal_model.rollout(problem_idx=0)
        return np.asarray(thermal_solution), thermal_problem

    def solve_thermal_case(self, case: ThermalCase, source: np.ndarray):
        if case.thermal_key == "fem":
            return self.solve_thermal_fem(source)
        if case.thermal_key == "pignn":
            return self.solve_thermal_pignn(source)
        raise ValueError(f"Unknown thermal solver key: {case.thermal_key}")


def create_default_solver(save_dir: Path = DEFAULT_SAVE_DIR):
    path_to_thermal = Path(
        "results/physics_informed/thermal_ih_problem/pimgn_trained_model.pth"
    )
    path_to_em = Path("results/physics_informed/em_aluminum/pimgn_trained_model.pth")

    em_problem = eddy_current_problem_different_mu_r(
        mu_r_workpiece=1,
        sigma_workpiece=37037037,
        a_star=2e-3,
    )
    mesh = em_problem.mesh
    material_properties_heat = MaterialPropertiesHeat(
        rho=2700,
        cp=933.3,
        k=211,
    )
    material_properties_em = {
        "mat_workpiece": MaterialPropertiesEM(sigma=37037037, mu=1),
        "mat_air": MaterialPropertiesEM(sigma=0, mu=1),
        "mat_coil": MaterialPropertiesEM(sigma=0, mu=1),
    }
    boundary_conditions_heat: dict[str, BoundaryCondition] = {
        "bc_workpiece_top": ConvectionBC(value=(10, 20)),
        "bc_workpiece_right": ConvectionBC(value=(10, 20)),
        "bc_workpiece_bottom": ConvectionBC(value=(10, 20)),
    }
    boundary_conditions_em: dict[str, BoundaryCondition] = {
        "bc_air": DirichletBC(value=0.0),
        "bc_axis": DirichletBC(value=0.0),
        "bc_workpiece_left": DirichletBC(value=0.0),
    }
    source_properties = SourceProperties(
        frequency=3000,
        current=3000,
        fill_factor=1.0,
    )
    time_config = TimeConfig(
        dt=0.1,
        t_final=10.0,
    )

    return IHSensitivityStudySolver(
        path_to_thermal_model=path_to_thermal,
        path_to_em_model=path_to_em,
        mesh=mesh,
        boundary_conditions_heat=boundary_conditions_heat,
        boundary_conditions_em=boundary_conditions_em,
        initial_condition=22,
        material_properties=material_properties_heat,
        material_properties_em=material_properties_em,
        source_properties=source_properties,
        time_config=time_config,
        save_dir=save_dir,
    )


def run_four_case_study(
    save_dir: Path = DEFAULT_SAVE_DIR,
    export_vtk: bool = True,
) -> dict[str, ThermalCaseResult]:
    save_dir.mkdir(parents=True, exist_ok=True)
    solver = create_default_solver(save_dir=save_dir)

    sources, em_fields = solver.solve_em_sources()

    results: dict[str, ThermalCaseResult] = {}
    thermal_problems = {}
    for case in CASES:
        print(f"Solving case {case.key}: {case.purpose}")
        solution, thermal_problem = solver.solve_thermal_case(
            case,
            source=sources[case.source_key],
        )
        results[case.key] = ThermalCaseResult(
            case=case,
            temperature=solution,
            source=sources[case.source_key],
        )
        thermal_problems[case.key] = thermal_problem

    baseline = results["A"].temperature
    _save_source_data(save_dir, sources, em_fields, solver)
    for key, result in results.items():
        case_dir = save_dir / result.case.dir_name
        if export_vtk:
            _export_thermal_case_vtk(
                solver=solver,
                thermal_problem=thermal_problems[key],
                result=result,
                reference_temperature=baseline,
                case_dir=case_dir,
            )
        _save_case_npz(
            case_dir=case_dir,
            result=result,
            reference_temperature=baseline,
            time_steps=solver.time_config.time_steps_export,
        )

    if export_vtk:
        _export_em_vtk(save_dir, solver, em_fields)

    write_temperature_report(results, save_dir=save_dir)
    return results


def _save_case_npz(
    case_dir: Path,
    result: ThermalCaseResult,
    reference_temperature: np.ndarray,
    time_steps: np.ndarray,
):
    results_dir = case_dir / "results_data"
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        results_dir / "results_thermal.npz",
        exact=np.asarray(reference_temperature),
        predicted=np.asarray(result.temperature),
        source=np.asarray(result.source),
        time=np.asarray(time_steps),
        case=result.case.key,
        source_key=result.case.source_key,
        thermal_key=result.case.thermal_key,
        purpose=result.case.purpose,
    )


def _save_source_data(save_dir: Path, sources: dict, em_fields: dict, solver):
    results_dir = save_dir / "results_data"
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        results_dir / "sources.npz",
        q_fem=np.asarray(sources["q_fem"]),
        q_gnn=np.asarray(sources["q_gnn"]),
        a_fem=np.asarray(em_fields["a_fem"]),
        a_gnn=np.asarray(em_fields["a_gnn"]),
        frequency=solver.source_properties.frequency,
        current=solver.source_properties.current,
    )


def _export_thermal_case_vtk(
    solver: IHSensitivityStudySolver,
    thermal_problem,
    result: ThermalCaseResult,
    reference_temperature: np.ndarray,
    case_dir: Path,
):
    fem_solver = FEMSolver(solver.mesh, problem=thermal_problem)
    fem_solver.export_to_vtk(
        array_true=np.asarray(reference_temperature),
        array_pred=np.asarray(result.temperature),
        time_steps=solver.time_config.time_steps_export,
        filename=str(case_dir / "vtk/thermal_solution"),
        material_fields={"JouleHeating": np.asarray(result.source)},
    )


def _export_em_vtk(save_dir: Path, solver: IHSensitivityStudySolver, em_fields: dict):
    fem_em_solver = solver.em_model.all_fem_solvers[0]
    fem_em_solver.export_to_vtk_complex(
        array_true=np.asarray(em_fields["a_fem"]) * solver.em_problem.A_star,
        array_pred=np.asarray(em_fields["a_gnn"]) * solver.em_problem.A_star,
        filename=str(save_dir / "em/vtk/em_solution"),
    )


def _max_temperature_series(temperature: np.ndarray) -> np.ndarray:
    temperature = np.asarray(temperature)
    if temperature.ndim < 2:
        raise ValueError(
            f"Thermal arrays must have shape (time_steps, nodes). Got {temperature.shape}."
        )
    return np.max(temperature, axis=1)


def _final_tmax(temperature: np.ndarray) -> float:
    return float(_max_temperature_series(temperature)[-1])


def _relative_delta_percent(value: float, baseline: float) -> float:
    return (value - baseline) / (abs(baseline) + 1e-8) * 100.0


def _ratio(value: float, baseline: float) -> float:
    if abs(baseline) < 1e-12:
        return np.nan
    return value / baseline


def write_temperature_report(
    results: dict[str, ThermalCaseResult],
    save_dir: Path = DEFAULT_SAVE_DIR,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    baseline_tmax = _final_tmax(results["A"].temperature)

    case_rows = []
    for case in CASES:
        result = results[case.key]
        tmax = _final_tmax(result.temperature)
        case_rows.append(
            {
                "case": case.key,
                "source": case.source_key,
                "thermal_solver": case.thermal_key,
                "purpose": case.purpose,
                "final_tmax_C": tmax,
                "delta_vs_A_C": tmax - baseline_tmax,
                "relative_delta_vs_A_percent": _relative_delta_percent(
                    tmax, baseline_tmax
                ),
                "ratio_vs_A": _ratio(tmax, baseline_tmax),
            }
        )

    csv_path = save_dir / "temperature_decomposition_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    report_lines = [
        "Induction-heating source/thermal decomposition",
        "",
        f"Reference: Case A = T_max^FEM(Q_FEM) = {baseline_tmax:.6g} C",
        "",
    ]
    for _, case_key, title, formula in DECOMPOSITION:
        tmax = _final_tmax(results[case_key].temperature)
        delta = tmax - baseline_tmax
        rel_delta = _relative_delta_percent(tmax, baseline_tmax)
        ratio = _ratio(tmax, baseline_tmax)
        report_lines.extend(
            [
                title,
                formula,
                f"case = {case_key}",
                f"comparison T_max = {tmax:.6g} C",
                f"delta = {delta:.6g} C",
                f"relative delta = {rel_delta:.6g} %",
                f"ratio = {ratio:.8g}",
                "",
            ]
        )

    report_path = save_dir / "temperature_decomposition_report.txt"
    report_path.write_text("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"Saved summary CSV to {csv_path}")
    print(f"Saved report to {report_path}")


def last_max_temp_error(path_to_thermal_npz: Path):
    temp_case, temp_reference = _load_thermal_npz(path_to_thermal_npz)

    max_temp_case = _final_tmax(temp_case)
    max_temp_reference = _final_tmax(temp_reference)

    abs_delta = max_temp_case - max_temp_reference
    rel_delta = _relative_delta_percent(max_temp_case, max_temp_reference)

    print(f"Max temperature (case): {max_temp_case}")
    print(f"Max temperature (reference A): {max_temp_reference}")
    print(f"Delta: {abs_delta}")
    print(f"Relative delta: {rel_delta:.2f}%")


def _load_thermal_npz(thermal_npz: Path):
    loaded_thermal = np.load(thermal_npz)
    temp_reference = loaded_thermal["exact"]
    temp_case = loaded_thermal["predicted"]

    if temp_case.shape != temp_reference.shape:
        raise ValueError(
            f"Case and reference arrays must have same shape. "
            f"Got {temp_case.shape} and {temp_reference.shape}."
        )
    if temp_case.ndim < 2:
        raise ValueError(
            f"Thermal arrays must have shape (time_steps, nodes). Got {temp_case.shape}."
        )

    return temp_case, temp_reference


def _time_axis(n_steps: int, dt: float | None = None, time_values=None):
    if time_values is not None:
        time_values = np.asarray(time_values)
        if time_values.shape[0] != n_steps:
            raise ValueError(
                f"time_values length must match rollout length. "
                f"Got {time_values.shape[0]} and {n_steps}."
            )
        return time_values, "Time [s]"

    if dt is not None:
        return np.arange(n_steps) * dt, "Time [s]"

    return np.arange(n_steps), "Time step"


def plot_temperature_over_time(
    thermal_npz,
    ax,
    ax_err,
    label=None,
):
    temp_case, temp_reference = _load_thermal_npz(thermal_npz)

    max_temp_case = _max_temperature_series(temp_case)
    max_temp_reference = _max_temperature_series(temp_reference)

    temp_delta = max_temp_case - max_temp_reference
    n_steps = max_temp_case.shape[0]
    line_idx = len(ax.lines)
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    markers = ["o", "s", "^", "D", "v", "P"]
    markevery = max(1, n_steps // 12)

    line_kwargs = {
        "linewidth": 1.4,
        "linestyle": line_styles[line_idx % len(line_styles)],
        "marker": markers[line_idx % len(markers)],
        "markersize": 3.0,
        "markevery": markevery,
        "alpha": 0.95,
    }

    (line,) = ax.plot(max_temp_case, label=label, **line_kwargs)
    ax.set_ylabel(r"$T_{\max}$ [C]")
    ax.legend(
        ncols=1,
        fontsize="small",
    )

    if len(ax_err.lines) == 0:
        ax_err.axhline(0.0, linewidth=0.8, color="0.35", linestyle="--", zorder=0)

    err_line_kwargs = {key: value for key, value in line_kwargs.items() if key != "color"}
    ax_err.plot(
        temp_delta,
        color=line.get_color(),
        **err_line_kwargs,
    )
    ax_err.set_xlabel("Time step")
    ax_err.set_ylabel(r"$\Delta T_{\max}$ [C]")
    ax_err.margins(y=0.15)


def plot_l2_error_over_time(
    thermal_npz,
    ax,
    label=None,
):
    temp_case, temp_reference = _load_thermal_npz(thermal_npz)

    l2_errors = np.linalg.norm(temp_case - temp_reference, axis=1) / (
        np.linalg.norm(temp_reference, axis=1) + 1e-8
    )

    ax.plot(l2_errors, linewidth=1, label=label)
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"Relative $L_2$ Error vs Case A")
    if label is not None:
        ax.legend(ncols=1, fontsize="small")


def _load_saved_case_results(save_dir: Path) -> dict[str, ThermalCaseResult]:
    results = {}
    for case in CASES:
        thermal_npz = save_dir / case.dir_name / "results_data/results_thermal.npz"
        if not thermal_npz.is_file():
            continue
        temp_case, _ = _load_thermal_npz(thermal_npz)
        loaded = np.load(thermal_npz)
        source = loaded["source"] if "source" in loaded.files else None
        results[case.key] = ThermalCaseResult(
            case=case,
            temperature=temp_case,
            source=source,
        )
    return results


def read_data_and_plot(
    save_dir: Path = DEFAULT_SAVE_DIR,
    plot_dir: Path = DEFAULT_PLOT_DIR,
):
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax, ax_err) = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.0),
        sharex=True,
        height_ratios=[3.0, 1.4],
        constrained_layout=True,
    )
    fig_l2, ax_l2 = plt.subplots(figsize=(6, 4))

    plotted_any = False
    for case in CASES:
        thermal_npz = save_dir / case.dir_name / "results_data/results_thermal.npz"
        if not thermal_npz.is_file():
            print(f"Skipping missing case {case.key}: {thermal_npz}")
            continue

        print(f"Results for {case.label}:")
        last_max_temp_error(thermal_npz)
        plot_temperature_over_time(
            thermal_npz,
            ax,
            ax_err,
            label=case.label,
        )
        plot_l2_error_over_time(
            thermal_npz,
            ax_l2,
            label=case.label,
        )
        plotted_any = True

    if not plotted_any:
        raise FileNotFoundError(
            f"No saved case results found under {save_dir}. "
            "Run with --run to generate the A-D study first."
        )

    fig.savefig(plot_dir / "max_temperature_over_time_study.pdf", dpi=300)
    fig_l2.savefig(plot_dir / "l2_error_over_time_study.pdf", dpi=300)

    saved_results = _load_saved_case_results(save_dir)
    if "A" in saved_results and all(key in saved_results for _, key, _, _ in DECOMPOSITION):
        write_temperature_report(saved_results, save_dir=save_dir)


def main():
    """
    python3 verification_plots/ih_source_sensitivity_study.py --run
    python3 verification_plots/ih_source_sensitivity_study.py --plot-only
    """
    parser = argparse.ArgumentParser(
        description="Run and plot the IH source/thermal sensitivity decomposition."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        default=False,
        help="Generate the four A-D thermal cases before plotting.",
    )
    parser.add_argument(
        "--no-vtk",
        action="store_true",
        default=True,
        help="Skip VTK export when generating the cases.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        default=True,
        help="Only read saved results and regenerate plots/reports.",
    )
    args = parser.parse_args()

    if args.run and args.plot_only:
        raise ValueError("Use either --run or --plot-only, not both.")

    if args.run:
        run_four_case_study(export_vtk=not args.no_vtk)

    read_data_and_plot()


if __name__ == "__main__":
    main()
