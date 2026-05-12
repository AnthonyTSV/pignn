from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for import_path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import numpy as np
import torch

from new_pignn.em_eddy_problems import em_team_36_problem
from new_pignn.fem import FEMSolver
from new_pignn.fem_em import FEMSolverEM
from new_pignn.thermal_problems import ih_team_36_problem
from new_pignn.trainer import PIMGNTrainer
from new_pignn.trainer_em import PIMGNTrainerEM

try:
    import ngsolve as ng
except ImportError:  # pragma: no cover - benchmark environment should have NGSolve.
    ng = None


DEFAULT_EM_CHECKPOINT = (
    PROJECT_ROOT
    / "results/physics_informed/em_team_36_problem/pimgn_trained_model.pth"
)
DEFAULT_THERMAL_CHECKPOINT = (
    PROJECT_ROOT
    / "results/physics_informed/thermal_team_36_problem/pimgn_trained_model.pth"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "computational_speedup"


class BenchmarkSetupError(RuntimeError):
    pass


@dataclass(frozen=True)
class TimingResult:
    problem: str
    solver: str
    device: str
    mean_s: float
    std_s: float
    repeats: int
    warmups: int
    samples_s: tuple[float, ...]
    n_nodes: int
    n_time_steps: int | None = None
    speedup_vs_fem: float | None = None


@contextlib.contextmanager
def _forced_cuda_availability(enable_cuda: bool):
    """Force trainer device selection without changing the rest of PyTorch."""
    original = torch.cuda.is_available
    torch.cuda.is_available = lambda: bool(enable_cuda) and bool(original())
    try:
        yield
    finally:
        torch.cuda.is_available = original


def _resolve_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _require_readable_checkpoint(path: Path, label: str) -> None:
    try:
        torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        torch.load(path, map_location="cpu")
    except RuntimeError as exc:
        if "failed finding central directory" in str(exc):
            raise BenchmarkSetupError(
                f"{label} at {path} is corrupted or incomplete. "
                "PyTorch could not read the zip central directory; replace the "
                "file with a valid checkpoint or retrain to regenerate it."
            ) from exc
        raise


def _synchronize(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _call_quietly(fn: Callable[[], object], quiet: bool) -> object:
    if not quiet:
        return fn()
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return fn()


def _measure(
    fn: Callable[[], object],
    repeats: int,
    warmups: int,
    device: str,
    quiet: bool,
) -> tuple[float, float, tuple[float, ...]]:
    for _ in range(warmups):
        _call_quietly(fn, quiet=quiet)
        _synchronize(device)

    samples = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter()
        _call_quietly(fn, quiet=quiet)
        _synchronize(device)
        samples.append(time.perf_counter() - start)

    samples_arr = np.asarray(samples, dtype=np.float64)
    std = float(np.std(samples_arr, ddof=1)) if len(samples) > 1 else 0.0
    return float(np.mean(samples_arr)), std, tuple(float(v) for v in samples)


def _set_threads(torch_threads: int | None, ngsolve_threads: int | None) -> None:
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    if ngsolve_threads is not None and ng is not None:
        ng.SetNumThreads(ngsolve_threads)


def _make_em_problem(mu_r: float, sigma: float):
    return em_team_36_problem()


def _make_thermal_problem(mu_r: float, sigma: float):
    # This setup step computes the heat source and is intentionally outside
    # the timed thermal surrogate/FEM regions.
    return ih_team_36_problem()


def _make_em_trainer(
    problem,
    checkpoint: Path,
    device: str,
    output_dir: Path,
    strict_checkpoint: bool,
):
    with _forced_cuda_availability(device == "cuda"):
        config = {
            "epochs": 1,
            "lr": 1e-3,
            "generate_ground_truth_for_validation": False,
            "save_dir": str(output_dir / f"em_pignn_{device}"),
            "enforce_axis_regularity": True,
            "training_mode": "physics",
            "data_weight": 0.0,
            "physics_loss": True,
            "batch_size": 1,
            "resume_from": str(checkpoint),
            "require_checkpoint": True,
            "strict_checkpoint": strict_checkpoint,
        }
        return PIMGNTrainerEM([problem], config=config)


def _make_thermal_trainer(problem, checkpoint: Path, device: str, output_dir: Path):
    with _forced_cuda_availability(device == "cuda"):
        config = {
            "epochs": 1,
            "lr": 1e-3,
            "time_window": 10,
            "noise_sigma": 1e-2,
            "generate_ground_truth_for_validation": False,
            "save_dir": str(output_dir / f"thermal_pignn_{device}"),
            "training_mode": "physics",
            "data_weight": 0.0,
            "physics_loss": True,
            "batch_size": 1,
            "resume_from": str(checkpoint),
        }
        return PIMGNTrainer([problem], config=config)


def _time_pignn_em(
    problem,
    checkpoint: Path,
    device: str,
    output_dir: Path,
    repeats: int,
    warmups: int,
    quiet: bool,
    strict_checkpoint: bool,
) -> TimingResult:
    trainer = _call_quietly(
        lambda: _make_em_trainer(
            problem=problem,
            checkpoint=checkpoint,
            device=device,
            output_dir=output_dir,
            strict_checkpoint=strict_checkpoint,
        ),
        quiet=quiet,
    )

    with _forced_cuda_availability(device == "cuda"):
        mean_s, std_s, samples = _measure(
            lambda: trainer.predict(problem_idx=0),
            repeats=repeats,
            warmups=warmups,
            device=device,
            quiet=quiet,
        )

    return TimingResult(
        problem="Electromagnetic",
        solver="PI-GNN",
        device=device.upper(),
        mean_s=mean_s,
        std_s=std_s,
        repeats=repeats,
        warmups=warmups,
        samples_s=samples,
        n_nodes=int(problem.n_nodes),
    )


def _time_pignn_thermal(
    problem,
    checkpoint: Path,
    device: str,
    output_dir: Path,
    repeats: int,
    warmups: int,
    quiet: bool,
) -> TimingResult:
    trainer = _call_quietly(
        lambda: _make_thermal_trainer(
            problem=problem,
            checkpoint=checkpoint,
            device=device,
            output_dir=output_dir,
        ),
        quiet=quiet,
    )

    with _forced_cuda_availability(device == "cuda"):
        mean_s, std_s, samples = _measure(
            lambda: trainer.rollout(problem_idx=0),
            repeats=repeats,
            warmups=warmups,
            device=device,
            quiet=quiet,
        )

    return TimingResult(
        problem="Thermal",
        solver="PI-GNN",
        device=device.upper(),
        mean_s=mean_s,
        std_s=std_s,
        repeats=repeats,
        warmups=warmups,
        samples_s=samples,
        n_nodes=int(problem.n_nodes),
        n_time_steps=len(problem.time_config.time_steps_export),
    )


def _time_fem_em(
    problem,
    repeats: int,
    warmups: int,
    quiet: bool,
) -> TimingResult:
    mean_s, std_s, samples = _measure(
        lambda: FEMSolverEM(
            problem.mesh,
            order=int(getattr(problem.mesh_config, "order", 1)),
            problem=problem,
            device=torch.device("cpu"),
        ).solve(problem),
        repeats=repeats,
        warmups=warmups,
        device="cpu",
        quiet=quiet,
    )
    return TimingResult(
        problem="Electromagnetic",
        solver="FEM",
        device="CPU",
        mean_s=mean_s,
        std_s=std_s,
        repeats=repeats,
        warmups=warmups,
        samples_s=samples,
        n_nodes=int(problem.n_nodes),
    )


def _time_fem_thermal(
    problem,
    repeats: int,
    warmups: int,
    quiet: bool,
) -> TimingResult:
    mean_s, std_s, samples = _measure(
        lambda: FEMSolver(
            problem.mesh,
            order=int(getattr(problem.mesh_config, "order", 1)),
            problem=problem,
        ).solve_transient_problem(problem),
        repeats=repeats,
        warmups=warmups,
        device="cpu",
        quiet=quiet,
    )
    return TimingResult(
        problem="Thermal",
        solver="FEM",
        device="CPU",
        mean_s=mean_s,
        std_s=std_s,
        repeats=repeats,
        warmups=warmups,
        samples_s=samples,
        n_nodes=int(problem.n_nodes),
        n_time_steps=len(problem.time_config.time_steps_export),
    )


def _with_speedups(results: Iterable[TimingResult]) -> list[TimingResult]:
    results = list(results)
    fem_baselines = {
        result.problem: result.mean_s
        for result in results
        if result.solver == "FEM" and result.device == "CPU"
    }
    enriched = []
    for result in results:
        baseline = fem_baselines.get(result.problem)
        speedup = None if baseline is None else baseline / result.mean_s
        enriched.append(
            TimingResult(
                problem=result.problem,
                solver=result.solver,
                device=result.device,
                mean_s=result.mean_s,
                std_s=result.std_s,
                repeats=result.repeats,
                warmups=result.warmups,
                samples_s=result.samples_s,
                n_nodes=result.n_nodes,
                n_time_steps=result.n_time_steps,
                speedup_vs_fem=speedup,
            )
        )
    return enriched


def _write_summary_csv(results: list[TimingResult], path: Path) -> None:
    fieldnames = [
        "problem",
        "solver",
        "device",
        "mean_s",
        "std_s",
        "speedup_vs_fem_cpu",
        "repeats",
        "warmups",
        "n_nodes",
        "n_time_steps",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "problem": result.problem,
                    "solver": result.solver,
                    "device": result.device,
                    "mean_s": f"{result.mean_s:.12g}",
                    "std_s": f"{result.std_s:.12g}",
                    "speedup_vs_fem_cpu": (
                        "" if result.speedup_vs_fem is None else f"{result.speedup_vs_fem:.12g}"
                    ),
                    "repeats": result.repeats,
                    "warmups": result.warmups,
                    "n_nodes": result.n_nodes,
                    "n_time_steps": "" if result.n_time_steps is None else result.n_time_steps,
                }
            )


def _write_raw_csv(results: list[TimingResult], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "problem",
                "solver",
                "device",
                "repeat",
                "runtime_s",
            ],
        )
        writer.writeheader()
        for result in results:
            for repeat_idx, runtime_s in enumerate(result.samples_s, start=1):
                writer.writerow(
                    {
                        "problem": result.problem,
                        "solver": result.solver,
                        "device": result.device,
                        "repeat": repeat_idx,
                        "runtime_s": f"{runtime_s:.12g}",
                    }
                )


def _tex_runtime(result: TimingResult) -> str:
    return f"{result.mean_s:.3e} $\\pm$ {result.std_s:.1e}"


def _tex_speedup(result: TimingResult) -> str:
    if result.speedup_vs_fem is None:
        return "--"
    return f"{result.speedup_vs_fem:.2f}"


def _write_latex_table(results: list[TimingResult], path: Path) -> None:
    rows = [
        (
            result.problem,
            f"{result.solver} ({result.device})",
            _tex_runtime(result),
            _tex_speedup(result),
        )
        for result in results
    ]
    with path.open("w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("    \\centering\n")
        f.write("    \\begin{tabular}{llcc}\n")
        f.write("        \\hline\n")
        f.write("        Problem & Solver & Runtime [s] & Speed-up vs. FEM CPU \\\\\n")
        f.write("        \\hline\n")
        for problem, solver, runtime, speedup in rows:
            f.write(
                f"        {problem} & {solver} & {runtime} & {speedup} \\\\\n"
            )
        f.write("        \\hline\n")
        f.write("    \\end{tabular}\n")
        f.write(
            "    \\caption{Computational runtime of the trained PI-GNN surrogates "
            "after training. The reported mean and standard deviation are computed "
            "over repeated inference runs after one warm-up run. PI-GNN timings "
            "exclude checkpoint loading and include graph construction, data "
            "transfer, and neural-network inference. FEM timings are measured on "
            "the CPU.}\n"
        )
        f.write("    \\label{tab:computational_speedup}\n")
        f.write("\\end{table}\n")


def _write_metadata(args, output_dir: Path, em_checkpoint: Path, thermal_checkpoint: Path) -> None:
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "torch_num_threads": torch.get_num_threads(),
        "ngsolve_available": ng is not None,
        "em_checkpoint": str(em_checkpoint),
        "thermal_checkpoint": str(thermal_checkpoint),
        "mu_r": args.mu_r,
        "sigma": args.sigma,
        "repeats": args.repeats,
        "warmups": args.warmups,
    }
    with (output_dir / "timing_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def _print_result(result: TimingResult) -> None:
    speedup = (
        ""
        if result.speedup_vs_fem is None
        else f", speed-up={result.speedup_vs_fem:.2f}x"
    )
    print(
        f"{result.problem:16s} {result.solver:6s} {result.device:4s}: "
        f"{result.mean_s:.6g} +/- {result.std_s:.2g} s{speedup}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark trained PI-GNN surrogates against CPU FEM. PI-GNN timings "
            "exclude model loading and include graph construction, data transfer, "
            "and neural-network inference."
        )
    )
    parser.add_argument(
        "--em-checkpoint",
        type=Path,
        default=DEFAULT_EM_CHECKPOINT,
        help="Path to the electromagnetic PI-GNN checkpoint.",
    )
    parser.add_argument(
        "--thermal-checkpoint",
        type=Path,
        default=DEFAULT_THERMAL_CHECKPOINT,
        help="Path to the thermal PI-GNN checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where timing CSV/TeX files are written.",
    )
    parser.add_argument(
        "--only",
        choices=("both", "em", "thermal"),
        default="both",
        help="Benchmark both models, or only one model family.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of timed repetitions after warm-up.",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Number of warm-up runs before timing.",
    )
    parser.add_argument(
        "--mu-r",
        type=float,
        default=50.0,
        help="Relative permeability for the representative query.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=6.0e6,
        help="Electrical conductivity [S/m] for the representative query.",
    )
    parser.add_argument(
        "--skip-fem",
        action="store_true",
        help="Do not run the CPU FEM baseline.",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Do not run PI-GNN GPU timing even if CUDA is available.",
    )
    parser.add_argument(
        "--strict-em-checkpoint",
        action="store_true",
        help="Require exact EM checkpoint/model compatibility.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Set PyTorch CPU thread count for reproducible CPU timings.",
    )
    parser.add_argument(
        "--ngsolve-threads",
        type=int,
        default=None,
        help="Set NGSolve CPU thread count for reproducible FEM timings.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show solver/trainer output during setup and timing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    em_checkpoint = _resolve_path(args.em_checkpoint)
    thermal_checkpoint = _resolve_path(args.thermal_checkpoint)
    if args.only in {"both", "em"}:
        _require_file(em_checkpoint, "EM checkpoint")
        _require_readable_checkpoint(em_checkpoint, "EM checkpoint")
    if args.only in {"both", "thermal"}:
        _require_file(thermal_checkpoint, "Thermal checkpoint")
        _require_readable_checkpoint(thermal_checkpoint, "Thermal checkpoint")

    _set_threads(args.torch_threads, args.ngsolve_threads)
    quiet = not args.verbose

    results: list[TimingResult] = []
    devices = ["cpu"]
    if not args.skip_gpu and torch.cuda.is_available():
        devices.append("cuda")
    elif not args.skip_gpu:
        print("CUDA is not available; skipping PI-GNN GPU timing.")

    if args.only in {"both", "em"}:
        print("Preparing electromagnetic problem...")
        em_problem = _call_quietly(
            lambda: _make_em_problem(args.mu_r, args.sigma),
            quiet=quiet,
        )
        if not args.skip_fem:
            print("Timing electromagnetic FEM on CPU...")
            results.append(
                _time_fem_em(
                    problem=em_problem,
                    repeats=args.repeats,
                    warmups=args.warmups,
                    quiet=quiet,
                )
            )
        for device in devices:
            print(f"Timing electromagnetic PI-GNN on {device.upper()}...")
            results.append(
                _time_pignn_em(
                    problem=em_problem,
                    checkpoint=em_checkpoint,
                    device=device,
                    output_dir=output_dir,
                    repeats=args.repeats,
                    warmups=args.warmups,
                    quiet=quiet,
                    strict_checkpoint=args.strict_em_checkpoint,
                )
            )

    if args.only in {"both", "thermal"}:
        print("Preparing thermal problem...")
        thermal_problem = _call_quietly(
            lambda: _make_thermal_problem(args.mu_r, args.sigma),
            quiet=quiet,
        )
        if not args.skip_fem:
            print("Timing thermal FEM on CPU...")
            results.append(
                _time_fem_thermal(
                    problem=thermal_problem,
                    repeats=args.repeats,
                    warmups=args.warmups,
                    quiet=quiet,
                )
            )
        for device in devices:
            print(f"Timing thermal PI-GNN on {device.upper()}...")
            results.append(
                _time_pignn_thermal(
                    problem=thermal_problem,
                    checkpoint=thermal_checkpoint,
                    device=device,
                    output_dir=output_dir,
                    repeats=args.repeats,
                    warmups=args.warmups,
                    quiet=quiet,
                )
            )

    results = _with_speedups(results)
    _write_summary_csv(results, output_dir / "runtime_summary.csv")
    _write_raw_csv(results, output_dir / "runtime_raw.csv")
    _write_latex_table(results, output_dir / "runtime_table.tex")
    _write_metadata(args, output_dir, em_checkpoint, thermal_checkpoint)

    print("\nTiming summary:")
    for result in results:
        _print_result(result)
    print(f"\nWrote timing files to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (BenchmarkSetupError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
