import time
import os
from typing import List, Tuple
import numpy as np
import torch
import torch.optim as optim
from trainer_em import PIMGNTrainerEM
from containers import MeshProblemEM

from train_problems import create_em_problem_complex

def grid_search_hyperparameters(
    problem_fn,
    learning_rates: list = None,
    scheduler_configs: list = None,
    base_config: dict = None,
    base_save_dir: str = "results/grid_search",
):
    """
    Perform grid search over learning rates and scheduler configurations.

    Args:
        problem_fn: Function that returns a MeshProblemEM instance
        learning_rates: List of learning rates to try (default: [1e-3, 1e-4, 1e-5])
        scheduler_configs: List of scheduler config dicts with keys:
            - "name": str (e.g., "StepLR", "CosineAnnealingLR", "ExponentialLR", "ReduceLROnPlateau")
            - "params": dict of scheduler-specific parameters
        base_config: Base training config dict
        base_save_dir: Base directory for saving results

    Returns:
        dict: Results for each configuration with final loss and errors
    """
    import itertools
    import json

    if learning_rates is None:
        learning_rates = [1e-3, 1e-4, 1e-5]

    if scheduler_configs is None:
        raise ValueError("scheduler_configs must be provided for grid search.")

    if base_config is None:
        raise ValueError("base_config must be provided for grid search.")

    results = {}
    all_combinations = list(itertools.product(learning_rates, scheduler_configs))
    total_experiments = len(all_combinations)

    print("=" * 70)
    print(f"GRID SEARCH: {total_experiments} experiments")
    print(f"Learning rates: {learning_rates}")
    print(f"Schedulers: {[s['name'] for s in scheduler_configs]}")
    print("=" * 70)

    for exp_idx, (lr, sched_config) in enumerate(all_combinations):
        exp_name = f"lr_{lr}_sched_{sched_config['name']}"
        print(f"\n[{exp_idx + 1}/{total_experiments}] Running: {exp_name}")
        print("-" * 50)

        # Create fresh problem instance for each experiment
        problem = problem_fn()

        # Build config for this experiment
        config = base_config.copy()
        config["lr"] = lr
        config["scheduler_config"] = sched_config
        config["save_dir"] = f"{base_save_dir}/{exp_name}"

        os.makedirs(config["save_dir"], exist_ok=True)

        try:
            # Create trainer with custom scheduler
            trainer = PIMGNTrainerEMGridSearch([problem], config)

            # Train
            trainer.train(train_problems_indices=[0])

            # Get final loss
            final_loss = trainer.losses[-1] if trainer.losses else float("inf")

            # Evaluate
            try:
                predictions, ground_truth, errors = trainer.evaluate_with_ground_truth(
                    problem_indices=[0]
                )
                final_error = errors[0] if errors else float("inf")
            except Exception as e:
                print(f"Evaluation failed: {e}")
                final_error = float("inf")

            results[exp_name] = {
                "lr": lr,
                "scheduler": sched_config,
                "final_loss": final_loss,
                "final_error": final_error,
                "loss_history": trainer.losses,
            }

            print(f"Final loss: {final_loss:.6e}, L2 error: {final_error:.6e}")

            # Save individual experiment results
            trainer.save_logs()

        except Exception as e:
            print(f"Experiment failed: {e}")
            results[exp_name] = {
                "lr": lr,
                "scheduler": sched_config,
                "final_loss": float("inf"),
                "final_error": float("inf"),
                "error_message": str(e),
            }

    # Find best configuration
    best_config = min(results.keys(), key=lambda k: results[k]["final_error"])
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)
    print(f"Best configuration: {best_config}")
    print(f"  Learning rate: {results[best_config]['lr']}")
    print(f"  Scheduler: {results[best_config]['scheduler']['name']}")
    print(f"  Final loss: {results[best_config]['final_loss']:.6e}")
    print(f"  L2 error: {results[best_config]['final_error']:.6e}")

    # Save summary
    summary_path = f"{base_save_dir}/grid_search_summary.json"
    with open(summary_path, "w") as f:
        # Convert to serializable format
        serializable_results = {}
        for k, v in results.items():
            serializable_results[k] = {
                "lr": v["lr"],
                "scheduler": v["scheduler"],
                "final_loss": float(v["final_loss"]),
                "final_error": float(v["final_error"]),
            }
        json.dump(
            {"results": serializable_results, "best_config": best_config},
            f,
            indent=2,
        )
    print(f"\nSummary saved to: {summary_path}")

    return results


class PIMGNTrainerEMGridSearch(PIMGNTrainerEM):
    """Extended trainer with configurable scheduler for grid search."""

    def __init__(self, problems: List[MeshProblemEM], config: dict):
        # Temporarily remove scheduler_config to let parent init
        scheduler_config = config.pop("scheduler_config", None)
        super().__init__(problems, config)

        # Replace scheduler if custom config provided
        if scheduler_config is not None:
            self.scheduler = self._create_scheduler(scheduler_config)

    def _create_scheduler(self, sched_config: dict):
        """Create a scheduler based on config dict."""
        name = sched_config["name"]
        params = sched_config.get("params", {})

        if name == "StepLR":
            return optim.lr_scheduler.StepLR(self.optimizer, **params)
        elif name == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **params)
        elif name == "ExponentialLR":
            return optim.lr_scheduler.ExponentialLR(self.optimizer, **params)
        elif name == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **params)
        elif name == "CosineAnnealingWarmRestarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, **params)
        elif name == "OneCycleLR":
            # OneCycleLR needs total_steps
            if "total_steps" not in params:
                params["total_steps"] = self.config.get("epochs", 1000)
            return optim.lr_scheduler.OneCycleLR(self.optimizer, **params)
        else:
            raise ValueError(f"Unknown scheduler: {name}")

    def train(self, train_problems_indices, val_problems_indices=None):
        """Override train to handle ReduceLROnPlateau scheduler."""
        print(f"Starting PIMGN-EM training on {self.device}")
        print(f"Training on problems: {train_problems_indices}")

        problem = self.problems[train_problems_indices[0]]
        prediction = np.zeros(problem.n_nodes, dtype=np.float64)

        is_reduce_on_plateau = isinstance(
            self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
        )

        for epoch in range(self.start_epoch, self.config["epochs"]):
            epoch_start = time.time()

            physics_loss, prediction_next = self.train_step(0, prediction=prediction)
            prediction = prediction_next

            self.losses.append(physics_loss)

            # Validation
            val_loss = None
            if val_problems_indices and self.all_ground_truth is not None:
                val_loss = self.validate(val_problems_indices)
                self.val_losses.append(val_loss)

            elapsed = time.time() - epoch_start
            self.logger.log_epoch(epoch, physics_loss, val_loss, elapsed)

            if epoch % 10 == 0:
                val_str = f" | Val Loss: {val_loss:.3e}" if val_loss is not None else ""
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1:4d} | Loss: {physics_loss:.3e}{val_str} | LR: {current_lr:.2e} | Time: {elapsed:.2f}s"
                )

            # Step scheduler (handle ReduceLROnPlateau differently)
            if is_reduce_on_plateau:
                self.scheduler.step(physics_loss)
            else:
                self.scheduler.step()

        print("Training completed!")


def run_grid_search_example():
    """Example of running grid search."""
    grid_search_hyperparameters(
        problem_fn=create_em_problem_complex,
        learning_rates=[1e-3, 1e-4, 1e-5],
        scheduler_configs=[
            {"name": "StepLR", "params": {"step_size": 1000, "gamma": 0.9}},
            {"name": "CosineAnnealingLR", "params": {"T_max": 500, "eta_min": 1e-6}},
            {"name": "ExponentialLR", "params": {"gamma": 0.999}},
            {"name": "ReduceLROnPlateau", "params": {"factor": 0.5, "patience": 100}},
        ],
        base_config={
            "epochs": 10000,
            "generate_ground_truth_for_validation": False,
        },
        base_save_dir="results/grid_search_em",
    )

if __name__ == "__main__":
    run_grid_search_example()