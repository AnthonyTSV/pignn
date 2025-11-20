import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import dataclasses

from containers import MeshProblem

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class TrainingLogger:
    """
    Logger class to store training information and write it to a JSON file.
    Handles numpy types automatically.
    """
    def __init__(self, save_dir: str = "results", filename: str = "training_log.json", save_interval: Optional[float] = None, save_epoch_interval: Optional[int] = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.save_interval = save_interval
        self.save_epoch_interval = save_epoch_interval
        self.last_save_time = time.time()
        self.log_data = {
            "config": {},
            "training_history": {
                "train_loss": [],
                "val_loss": [],
                "epoch_times": []
            },
            "evaluation": {},
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": "unknown"
            },
            "problems": []
        }
        self.start_time = time.time()

    def log_config(self, config: Dict[str, Any]):
        """Log configuration dictionary."""
        self.log_data["config"] = config

    def set_device(self, device: str):
        """Log the device being used."""
        self.log_data["metadata"]["device"] = str(device)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None, epoch_time: float = 0.0):
        """Log metrics for a single epoch."""
        self.log_data["training_history"]["train_loss"].append(train_loss)
        self.log_data["training_history"]["val_loss"].append(val_loss)
        self.log_data["training_history"]["epoch_times"].append(epoch_time)
        
        # Periodic save
        should_save = False
        if self.save_interval and (time.time() - self.last_save_time > self.save_interval):
            should_save = True
            self.last_save_time = time.time()
            
        if self.save_epoch_interval and (epoch + 1) % self.save_epoch_interval == 0:
            should_save = True
            
        if should_save:
            self.save()

    def log_evaluation(self, data: Any, metric_name: str):
        """Log evaluation metrics."""
        self.log_data["evaluation"][metric_name] = data

    def log_problems(self, problems: List[Any]):
        """Log problem configurations."""
        self.log_data["problems"] = [self._serialize_problem(p) for p in problems]

    def _serialize_problem(self, problem: MeshProblem) -> Dict[str, Any]:
        """Extract serializable data from a MeshProblem instance."""
        # Basic attributes
        data = {
            "problem_id": getattr(problem, "problem_id", None),
            "n_nodes": getattr(problem, "n_nodes", None),
            "n_edges": getattr(problem, "n_edges", None),
            "alpha": getattr(problem, "alpha", None),
            "boundary_values": getattr(problem, "boundary_values", None),
            "neumann_values": getattr(problem, "neumann_values", None),
            "robin_values": getattr(problem, "robin_values", None),
            "nonlinear_source_params": getattr(problem, "nonlinear_source_params", None),
        }
        
        # Config objects
        if hasattr(problem, "time_config"):
            tc = problem.time_config
            data["time_config"] = {
                "dt": tc.dt,
                "t_final": tc.t_final,
                "num_steps": tc.num_steps
            }
            
        if hasattr(problem, "mesh_config"):
            mc = problem.mesh_config
            # Convert dataclass to dict if possible, otherwise manual extraction
            if dataclasses.is_dataclass(mc):
                mc_dict = dataclasses.asdict(mc)
                # Remove potentially large or non-serializable objects if any
                data["mesh_config"] = mc_dict
            else:
                data["mesh_config"] = {
                    "maxh": getattr(mc, "maxh", None),
                    "order": getattr(mc, "order", None),
                    "dim": getattr(mc, "dim", None),
                    "mesh_type": getattr(mc, "mesh_type", None),
                    "dirichlet_boundaries": getattr(mc, "dirichlet_boundaries", None),
                    "neumann_boundaries": getattr(mc, "neumann_boundaries", None),
                    "robin_boundaries": getattr(mc, "robin_boundaries", None)
                }
            
        return data

    def save(self, filename: Optional[str] = None):
        """Save the log data to a JSON file."""
        self.log_data["metadata"]["total_duration"] = time.time() - self.start_time
        
        target_filename = filename if filename else self.filename
        filepath = self.save_dir / target_filename
        try:
            with open(filepath, 'w') as f:
                json.dump(self.log_data, f, indent=4, cls=NumpyEncoder)
            print(f"Training log saved to {filepath}")
        except Exception as e:
            print(f"Failed to save training log: {e}")
