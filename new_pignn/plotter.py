import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

def load_log(log_path: Path) -> Dict:
	"""Load training log JSON file."""
	if not log_path.exists():
		raise FileNotFoundError(f"Log file not found: {log_path}")

	with log_path.open("r") as fh:
		return json.load(fh)


def extract_losses(log_data: Dict) -> Tuple[List[float], List[float], List[int]]:
	"""Extract train/validation losses and epoch indices from log data."""
	history = log_data.get("training_history", {})
	train_loss = history.get("train_loss", [])
	val_loss = history.get("val_loss", [])
	epochs = list(range(1, len(train_loss) + 1))
	return train_loss, val_loss, epochs


def plot_losses(train_loss: List[float], val_loss: List[float], epochs: List[int], *,
				title: str = "Training History", y_log: bool = True,
				output_path: Path | None = None, show: bool = True) -> None:
	"""Plot training/validation loss curves."""
	if not train_loss:
		raise ValueError("Training loss list is empty – nothing to plot.")

	plt.figure(figsize=(8, 4.5))
	plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)

	if val_loss and any(v is not None for v in val_loss):
		valid_epochs = epochs[:len(val_loss)]
		filtered_val = [v for v in val_loss]
		plt.plot(valid_epochs, filtered_val, label="Validation Loss", linewidth=2)

	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title(title)
	plt.grid(True, linestyle="--", alpha=0.4)
	if y_log:
		plt.yscale("log")
	plt.legend()
	plt.tight_layout()

	if output_path:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(output_path, dpi=150)
		print(f"Saved plot to {output_path}")

	if show:
		plt.show()
	else:
		plt.close()


def main() -> None:
	log_path = Path("results/pimgn_test_problem_log.json")
	log_data = load_log(log_path)
	train_loss, val_loss, epochs = extract_losses(log_data)
	plot_losses(
		train_loss,
		val_loss,
		epochs,
		title="PIMGN Training History",
		y_log=True,
		output_path=None,
		show=True,
	)


if __name__ == "__main__":
	main()
