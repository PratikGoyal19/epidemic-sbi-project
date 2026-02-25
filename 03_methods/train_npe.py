
import os
from google.colab import files

# Create the expected folder structure
os.makedirs('02_data', exist_ok=True)
os.makedirs('03_methods/artifacts', exist_ok=True)

print("Upload sir_dataset.npz:")
uploaded_data = files.upload()
for fn in uploaded_data.keys():
    os.rename(fn, f'02_data/{fn}')

print("Upload train_npe.py (or create it in the file explorer):")
uploaded_script = files.upload()

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from sbi.inference import SNPE
from sbi.utils import BoxUniform

def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path)
    if "theta" not in data or "x" not in data:
        raise KeyError("Dataset must contain 'theta' and 'x' arrays.")
    theta = data["theta"]
    x = data["x"]
    if theta.ndim != 2 or theta.shape[1] != 2:
        raise ValueError(f"Expected theta shape (N, 2), got {theta.shape}")
    if x.ndim != 2:
        raise ValueError(f"Expected x shape (N, T), got {x.shape}")
    if x.shape[0] != theta.shape[0]:
        raise ValueError("theta and x must have the same number of samples.")
    return theta.astype(np.float32), x.astype(np.float32)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NPE on SIR dataset using sbi")
    parser.add_argument(
        "--data",
        type=str,
        default="02_data/sir_dataset.npz",
        help="Path to generated dataset (.npz).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="03_methods/artifacts",
        help="Output directory for posterior and metrics.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=50)
    parser.add_argument(
        "--density-estimator", 
        type=str, 
        default="maf", 
        choices=["maf", "nsf", "mdn"],
        help="Type of density estimator for the posterior."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
  # parse_known_args() parses your arguments and ignores the extra Colab '-f' flag
    args, unknown = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data)
    out_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    theta, x = load_dataset(data_path)
    theta_tensor = torch.from_numpy(theta)
    x_tensor = torch.from_numpy(x)

    # 2. Define Prior dynamically based on dataset bounds
    # (Extracting min/max slightly expanded to prevent edge-case clipping)
    theta_min = theta_tensor.min(dim=0)[0] - 1e-4
    theta_max = theta_tensor.max(dim=0)[0] + 1e-4
    prior = BoxUniform(low=theta_min, high=theta_max, device=args.device)

    # 3. Setup and Train NPE via sbi
    print(f"Initializing SNPE with {args.density_estimator.upper()} estimator...")
    inference = SNPE(
        prior=prior, 
        density_estimator=args.density_estimator, 
        device=args.device
    )
    
    inference = inference.append_simulations(theta_tensor, x_tensor)

    start = time.time()
    density_estimator = inference.train(
        training_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_num_epochs=args.epochs,
        show_train_summary=True
    )
    runtime_sec = time.time() - start

    # 4. Build and Save Posterior
    posterior = inference.build_posterior(density_estimator)
    
    # Crucial: Map to CPU before pickling to allow cross-architecture local loading
    posterior.set_default_device("cpu")
    
    posterior_path = out_dir / "npe_posterior.pkl"
    with open(posterior_path, "wb") as f:
        pickle.dump(posterior, f)

    # 5. Save Metrics
    metrics = {
        "dataset_path": str(data_path),
        "n_samples": int(theta.shape[0]),
        "trajectory_length_T": int(x.shape[1]),
        "device": str(args.device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "density_estimator": args.density_estimator,
        "prior_bounds": {
            "low": theta_min.tolist(),
            "high": theta_max.tolist()
        },
        "runtime_sec": float(runtime_sec),
    }
    
    metrics_path = out_dir / "npe_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved CPU-mapped posterior to: {posterior_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Runtime: {runtime_sec:.2f}s")

if __name__ == "__main__":
    main()
