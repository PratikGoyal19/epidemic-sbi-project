"""Setup the compatible Python 3.10 Environment"""

# Force install Python 3.10 and compatible TensorFlow/Bayesflow
!sudo apt-get update -y
!sudo apt-get install python3.10 python3.10-distutils -y
!curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
!python3.10 -m pip install tensorflow==2.15.0 bayesflow==1.1.6

"""Mount Drive & Run the Script"""

from google.colab import drive
drive.mount('/content/drive')

# Run using the Python 3.10 environment we just built
!python3.10 /content/drive/MyDrive/epidemic-sbi-project/03_methods/train_npe.py \
  --data /content/drive/MyDrive/epidemic-sbi-project/02_data/sir_dataset.npz \
  --artifacts-dir /content/drive/MyDrive/epidemic-sbi-project/03_methods/artifacts

"""Zip and Download Artifacts"""

from google.colab import files

# Zip the artifacts directory
!zip -r /content/npe_artifacts.zip /content/drive/MyDrive/epidemic-sbi-project/03_methods/artifacts/

# Download the zip directly to your computer
files.download('/content/npe_artifacts.zip')

    

"""
Train a Neural Posterior Estimation (NPE) model for SIR data using BayesFlow.

Model:
- Learns p(theta | x), where:
  - theta = [beta, gamma] (shape: 2)
  - x = infected trajectory over time (shape: T)
- Uses an Amortized Posterior backed by an Invertible Network (Normalizing Flow).

Outputs:
- 03_methods/artifacts/npe_weights (TensorFlow Checkpoint files)
- 03_methods/artifacts/npe_metrics.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import bayesflow as bf



def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Loads and validates the SIR dataset."""
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
    parser = argparse.ArgumentParser(description="Train NPE on SIR dataset using BayesFlow")
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
        "--num-coupling-layers", 
        type=int, 
        default=4,
        help="Number of coupling layers for the Invertible Network."
    )
    # parse_known_args() ignores hidden Jupyter flags like '-f' if run in a notebook
    args, _ = parser.parse_known_args()
    return args

def main() -> None:
    args = parse_args()
    
    # Set seeds for reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data)
    out_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    theta, x = load_dataset(data_path)
    
    # Format data specifically for BayesFlow offline training
    sim_data = {
        "prior_draws": theta,
        "sim_data": x
    }

    print(f"Initializing BayesFlow NPE with {args.num_coupling_layers} coupling layers...")
    # 1. Define the Inference Network (Normalizing Flow)
    inference_net = bf.networks.InvertibleNetwork(
        num_params=2, # beta and gamma
        num_coupling_layers=args.num_coupling_layers,
        coupling_net_settings={
            "dense_args": {"units": args.hidden_dim, "activation": "relu"}
        }
    )

    # 2. Set up the Amortized Posterior
    amortizer = bf.amortizers.AmortizedPosterior(inference_net)

    # 3. Configure the Trainer
    trainer = bf.trainers.Trainer(
        amortizer=amortizer,
        default_lr=args.lr
    )

    print("Starting offline training...")
    start = time.time()
    # 4. Execute Offline Training
    history = trainer.train_offline(
        simulations_dict=sim_data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    runtime_sec = time.time() - start

    # 5. Save Weights (Without .h5 extension to force standard TF Checkpoint format)
    weights_prefix = out_dir / "npe_weights"
    trainer.amortizer.save_weights(str(weights_prefix))

    # 6. Save Metrics
    metrics = {
        "dataset_path": str(data_path),
        "n_samples": int(theta.shape[0]),
        "trajectory_length_T": int(x.shape[1]),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_coupling_layers": args.num_coupling_layers,
        "hidden_dim": args.hidden_dim,
        "runtime_sec": float(runtime_sec)
    }
    
    metrics_path = out_dir / "npe_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved BayesFlow weights to: {out_dir} (as TF Checkpoints)")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Runtime: {runtime_sec:.2f}s")

if __name__ == "__main__":
    main()
