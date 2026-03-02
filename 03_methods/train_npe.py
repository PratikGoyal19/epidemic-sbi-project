"""
Train a Neural Posterior Estimation (NPE) model for SIR data using BayesFlow.

Model:
- Learns p(theta | x), where:
  - theta = [beta, gamma] (shape: 2)
  - x = infected trajectory over time (shape: T)
- Uses an Amortized Posterior backed by an Invertible Network (Normalizing Flow).

Outputs:
- 03_methods/artifacts/npe_checkpoint/ (TensorFlow Checkpoint)
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
    parser.add_argument("--hidden-dim", type=int, default=128)  #  FIX: Increased from 50 to 128 (better for time series)
    parser.add_argument(
        "--num-coupling-layers", 
        type=int, 
        default=4,
        help="Number of coupling layers for the Invertible Network."
    )
    parser.add_argument(
        "--summary-dim",
        type=int,
        default=32,
        help="Output dimension of summary network."  #  FIX: Added parameter
    )
    args = parser.parse_args()  #  FIX 1: Changed from parse_known_args()
    return args

def main() -> None:
    args = parse_args()
    
    # Set seeds for reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data)
    out_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Training NPE with BayesFlow 1.1.6")
    print("="*70)
    print("Loading dataset...")
    theta, x = load_dataset(data_path)
    print(f" Loaded {len(theta):,} samples")
    print(f"   Theta shape: {theta.shape}")
    print(f"   X shape: {x.shape}")
    
    # Format data specifically for BayesFlow offline training
    sim_data = {
        "prior_draws": theta,
        "sim_data": x
    }

    print(f"\nInitializing BayesFlow NPE...")
    print(f"  Coupling layers: {args.num_coupling_layers}")
    print(f"  Hidden units: {args.hidden_dim}")
    print(f"  Summary dim: {args.summary_dim}")
    
    # 🔧 FIX 2: ADD SUMMARY NETWORK (CRITICAL!)
    # Summary network processes time series x into fixed-size representation
    summary_net = bf.networks.DeepSet(summary_dim=args.summary_dim)
    print(" Summary network created")
    
    # 1. Define the Inference Network (Normalizing Flow)
    inference_net = bf.networks.InvertibleNetwork(
        num_params=2,  # beta and gamma
        num_coupling_layers=args.num_coupling_layers,
        coupling_net_settings={
            "dense_args": {"units": args.hidden_dim, "activation": "relu"}
        }
    )
    print(" Inference network created")

    # 2. Set up the Amortized Posterior (WITH summary network)
    amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)  #  FIX 2: Added summary_net
    print(" Amortizer created")

    # 3. Configure the Trainer
    trainer = bf.trainers.Trainer(
        amortizer=amortizer,
        checkpoint_path=str(out_dir / "npe_checkpoint")  #  FIX 3: Use checkpoint_path instead of default_lr
    )
    print(" Trainer ready")

    print("\n" + "="*70)
    print("Starting offline training...")
    print("="*70)
    start = time.time()
    
    # 4. Execute Offline Training
    history = trainer.train_offline(
        simulations_dict=sim_data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    runtime_sec = time.time() - start

    print("\n" + "="*70)
    print(" Training complete!")
    print("="*70)
    print(f"Runtime: {runtime_sec:.2f}s ({runtime_sec/60:.1f} minutes)")

    # 5. Checkpoint is automatically saved by Trainer
    # No need for manual save_weights call

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
        "summary_dim": args.summary_dim,  #  FIX: Added
        "runtime_sec": float(runtime_sec),
        "runtime_minutes": float(runtime_sec / 60)  #  FIX: Added
    }
    
    metrics_path = out_dir / "npe_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n Saved checkpoint to: {out_dir / 'npe_checkpoint'}")
    print(f" Saved metrics to: {metrics_path}")
    print("\n" + "="*70)
    print("NPE training complete! ")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
