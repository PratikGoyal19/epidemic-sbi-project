"""
Train a Neural Posterior Estimation (NPE) model for SIR data using BayesFlow.

Model:
- Learns p(theta | x), where:
  - theta = [beta, gamma] (shape: 2)
  - x = infected trajectory over time (shape: T)
- Uses BayesFlow's AmortizedPosterior with Invertible Network (Normalizing Flow).

Outputs:
- 03_methods/artifacts/npe_checkpoint/ (TensorFlow Checkpoint)
- 03_methods/artifacts/npe_metrics.json
- 03_methods/artifacts/npe_history.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# BayesFlow imports
import bayesflow as bf
from bayesflow.networks import InvertibleNetwork, DeepSet
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.trainers import Trainer


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the SIR dataset."""
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


def prepare_bayesflow_data(theta: np.ndarray, x: np.ndarray, val_fraction: float = 0.1):
    """
    Prepare data in BayesFlow format for offline training.
    
    BayesFlow expects dictionaries with specific keys:
    - "parameters": parameter values (theta)
    - "summary_conditions": observed data (x)
    """
    n_total = len(theta)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val
    
    # Split into train/val
    train_data = {
        "parameters": theta[:n_train],
        "summary_conditions": x[:n_train]
    }
    
    val_data = {
        "parameters": theta[n_train:],
        "summary_conditions": x[n_train:]
    }
    
    print(f"Training samples: {n_train:,}")
    print(f"Validation samples: {n_val:,}")
    
    return train_data, val_data


def create_npe_amortizer(
    num_params: int = 2,
    summary_dim: int = 32,
    num_coupling_layers: int = 4,
    coupling_hidden_units: int = 128
):
    """
    Create NPE amortizer with summary network and inference network.
    
    Architecture:
    1. Summary Network: Processes time series x → summary vector
    2. Inference Network: Normalizing flow for p(theta | summary)
    """
    print("\n🔧 Building NPE Architecture...")
    
    # 1. Summary Network (processes time series observations)
    summary_net = DeepSet(
        summary_dim=summary_dim,
        num_dense_s1=2,  # Layers before pooling
        num_dense_s2=2,  # Layers after pooling
        num_dense_s3=2   # Final dense layers
    )
    print(f"    Summary network: DeepSet with output dim={summary_dim}")
    
    # 2. Inference Network (normalizing flow)
    inference_net = InvertibleNetwork(
        num_params=num_params,
        num_coupling_layers=num_coupling_layers,
        coupling_settings={
            "dense_args": {
                "units": coupling_hidden_units,
                "activation": "relu"
            }
        }
    )
    print(f"    Inference network: {num_coupling_layers} coupling layers, {coupling_hidden_units} hidden units")
    
    # 3. Amortized Posterior (combines both networks)
    amortizer = AmortizedPosterior(inference_net, summary_net)
    print(f"    Amortized Posterior created\n")
    
    return amortizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NPE on SIR dataset using BayesFlow"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="02_data/sir_dataset.npz",
        help="Path to generated dataset (.npz)"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="03_methods/artifacts",
        help="Output directory for checkpoints and metrics"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation"
    )
    parser.add_argument(
        "--summary-dim",
        type=int,
        default=32,
        help="Output dimension of summary network"
    )
    parser.add_argument(
        "--num-coupling-layers",
        type=int,
        default=4,
        help="Number of coupling layers in normalizing flow"
    )
    parser.add_argument(
        "--coupling-hidden-units",
        type=int,
        default=128,
        help="Hidden units in coupling networks"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Set seeds
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup paths
    data_path = Path(args.data)
    out_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(" TRAINING NPE WITH BAYESFLOW")
    print("="*70)
    print(f"Dataset: {data_path}")
    print(f"Output: {out_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*70 + "\n")
    
    # Load data
    print(" Loading dataset...")
    theta, x = load_dataset(data_path)
    print(f"   Loaded {len(theta):,} samples")
    print(f"   Theta shape: {theta.shape} (beta, gamma)")
    print(f"   X shape: {x.shape} (time series)\n")
    
    # Prepare for BayesFlow
    print(" Preparing data for BayesFlow...")
    train_data, val_data = prepare_bayesflow_data(theta, x, args.val_fraction)
    print()
    
    # Create amortizer
    amortizer = create_npe_amortizer(
        num_params=2,
        summary_dim=args.summary_dim,
        num_coupling_layers=args.num_coupling_layers,
        coupling_hidden_units=args.coupling_hidden_units
    )
    
    # Create trainer
    print(" Initializing trainer...")
    trainer = Trainer(
        amortizer=amortizer,
        checkpoint_path=str(out_dir / "npe_checkpoint")
    )
    print("    Trainer ready\n")
    
    # Train
    print("="*70)
    print(" STARTING TRAINING")
    print("="*70)
    print(f"  Estimated time: {args.epochs * len(train_data['parameters']) / (args.batch_size * 60):.1f} minutes\n")
    
    start_time = time.time()
    
    history = trainer.train_offline(
        simulations_dict=train_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_sims=val_data
    )
    
    runtime_sec = time.time() - start_time
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETE!")
    print("="*70)
    print(f"Runtime: {runtime_sec:.2f}s ({runtime_sec/60:.1f} minutes)")
    print()
    
    # Save metrics
    metrics = {
        "dataset_path": str(data_path),
        "n_samples": int(theta.shape[0]),
        "n_train": len(train_data["parameters"]),
        "n_val": len(val_data["parameters"]),
        "trajectory_length_T": int(x.shape[1]),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "val_fraction": args.val_fraction,
        "summary_dim": args.summary_dim,
        "num_coupling_layers": args.num_coupling_layers,
        "coupling_hidden_units": args.coupling_hidden_units,
        "runtime_sec": float(runtime_sec),
        "runtime_minutes": float(runtime_sec / 60),
        "final_train_loss": float(history['train_losses'][-1]) if history else None,
        "final_val_loss": float(history['val_losses'][-1]) if history and 'val_losses' in history else None
    }
    
    metrics_path = out_dir / "npe_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    # Save training history
    if history:
        history_path = out_dir / "npe_history.json"
        history_dict = {k: [float(v) for v in vals] for k, vals in history.items()}
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_dict, f, indent=2)
        print(f" Saved training history to: {history_path}")
    
    print(f" Saved metrics to: {metrics_path}")
    print(f" Saved checkpoint to: {out_dir / 'npe_checkpoint'}")
    
    print("\n" + "="*70)
    print(" NPE TRAINING COMPLETE!")
    print("="*70)
    print("\n Next steps:")
    print("   1. Train NLE model (train_nle.py)")
    print("   2. Compare NPE vs NLE performance")
    print("   3. Run evaluation metrics")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
