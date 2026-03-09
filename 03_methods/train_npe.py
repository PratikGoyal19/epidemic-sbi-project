"""
Train a Neural Posterior Estimation (NPE) model for SIR data using BayesFlow.
Parameters inferred: beta, gamma, I0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import bayesflow as bf


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path)
    theta = data["theta"].astype(np.float32)
    x = data["x"].astype(np.float32)
    return theta, x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="02_data/sir_dataset.npz")
    parser.add_argument("--artifacts-dir", type=str, default="03_methods/artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-coupling-layers", type=int, default=4)
    parser.add_argument("--summary-dim", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data)
    out_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Training NPE with BayesFlow 1.1.6")
    print("Parameters: beta, gamma, I0")
    print("=" * 70)
    print("Loading dataset...")
    theta, x = load_dataset(data_path)
    print(f"  Loaded {len(theta):,} samples")
    print(f"  Theta shape: {theta.shape}  [beta, gamma, I0]")
    print(f"  X shape: {x.shape}")

    # BayesFlow expects 3D input for sequence: (batch, time_steps, features)
    x_3d = x[:, :, np.newaxis]
    print(f"  X reshaped to: {x_3d.shape}")

    sim_data = {
        "prior_draws": theta,
        "sim_data": x_3d
    }

    print(f"\nInitializing BayesFlow NPE...")
    print(f"  Coupling layers: {args.num_coupling_layers}")
    print(f"  Hidden units: {args.hidden_dim}")
    print(f"  Summary dim: {args.summary_dim}")

    summary_net = bf.networks.SequenceNetwork(
        summary_dim=args.summary_dim
    )
    print("  Summary network (SequenceNetwork) created")

    inference_net = bf.networks.InvertibleNetwork(
        num_params=3,  # beta, gamma, I0
        num_coupling_layers=args.num_coupling_layers,
        coupling_settings={
            "dense_args": {"units": args.hidden_dim, "activation": "relu"}
        }
    )
    print("  Inference network created (3 parameters: beta, gamma, I0)")

    amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)
    print("  Amortizer created")

    trainer = bf.trainers.Trainer(
        amortizer=amortizer,
        checkpoint_path=str(out_dir / "npe_checkpoint")
    )
    print("  Trainer ready")

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    start = time.time()

    history = trainer.train_offline(
        simulations_dict=sim_data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    runtime_sec = time.time() - start

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"Runtime: {runtime_sec:.2f}s ({runtime_sec/60:.1f} minutes)")

    metrics = {
        "n_samples": int(theta.shape[0]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_coupling_layers": args.num_coupling_layers,
        "hidden_dim": args.hidden_dim,
        "summary_dim": args.summary_dim,
        "runtime_sec": float(runtime_sec),
        "num_params": 3,
        "param_names": ["beta", "gamma", "I0"],
    }
    metrics_path = out_dir / "npe_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Checkpoint saved to: {out_dir / 'npe_checkpoint'}")
    print(f"  Metrics saved to: {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
