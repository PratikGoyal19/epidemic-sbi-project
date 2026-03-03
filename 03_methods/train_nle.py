"""
Train a Neural Likelihood Estimation (NLE) model for SIR data using BayesFlow.

Model:
- Learns p(x | theta), where:
  - theta = [beta, gamma] (shape: 2)
  - x = infected trajectory over time (shape: T)
- Uses BayesFlow's amortized likelihood with a conditional invertible network.

Outputs:
- 03_methods/artifacts/nle_checkpoint/ (TensorFlow checkpoint files)
- 03_methods/artifacts/nle_metrics.json
- 03_methods/artifacts/nle_normalization.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required for BayesFlow NLE training. "
        "Install it with: pip install tensorflow"
    ) from exc

try:
    import bayesflow as bf
except ImportError as exc:
    raise ImportError(
        "BayesFlow is required for NLE training. "
        "Install it with: pip install bayesflow==1.1.6"
    ) from exc


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def compute_standardization(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NLE on SIR dataset using BayesFlow"
    )
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
        help="Output directory for checkpoint and metrics.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-coupling-layers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--normalize-x",
        action="store_true",
        help="Standardize x with train-set mean/std before training.",
    )
    return parser.parse_args()


def _to_list(values: Any) -> list[float]:
    arr = np.asarray(values).reshape(-1)
    return [float(v) for v in arr.tolist()]


def coerce_history(raw_history: Any) -> dict[str, list[float]]:
    if raw_history is None:
        return {}

    if isinstance(raw_history, dict):
        coerced: dict[str, list[float]] = {}
        for key, value in raw_history.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                coerced[key] = _to_list(value)
        return coerced

    if hasattr(raw_history, "to_dict"):
        try:
            as_dict = raw_history.to_dict(orient="list")
            return {
                str(key): _to_list(value)
                for key, value in as_dict.items()
                if isinstance(value, (list, tuple, np.ndarray))
            }
        except TypeError:
            as_dict = raw_history.to_dict()
            if isinstance(as_dict, dict):
                return {
                    str(key): _to_list(value)
                    for key, value in as_dict.items()
                    if isinstance(value, (list, tuple, np.ndarray))
                }

    return {}


def pick_metric_series(
    history: dict[str, list[float]], candidates: list[str]
) -> tuple[str | None, list[float]]:
    for key in candidates:
        values = history.get(key, [])
        if values:
            return key, values
    return None, []


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_path = Path(args.data)
    out_dir = Path(args.artifacts_dir)
    checkpoint_path = out_dir / "nle_checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Training NLE with BayesFlow 1.1.6")
    print("=" * 70)
    print("Loading dataset...")
    theta, x = load_dataset(data_path)
    print(f" Loaded {len(theta):,} samples")
    print(f"   Theta shape: {theta.shape}")
    print(f"   X shape: {x.shape}")

    idx = np.arange(theta.shape[0])
    train_idx, val_idx = train_test_split(
        idx, test_size=args.val_fraction, random_state=args.seed, shuffle=True
    )
    train_theta, val_theta = theta[train_idx], theta[val_idx]
    train_x, val_x = x[train_idx], x[val_idx]

    x_mean = np.zeros((1, x.shape[1]), dtype=np.float32)
    x_std = np.ones((1, x.shape[1]), dtype=np.float32)
    if args.normalize_x:
        x_mean, x_std = compute_standardization(train_x)
        train_x = (train_x - x_mean) / x_std
        val_x = (val_x - x_mean) / x_std

    train_sims = {"prior_draws": train_theta, "sim_data": train_x}
    val_sims = {"prior_draws": val_theta, "sim_data": val_x}

    print("\nInitializing BayesFlow NLE...")
    print(f"  Coupling layers: {args.num_coupling_layers}")
    print(f"  Hidden units: {args.hidden_dim}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")

    likelihood_net = bf.networks.InvertibleNetwork(
        num_params=x.shape[1],
        num_coupling_layers=args.num_coupling_layers,
        coupling_settings={"dense_args": {"units": args.hidden_dim, "activation": "relu"}},
    )
    amortizer = bf.amortizers.AmortizedLikelihood(likelihood_net)
    configurator = bf.configuration.DefaultLikelihoodConfigurator()

    trainer = bf.trainers.Trainer(
        amortizer=amortizer,
        configurator=configurator,
        checkpoint_path=str(checkpoint_path),
        default_lr=args.lr,
    )

    if hasattr(tf.keras.optimizers, "AdamW"):
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    print("\n" + "=" * 70)
    print("Starting offline training...")
    print("=" * 70)
    start = time.time()
    raw_history = trainer.train_offline(
        simulations_dict=train_sims,
        validation_sims=val_sims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=optimizer,
    )
    runtime_sec = time.time() - start

    history = coerce_history(raw_history)
    train_key, train_series = pick_metric_series(
        history, ["loss", "train_loss", "train_losses"]
    )
    val_key, val_series = pick_metric_series(
        history, ["val_loss", "validation_loss", "val_losses", "validation_losses"]
    )

    if val_series:
        best_epoch = int(np.argmin(val_series) + 1)
        best_val_loss = float(np.min(val_series))
        final_train_loss = float(train_series[-1]) if train_series else float("nan")
        final_val_loss = float(val_series[-1])
    elif train_series:
        best_epoch = int(np.argmin(train_series) + 1)
        best_val_loss = float("nan")
        final_train_loss = float(train_series[-1])
        final_val_loss = float("nan")
    else:
        best_epoch = -1
        best_val_loss = float("nan")
        final_train_loss = float("nan")
        final_val_loss = float("nan")

    np.savez_compressed(
        out_dir / "nle_normalization.npz",
        normalize_x=np.array([args.normalize_x], dtype=np.bool_),
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        train_idx=train_idx.astype(np.int32),
        val_idx=val_idx.astype(np.int32),
    )

    metrics = {
        "dataset_path": str(data_path),
        "n_samples": int(theta.shape[0]),
        "trajectory_length_T": int(x.shape[1]),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "num_coupling_layers": args.num_coupling_layers,
        "hidden_dim": args.hidden_dim,
        "val_fraction": args.val_fraction,
        "normalize_x": args.normalize_x,
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "train_metric_key": train_key,
        "val_metric_key": val_key,
        "runtime_sec": float(runtime_sec),
        "history": history,
    }
    with open(out_dir / "nle_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f" Saved checkpoint to: {checkpoint_path}")
    print(f" Saved metrics to: {out_dir / 'nle_metrics.json'}")
    print(f" Saved normalization to: {out_dir / 'nle_normalization.npz'}")
    print(f" Runtime: {runtime_sec:.2f}s")


if __name__ == "__main__":
    main()
