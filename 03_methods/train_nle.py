"""
Train a simple Neural Likelihood Estimation (NLE) model for SIR data.

Model:
- Learns p(x | theta), where:
  - theta = [beta, gamma] (shape: 2)
  - x = infected trajectory over time (shape: T)
- Uses a conditional diagonal-Gaussian likelihood parameterized by an MLP.

Outputs:
- 03_methods/artifacts/nle_model.pt
- 03_methods/artifacts/nle_metrics.json
- 03_methods/artifacts/nle_normalization.npz
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TrajectoryDataset(Dataset):
    """Torch dataset of (theta, x) pairs."""

    def __init__(self, theta: np.ndarray, x: np.ndarray) -> None:
        self.theta = torch.from_numpy(theta.astype(np.float32))
        self.x = torch.from_numpy(x.astype(np.float32))

    def __len__(self) -> int:
        return self.theta.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.theta[idx], self.x[idx]


class ConditionalGaussianLikelihood(nn.Module):
    """Maps theta -> Gaussian parameters (mu, log_sigma2) over x."""

    def __init__(self, theta_dim: int, x_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(theta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, x_dim)
        self.logvar_head = nn.Linear(hidden_dim, x_dim)

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(theta)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(min=-8.0, max=8.0)
        return mu, logvar


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gaussian_nll(
    x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Average NLL under diagonal Gaussian."""
    nll = 0.5 * (np.log(2.0 * np.pi) + logvar + ((x - mu) ** 2) / torch.exp(logvar))
    return nll.sum(dim=1).mean()


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


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for theta, x in loader:
        theta = theta.to(device)
        x = x.to(device)
        optimizer.zero_grad()
        mu, logvar = model(theta)
        loss = gaussian_nll(x, mu, logvar)
        loss.backward()
        optimizer.step()
        batch_size = theta.shape[0]
        total_loss += loss.item() * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    for theta, x in loader:
        theta = theta.to(device)
        x = x.to(device)
        mu, logvar = model(theta)
        loss = gaussian_nll(x, mu, logvar)
        batch_size = theta.shape[0]
        total_loss += loss.item() * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NLE on SIR dataset")
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
        help="Output directory for model and metrics.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--normalize-x",
        action="store_true",
        help="Standardize x with train-set mean/std before training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    data_path = Path(args.data)
    out_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    theta, x = load_dataset(data_path)

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

    train_ds = TrajectoryDataset(train_theta, train_x)
    val_ds = TrajectoryDataset(val_theta, val_x)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ConditionalGaussianLikelihood(
        theta_dim=2, x_dim=x.shape[1], hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val = float("inf")
    best_epoch = -1
    history: dict[str, list[float]] = {"train_nll": [], "val_nll": []}
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_nll = train_epoch(model, train_loader, optimizer, device)
        val_nll = eval_epoch(model, val_loader, device)
        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)

        if val_nll < best_val:
            best_val = val_nll
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "nle_model.pt")

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"[Epoch {epoch:03d}/{args.epochs}] "
                f"train_nll={train_nll:.4f} val_nll={val_nll:.4f}"
            )

    runtime_sec = time.time() - start

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
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "val_fraction": args.val_fraction,
        "normalize_x": args.normalize_x,
        "best_epoch": best_epoch,
        "best_val_nll": float(best_val),
        "final_train_nll": float(history["train_nll"][-1]),
        "final_val_nll": float(history["val_nll"][-1]),
        "runtime_sec": float(runtime_sec),
        "history": history,
    }
    with open(out_dir / "nle_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved best model to: {out_dir / 'nle_model.pt'}")
    print(f"Saved metrics to: {out_dir / 'nle_metrics.json'}")
    print(f"Saved normalization to: {out_dir / 'nle_normalization.npz'}")
    print(f"Best val NLL: {best_val:.4f} at epoch {best_epoch}")
    print(f"Runtime: {runtime_sec:.2f}s")


if __name__ == "__main__":
    main()
