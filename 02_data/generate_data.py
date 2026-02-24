"""
Generate training data for SIR simulation-based inference.

Outputs:
- theta: shape (n_samples, 2), columns [beta, gamma]
- x: shape (n_samples, T), infected trajectories
"""

from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util

import numpy as np
from tqdm import tqdm


def _load_sir_simulator():
    """Load SIRSimulator from 01_simulator/sir_model.py."""
    project_root = Path(__file__).resolve().parents[1]
    simulator_path = project_root / "01_simulator" / "sir_model.py"
    spec = importlib.util.spec_from_file_location("sir_model", simulator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load simulator from {simulator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SIRSimulator


def sample_prior(
    rng: np.random.Generator,
    n_samples: int,
    beta_min: float,
    beta_max: float,
    gamma_min: float,
    gamma_max: float,
) -> np.ndarray:
    """Sample parameters from independent uniform priors."""
    beta = rng.uniform(beta_min, beta_max, size=n_samples)
    gamma = rng.uniform(gamma_min, gamma_max, size=n_samples)
    return np.column_stack([beta, gamma])


def generate_dataset(
    n_samples: int,
    T: int,
    N: int,
    I0: int,
    R0: int,
    beta_min: float,
    beta_max: float,
    gamma_min: float,
    gamma_max: float,
    normalize_by_population: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate SIR trajectories for sampled parameters."""
    rng = np.random.default_rng(seed)
    theta = sample_prior(
        rng=rng,
        n_samples=n_samples,
        beta_min=beta_min,
        beta_max=beta_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )

    SIRSimulator = _load_sir_simulator()
    simulator = SIRSimulator(N=N, I0=I0, R0=R0, T=T)
    x = np.zeros((n_samples, T), dtype=np.float32)

    for i in tqdm(range(n_samples), desc="Simulating SIR data"):
        beta_i, gamma_i = theta[i]
        sim_seed = int(rng.integers(0, 2**32 - 1))
        trajectory = simulator.simulate(beta=beta_i, gamma=gamma_i, seed=sim_seed)
        if normalize_by_population:
            trajectory = trajectory / float(N)
        x[i] = trajectory.astype(np.float32)

    return theta.astype(np.float32), x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SIR SBI dataset")
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--T", type=int, default=160)
    parser.add_argument("--population", type=int, default=10_000)
    parser.add_argument("--i0", type=int, default=10)
    parser.add_argument("--r0", type=int, default=0)
    parser.add_argument("--beta-min", type=float, default=0.10)
    parser.add_argument("--beta-max", type=float, default=1.00)
    parser.add_argument("--gamma-min", type=float, default=0.01)
    parser.add_argument("--gamma-max", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalize-by-population",
        action="store_true",
        help="Store infected ratio I/N instead of absolute infected counts.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="02_data/sir_dataset.npz",
        help="Output .npz file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    theta, x = generate_dataset(
        n_samples=args.n_samples,
        T=args.T,
        N=args.population,
        I0=args.i0,
        R0=args.r0,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        normalize_by_population=args.normalize_by_population,
        seed=args.seed,
    )

    np.savez_compressed(
        out_path,
        theta=theta,
        x=x,
        n_samples=np.int32(args.n_samples),
        T=np.int32(args.T),
        N=np.int32(args.population),
        I0=np.int32(args.i0),
        R0=np.int32(args.r0),
    )

    print(f"Saved dataset to: {out_path}")
    print(f"theta shape: {theta.shape}")
    print(f"x shape: {x.shape}")


if __name__ == "__main__":
    main()
