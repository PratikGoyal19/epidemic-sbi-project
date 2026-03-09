"""
04_evaluation/metrics.py

Evaluation and comparison of NPE vs NLE for SIR parameter estimation.
Metrics: MAE, Coverage, Posterior Recovery Plots, SBC

Team: Pratik Goyal, Suryansh Chaturvedi, Mayank Choudhary
Course: Generative Neural Networks for the Sciences
University of Heidelberg, Winter 2025/26
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json
import importlib.util
import bayesflow as bf
import tensorflow as tf


# ============================================================================
# CONFIG
# ============================================================================

ARTIFACTS_DIR = Path("03_methods/artifacts")
DATA_PATH     = Path("02_data/sir_dataset.npz")
OUT_DIR       = Path("04_evaluation/results")
N_TEST        = 200       # number of test samples
N_POSTERIOR   = 500       # posterior samples per observation
CREDIBLE_LEVELS = [0.5, 0.9]  # credible interval levels for coverage
SEED          = 42


# ============================================================================
# HELPERS
# ============================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(path: Path, n_test: int, seed: int):
    """Load dataset and return a held-out test split."""
    data = np.load(path)
    theta = data["theta"].astype(np.float32)
    x     = data["x"].astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(theta), size=n_test, replace=False)
    return theta[idx], x[idx]


def _load_sir_simulator():
    project_root = Path(__file__).resolve().parents[1]
    simulator_path = project_root / "01_simulator" / "sir_model.py"
    spec = importlib.util.spec_from_file_location("sir_model", simulator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SIRSimulator


# ============================================================================
# LOAD MODELS
# ============================================================================

def load_npe(artifacts_dir: Path):
    """Reconstruct and load the trained NPE model."""
    print("  Loading NPE...")
    summary_net   = bf.networks.SequenceNetwork(summary_dim=32)
    inference_net = bf.networks.InvertibleNetwork(
        num_params=2,
        num_coupling_layers=4,
        coupling_settings={"dense_args": {"units": 128, "activation": "relu"}}
    )
    amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)
    trainer = bf.trainers.Trainer(
        amortizer=amortizer,
        checkpoint_path=str(artifacts_dir / "npe_checkpoint")
    )
    print("  NPE loaded successfully")
    return amortizer


def load_nle(artifacts_dir: Path):
    """Reconstruct and load the trained NLE model."""
    print("  Loading NLE...")
    likelihood_net = bf.networks.InvertibleNetwork(
        num_params=160,
        num_coupling_layers=4,
        coupling_settings={"dense_args": {"units": 128, "activation": "relu"}}
    )
    amortizer = bf.amortizers.AmortizedLikelihood(likelihood_net)
    trainer = bf.trainers.Trainer(
        amortizer=amortizer,
        checkpoint_path=str(artifacts_dir / "nle_checkpoint")
    )

    # Load normalization stats
    norm_path = artifacts_dir / "nle_normalization.npz"
    norm = np.load(norm_path)
    x_mean = norm["x_mean"].astype(np.float32)
    x_std  = norm["x_std"].astype(np.float32)

    print("  NLE loaded successfully")
    return amortizer, x_mean, x_std


# ============================================================================
# POSTERIOR SAMPLING
# ============================================================================

def get_npe_posterior_samples(amortizer, x: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Sample from NPE posterior.
    x shape: (n_test, 160)
    returns: (n_test, n_samples, 2)
    """
    x_3d = x[:, :, np.newaxis]  # (n_test, 160, 1)
    samples = amortizer.sample({"summary_conditions": x_3d}, n_samples=n_samples)
    return samples  # (n_test, n_samples, 2)


def get_nle_posterior_samples(
    amortizer, x: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray,
    n_samples: int, seed: int
) -> np.ndarray:
    """
    Sample from NLE posterior using rejection sampling with a uniform prior.
    x shape: (n_test, 160)
    returns: (n_test, n_samples, 2)
    """
    rng = np.random.default_rng(seed)
    n_test = x.shape[0]

    # Normalize x
    x_norm = (x - x_mean) / (x_std + 1e-8)

    # Prior bounds
    beta_min,  beta_max  = 0.10, 0.60
    gamma_min, gamma_max = 0.01, 0.10

    all_samples = np.zeros((n_test, n_samples, 2), dtype=np.float32)

    for i in range(n_test):
        collected = []
        batch     = n_samples * 20  # oversample for rejection

        while len(collected) < n_samples:
            # Sample from prior
            betas  = rng.uniform(beta_min,  beta_max,  size=batch).astype(np.float32)
            gammas = rng.uniform(gamma_min, gamma_max, size=batch).astype(np.float32)
            theta_prop = np.column_stack([betas, gammas])  # (batch, 2)

            # Tile observation
            x_rep = np.tile(x_norm[i], (batch, 1))  # (batch, 160)

            # Compute log-likelihood under NLE
            # BayesFlow 1.1.6 AmortizedLikelihood.log_likelihood uses "observables" and "conditions"
            input_dict = {
                "observables":  x_rep,
                "conditions":   theta_prop,
            }
            log_liks = amortizer.log_likelihood(input_dict)  # (batch,)
            log_liks = np.array(log_liks).flatten()

            # Importance weights (unnormalized)
            log_liks -= np.max(log_liks)
            weights = np.exp(log_liks)
            weights /= weights.sum()

            # Resample
            chosen_idx = rng.choice(batch, size=min(n_samples, batch), replace=True, p=weights)
            collected.append(theta_prop[chosen_idx])

        all_samples[i] = np.concatenate(collected, axis=0)[:n_samples]

        if (i + 1) % 25 == 0:
            print(f"    NLE posterior sampling: {i+1}/{n_test}")

    return all_samples


# ============================================================================
# METRICS
# ============================================================================

def compute_mae(theta_true: np.ndarray, posterior_samples: np.ndarray) -> dict:
    """
    Compute Mean Absolute Error using posterior mean as point estimate.
    theta_true:        (n_test, 2)
    posterior_samples: (n_test, n_samples, 2)
    """
    posterior_mean = posterior_samples.mean(axis=1)  # (n_test, 2)
    mae_beta  = float(np.mean(np.abs(posterior_mean[:, 0] - theta_true[:, 0])))
    mae_gamma = float(np.mean(np.abs(posterior_mean[:, 1] - theta_true[:, 1])))
    mae_mean  = float((mae_beta + mae_gamma) / 2)
    return {"mae_beta": mae_beta, "mae_gamma": mae_gamma, "mae_mean": mae_mean}


def compute_coverage(theta_true: np.ndarray, posterior_samples: np.ndarray,
                     levels: list) -> dict:
    """
    Compute empirical coverage at given credible interval levels.
    Ideal coverage at level α = α (e.g. 90% CI should contain true value 90% of the time).
    """
    results = {}
    for level in levels:
        alpha = (1 - level) / 2
        lower = np.quantile(posterior_samples, alpha,     axis=1)  # (n_test, 2)
        upper = np.quantile(posterior_samples, 1 - alpha, axis=1)  # (n_test, 2)

        in_interval = (theta_true >= lower) & (theta_true <= upper)  # (n_test, 2)
        cov_beta  = float(in_interval[:, 0].mean())
        cov_gamma = float(in_interval[:, 1].mean())
        cov_mean  = float(in_interval.mean())

        results[f"coverage_{int(level*100)}"] = {
            "beta":  cov_beta,
            "gamma": cov_gamma,
            "mean":  cov_mean
        }
    return results


def compute_rmse(theta_true: np.ndarray, posterior_samples: np.ndarray) -> dict:
    """Compute RMSE using posterior mean as point estimate."""
    posterior_mean = posterior_samples.mean(axis=1)
    rmse_beta  = float(np.sqrt(np.mean((posterior_mean[:, 0] - theta_true[:, 0])**2)))
    rmse_gamma = float(np.sqrt(np.mean((posterior_mean[:, 1] - theta_true[:, 1])**2)))
    return {"rmse_beta": rmse_beta, "rmse_gamma": rmse_gamma}


# ============================================================================
# PLOTS
# ============================================================================

def plot_posterior_recovery(theta_true: np.ndarray, posterior_samples: np.ndarray,
                             method_name: str, out_dir: Path) -> None:
    """
    Posterior recovery plot: true vs posterior mean with uncertainty.
    """
    param_names  = ["β (beta)", "γ (gamma)"]
    param_labels = ["beta", "gamma"]
    posterior_mean = posterior_samples.mean(axis=1)
    posterior_std  = posterior_samples.std(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{method_name} — Posterior Recovery", fontsize=14, fontweight="bold")

    for j, (name, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[j]
        ax.errorbar(
            theta_true[:, j], posterior_mean[:, j],
            yerr=posterior_std[:, j],
            fmt="o", alpha=0.4, markersize=3, linewidth=0.5,
            color="steelblue", ecolor="lightblue", label="Posterior mean ± std"
        )
        # Identity line
        lo = theta_true[:, j].min()
        hi = theta_true[:, j].max()
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect recovery")
        ax.set_xlabel(f"True {name}", fontsize=11)
        ax.set_ylabel(f"Predicted {name}", fontsize=11)
        ax.set_title(name, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"{method_name.lower()}_posterior_recovery.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_npe_vs_nle_comparison(
    theta_true: np.ndarray,
    npe_samples: np.ndarray,
    nle_samples: np.ndarray,
    out_dir: Path
) -> None:
    """
    Side-by-side comparison of NPE vs NLE posterior recovery.
    """
    methods      = ["NPE", "NLE"]
    all_samples  = [npe_samples, nle_samples]
    colors       = ["steelblue", "darkorange"]
    param_names  = ["β (beta)", "γ (gamma)"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NPE vs NLE — Posterior Recovery Comparison", fontsize=14, fontweight="bold")

    for j, param in enumerate(param_names):
        for m, (method, samples, color) in enumerate(zip(methods, all_samples, colors)):
            ax = axes[j][m]
            post_mean = samples.mean(axis=1)
            post_std  = samples.std(axis=1)

            ax.errorbar(
                theta_true[:, j], post_mean[:, j],
                yerr=post_std[:, j],
                fmt="o", alpha=0.4, markersize=3, linewidth=0.5,
                color=color, ecolor="lightgray"
            )
            lo = theta_true[:, j].min()
            hi = theta_true[:, j].max()
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=2)
            ax.set_xlabel(f"True {param}", fontsize=10)
            ax.set_ylabel(f"Predicted {param}", fontsize=10)
            ax.set_title(f"{method} — {param}", fontsize=11)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "npe_vs_nle_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_sbc(posterior_samples: np.ndarray, theta_true: np.ndarray,
             method_name: str, out_dir: Path) -> None:
    """
    Simulation-Based Calibration (SBC) rank histogram.
    For a well-calibrated posterior, ranks should be uniform.
    """
    param_names  = ["β (beta)", "γ (gamma)"]
    param_labels = ["beta", "gamma"]
    n_test, n_samples, _ = posterior_samples.shape

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{method_name} — SBC Rank Histograms", fontsize=13, fontweight="bold")

    for j, (name, label) in enumerate(zip(param_names, param_labels)):
        ranks = np.array([
            np.sum(posterior_samples[i, :, j] < theta_true[i, j])
            for i in range(n_test)
        ])
        ax = axes[j]
        n_bins = 20
        ax.hist(ranks, bins=n_bins, density=True,
                color="steelblue", edgecolor="white", alpha=0.8)
        # Expected uniform line — density = 1/n_bins matches histogram scale
        ax.axhline(1.0 / n_bins, color="red", linestyle="--",
                   linewidth=2, label="Uniform (ideal)")
        ax.set_xlabel("Rank", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{name}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"{method_name.lower()}_sbc.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_metrics_summary(npe_metrics: dict, nle_metrics: dict, out_dir: Path) -> None:
    """
    Bar chart comparing MAE and coverage between NPE and NLE.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("NPE vs NLE — Metrics Summary", fontsize=14, fontweight="bold")

    methods = ["NPE", "NLE"]
    colors  = ["steelblue", "darkorange"]

    # MAE beta
    ax = axes[0]
    vals = [npe_metrics["mae"]["mae_beta"], nle_metrics["mae"]["mae_beta"]]
    bars = ax.bar(methods, vals, color=colors, alpha=0.8, edgecolor="black")
    ax.set_title("MAE — β (beta)", fontsize=12)
    ax.set_ylabel("MAE", fontsize=11)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # MAE gamma
    ax = axes[1]
    vals = [npe_metrics["mae"]["mae_gamma"], nle_metrics["mae"]["mae_gamma"]]
    bars = ax.bar(methods, vals, color=colors, alpha=0.8, edgecolor="black")
    ax.set_title("MAE — γ (gamma)", fontsize=12)
    ax.set_ylabel("MAE", fontsize=11)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # 90% Coverage
    ax = axes[2]
    vals = [
        npe_metrics["coverage"]["coverage_90"]["mean"],
        nle_metrics["coverage"]["coverage_90"]["mean"]
    ]
    bars = ax.bar(methods, vals, color=colors, alpha=0.8, edgecolor="black")
    ax.axhline(0.90, color="red", linestyle="--", linewidth=2, label="Ideal (90%)")
    ax.set_title("90% Credible Interval Coverage", fontsize=12)
    ax.set_ylabel("Coverage", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = out_dir / "metrics_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EVALUATION: NPE vs NLE for SIR Parameter Estimation")
    print("=" * 70)

    # ── Load test data ──────────────────────────────────────────────────────
    print("\nLoading test data...")
    theta_true, x_test = load_dataset(DATA_PATH, N_TEST, SEED)
    print(f"  Test samples: {N_TEST}")
    print(f"  theta shape: {theta_true.shape}")
    print(f"  x shape:     {x_test.shape}")

    # ── Load models ─────────────────────────────────────────────────────────
    print("\nLoading trained models...")
    npe_amortizer = load_npe(ARTIFACTS_DIR)
    nle_amortizer, x_mean, x_std = load_nle(ARTIFACTS_DIR)

    # ── Sample posteriors ───────────────────────────────────────────────────
    print(f"\nSampling NPE posteriors ({N_POSTERIOR} samples x {N_TEST} observations)...")
    npe_samples = get_npe_posterior_samples(npe_amortizer, x_test, N_POSTERIOR)
    print(f"  NPE samples shape: {npe_samples.shape}")

    print(f"\nSampling NLE posteriors ({N_POSTERIOR} samples x {N_TEST} observations)...")
    nle_samples = get_nle_posterior_samples(
        nle_amortizer, x_test, x_mean, x_std, N_POSTERIOR, SEED
    )
    print(f"  NLE samples shape: {nle_samples.shape}")

    # ── Compute metrics ─────────────────────────────────────────────────────
    print("\nComputing metrics...")

    npe_mae      = compute_mae(theta_true, npe_samples)
    npe_coverage = compute_coverage(theta_true, npe_samples, CREDIBLE_LEVELS)
    npe_rmse     = compute_rmse(theta_true, npe_samples)

    nle_mae      = compute_mae(theta_true, nle_samples)
    nle_coverage = compute_coverage(theta_true, nle_samples, CREDIBLE_LEVELS)
    nle_rmse     = compute_rmse(theta_true, nle_samples)

    npe_metrics = {"mae": npe_mae, "coverage": npe_coverage, "rmse": npe_rmse}
    nle_metrics = {"mae": nle_mae, "coverage": nle_coverage, "rmse": nle_rmse}

    # ── Print results table ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<35} {'NPE':>12} {'NLE':>12}")
    print("-" * 60)
    print(f"{'MAE beta':<35} {npe_mae['mae_beta']:>12.4f} {nle_mae['mae_beta']:>12.4f}")
    print(f"{'MAE gamma':<35} {npe_mae['mae_gamma']:>12.4f} {nle_mae['mae_gamma']:>12.4f}")
    print(f"{'MAE mean':<35} {npe_mae['mae_mean']:>12.4f} {nle_mae['mae_mean']:>12.4f}")
    print(f"{'RMSE beta':<35} {npe_rmse['rmse_beta']:>12.4f} {nle_rmse['rmse_beta']:>12.4f}")
    print(f"{'RMSE gamma':<35} {npe_rmse['rmse_gamma']:>12.4f} {nle_rmse['rmse_gamma']:>12.4f}")
    print(f"{'50% Coverage beta':<35} {npe_coverage['coverage_50']['beta']:>12.3f} {nle_coverage['coverage_50']['beta']:>12.3f}")
    print(f"{'50% Coverage gamma':<35} {npe_coverage['coverage_50']['gamma']:>12.3f} {nle_coverage['coverage_50']['gamma']:>12.3f}")
    print(f"{'90% Coverage beta':<35} {npe_coverage['coverage_90']['beta']:>12.3f} {nle_coverage['coverage_90']['beta']:>12.3f}")
    print(f"{'90% Coverage gamma':<35} {npe_coverage['coverage_90']['gamma']:>12.3f} {nle_coverage['coverage_90']['gamma']:>12.3f}")
    print("=" * 70)

    # ── Save metrics JSON ───────────────────────────────────────────────────
    all_metrics = {
        "n_test": N_TEST,
        "n_posterior_samples": N_POSTERIOR,
        "NPE": npe_metrics,
        "NLE": nle_metrics
    }
    metrics_path = OUT_DIR / "comparison_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved to: {metrics_path}")

    # ── Generate plots ──────────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_posterior_recovery(theta_true, npe_samples, "NPE", OUT_DIR)
    plot_posterior_recovery(theta_true, nle_samples, "NLE", OUT_DIR)
    plot_npe_vs_nle_comparison(theta_true, npe_samples, nle_samples, OUT_DIR)
    plot_sbc(npe_samples, theta_true, "NPE", OUT_DIR)
    plot_sbc(nle_samples, theta_true, "NLE", OUT_DIR)
    plot_metrics_summary(npe_metrics, nle_metrics, OUT_DIR)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {OUT_DIR}/")
    print("  - comparison_metrics.json")
    print("  - npe_posterior_recovery.png")
    print("  - nle_posterior_recovery.png")
    print("  - npe_vs_nle_comparison.png")
    print("  - npe_sbc.png")
    print("  - nle_sbc.png")
    print("  - metrics_summary.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
