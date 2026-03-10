"""
04_evaluation/real_data.py

Apply trained NPE and NLE models to real COVID-19 data (Italy, first wave).
Estimates SIR parameters (beta, gamma, I0) from real observed case counts.

Team: Pratik Goyal, Suryansh Chaturvedi, Mayank Choudhary
Course: Generative Neural Networks for the Sciences
University of Heidelberg, Winter 2025/26
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import bayesflow as bf
import tensorflow as tf


# ============================================================================
# CONFIG
# ============================================================================

DATA_CSV      = Path("02_data/owid-covid-data.csv")
ARTIFACTS_DIR = Path("03_methods/artifacts")
OUT_DIR       = Path("04_evaluation/results/real_data")

COUNTRY    = "Italy"
WAVE_START = "2020-02-23"
T          = 160
N_POSTERIOR = 1000

# Prior bounds — must match training
BETA_MIN,  BETA_MAX  = 0.10, 0.60
GAMMA_MIN, GAMMA_MAX = 0.01, 0.10
I0_MIN,    I0_MAX    = 1.0,  50.0

SEED = 42


# ============================================================================
# LOAD AND PREPROCESS REAL DATA
# ============================================================================

def load_italy_wave(csv_path: Path, country: str, wave_start: str, T: int) -> tuple:
    """
    Extract T days of smoothed new cases starting from wave_start.
    Rescales to match SIR model population of N=10,000.
    Returns: scaled_cases (T,), real_population (float)
    """
    df = pd.read_csv(csv_path)
    country_df = df[df["location"] == country][
        ["date", "new_cases_smoothed", "population"]
    ].dropna().reset_index(drop=True)

    country_df["date"] = pd.to_datetime(country_df["date"])
    wave_df = country_df[country_df["date"] >= pd.to_datetime(wave_start)].head(T).reset_index(drop=True)

    if len(wave_df) < T:
        raise ValueError(f"Not enough data: got {len(wave_df)} days, need {T}")

    real_population = float(wave_df["population"].iloc[0])
    raw_cases       = np.maximum(wave_df["new_cases_smoothed"].values.astype(np.float32), 0)

    model_N      = 10000.0
    scaled_cases = np.maximum(raw_cases * (model_N / real_population), 0)

    print(f"  Country: {country}")
    print(f"  Wave start: {wave_start}")
    print(f"  Days extracted: {len(wave_df)}")
    print(f"  Real population: {real_population:,.0f}")
    print(f"  Raw peak cases/day: {raw_cases.max():.0f}")
    print(f"  Scaled peak cases/day (N=10k): {scaled_cases.max():.4f}")

    return scaled_cases, real_population


# ============================================================================
# LOAD MODELS
# ============================================================================

def load_npe(artifacts_dir: Path):
    summary_net   = bf.networks.SequenceNetwork(summary_dim=32)
    inference_net = bf.networks.InvertibleNetwork(
        num_params=3,  # beta, gamma, I0
        num_coupling_layers=4,
        coupling_settings={"dense_args": {"units": 128, "activation": "relu"}}
    )
    amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)
    bf.trainers.Trainer(
        amortizer=amortizer,
        checkpoint_path=str(artifacts_dir / "npe_checkpoint")
    )
    return amortizer


def load_nle(artifacts_dir: Path):
    likelihood_net = bf.networks.InvertibleNetwork(
        num_params=160,
        num_coupling_layers=4,
        coupling_settings={"dense_args": {"units": 128, "activation": "relu"}}
    )
    amortizer = bf.amortizers.AmortizedLikelihood(likelihood_net)
    bf.trainers.Trainer(
        amortizer=amortizer,
        checkpoint_path=str(artifacts_dir / "nle_checkpoint")
    )
    norm   = np.load(artifacts_dir / "nle_normalization.npz")
    x_mean = norm["x_mean"].astype(np.float32)
    x_std  = norm["x_std"].astype(np.float32)
    return amortizer, x_mean, x_std


# ============================================================================
# POSTERIOR SAMPLING
# ============================================================================

def sample_npe_posterior(amortizer, x: np.ndarray, n_samples: int) -> np.ndarray:
    """
    x: (T,) → returns (n_samples, 3)  [beta, gamma, I0]
    """
    x_3d    = x[np.newaxis, :, np.newaxis].astype(np.float32)  # (1, T, 1)
    raw     = amortizer.sample({"summary_conditions": x_3d}, n_samples=n_samples)
    samples = np.array(raw)

    if samples.ndim == 3 and samples.shape[0] == 1:
        samples = samples[0]

    assert samples.shape == (n_samples, 3), \
        f"Unexpected NPE samples shape: {samples.shape}, expected ({n_samples}, 3)"

    return samples  # (n_samples, 3)


def sample_nle_posterior(amortizer, x: np.ndarray, x_mean: np.ndarray,
                          x_std: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """
    Importance-weighted resampling using NLE.
    x: (T,) → returns (n_samples, 3)  [beta, gamma, I0]
    """
    rng    = np.random.default_rng(seed)
    x_norm = ((x - x_mean) / (x_std + 1e-8)).astype(np.float32)

    batch        = n_samples * 20
    max_attempts = 10
    result       = None

    for attempt in range(max_attempts):
        betas  = rng.uniform(BETA_MIN,  BETA_MAX,  size=batch).astype(np.float32)
        gammas = rng.uniform(GAMMA_MIN, GAMMA_MAX, size=batch).astype(np.float32)
        i0s    = rng.uniform(I0_MIN,    I0_MAX,    size=batch).astype(np.float32)
        theta_prop = np.column_stack([betas, gammas, i0s])  # (batch, 3)

        x_rep    = np.tile(x_norm, (batch, 1))
        log_liks = np.array(amortizer.log_likelihood({
            "observables": x_rep,
            "conditions":  theta_prop,
        })).flatten()

        if np.any(np.isnan(log_liks)) or np.all(log_liks == log_liks[0]):
            print(f"  Warning: degenerate log-likelihoods on attempt {attempt+1}, retrying...")
            continue

        log_liks -= np.max(log_liks)
        weights   = np.exp(log_liks)
        weight_sum = weights.sum()

        if weight_sum == 0 or np.isnan(weight_sum):
            print(f"  Warning: zero weight sum on attempt {attempt+1}, retrying...")
            continue

        weights /= weight_sum
        chosen_idx = rng.choice(batch, size=n_samples, replace=True, p=weights)
        result = theta_prop[chosen_idx]
        break

    if result is None:
        print("  Warning: all attempts failed, falling back to uniform sampling")
        betas  = rng.uniform(BETA_MIN,  BETA_MAX,  size=n_samples).astype(np.float32)
        gammas = rng.uniform(GAMMA_MIN, GAMMA_MAX, size=n_samples).astype(np.float32)
        i0s    = rng.uniform(I0_MIN,    I0_MAX,    size=n_samples).astype(np.float32)
        result = np.column_stack([betas, gammas, i0s])

    return result  # (n_samples, 3)


# ============================================================================
# SIR FORWARD MODEL (for posterior predictive check)
# ============================================================================

def run_sir(beta: float, gamma: float, N: float = 10000,
            I0: float = 1.0, T: int = 160) -> np.ndarray:
    """Run deterministic SIR model, return I(t)."""
    from scipy.integrate import odeint

    def deriv(y, t):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt =  beta * S * I / N - gamma * I
        dRdt =  gamma * I
        return dSdt, dIdt, dRdt

    y0  = (N - I0, I0, 0.0)
    t   = np.linspace(0, T - 1, T)
    sol = odeint(deriv, y0, t)
    return sol[:, 1]


# ============================================================================
# PLOTS
# ============================================================================

def plot_posterior_distributions(npe_samples: np.ndarray, nle_samples: np.ndarray,
                                  out_dir: Path) -> None:
    """Plot posterior distributions of beta, gamma, I0 for NPE and NLE."""
    param_names = ["β (infection rate)", "γ (recovery rate)", "I₀ (initial infected)"]
    methods     = ["NPE", "NLE"]
    all_samples = [npe_samples, nle_samples]
    colors      = ["steelblue", "darkorange"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("Posterior Distributions — Italy COVID-19 First Wave\n(NPE vs NLE)",
                 fontsize=14, fontweight="bold")

    for j, param in enumerate(param_names):
        for m, (method, samples, color) in enumerate(zip(methods, all_samples, colors)):
            ax         = axes[j][m]
            mean_val   = samples[:, j].mean()
            median_val = np.median(samples[:, j])
            ci_low     = np.percentile(samples[:, j], 5)
            ci_high    = np.percentile(samples[:, j], 95)

            ax.hist(samples[:, j], bins=40, color=color, alpha=0.8,
                    edgecolor="white", density=True)
            ax.axvline(mean_val,   color="red",   linestyle="--", linewidth=2,
                       label=f"Mean: {mean_val:.4f}")
            ax.axvline(median_val, color="black", linestyle=":",  linewidth=2,
                       label=f"Median: {median_val:.4f}")
            ax.axvspan(ci_low, ci_high, alpha=0.15, color=color,
                       label=f"90% CI: [{ci_low:.4f}, {ci_high:.4f}]")
            ax.set_xlabel(param, fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(f"{method} — {param}", fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "real_data_posteriors.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_posterior_predictive(scaled_cases: np.ndarray, npe_samples: np.ndarray,
                               nle_samples: np.ndarray, out_dir: Path) -> None:
    """Overlay SIR trajectories from posterior samples on real observed data.
    Uses inferred I0 from each posterior sample instead of a fixed value.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Posterior Predictive Check — Italy COVID-19 First Wave",
                 fontsize=14, fontweight="bold")

    methods     = ["NPE", "NLE"]
    all_samples = [npe_samples, nle_samples]
    colors      = ["steelblue", "darkorange"]
    t_axis      = np.arange(len(scaled_cases))

    for m, (method, samples, color) in enumerate(zip(methods, all_samples, colors)):
        ax     = axes[m]
        n_plot = min(100, len(samples))
        idx    = np.random.choice(len(samples), size=n_plot, replace=False)

        for i in idx:
            beta_i  = float(samples[i, 0])
            gamma_i = float(samples[i, 1])
            i0_i    = float(samples[i, 2])  # use inferred I0
            traj = run_sir(beta_i, gamma_i, N=10000, I0=i0_i, T=len(scaled_cases))
            ax.plot(t_axis, traj, color=color, alpha=0.05, linewidth=0.8)

        beta_mean  = float(samples[:, 0].mean())
        gamma_mean = float(samples[:, 1].mean())
        i0_mean    = float(samples[:, 2].mean())
        mean_traj  = run_sir(beta_mean, gamma_mean, N=10000, I0=i0_mean, T=len(scaled_cases))
        ax.plot(t_axis, mean_traj, color="red", linewidth=2.5,
                label=f"Posterior mean\n(β={beta_mean:.3f}, γ={gamma_mean:.3f}, I₀={i0_mean:.1f})")

        ax.plot(t_axis, scaled_cases, color="black", linewidth=2,
                linestyle="--", label="Observed (scaled to N=10k)", zorder=5)

        ax.set_xlabel("Days since Feb 23, 2020", fontsize=11)
        ax.set_ylabel("Infected (scaled)", fontsize=11)
        ax.set_title(f"{method} Posterior Predictive", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "real_data_predictive.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_npe_vs_nle_comparison(npe_samples: np.ndarray, nle_samples: np.ndarray,
                                out_dir: Path) -> None:
    """Overlay NPE and NLE posteriors on same axes for direct comparison."""
    param_names = ["β (infection rate)", "γ (recovery rate)", "I₀ (initial infected)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("NPE vs NLE Posterior Comparison — Italy COVID-19 First Wave",
                 fontsize=13, fontweight="bold")

    for j, param in enumerate(param_names):
        ax = axes[j]
        ax.hist(npe_samples[:, j], bins=40, color="steelblue", alpha=0.6,
                density=True, edgecolor="white", label="NPE")
        ax.hist(nle_samples[:, j], bins=40, color="darkorange", alpha=0.6,
                density=True, edgecolor="white", label="NLE")
        ax.axvline(npe_samples[:, j].mean(), color="steelblue", linestyle="--",
                   linewidth=2, label=f"NPE mean: {npe_samples[:, j].mean():.4f}")
        ax.axvline(nle_samples[:, j].mean(), color="darkorange", linestyle="--",
                   linewidth=2, label=f"NLE mean: {nle_samples[:, j].mean():.4f}")
        ax.set_xlabel(param, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(param, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "real_data_npe_vs_nle.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REAL DATA INFERENCE: Italy COVID-19 First Wave")
    print("Parameters: beta, gamma, I0")
    print("=" * 70)

    print(f"\nLoading {COUNTRY} COVID-19 data...")
    scaled_cases, real_pop = load_italy_wave(DATA_CSV, COUNTRY, WAVE_START, T)

    print("\nLoading trained models...")
    print("  Loading NPE...")
    npe_amortizer = load_npe(ARTIFACTS_DIR)
    print("  NPE loaded successfully")

    print("  Loading NLE...")
    nle_amortizer, x_mean, x_std = load_nle(ARTIFACTS_DIR)
    print("  NLE loaded successfully")

    print(f"\nSampling NPE posterior ({N_POSTERIOR} samples)...")
    npe_samples = sample_npe_posterior(npe_amortizer, scaled_cases, N_POSTERIOR)
    print(f"  NPE samples shape: {npe_samples.shape}")

    print(f"\nSampling NLE posterior ({N_POSTERIOR} samples)...")
    nle_samples = sample_nle_posterior(
        nle_amortizer, scaled_cases, x_mean, x_std, N_POSTERIOR, SEED
    )
    print(f"  NLE samples shape: {nle_samples.shape}")

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INFERRED PARAMETERS — Italy COVID-19 First Wave")
    print("=" * 70)
    print(f"\n{'Parameter':<20} {'NPE Mean':>12} {'NPE 90% CI':>22} "
          f"{'NLE Mean':>12} {'NLE 90% CI':>22}")
    print("-" * 90)

    for j, param in enumerate(["beta", "gamma", "I0"]):
        npe_mean = float(npe_samples[:, j].mean())
        npe_lo   = float(np.percentile(npe_samples[:, j], 5))
        npe_hi   = float(np.percentile(npe_samples[:, j], 95))
        nle_mean = float(nle_samples[:, j].mean())
        nle_lo   = float(np.percentile(nle_samples[:, j], 5))
        nle_hi   = float(np.percentile(nle_samples[:, j], 95))
        print(f"{param:<20} {npe_mean:>12.4f} [{npe_lo:.4f}, {npe_hi:.4f}]"
              f" {nle_mean:>12.4f} [{nle_lo:.4f}, {nle_hi:.4f}]")

    npe_R0 = npe_samples[:, 0] / npe_samples[:, 1]
    nle_R0 = nle_samples[:, 0] / nle_samples[:, 1]
    print(f"\n{'R0 = beta/gamma':<20} {npe_R0.mean():>12.2f} "
          f"[{np.percentile(npe_R0,5):.2f}, {np.percentile(npe_R0,95):.2f}]"
          f" {nle_R0.mean():>12.2f} "
          f"[{np.percentile(nle_R0,5):.2f}, {np.percentile(nle_R0,95):.2f}]")
    print("=" * 70)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results = {
        "country": COUNTRY,
        "wave_start": WAVE_START,
        "T": T,
        "real_population": float(real_pop),
        "NPE": {
            "beta_mean":  float(npe_samples[:, 0].mean()),
            "beta_ci90":  [float(np.percentile(npe_samples[:, 0], 5)),
                           float(np.percentile(npe_samples[:, 0], 95))],
            "gamma_mean": float(npe_samples[:, 1].mean()),
            "gamma_ci90": [float(np.percentile(npe_samples[:, 1], 5)),
                           float(np.percentile(npe_samples[:, 1], 95))],
            "I0_mean":    float(npe_samples[:, 2].mean()),
            "I0_ci90":    [float(np.percentile(npe_samples[:, 2], 5)),
                           float(np.percentile(npe_samples[:, 2], 95))],
            "R0_mean":    float(npe_R0.mean()),
            "R0_ci90":    [float(np.percentile(npe_R0, 5)),
                           float(np.percentile(npe_R0, 95))],
        },
        "NLE": {
            "beta_mean":  float(nle_samples[:, 0].mean()),
            "beta_ci90":  [float(np.percentile(nle_samples[:, 0], 5)),
                           float(np.percentile(nle_samples[:, 0], 95))],
            "gamma_mean": float(nle_samples[:, 1].mean()),
            "gamma_ci90": [float(np.percentile(nle_samples[:, 1], 5)),
                           float(np.percentile(nle_samples[:, 1], 95))],
            "I0_mean":    float(nle_samples[:, 2].mean()),
            "I0_ci90":    [float(np.percentile(nle_samples[:, 2], 2)),
                           float(np.percentile(nle_samples[:, 2], 95))],
            "R0_mean":    float(nle_R0.mean()),
            "R0_ci90":    [float(np.percentile(nle_R0, 5)),
                           float(np.percentile(nle_R0, 95))],
        }
    }

    results_path = OUT_DIR / "italy_inference_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    print("\nGenerating plots...")
    plot_posterior_distributions(npe_samples, nle_samples, OUT_DIR)
    plot_posterior_predictive(scaled_cases, npe_samples, nle_samples, OUT_DIR)
    plot_npe_vs_nle_comparison(npe_samples, nle_samples, OUT_DIR)

    print("\n" + "=" * 70)
    print("REAL DATA INFERENCE COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {OUT_DIR}/")
    print("  - italy_inference_results.json")
    print("  - real_data_posteriors.png")
    print("  - real_data_predictive.png")
    print("  - real_data_npe_vs_nle.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
