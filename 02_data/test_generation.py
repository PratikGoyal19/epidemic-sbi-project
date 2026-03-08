"""
Quick test of data generation with small sample size (100 samples)
Test before generating full 10,000 samples
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from tqdm import tqdm
import importlib.util


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


def sample_prior(rng, n_samples, beta_min, beta_max, gamma_min, gamma_max):
    """Sample parameters from uniform priors."""
    beta = rng.uniform(beta_min, beta_max, size=n_samples)
    gamma = rng.uniform(gamma_min, gamma_max, size=n_samples)
    return np.column_stack([beta, gamma])


def generate_dataset(n_samples, T, N, I0, R0, beta_min, beta_max, 
                     gamma_min, gamma_max, seed):
    """Generate test dataset."""
    rng = np.random.default_rng(seed)
    
    theta = sample_prior(rng, n_samples, beta_min, beta_max, gamma_min, gamma_max)
    
    SIRSimulator = _load_sir_simulator()
    simulator = SIRSimulator(N=N, I0=I0, R0=R0, T=T)
    x = np.zeros((n_samples, T), dtype=np.float32)
    
    for i in tqdm(range(n_samples), desc="Simulating SIR data"):
        beta_i, gamma_i = theta[i]
        sim_seed = int(rng.integers(0, 2**32 - 1))
        trajectory = simulator.simulate(beta=beta_i, gamma=gamma_i, seed=sim_seed)
        x[i] = trajectory.astype(np.float32)
    
    return theta.astype(np.float32), x


def print_statistics(theta, x, N):
    """Print dataset statistics."""
    print("\n" + "="*70)
    print("📊 DATASET STATISTICS")
    print("="*70)
    
    print(f"\n📐 Parameters:")
    print(f"  Beta:  min={theta[:, 0].min():.4f}, max={theta[:, 0].max():.4f}, mean={theta[:, 0].mean():.4f}")
    print(f"  Gamma: min={theta[:, 1].min():.4f}, max={theta[:, 1].max():.4f}, mean={theta[:, 1].mean():.4f}")
    
    R0 = theta[:, 0] / theta[:, 1]
    print(f"  R₀:    min={R0.min():.2f}, max={R0.max():.2f}, mean={R0.mean():.2f}")
    
    print(f"\n📈 Observations:")
    peak_infected = x.max(axis=1)
    print(f"  Peak infected: min={peak_infected.min():.0f}, max={peak_infected.max():.0f}, mean={peak_infected.mean():.0f}")
    print(f"  (As % of population: min={peak_infected.min()/N*100:.1f}%, max={peak_infected.max()/N*100:.1f}%, mean={peak_infected.mean()/N*100:.1f}%)")
    
    print(f"\n✅ Quality Checks:")
    print(f"  NaN values:      {'❌ FOUND!' if np.any(np.isnan(x)) else '✅ None'}")
    print(f"  Negative values: {'❌ FOUND!' if np.any(x < 0) else '✅ None'}")
    print(f"  Zero epidemics:  {np.sum(peak_infected == 0)} / {len(x)}")
    
    print("="*70 + "\n")


def main():
    print("="*70)
    print("🧪 TESTING DATA GENERATION (100 SAMPLES)")
    print("="*70)
    print("This is a quick test before generating full 10,000 samples")
    print("Expected time: ~1 minute")
    print("="*70 + "\n")
    
    try:
        # Generate small test dataset
        theta, x = generate_dataset(
            n_samples=100,
            T=160,
            N=10000,
            I0=10,
            R0=0,
            beta_min=0.10,
            beta_max=0.60,
            gamma_min=0.01,
            gamma_max=0.10,
            seed=42
        )
        
        print("\n✅ Generation successful!")
        print(f"Theta shape: {theta.shape}")
        print(f"X shape: {x.shape}")
        
        # Print statistics
        print_statistics(theta, x, 10000)
        
        print("="*70)
        print("✅ TEST 2 PASSED!")
        print("="*70)
        print("\n✅ Data generation works correctly!")
        print("\n🚀 Next step: Run full data generation:")
        print("   python generate_data.py --n-samples 10000")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST 2 FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)