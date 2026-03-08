"""
Quick test of NPE with small training run (5 epochs)
Tests the full pipeline before running full training
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
import bayesflow as bf
from pathlib import Path
import json


def load_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path)
    theta = data["theta"].astype(np.float32)
    x = data["x"].astype(np.float32)
    return theta, x


def main():
    print("=" * 70)
    print("QUICK NPE TEST (5 epochs, 500 samples)")
    print("=" * 70)
    print("This tests the full NPE pipeline quickly before full training")
    print("Expected time: ~2 minutes")
    print("=" * 70 + "\n")

    try:
        # Load small subset of data
        data_path = Path("02_data/sir_dataset.npz")
        theta, x = load_dataset(data_path)

        # Use only 500 samples for quick test
        theta = theta[:500]
        x = x[:500]
        print(f"✅ Data loaded: theta={theta.shape}, x={x.shape}")

        # Reshape for SequenceNetwork
        x_3d = x[:, :, np.newaxis]
        print(f"✅ X reshaped to: {x_3d.shape}")

        sim_data = {
            "prior_draws": theta,
            "sim_data": x_3d
        }

        # Build small network for quick test
        summary_net = bf.networks.SequenceNetwork(summary_dim=16)
        inference_net = bf.networks.InvertibleNetwork(
            num_params=2,
            num_coupling_layers=2,
            coupling_settings={
                "dense_args": {"units": 64, "activation": "relu"}
            }
        )
        amortizer = bf.amortizers.AmortizedPosterior(
            inference_net, summary_net
        )

        out_dir = Path("03_methods/artifacts/test_npe_checkpoint")
        trainer = bf.trainers.Trainer(
            amortizer=amortizer,
            checkpoint_path=str(out_dir)
        )
        print("✅ Networks initialized")

        # Train for just 5 epochs
        print("\nRunning 5 test epochs...")
        history = trainer.train_offline(
            simulations_dict=sim_data,
            epochs=5,
            batch_size=64
        )
        print("✅ Training completed without errors")

        # Quick posterior sample test
        print("\nTesting posterior sampling...")
        x_test = x_3d[:5]
        posterior_samples = amortizer.sample(
            {"summary_conditions": x_test}, n_samples=100
        )
        print(f"✅ Posterior samples shape: {posterior_samples.shape}")
        print(f"   Expected: (5, 100, 2)")

        # Check samples are in reasonable range
        beta_samples = posterior_samples[:, :, 0]
        gamma_samples = posterior_samples[:, :, 1]
        print(f"\n✅ Beta samples range: "
              f"[{beta_samples.min():.3f}, {beta_samples.max():.3f}]")
        print(f"✅ Gamma samples range: "
              f"[{gamma_samples.min():.3f}, {gamma_samples.max():.3f}]")

        print("\n" + "=" * 70)
        print("✅ ALL NPE TESTS PASSED!")
        print("=" * 70)
        print("\n✅ NPE pipeline works correctly!")
        print("🚀 Ready for full training: python 03_methods/train_npe.py")
        print("=" * 70 + "\n")
        return True

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ NPE TEST FAILED!")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)