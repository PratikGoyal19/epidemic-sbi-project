"""
SIR Epidemic Simulator
Team: Pratik Goyal, Suryansh Chaturvedi, Mayank Choudhary
Date: February 2026
Course: Generative Neural Networks for the Sciences
University of Heidelberg
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class SIRSimulator:
    """
    SIR epidemic model simulator for parameter inference using BayesFlow
    
    The SIR model divides population into three compartments:
    - S (Susceptible): Healthy individuals who can get infected
    - I (Infected): Sick individuals who can spread the disease
    - R (Recovered): Immune individuals who cannot get infected again
    
    Model equations:
        dS/dt = -beta * S * I / N
        dI/dt = beta * S * I / N - gamma * I
        dR/dt = gamma * I
    
    Where:
        beta  = infection rate (how fast disease spreads)
        gamma = recovery rate (how fast people recover)
        I0    = initial number of infected individuals (now inferred)
        R0    = beta/gamma = basic reproduction number
    """
    
    def __init__(self, N=10000, T=160):
        """
        Initialize the SIR simulator.
        Note: I0 is no longer fixed here — it is passed to simulate() as a parameter.
        
        Parameters:
        -----------
        N : int
            Total population size (default: 10,000)
        T : int
            Number of days to simulate (default: 160)
        """
        self.N = N
        self.T = T
        
    def _deriv(self, y, t, N, beta, gamma):
        """SIR differential equations."""
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt =  beta * S * I / N - gamma * I
        dRdt =  gamma * I
        return dSdt, dIdt, dRdt
    
    def simulate(self, beta, gamma, I0=10, noise_level=0.1, seed=None):
        """
        Run one SIR simulation with given parameters.
        
        Parameters:
        -----------
        beta : float
            Infection rate (range: 0.10 to 0.60)
        gamma : float
            Recovery rate (range: 0.01 to 0.10)
        I0 : float
            Initial number of infected individuals (range: 1 to 50)
            Now an inferred parameter instead of a fixed constant.
        noise_level : float
            Observation noise level (default: 0.1)
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        I_observed : numpy array
            Noisy observed infected counts over T days (length T)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initial conditions — I0 is now a parameter
        S0 = self.N - I0
        y0 = (S0, I0, 0.0)
        
        # Time points (one per day)
        t = np.linspace(0, self.T, self.T)
        
        # Solve the ODE system
        solution = odeint(self._deriv, y0, t, args=(self.N, beta, gamma))
        
        # Extract infected counts
        I_true = solution[:, 1]
        
        # Add Poisson observation noise
        I_observed = np.random.poisson(np.maximum(1, I_true))
        I_observed = np.maximum(0, I_observed).astype(float)
        
        return I_observed
    
    def plot_simulation(self, beta, gamma, I0=10, save_path=None):
        """Visualize one epidemic simulation."""
        I_observed = self.simulate(beta, gamma, I0=I0)
        R0_val = beta / gamma
        
        plt.figure(figsize=(10, 6))
        plt.plot(I_observed, linewidth=2, color='darkred', label='Infected')
        plt.xlabel('Days', fontsize=12)
        plt.ylabel('Number of Infected Individuals', fontsize=12)
        plt.title(f'SIR Epidemic Simulation\n'
                  f'β={beta:.3f}, γ={gamma:.3f}, I₀={I0:.0f}, R₀={R0_val:.2f}',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Figure saved to {save_path}")
        
        plt.show()
        return I_observed


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTING SIR SIMULATOR (with I0 as inferred parameter)")
    print("="*70)
    
    sim = SIRSimulator(N=10000, T=160)
    print("✅ Simulator created successfully")
    print(f"   Population: {sim.N:,}")
    print(f"   Simulation days: {sim.T}")
    print(f"   I0: now an inferred parameter (range: 1-50)")
    
    # Test 1: Basic simulation
    print("\n" + "-"*70)
    print("TEST 1: Basic Simulation with I0=10")
    print("-"*70)
    I = sim.simulate(beta=0.3, gamma=0.05, I0=10, seed=42)
    print(f"✅ beta=0.3, gamma=0.05, I0=10 → Peak={I.max():.0f}")

    # Test 2: Different I0 values
    print("\n" + "-"*70)
    print("TEST 2: Effect of I0 on epidemic")
    print("-"*70)
    for i0 in [1, 10, 25, 50]:
        I = sim.simulate(beta=0.3, gamma=0.05, I0=i0, seed=42)
        print(f"✅ I0={i0:2d} → Peak={I.max():.0f}, Final={I[-1]:.0f}")

    # Test 3: Data quality
    print("\n" + "-"*70)
    print("TEST 3: Data Quality Checks")
    print("-"*70)
    I = sim.simulate(beta=0.3, gamma=0.05, I0=10, seed=42)
    assert np.all(I >= 0), "ERROR: Negative values!"
    assert not np.any(np.isnan(I)), "ERROR: NaN values!"
    assert len(I) == 160, "ERROR: Wrong length!"
    print("✅ No negative values")
    print("✅ No NaN values")
    print("✅ Correct length (160 days)")

    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED!")
    print("="*70)
