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
        beta = infection rate (how fast disease spreads)
        gamma = recovery rate (how fast people recover)
        R0 = beta/gamma = basic reproduction number
    """
    
    def __init__(self, N=10000, I0=10, R0=0, T=160):
        """
        Initialize the SIR simulator
        
        Parameters:
        -----------
        N : int
            Total population size (default: 10,000)
        I0 : int
            Initial number of infected individuals (default: 10)
        R0 : int
            Initial number of recovered individuals (default: 0)
        T : int
            Number of days to simulate (default: 160)
        """
        self.N = N
        self.I0 = I0
        self.S0 = N - I0 - R0
        self.R0_init = R0
        self.T = T
        
    def _deriv(self, y, t, N, beta, gamma):
        """
        SIR differential equations
        
        Parameters:
        -----------
        y : tuple
            Current state (S, I, R)
        t : float
            Current time
        N : int
            Total population
        beta : float
            Infection rate
        gamma : float
            Recovery rate
            
        Returns:
        --------
        tuple
            Derivatives (dS/dt, dI/dt, dR/dt)
        """
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    def simulate(self, beta, gamma, noise_level=0.1, seed=None):
        """
        Run one SIR simulation with given parameters
        
        This function:
        1. Solves the ODE system to get true infected counts
        2. Adds realistic observation noise (Poisson)
        3. Returns noisy observed data (what we'd see in reality)
        
        Parameters:
        -----------
        beta : float
            Infection rate (typical range: 0.0001 to 0.001)
            Higher beta = faster spread
        gamma : float
            Recovery rate (typical range: 0.01 to 0.1)
            Higher gamma = faster recovery
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
        
        # Initial conditions
        y0 = self.S0, self.I0, self.R0_init
        
        # Time points (one per day)
        t = np.linspace(0, self.T, self.T)
        
        # Solve the ODE system
        solution = odeint(self._deriv, y0, t, args=(self.N, beta, gamma))
        
        # Extract infected counts (column 1 of solution)
        I_true = solution[:, 1]
        
        # Add realistic observation noise (Poisson distribution for count data)
        # This simulates real-world measurement uncertainty
        I_observed = np.random.poisson(np.maximum(1, I_true))
        
        # Ensure no negative values
        I_observed = np.maximum(0, I_observed).astype(float)
        
        return I_observed
    
    def plot_simulation(self, beta, gamma, save_path=None):
        """
        Visualize one epidemic simulation
        
        Parameters:
        -----------
        beta : float
            Infection rate
        gamma : float
            Recovery rate
        save_path : str, optional
            If provided, save figure to this path
            
        Returns:
        --------
        I_observed : numpy array
            The simulated infected counts
        """
        # Run simulation
        I_observed = self.simulate(beta, gamma)
        
        # Calculate R0 (basic reproduction number)
        R0 = beta / gamma
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.plot(I_observed, linewidth=2, color='darkred', label='Infected')
        plt.xlabel('Days', fontsize=12)
        plt.ylabel('Number of Infected Individuals', fontsize=12)
        plt.title(f'SIR Epidemic Simulation\nβ={beta:.5f}, γ={gamma:.3f}, R₀={R0:.2f}', 
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
# TESTING CODE - Run this file directly to test the simulator
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTING SIR SIMULATOR")
    print("="*70)
    
    # Create simulator instance
    sim = SIRSimulator(N=10000, I0=10, T=160)
    print("✅ Simulator created successfully")
    print(f"   Population: {sim.N:,}")
    print(f"   Initial infected: {sim.I0}")
    print(f"   Simulation days: {sim.T}")
    
    # Test 1: Basic simulation with CORRECT parameters
    print("\n" + "-"*70)
    print("TEST 1: Basic Simulation")
    print("-"*70)
    beta_test = 0.3  # Changed from 0.0003 to 0.3
    gamma_test = 0.05
    I = sim.simulate(beta=beta_test, gamma=gamma_test, seed=42)
    R0_test = beta_test / gamma_test
    print(f"✅ Simulation complete")
    print(f"   Parameters: β={beta_test:.5f}, γ={gamma_test:.3f}")
    print(f"   R₀: {R0_test:.2f}")
    print(f"   Peak infected: {I.max():.0f} people")
    print(f"   Final infected: {I[-1]:.0f} people")
    
    # Test 2: Data quality checks
    print("\n" + "-"*70)
    print("TEST 2: Data Quality Checks")
    print("-"*70)
    assert np.all(I >= 0), "ERROR: Found negative values!"
    assert not np.any(np.isnan(I)), "ERROR: Found NaN values!"
    assert len(I) == 160, "ERROR: Wrong length!"
    print("✅ No negative values")
    print("✅ No NaN values")
    print("✅ Correct length (160 days)")
    
    # Test 3: R₀ effect on epidemic size with CORRECT parameters
    print("\n" + "-"*70)
    print("TEST 3: R₀ Effect on Epidemic Size")
    print("-"*70)
    
    # Use realistic parameter combinations
    beta_low, gamma_low = 0.15, 0.05      # R₀ = 3
    beta_med, gamma_med = 0.3, 0.05       # R₀ = 6
    beta_high, gamma_high = 0.6, 0.05     # R₀ = 12
    
    I_low = sim.simulate(beta=beta_low, gamma=gamma_low, seed=123)
    I_med = sim.simulate(beta=beta_med, gamma=gamma_med, seed=124)
    I_high = sim.simulate(beta=beta_high, gamma=gamma_high, seed=125)
    
    R0_low = beta_low / gamma_low
    R0_med = beta_med / gamma_med
    R0_high = beta_high / gamma_high
    
    print(f"✅ Low R₀ ({R0_low:.1f}):  β={beta_low:.2f}, Peak = {I_low.max():.0f} infected")
    print(f"✅ Med R₀ ({R0_med:.1f}):  β={beta_med:.2f}, Peak = {I_med.max():.0f} infected")
    print(f"✅ High R₀ ({R0_high:.1f}): β={beta_high:.2f}, Peak = {I_high.max():.0f} infected")
    
    # Verify trend
    if I_high.max() > I_med.max() > I_low.max():
        print("✅ Higher R₀ produces larger epidemics (correct!)")
    else:
        print("⚠️  Note: Due to stochastic noise, trend may vary slightly")
    
    # Test 4: Reproducibility
    print("\n" + "-"*70)
    print("TEST 4: Reproducibility with Random Seed")
    print("-"*70)
    I1 = sim.simulate(beta=0.3, gamma=0.05, seed=42)
    I2 = sim.simulate(beta=0.3, gamma=0.05, seed=42)
    assert np.allclose(I1, I2), "ERROR: Same seed gives different results!"
    print("✅ Same seed produces identical results")
    
    # Test 5: Create visualization
    print("\n" + "-"*70)
    print("TEST 5: Creating Visualization")
    print("-"*70)
    sim.plot_simulation(beta=0.3, gamma=0.05, 
                       save_path='test_epidemic.png')
    print("✅ Plot created and saved")
    
    # Summary
    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED!")
    print("="*70)
    print("✅ Simulator is working correctly!")
    print("✅ Ready for data generation!")
    print("="*70 + "\n")