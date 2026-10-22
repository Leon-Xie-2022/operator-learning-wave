"""
Visualization Tools for Solver Self-Checks

This module generates diagnostic plots for the wave equation solver to verify:
1. Wave propagation in constant coefficient case against an analytic solution
2. Final state solutions for variable coefficient cases
3. Energy-like diagnostic quantity evolution over time
4. Comparison of smooth vs rough wave speed functions

All plots are saved to results/figures/solver_checks/ for inspection and potential
use in paper figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add solver directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from solver.solver_core import solve_wave_equation
from solver.solver_utils import compute_energy_like, compute_cfl_condition

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'solver_checks')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _effective_dt(T, requested_dt):
    """Match the adjusted time step used internally by solve_wave_equation."""
    Nt = int(np.ceil(T / requested_dt)) + 1
    return T / (Nt - 1), Nt


def plot_constant_coefficient_propagation():
    """
    Plot wave propagation against the analytic single-mode solution for c=1.
    """
    print("Generating constant coefficient propagation plot...")

    # Grid setup
    N = 128
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]

    # Constant wave speed
    c = np.ones(N)

    # Initial condition: single mode. With u_t(x,0)=0 and c=1, the exact
    # solution is u(x,t)=sin(pi*x)*cos(pi*t).
    u0 = np.sin(np.pi * x)

    # Time points to plot
    T_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    dt = compute_cfl_condition(dx, c, cfl=0.4)

    max_errors = []
    plt.figure(figsize=(12, 8))

    for i, T in enumerate(T_values):
        if T == 0.0:
            u = u0
        else:
            u = solve_wave_equation(x, u0, c, T, dt=dt, return_intermediate=False)
        u_exact = np.sin(np.pi * x) * np.cos(np.pi * T)
        max_errors.append(np.max(np.abs(u - u_exact)))

        plt.subplot(2, 3, i+1)
        plt.plot(x, u, color='#4C78A8', linewidth=1.8, label='Numerical')
        plt.plot(x, u_exact, 'k--', linewidth=2.6, label='Analytic')
        plt.title(f't = {T:.2f}, max err = {max_errors[-1]:.2e}')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        if i == 0:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-1.2, 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'constant_coefficient_propagation.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved constant_coefficient_propagation.png (max analytic error {max(max_errors):.3e})")


def plot_variable_coefficient_final_states():
    """
    Plot final state solutions for different variable coefficient cases.
    """
    print("Generating variable coefficient final states plot...")

    # Grid setup
    N = 128
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]

    # Initial condition
    u0 = 0.5 * np.sin(np.pi * x) + 0.3 * np.sin(2 * np.pi * x)

    # Different wave speed profiles
    c_cases = {
        'Smooth': 1.0 + 0.3 * np.sin(2 * np.pi * x),
        'Medium': 1.0 + 0.5 * np.sin(2 * np.pi * x) + 0.2 * np.sin(4 * np.pi * x),
        'Rough': 1.0 + 0.4 * np.sin(4 * np.pi * x) + 0.3 * np.sin(8 * np.pi * x) + 0.2 * np.sin(16 * np.pi * x)
    }

    T = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), sharex=True)

    for i, (label, c) in enumerate(c_cases.items()):
        dt = compute_cfl_condition(dx, c, cfl=0.4)
        uT = solve_wave_equation(x, u0, c, T, dt=dt, return_intermediate=False)

        axes[0, i].plot(x, u0, 'k--', label='same initial u0', alpha=0.9, linewidth=2.0)
        axes[0, i].plot(x, uT, color='#E45756', label='terminal u(T)', linewidth=2.2)
        axes[0, i].set_title(f'{label}: solution')
        axes[0, i].set_ylabel('u')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_ylim(-1.2, 1.2)

        axes[1, i].plot(x, c, color='#4C78A8', label='c(x)', linewidth=2)
        axes[1, i].set_title(f'{label}: wave speed')
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel('c(x)')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_ylim(0.3, 1.8)

    fig.suptitle('Effect of coefficient smoothness on terminal solution\nSame initial condition in all columns; only c(x) changes',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(OUTPUT_DIR, 'variable_coefficient_final_states.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Saved variable_coefficient_final_states.png")


def plot_energy_evolution():
    """
    Plot energy-like quantity evolution over time.
    """
    print("Generating energy evolution plot...")

    # Grid setup
    N = 64
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]

    # Test cases
    cases = {
        'Constant c=1': {
            'u0': np.sin(np.pi * x),
            'c': np.ones(N)
        },
        'Variable c': {
            'u0': 0.5 * np.sin(np.pi * x) + 0.3 * np.sin(2 * np.pi * x),
            'c': 1.0 + 0.4 * np.sin(2 * np.pi * x)
        }
    }

    T = 2.0

    plt.figure(figsize=(12, 5))

    for i, (label, params) in enumerate(cases.items()):
        u0 = params['u0']
        c = params['c']

        requested_dt = compute_cfl_condition(dx, c, cfl=0.4)
        dt, _ = _effective_dt(T, requested_dt)
        uT, u_full = solve_wave_equation(x, u0, c, T, dt=requested_dt, return_intermediate=True)

        # Compute energy evolution
        times = []
        energies = []
        Nt = u_full.shape[1]

        for n in range(1, Nt):
            t = n * dt
            energy = compute_energy_like(u_full[:, n], u_full[:, n-1], c, dx, dt)
            times.append(t)
            energies.append(energy)

        plt.subplot(1, 2, i+1)
        energies = np.asarray(energies)
        rel_variation = 100.0 * (np.max(energies) - np.min(energies)) / (np.mean(energies) + 1e-12)

        plt.plot(times, energies, color='#4C78A8', linewidth=2)
        plt.title(f'Energy-like Diagnostic: {label}')
        plt.xlabel('Time t')
        plt.ylabel('Energy-like diagnostic')
        plt.grid(True, alpha=0.3)
        margin = 0.08 * (np.max(energies) - np.min(energies) + 1e-12)
        plt.ylim(np.min(energies) - margin, np.max(energies) + margin)
        plt.text(
            0.97,
            0.05,
            f'range/mean = {rel_variation:.2f}%',
            transform=plt.gca().transAxes,
            ha='right',
            va='bottom',
            fontsize=9,
            bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.85),
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'energy_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Saved energy_evolution.png")


def plot_wave_speed_comparison():
    """
    Plot comparison of smooth vs rough wave speed functions.
    """
    print("Generating wave speed comparison plot...")

    # Grid setup
    N = 128
    x = np.linspace(0, 1, N)

    # Generate different smoothness levels
    np.random.seed(42)

    # Smooth case
    c_smooth = 1.0 + 0.3 * np.sin(2 * np.pi * x)

    # Medium roughness
    c_medium = c_smooth + 0.2 * np.sin(4 * np.pi * x)

    # Rough case
    c_rough = c_medium + 0.15 * np.sin(8 * np.pi * x) + 0.1 * np.sin(16 * np.pi * x)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(x, c_smooth, 'g-', label='Smooth', linewidth=2)
    plt.plot(x, c_medium, 'b-', label='Medium', linewidth=2)
    plt.plot(x, c_rough, 'r-', label='Rough', linewidth=2)
    plt.title('Wave Speed Functions c(x) with Different Smoothness')
    plt.xlabel('x')
    plt.ylabel('c(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 1.8)

    # Show normalized Fourier-mode content after removing the mean.
    plt.subplot(2, 1, 2)
    for c, label, color in [(c_smooth, 'Smooth', 'g'), (c_medium, 'Medium', 'b'), (c_rough, 'Rough', 'r')]:
        c_centered = c - np.mean(c)
        magnitude = np.abs(np.fft.rfft(c_centered))
        mode_index = np.arange(len(magnitude))
        positive = mode_index > 0
        normalized = magnitude / (np.sum(magnitude[positive]) + 1e-12)
        plt.plot(mode_index[positive][:N//4], normalized[positive][:N//4], f'{color}-', label=label, linewidth=2)

    plt.title('Normalized Fourier-mode Content of c(x)')
    plt.xlabel('Fourier mode index k')
    plt.ylabel('Normalized FFT magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'wave_speed_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("[OK] Saved wave_speed_comparison.png")


def generate_all_plots():
    """
    Generate all diagnostic plots for solver verification.
    """
    print("=" * 50)
    print("GENERATING SOLVER DIAGNOSTIC PLOTS")
    print("=" * 50)

    try:
        plot_constant_coefficient_propagation()
        plot_variable_coefficient_final_states()
        plot_energy_evolution()
        plot_wave_speed_comparison()

        print("=" * 50)
        print("[OK] All plots generated successfully!")
        print(f"Plots saved to: {OUTPUT_DIR}")

    except Exception as e:
        print(f"[FAIL] Error generating plots: {e}")
        raise


if __name__ == "__main__":
    generate_all_plots()
