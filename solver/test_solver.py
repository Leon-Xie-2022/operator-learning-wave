"""
Test Suite for the 1D Variable-Coefficient Wave Equation Solver

This module contains systematic tests for the reference solver to ensure:
1. Correct behavior in constant coefficient case (sanity check)
2. Strict enforcement of boundary conditions
3. Clear rejection of invalid inputs
4. Numerical stability across the training distribution
5. Reasonable evolution of energy-like quantities

All tests should pass before proceeding to data generation and model training.
"""

import numpy as np
import sys
import os

# Add solver directory to path
sys.path.append(os.path.dirname(__file__))

from solver_core import solve_wave_equation
from solver_utils import compute_energy_like, compute_cfl_condition, compute_interface_coefficients


def test_constant_coefficient_sanity():
    """
    Sanity check with constant wave speed c(x) = 1.

    Uses a single-mode sinusoidal initial condition and checks that the solution
    behaves like a standard wave equation (propagation, reflection at boundaries).
    """
    print("Running constant coefficient sanity check...")

    # Grid setup
    N = 128
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]

    # Constant wave speed
    c = np.ones(N)

    # Single-mode initial condition: u0 = sin(pi*x).
    # This should satisfy boundary conditions u(0)=u(1)=0
    u0 = np.sin(np.pi * x)

    # Time parameters
    T = 2.0  # Two periods for a full cycle
    dt = compute_cfl_condition(dx, c, cfl=0.4)

    # Solve
    uT, u_full = solve_wave_equation(x, u0, c, T, dt=dt, return_intermediate=True)

    # Check boundary conditions
    assert np.allclose(u_full[0, :], 0.0, atol=1e-10), "Left boundary not satisfied"
    assert np.allclose(u_full[-1, :], 0.0, atol=1e-10), "Right boundary not satisfied"

    # Check that solution is not trivial (should have some oscillation)
    max_amplitude = np.max(np.abs(u_full))
    assert max_amplitude > 0.1, f"Solution amplitude too small: {max_amplitude}"

    # Check energy-like quantity evolution
    energies = []
    Nt = u_full.shape[1]
    for n in range(1, Nt):
        energy = compute_energy_like(u_full[:, n], u_full[:, n-1], c, dx, dt)
        energies.append(energy)

    energies = np.array(energies)
    # Energy should not grow exponentially (basic stability check)
    max_energy = np.max(energies)
    min_energy = np.min(energies)
    energy_ratio = max_energy / (min_energy + 1e-10)  # Avoid division by zero
    assert energy_ratio < 10.0, f"Energy growth too large: ratio {energy_ratio}"

    print("[OK] Constant coefficient sanity check passed")
    return True


def test_boundary_conditions():
    """
    Test that boundary conditions are strictly enforced for arbitrary samples.
    """
    print("Running boundary condition checks...")

    # Grid setup
    N = 64
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]

    # Test multiple random samples
    np.random.seed(42)  # For reproducibility
    num_samples = 10

    for i in range(num_samples):
        # Random initial condition (satisfying boundaries)
        u0 = np.random.randn(N) * 0.1
        u0[0] = 0.0
        u0[-1] = 0.0

        # Random wave speed
        c = 0.5 + 0.5 * np.random.rand(N)

        # Solve
        T = 0.5
        dt = compute_cfl_condition(dx, c, cfl=0.4)
        uT, u_full = solve_wave_equation(x, u0, c, T, dt=dt, return_intermediate=True)

        # Check boundaries at all time steps
        assert np.allclose(u_full[0, :], 0.0, atol=1e-12), f"Sample {i}: Left boundary violated"
        assert np.allclose(u_full[-1, :], 0.0, atol=1e-12), f"Sample {i}: Right boundary violated"

        # Check final state boundaries
        assert np.abs(uT[0]) < 1e-12, f"Sample {i}: Final left boundary violated"
        assert np.abs(uT[-1]) < 1e-12, f"Sample {i}: Final right boundary violated"

    print(f"[OK] Boundary conditions satisfied for {num_samples} random samples")
    return True


def test_input_validation():
    """
    Test that invalid solver inputs are rejected with clear ValueError exceptions.
    """
    print("Running input validation checks...")

    x = np.linspace(0, 1, 8)
    u0 = np.sin(np.pi * x)
    c = np.ones_like(x)

    invalid_cases = [
        ("too few grid points", x[:2], u0[:2], c[:2], 0.5, None),
        ("non-positive final time", x, u0, c, 0.0, None),
        ("non-uniform grid", np.array([0.0, 0.1, 0.25, 0.5, 0.7, 0.85, 0.95, 1.0]), u0, c, 0.5, None),
        ("non-positive wave speed", x, u0, np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 0.5, None),
        ("negative dt", x, u0, c, 0.5, -0.01),
    ]

    for name, x_bad, u0_bad, c_bad, T_bad, dt_bad in invalid_cases:
        try:
            solve_wave_equation(x_bad, u0_bad, c_bad, T_bad, dt=dt_bad)
        except ValueError:
            continue
        print(f"[FAIL] Input validation did not reject {name}")
        return False

    print("[OK] Invalid inputs rejected with ValueError")
    return True


def test_solver_utils_validation():
    """
    Test that solver utility functions reject invalid numerical inputs.
    """
    print("Running solver utility validation checks...")

    x = np.linspace(0, 1, 8)
    u = np.sin(np.pi * x)
    c = np.ones_like(x)
    dx = x[1] - x[0]
    dt = 0.01

    invalid_calls = [
        ("CFL negative dx", lambda: compute_cfl_condition(-dx, c)),
        ("CFL non-positive c", lambda: compute_cfl_condition(dx, np.array([1.0, 0.0, 1.0]))),
        ("interface too short", lambda: compute_interface_coefficients(np.array([1.0]))),
        ("interface non-positive c", lambda: compute_interface_coefficients(np.array([1.0, -0.5]))),
        ("energy length mismatch", lambda: compute_energy_like(u, u[:-1], c, dx, dt)),
        ("energy negative dt", lambda: compute_energy_like(u, u, c, dx, -dt)),
        ("energy non-positive c", lambda: compute_energy_like(u, u, np.zeros_like(c), dx, dt)),
    ]

    for name, call in invalid_calls:
        try:
            call()
        except ValueError:
            continue
        print(f"[FAIL] Solver utility validation did not reject {name}")
        return False

    print("[OK] Solver utility invalid inputs rejected with ValueError")
    return True


def test_numerical_stability():
    """
    Test numerical stability across training-like distribution.
    """
    print("Running numerical stability checks...")

    # Grid setup
    N = 64
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]

    # Test parameters
    np.random.seed(123)  # For reproducibility
    num_samples = 20
    T = 1.0
    max_allowed_amplitude = 100.0  # Reasonable upper bound

    unstable_count = 0

    for i in range(num_samples):
        # Generate training-like initial condition
        # Use a few sinusoidal modes
        u0 = np.zeros(N)
        for k in range(1, 4):  # modes 1,2,3
            amp = np.random.uniform(0.1, 0.5)
            phase = np.random.uniform(0, 2*np.pi)
            u0 += amp * np.sin(k * np.pi * x + phase)
        u0[0] = 0.0
        u0[-1] = 0.0

        # Generate training-like wave speed
        # Smooth random field
        c_base = 1.0
        c_perturbation = 0.3 * np.sin(2 * np.pi * x) + 0.1 * np.sin(4 * np.pi * x)
        c = c_base + c_perturbation
        c = np.clip(c, 0.5, 1.5)  # Keep positive and reasonable

        # Solve
        dt = compute_cfl_condition(dx, c, cfl=0.4)
        try:
            uT, u_full = solve_wave_equation(x, u0, c, T, dt=dt, return_intermediate=True)

            # Check for NaN or infinite values
            if np.any(np.isnan(u_full)) or np.any(np.isinf(u_full)):
                print(f"[FAIL] Sample {i}: NaN or Inf detected")
                unstable_count += 1
                continue

            # Check amplitude bounds
            max_amp = np.max(np.abs(u_full))
            if max_amp > max_allowed_amplitude:
                print(f"[FAIL] Sample {i}: Amplitude explosion: {max_amp}")
                unstable_count += 1
                continue

            # Check energy-like quantity
            energies = []
            Nt = u_full.shape[1]
            for n in range(1, Nt):
                energy = compute_energy_like(u_full[:, n], u_full[:, n-1], c, dx, dt)
                if np.isnan(energy) or np.isinf(energy):
                    print(f"[FAIL] Sample {i}: Invalid energy at step {n}")
                    unstable_count += 1
                    break
                energies.append(energy)

            if len(energies) == Nt - 1:
                energies = np.array(energies)
                if np.max(energies) / (np.min(energies) + 1e-10) > 50.0:
                    print(f"[FAIL] Sample {i}: Energy growth too large")
                    unstable_count += 1

        except Exception as e:
            print(f"[FAIL] Sample {i}: Exception during solving: {e}")
            unstable_count += 1

    if unstable_count == 0:
        print(f"[OK] All {num_samples} stability tests passed")
        return True
    else:
        print(f"[FAIL] {unstable_count}/{num_samples} stability tests failed")
        return False


def test_energy_like_evolution():
    """
    Test that energy-like quantities evolve reasonably over time.
    """
    print("Running energy-like quantity evolution checks...")

    # Grid setup
    N = 64
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]

    # Test case
    u0 = 0.5 * np.sin(np.pi * x)  # Single mode
    c = np.ones(N)  # Constant speed

    T = 3.0
    dt = compute_cfl_condition(dx, c, cfl=0.4)
    uT, u_full = solve_wave_equation(x, u0, c, T, dt=dt, return_intermediate=True)

    # Compute energy evolution
    energies = []
    Nt = u_full.shape[1]
    for n in range(1, Nt):
        energy = compute_energy_like(u_full[:, n], u_full[:, n-1], c, dx, dt)
        energies.append(energy)

    energies = np.array(energies)

    # Check for reasonable behavior
    # Energy should oscillate but not grow uncontrollably
    energy_std = np.std(energies)
    energy_mean = np.mean(energies)

    # Coefficient of variation should be reasonable
    cv = energy_std / (energy_mean + 1e-10)
    assert cv < 2.0, f"Energy variation too large: CV = {cv}"

    # No NaN or Inf
    assert not np.any(np.isnan(energies)), "NaN in energy values"
    assert not np.any(np.isinf(energies)), "Inf in energy values"

    # Energy should stay positive
    assert np.all(energies > 0), "Negative energy values detected"

    print("[OK] Energy-like quantity evolution reasonable")
    return True


def run_all_tests():
    """
    Run all solver tests and report results.
    """
    print("=" * 50)
    print("SOLVER TEST SUITE")
    print("=" * 50)

    tests = [
        test_constant_coefficient_sanity,
        test_boundary_conditions,
        test_input_validation,
        test_solver_utils_validation,
        test_numerical_stability,
        test_energy_like_evolution
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"[FAIL] {test.__name__} failed")
        except Exception as e:
            print(f"[FAIL] {test.__name__} failed with exception: {e}")

    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("[OK] All tests passed! Solver is ready for use.")
        return True
    else:
        print("[FAIL] Some tests failed. Please debug before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
