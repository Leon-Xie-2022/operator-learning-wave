"""Utility functions for the 1D variable-coefficient wave solver.

The energy-like diagnostic used here is

    E ~= sum_{i=1}^{N-2} [(u_t[i])^2 + c[i]^2 (u_x[i])^2] * dx,

where velocity is approximated from consecutive time levels and u_x is
approximated by a centered difference. It is used to monitor numerical
behavior, not as a proof of exact energy conservation.
"""

import numpy as np


def compute_cfl_condition(dx, c, cfl=0.5):
    """
    Compute the maximum allowable time step from an approximate CFL condition.

    For u_tt = d_x(c^2 u_x), the condition is approximately
    dt <= cfl * dx / max(c).
    """
    c = np.asarray(c)
    if dx <= 0:
        raise ValueError("dx must be positive")
    if cfl <= 0:
        raise ValueError("cfl must be positive")
    if c.size == 0 or not np.all(np.isfinite(c)):
        raise ValueError("c must be a non-empty array of finite values")
    if not np.all(c > 0):
        raise ValueError("Wave speed c must be strictly positive")

    return cfl * dx / np.max(c)


def compute_interface_coefficients(c):
    """
    Compute c^2 interface coefficients for the conservative discretization.

    Interface values c^2_{i+1/2} are approximated by
    (c[i]^2 + c[i+1]^2) / 2.
    """
    c = np.asarray(c)
    if c.size < 2:
        raise ValueError("c must contain at least two grid points")
    if not np.all(np.isfinite(c)):
        raise ValueError("c must contain only finite values")
    if not np.all(c > 0):
        raise ValueError("Wave speed c must be strictly positive")

    c2 = c**2
    return (c2[:-1] + c2[1:]) / 2


def compute_energy_like(u_current, u_previous, c, dx, dt):
    """
    Compute the energy-like stability diagnostic.

    Parameters:
    -----------
    u_current : array_like
        Current time step displacement, shape (N,)
    u_previous : array_like
        Previous time step displacement, shape (N,)
    c : array_like
        Wave speed array, shape (N,)
    dx : float
        Spatial grid spacing
    dt : float
        Time step size

    Returns:
    --------
    energy : float
        Energy-like quantity.
    """
    u_current = np.asarray(u_current)
    u_previous = np.asarray(u_previous)
    c = np.asarray(c)

    N = len(u_current)
    if len(u_previous) != N or len(c) != N:
        raise ValueError("u_current, u_previous, and c must have the same length")
    if N < 3:
        raise ValueError("Inputs must contain at least three grid points")
    if dx <= 0:
        raise ValueError("dx must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if not np.all(np.isfinite(u_current)) or not np.all(np.isfinite(u_previous)) or not np.all(np.isfinite(c)):
        raise ValueError("u_current, u_previous, and c must contain only finite values")
    if not np.all(c > 0):
        raise ValueError("Wave speed c must be strictly positive")

    u_t = (u_current[1:-1] - u_previous[1:-1]) / dt
    u_x = (u_current[2:] - u_current[:-2]) / (2 * dx)

    kinetic = u_t**2
    potential = c[1:-1] ** 2 * u_x**2
    return np.sum((kinetic + potential) * dx)


def compute_relative_l2_error(u_pred, u_true):
    """Compute ||u_pred - u_true||_2 / ||u_true||_2."""
    u_pred = np.asarray(u_pred)
    u_true = np.asarray(u_true)

    numerator = np.linalg.norm(u_pred - u_true)
    denominator = np.linalg.norm(u_true)

    if denominator == 0:
        return np.nan

    return numerator / denominator


def compute_mse(u_pred, u_true):
    """Compute the mean squared error."""
    u_pred = np.asarray(u_pred)
    u_true = np.asarray(u_true)
    return np.mean((u_pred - u_true) ** 2)


def interpolate_to_grid(u, x_old, x_new):
    """Linearly interpolate values from one grid to another."""
    return np.interp(x_new, x_old, u)
