"""Reference solver for the 1D variable-coefficient wave equation.

This module solves the conservative wave equation

    u_tt = d_x(c(x)^2 u_x),    x in [0, 1], t in [0, T],

with homogeneous Dirichlet boundary conditions. The solver uses a conservative
finite-difference spatial discretization and leapfrog time stepping. Interface
values of c^2 are approximated by arithmetic averages, which is adequate for
the smooth coefficient fields used in these experiments.

The implementation assumes a one-dimensional uniform grid, positive wave
speed, positive final time, and a CFL-stable time step.
"""

import numpy as np


def solve_wave_equation(x, u0, c, T, dt=None, cfl=0.5, return_intermediate=False):
    """
    Solve the 1D variable-coefficient wave equation using finite differences.

    Parameters:
    -----------
    x : array_like
        Spatial grid points, shape (N,)
    u0 : array_like
        Initial displacement at t=0, shape (N,)
    c : array_like
        Wave speed function evaluated at grid points, shape (N,)
    T : float
        Final time
    dt : float, optional
        Time step size. If None, computed from the CFL condition.
    cfl : float, optional
        CFL coefficient for automatic dt computation.
    return_intermediate : bool, optional
        If True, return solution at all time steps; otherwise only final time.

    Returns:
    --------
    uT : ndarray
        Solution at time T, shape (N,)
    u_full : ndarray, optional
        Full time history if return_intermediate=True, shape (N, Nt)
    """
    x = np.asarray(x)
    u0 = np.asarray(u0)
    c = np.asarray(c)

    if len(x) != len(u0) or len(x) != len(c):
        raise ValueError("x, u0, and c must have the same length")

    N = len(x)
    if N < 3:
        raise ValueError("x, u0, and c must contain at least three grid points")
    if T <= 0:
        raise ValueError("T must be positive")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(u0)) or not np.all(np.isfinite(c)):
        raise ValueError("x, u0, and c must contain only finite values")
    if not np.all(c > 0):
        raise ValueError("Wave speed c must be strictly positive at all grid points")

    dx_values = np.diff(x)
    if not np.all(dx_values > 0):
        raise ValueError("x must be strictly increasing")
    dx = dx_values[0]
    if not np.allclose(dx_values, dx, rtol=1e-6, atol=1e-12):
        raise ValueError("x must be a uniform grid")

    if dt is None:
        dt = cfl * dx / np.max(c)
    elif dt <= 0:
        raise ValueError("dt must be positive")

    Nt = int(np.ceil(T / dt)) + 1
    dt = T / (Nt - 1)

    u = np.zeros((N, Nt))
    u[:, 0] = u0.copy()
    u[0, :] = 0.0
    u[-1, :] = 0.0

    # Start leapfrog from a Taylor step with zero initial velocity.
    u_tt_0 = _compute_spatial_derivative(u0, c, dx)
    u[:, 1] = u0 + (dt**2 / 2) * u_tt_0
    u[0, 1] = 0.0
    u[-1, 1] = 0.0

    for n in range(1, Nt - 1):
        u_tt = _compute_spatial_derivative(u[:, n], c, dx)
        u[:, n + 1] = 2 * u[:, n] - u[:, n - 1] + dt**2 * u_tt
        u[0, n + 1] = 0.0
        u[-1, n + 1] = 0.0

    if return_intermediate:
        return u[:, -1], u
    return u[:, -1]


def _compute_spatial_derivative(u, c, dx):
    """
    Compute d_x(c^2 u_x) using conservative finite differences.

    Parameters:
    -----------
    u : ndarray
        Current displacement, shape (N,)
    c : ndarray
        Wave speed, shape (N,)
    dx : float
        Grid spacing

    Returns:
    --------
    u_tt : ndarray
        Spatial derivative d_x(c^2 u_x), shape (N,)
    """
    N = len(u)
    u_tt = np.zeros(N)

    for i in range(1, N - 1):
        c2_ip = (c[i] ** 2 + c[i + 1] ** 2) / 2
        c2_im = (c[i - 1] ** 2 + c[i] ** 2) / 2

        flux_p = c2_ip * (u[i + 1] - u[i]) / dx
        flux_m = c2_im * (u[i] - u[i - 1]) / dx
        u_tt[i] = (flux_p - flux_m) / dx

    return u_tt
