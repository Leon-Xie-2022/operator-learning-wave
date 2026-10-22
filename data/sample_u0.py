"""
Initial Displacement Sampling for 1D Wave Equation

This module samples initial displacement functions u0(x) for the wave equation.
The sampling is based on random linear combinations of sinusoidal modes, ensuring compatibility
with Dirichlet boundary conditions u0(0) = u0(1) = 0.

Training distribution:
- Frequency range: modes k = 1 to 5 (fundamental to 5th harmonic)
- Maximum modes per sample: 5
- Amplitude coefficients: uniform [0.1, 0.5] per mode
- No phase shifts; components are homogeneous Dirichlet sine modes sin(k*pi*x)
- Total amplitude normalized to [0.1, 1.0]

OOD high-frequency distribution:
- Frequency range: modes k = 6 to 10 (higher harmonics)
- Maximum modes per sample: 3
- Amplitude coefficients: uniform [0.05, 0.3] per mode (smaller for stability)
- No phase shifts; components are homogeneous Dirichlet sine modes sin(k*pi*x)
- Total amplitude normalized to [0.05, 0.5]

Design rationale:
- Sinusoidal modes naturally satisfy boundary conditions
- Limited frequency range ensures numerical stability
- Random combinations create diverse training data
- OOD frequencies test extrapolation to unseen high-frequency components
- Amplitude bounds prevent extreme values that could cause numerical issues
"""

import numpy as np


def _validate_grid(x):
    x = np.asarray(x)
    if x.ndim != 1 or len(x) < 3:
        raise ValueError("x must be a one-dimensional grid with at least three points")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    return x


def _normalize_amplitude(u0, min_amp, max_amp):
    total_amp = np.max(np.abs(u0))
    if total_amp < 1e-12:
        raise ValueError("Sampled initial displacement has near-zero amplitude")
    if total_amp > max_amp:
        u0 = u0 * (max_amp / total_amp)
    elif total_amp < min_amp:
        u0 = u0 * (min_amp / total_amp)
    return u0


def sample_u0_train(x, max_modes=5, seed=None):
    """
    Sample initial displacement from training distribution.

    Parameters:
    -----------
    x : array_like
        Spatial grid points, shape (N,)
    max_modes : int, optional
        Maximum number of modes to combine (default 5)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    u0 : ndarray
        Initial displacement, shape (N,)
    """
    rng = np.random.default_rng(seed)
    x = _validate_grid(x)
    N = len(x)
    u0 = np.zeros(N)
    allowed_modes = np.arange(1, 6)
    max_modes = min(max_modes, len(allowed_modes))

    # Number of modes for this sample
    num_modes = rng.integers(1, max_modes + 1)
    modes = rng.choice(allowed_modes, size=num_modes, replace=False)

    for k in modes:
        # Signed amplitude. Avoid arbitrary phase shifts so every component is
        # exactly in the homogeneous Dirichlet sine basis sin(k*pi*x).
        amp = rng.uniform(0.1, 0.5) * rng.choice([-1.0, 1.0])

        # Add mode
        u0 += amp * np.sin(k * np.pi * x)

    # Ensure boundary conditions (should be automatic for integer k, but enforce)
    u0[0] = 0.0
    u0[-1] = 0.0

    return _normalize_amplitude(u0, min_amp=0.1, max_amp=1.0)


def sample_u0_ood_frequency(x, max_modes=3, seed=None):
    """
    Sample initial displacement from OOD high-frequency distribution.

    Parameters:
    -----------
    x : array_like
        Spatial grid points, shape (N,)
    max_modes : int, optional
        Maximum number of modes to combine (default 3)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    u0 : ndarray
        Initial displacement, shape (N,)
    """
    rng = np.random.default_rng(seed)
    x = _validate_grid(x)
    N = len(x)
    u0 = np.zeros(N)
    allowed_modes = np.arange(6, 11)
    max_modes = min(max_modes, len(allowed_modes))

    # Number of modes for this sample
    num_modes = rng.integers(1, max_modes + 1)
    modes = rng.choice(allowed_modes, size=num_modes, replace=False)

    for k in modes:
        # Signed amplitude (smaller for stability). No phase shift, preserving
        # exact compatibility with the fixed-end boundary conditions.
        amp = rng.uniform(0.05, 0.3) * rng.choice([-1.0, 1.0])

        # Add mode
        u0 += amp * np.sin(k * np.pi * x)

    # Ensure boundary conditions
    u0[0] = 0.0
    u0[-1] = 0.0

    return _normalize_amplitude(u0, min_amp=0.05, max_amp=0.5)


def sample_u0(x, distribution='train', **kwargs):
    """
    Unified interface for sampling initial displacements.

    Parameters:
    -----------
    x : array_like
        Spatial grid points
    distribution : str
        'train' or 'ood_frequency'
    **kwargs
        Additional arguments passed to sampling functions

    Returns:
    --------
    u0 : ndarray
        Initial displacement
    """
    if distribution == 'train':
        return sample_u0_train(x, **kwargs)
    elif distribution == 'ood_frequency':
        return sample_u0_ood_frequency(x, **kwargs)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def get_frequency_range(distribution):
    """
    Get the frequency range for a given distribution.

    Parameters:
    -----------
    distribution : str
        'train' or 'ood_frequency'

    Returns:
    --------
    freq_range : tuple
        (min_k, max_k)
    """
    if distribution == 'train':
        return (1, 5)
    elif distribution == 'ood_frequency':
        return (6, 10)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def is_ood_sample(u0, x, threshold_freq=5.5):
    """
    Determine if a sample is OOD based on its frequency content.

    This projects the sample onto the Dirichlet sine basis and checks whether
    most spectral energy lies above the training cutoff.

    Parameters:
    -----------
    u0 : array_like
        Initial displacement
    x : array_like
        Spatial grid
    threshold_freq : float
        Mode threshold (default 5.5, between train max 5 and OOD min 6)

    Returns:
    --------
    is_ood : bool
        True if sample appears to be OOD
    """
    x = _validate_grid(x)
    u0 = np.asarray(u0)
    if u0.shape != x.shape:
        raise ValueError("u0 and x must have the same shape")
    if not np.all(np.isfinite(u0)):
        raise ValueError("u0 must contain only finite values")

    max_mode = min(20, len(x) - 2)
    modes = np.arange(1, max_mode + 1)
    basis = np.sin(np.pi * modes[:, None] * x[None, :])
    coeffs = basis @ u0
    energy = coeffs**2
    total_energy = np.sum(energy)
    if total_energy < 1e-14:
        return False

    high_energy = np.sum(energy[modes > threshold_freq])
    return high_energy / total_energy > 0.5
