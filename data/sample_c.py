"""
Wave Speed Sampling for 1D Variable-Coefficient Wave Equation

This module provides functions to sample wave speed functions c(x) for the wave equation.
The sampling generates smooth random fields to ensure numerical stability while providing
diverse training data.

Training distribution:
- Base value: 1.0
- Perturbation: sum of sinusoidal modes with frequencies k=1,2,3
- Amplitude range: [-0.3, 0.3] per mode
- Final range: smoothly rescaled to [0.5, 1.5] when needed

OOD smoothness distribution:
- Similar to training but with smaller amplitudes and only low frequencies (k=1,2)
- Final range: smoothly rescaled to [0.7, 1.3] when needed
- This is an easier smoothness-shift diagnostic, not a severe OOD shift

OOD rough distribution:
- Higher frequency components (k=1 to 6)
- Larger amplitude perturbations
- Final range: smoothly rescaled to [0.4, 1.6] when needed

All distributions ensure c(x) > 0 and are sufficiently smooth for numerical stability.
"""

import numpy as np


def _validate_grid(x):
    x = np.asarray(x)
    if x.ndim != 1 or len(x) < 3:
        raise ValueError("x must be a one-dimensional grid with at least three points")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    return x


def _rescale_to_range(c, min_value, max_value):
    """Smoothly rescale c into [min_value, max_value] without clipping plateaus."""
    current_min = np.min(c)
    current_max = np.max(c)
    if current_max - current_min < 1e-12:
        midpoint = 0.5 * (min_value + max_value)
        return np.full_like(c, midpoint, dtype=float)
    if current_min >= min_value and current_max <= max_value:
        return c
    c01 = (c - current_min) / (current_max - current_min)
    return min_value + (max_value - min_value) * c01


def sample_c_train(x, seed=None):
    """
    Sample wave speed from training distribution.

    Uses low-frequency sinusoidal perturbations around c=1.0.

    Parameters:
    -----------
    x : array_like
        Spatial grid points, shape (N,)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    c : ndarray
        Wave speed function, shape (N,)
    """
    rng = np.random.default_rng(seed)
    x = _validate_grid(x)
    N = len(x)

    # Base wave speed
    c = np.ones(N)

    # Add perturbations with modes k=1,2,3
    for k in [1, 2, 3]:
        amp = rng.uniform(-0.3, 0.3)
        phase = rng.uniform(0, 2 * np.pi)
        c += amp * np.sin(2 * np.pi * k * x + phase)

    return _rescale_to_range(c, 0.5, 1.5)


def sample_c_ood_smooth(x, seed=None):
    """
    Sample wave speed from OOD smooth distribution.

    Uses very low-frequency, small-amplitude perturbations.

    Parameters:
    -----------
    x : array_like
        Spatial grid points, shape (N,)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    c : ndarray
        Wave speed function, shape (N,)
    """
    rng = np.random.default_rng(seed)
    x = _validate_grid(x)
    N = len(x)

    # Base wave speed
    c = np.ones(N)

    # Add small, low-frequency perturbations
    for k in [1, 2]:
        amp = rng.uniform(-0.15, 0.15)
        phase = rng.uniform(0, 2 * np.pi)
        c += amp * np.sin(2 * np.pi * k * x + phase)

    return _rescale_to_range(c, 0.7, 1.3)


def sample_c_ood_rough(x, seed=None):
    """
    Sample wave speed from OOD rough distribution.

    Uses higher-frequency components for more variable wave speeds.

    Parameters:
    -----------
    x : array_like
        Spatial grid points, shape (N,)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    c : ndarray
        Wave speed function, shape (N,)
    """
    rng = np.random.default_rng(seed)
    x = _validate_grid(x)
    N = len(x)

    # Base wave speed
    c = np.ones(N)

    # Add higher-frequency perturbations
    for k in range(1, 7):  # k=1 to 6
        amp = rng.uniform(-0.2, 0.2) * (1.0 / k)  # Amplitude decreases with frequency
        phase = rng.uniform(0, 2 * np.pi)
        c += amp * np.sin(2 * np.pi * k * x + phase)

    return _rescale_to_range(c, 0.4, 1.6)


def sample_c(x, distribution='train', seed=None):
    """
    Unified interface for sampling wave speed functions.

    Parameters:
    -----------
    x : array_like
        Spatial grid points
    distribution : str
        'train', 'ood_smoothness', 'ood_smooth', or 'ood_rough'.
        'ood_smoothness' is the canonical project name; 'ood_smooth'
        remains accepted for legacy dataset filenames.
    seed : int, optional
        Random seed

    Returns:
    --------
    c : ndarray
        Wave speed function
    """
    if distribution == 'train':
        return sample_c_train(x, seed=seed)
    elif distribution in ('ood_smoothness', 'ood_smooth'):
        return sample_c_ood_smooth(x, seed=seed)
    elif distribution == 'ood_rough':
        return sample_c_ood_rough(x, seed=seed)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def get_c_range(distribution):
    """
    Get the typical range of c(x) for a given distribution.

    Parameters:
    -----------
    distribution : str
        'train', 'ood_smoothness', 'ood_smooth', or 'ood_rough'

    Returns:
    --------
    c_range : tuple
        (min_c, max_c)
    """
    ranges = {
        'train': (0.5, 1.5),
        'ood_smoothness': (0.7, 1.3),
        'ood_smooth': (0.7, 1.3),
        'ood_rough': (0.4, 1.6)
    }
    return ranges.get(distribution, (0.5, 1.5))


def compute_smoothness_measure(c, x):
    """
    Compute a simple smoothness measure for c(x).

    Uses the maximum absolute value of the second derivative as a roughness indicator.

    Parameters:
    -----------
    c : array_like
        Wave speed function
    x : array_like
        Spatial grid

    Returns:
    --------
    smoothness : float
        Lower values indicate smoother functions
    """
    x = _validate_grid(x)
    c = np.asarray(c)
    if c.shape != x.shape:
        raise ValueError("c and x must have the same shape")
    if not np.all(np.isfinite(c)):
        raise ValueError("c must contain only finite values")

    dx_values = np.diff(x)
    if not np.allclose(dx_values, dx_values[0], rtol=1e-6, atol=1e-12):
        raise ValueError("x must be a uniform grid")
    dx = dx_values[0]
    c_diff1 = np.diff(c) / dx
    c_diff2 = np.diff(c_diff1) / dx
    return np.max(np.abs(c_diff2))


def is_ood_c_sample(c, x, smooth_threshold=5.0, rough_threshold=15.0):
    """
    Heuristically label a c(x) sample based on a roughness diagnostic.

    Dataset labels are defined by the sampler used to generate the sample; this
    helper is only for diagnostics and should not be treated as ground truth.

    Parameters:
    -----------
    c : array_like
        Wave speed function
    x : array_like
        Spatial grid
    smooth_threshold : float
        Threshold for smooth OOD (default 5.0)
    rough_threshold : float
        Threshold for rough OOD (default 15.0)

    Returns:
    --------
    ood_type : str or None
        'ood_smoothness', 'ood_rough', or None
    """
    smoothness = compute_smoothness_measure(c, x)

    if smoothness < smooth_threshold:
        return 'ood_smoothness'
    elif smoothness > rough_threshold:
        return 'ood_rough'
    else:
        return None
