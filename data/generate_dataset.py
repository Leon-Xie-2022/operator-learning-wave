"""
Dataset Generation for 1D Variable-Coefficient Wave Equation

This module generates training and evaluation datasets by sampling initial conditions
and wave speeds, then solving the PDE to obtain terminal states.

Generated datasets:
- train: Training data from training distributions
- validation: Validation data from training distributions
- id_test: In-distribution test data
- ood_frequency: OOD test data with high-frequency initial conditions
- ood_smoothness: OOD smoothness-shift data with very smooth wave speeds
- ood_rough: auxiliary rough wave-speed diagnostic data, not part of the
  canonical main evaluation set

Each sample contains:
- u0: Initial displacement (N,)
- c: Wave speed function (N,)
- uT: Terminal displacement (N,)
- label: Dataset type string
- meta_pickle: Pickled list of metadata dictionaries with sampling parameters

Data is saved as .npz files with the following structure:
- u0: (num_samples, N)
- c: (num_samples, N)
- uT: (num_samples, N)
- labels: (num_samples,) string array
- meta_pickle: Pickled metadata list
- quality_pickle: Pickled dataset-level quality summary
"""

import numpy as np
import os
from tqdm import tqdm
import pickle
import sys

# Add project root to path before package imports when this file is executed
# as a script via `python data/generate_dataset.py`.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import sampling functions
from data.sample_u0 import sample_u0
from data.sample_c import sample_c

# Import solver
from solver.solver_core import solve_wave_equation
from solver.solver_utils import compute_cfl_condition


CANONICAL_DATASET_TYPES = ('train', 'validation', 'id_test', 'ood_frequency', 'ood_smoothness')
AUXILIARY_DATASET_TYPES = ('ood_rough',)
DATASET_TYPE_ALIASES = {'ood_smooth': 'ood_smoothness'}
DATASET_FILE_STEMS = {'ood_smoothness': 'ood_smooth'}


def canonical_dataset_type(dataset_type):
    """Return the canonical public dataset type for a split name or legacy alias."""
    return DATASET_TYPE_ALIASES.get(dataset_type, dataset_type)


def dataset_file_stem(dataset_type):
    """Return the on-disk file stem used for a canonical or legacy dataset type."""
    canonical_type = canonical_dataset_type(dataset_type)
    return DATASET_FILE_STEMS.get(canonical_type, canonical_type)


def _validate_grid(x):
    x = np.asarray(x)
    if x.ndim != 1 or len(x) < 3:
        raise ValueError("x must be a one-dimensional grid with at least three points")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    dx_values = np.diff(x)
    if not np.all(dx_values > 0):
        raise ValueError("x must be strictly increasing")
    if not np.allclose(dx_values, dx_values[0], rtol=1e-6, atol=1e-12):
        raise ValueError("x must be a uniform grid")
    return x, dx_values[0]


def _validate_generated_sample(u0, c, uT, sample_id, dataset_type):
    if not np.all(np.isfinite(u0)):
        raise ValueError(f"{dataset_type} sample {sample_id}: u0 contains non-finite values")
    if not np.all(np.isfinite(c)):
        raise ValueError(f"{dataset_type} sample {sample_id}: c contains non-finite values")
    if not np.all(np.isfinite(uT)):
        raise ValueError(f"{dataset_type} sample {sample_id}: uT contains non-finite values")
    if np.max(np.abs(u0)) < 1e-12:
        raise ValueError(f"{dataset_type} sample {sample_id}: u0 is near zero")
    if not np.all(c > 0):
        raise ValueError(f"{dataset_type} sample {sample_id}: c must be strictly positive")
    if np.max(np.abs(uT)) > 100.0:
        raise ValueError(f"{dataset_type} sample {sample_id}: uT amplitude is suspiciously large")


def _quality_summary(u0_array, c_array, uT_array):
    return {
        'u0_abs_max_min': float(np.min(np.max(np.abs(u0_array), axis=1))),
        'u0_abs_max_mean': float(np.mean(np.max(np.abs(u0_array), axis=1))),
        'c_min': float(np.min(c_array)),
        'c_max': float(np.max(c_array)),
        'uT_abs_max': float(np.max(np.abs(uT_array))),
        'uT_nan_count': int(np.isnan(uT_array).sum()),
    }


def generate_dataset(num_samples, x, dataset_type, seed_offset=0):
    """
    Generate a dataset of the specified type.

    Parameters:
    -----------
    num_samples : int
        Number of samples to generate
    x : array_like
        Spatial grid, shape (N,)
    dataset_type : str
        Dataset split. Canonical values are 'train', 'validation',
        'id_test', 'ood_frequency', and 'ood_smoothness'. The legacy alias
        'ood_smooth' is accepted and saved with canonical labels.
        'ood_rough' is supported as an auxiliary diagnostic split.
    seed_offset : int
        Offset for random seeds to ensure different datasets

    Returns:
    --------
    data : dict
        Dictionary containing all samples
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    dataset_type = canonical_dataset_type(dataset_type)
    valid_types = set(CANONICAL_DATASET_TYPES) | set(AUXILIARY_DATASET_TYPES)
    if dataset_type not in valid_types:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Valid types: {sorted(valid_types)}")

    x, dx = _validate_grid(x)
    N = len(x)
    rng = np.random.default_rng(seed_offset)

    # Initialize data arrays
    u0_array = np.zeros((num_samples, N))
    c_array = np.zeros((num_samples, N))
    uT_array = np.zeros((num_samples, N))
    labels = np.full(num_samples, dataset_type, dtype='<U20')
    meta_list = []

    print(f"Generating {dataset_type} dataset with {num_samples} samples...")

    for i in tqdm(range(num_samples)):
        # Draw per-sample seeds from a dataset-local RNG. This keeps each split
        # reproducible without mutating NumPy's global random state.
        seed = int(rng.integers(0, np.iinfo(np.int32).max))

        # Sample initial condition
        if dataset_type in ['train', 'validation', 'id_test']:
            u0_dist = 'train'
        elif dataset_type == 'ood_frequency':
            u0_dist = 'ood_frequency'
        else:  # ood_smoothness, ood_rough
            u0_dist = 'train'

        u0 = sample_u0(x, distribution=u0_dist, seed=seed)

        # Sample wave speed
        if dataset_type in ['train', 'validation', 'id_test', 'ood_frequency']:
            c_dist = 'train'
        elif dataset_type == 'ood_smoothness':
            c_dist = 'ood_smoothness'
        elif dataset_type == 'ood_rough':
            c_dist = 'ood_rough'

        c = sample_c(x, distribution=c_dist, seed=seed + 1000)

        # Solve PDE
        T = 1.0  # Fixed terminal time
        dt = compute_cfl_condition(dx, c, cfl=0.4)
        uT = solve_wave_equation(x, u0, c, T, dt=dt)
        _validate_generated_sample(u0, c, uT, i, dataset_type)

        # Store data
        u0_array[i] = u0
        c_array[i] = c
        uT_array[i] = uT

        # Store metadata
        meta = {
            'sample_id': i,
            'u0_distribution': u0_dist,
            'c_distribution': c_dist,
            'T': T,
            'dt': dt,
            'N': N,
            'dx': dx
        }
        meta_list.append(meta)

    data = {
        'u0': u0_array,
        'c': c_array,
        'uT': uT_array,
        'labels': labels,
        'meta': meta_list,
        'quality_summary': _quality_summary(u0_array, c_array, uT_array),
    }
    print(f"Quality summary for {dataset_type}: {data['quality_summary']}")

    return data


def save_dataset(data, filename):
    """
    Save dataset to .npz file.

    Parameters:
    -----------
    data : dict
        Dataset dictionary
    filename : str
        Output filename (without extension)
    """
    # Convert metadata to bytes for np.savez. Keep metadata out of the direct
    # array payload to avoid object-array duplication in the .npz file.
    meta_bytes = pickle.dumps(data['meta'])
    quality_bytes = pickle.dumps(data.get('quality_summary', {}))

    np.savez(
        filename,
        u0=data['u0'],
        c=data['c'],
        uT=data['uT'],
        labels=data['labels'],
        meta_pickle=meta_bytes,
        quality_pickle=quality_bytes,
    )
    print(f"Saved dataset to {filename}.npz")


def load_dataset(filename):
    """
    Load dataset from .npz file.

    Parameters:
    -----------
    filename : str
        Dataset filename (without extension)

    Returns:
    --------
    data : dict
        Loaded dataset
    """
    loaded = np.load(filename + '.npz', allow_pickle=True)
    data = {key: loaded[key] for key in loaded.keys() if key not in ('meta_pickle', 'quality_pickle')}
    data['meta'] = pickle.loads(loaded['meta_pickle'])
    if 'quality_pickle' in loaded:
        data['quality_summary'] = pickle.loads(loaded['quality_pickle'])
    return data


def generate_all_datasets(x, output_dir='data'):
    """
    Generate all required datasets.

    Parameters:
    -----------
    x : array_like
        Spatial grid
    output_dir : str
        Output directory for datasets
    """
    os.makedirs(output_dir, exist_ok=True)

    # Dataset configurations. Canonical public names follow project_plan.md.
    # The ood_smoothness split is still saved as ood_smooth_dataset.npz for
    # compatibility with existing generated data and configs.
    configs = {
        'train': 2000,
        'validation': 200,
        'id_test': 200,
        'ood_frequency': 200,
        'ood_smoothness': 200,
        'ood_rough': 200,
    }

    seed_offsets = {
        'train': 11,
        'validation': 23,
        'id_test': 37,
        'ood_frequency': 41,
        'ood_smoothness': 53,
        'ood_rough': 67,
    }

    for dataset_type, num_samples in configs.items():
        data = generate_dataset(num_samples, x, dataset_type, seed_offset=seed_offsets[dataset_type])
        filename = os.path.join(output_dir, f'{dataset_file_stem(dataset_type)}_dataset')
        save_dataset(data, filename)

    # Generate README
    generate_readme(output_dir)


def generate_readme(output_dir):
    """
    Generate README.md explaining the dataset format.
    """
    readme_content = """# Wave Equation Datasets

This directory contains the generated datasets for the 1D variable-coefficient wave equation operator learning task.

## Dataset Files

- `train_dataset.npz`: Training data (2000 samples)
- `validation_dataset.npz`: Validation data (200 samples)
- `id_test_dataset.npz`: In-distribution test data (200 samples)
- `ood_frequency_dataset.npz`: OOD test data with high-frequency initial conditions (200 samples)
- `ood_smooth_dataset.npz`: Canonical split `ood_smoothness`; OOD smoothness-shift data with very smooth wave speeds (200 samples)
- `ood_rough_dataset.npz`: Auxiliary rough wave-speed diagnostic data (200 samples)

## Data Format

Each .npz file contains the following arrays:

- `u0`: Initial displacement functions, shape (num_samples, N)
- `c`: Wave speed functions, shape (num_samples, N)
- `uT`: Terminal displacement functions, shape (num_samples, N)
- `labels`: Dataset type labels, shape (num_samples,)
- `meta_pickle`: Pickled list of metadata dictionaries
- `quality_pickle`: Pickled dataset-level quality summary

## Metadata

Each sample's metadata dictionary contains:
- `sample_id`: Sample index
- `u0_distribution`: Initial condition distribution type
- `c_distribution`: Wave speed distribution type
- `T`: Terminal time
- `dt`: Time step used in simulation
- `N`: Number of spatial grid points
- `dx`: Spatial grid spacing

## Loading Data

```python
import numpy as np
import pickle

# Load dataset
data = np.load('train_dataset.npz', allow_pickle=True)
u0 = data['u0']
c = data['c']
uT = data['uT']
labels = data['labels']
meta = pickle.loads(data['meta_pickle'])
```

## Grid Information

- Spatial domain: x in [0, 1]
- Boundary conditions: u(0,t) = u(1,t) = 0
- PDE: u_tt = d_x(c(x)^2 u_x)
- Terminal time: T = 1.0
"""

    with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print("Generated README.md")


if __name__ == "__main__":
    # Grid setup
    N = 128  # Spatial resolution
    x = np.linspace(0, 1, N)

    # Generate all datasets
    generate_all_datasets(x)
