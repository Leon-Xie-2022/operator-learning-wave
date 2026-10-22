"""
Dataset Loader for Wave Equation Operator Learning

This module provides unified data loading interfaces for different model architectures.
Supports FNO and the fixed-grid supervised DeepONet baseline used in this project,
with optional standardization.

Canonical public splits:
- train
- validation
- id_test
- ood_frequency
- ood_smoothness

The legacy on-disk file `ood_smooth_dataset.npz` is exposed publicly as
`ood_smoothness`. The auxiliary `ood_rough` split is supported but is not
returned by default because it is outside the main project_plan.md evaluation set.

Data shapes returned by DataLoader batches:
- FNO: input (batch, N, 2) [u0, c], output (batch, N, 1) [uT]
- DeepONet: branch (batch, 2*N) [u0;c], trunk (batch, N) fixed normalized grid x,
  target (batch, N)

Standardization:
- Optional z-score normalization using training set statistics
- Statistics are computed from the training split only and reused for all splits
"""

import json
import numpy as np
import os
import pickle
from torch.utils.data import Dataset, DataLoader
import torch


NORMALIZATION_EPS = 1e-8
SUPPORTED_MODEL_TYPES = {'fno', 'deeponet'}
CANONICAL_SPLITS = ('train', 'validation', 'id_test', 'ood_frequency', 'ood_smoothness')
AUXILIARY_SPLITS = ('ood_rough',)
SPLIT_ALIASES = {'ood_smooth': 'ood_smoothness'}
SPLIT_FILE_STEMS = {'ood_smoothness': 'ood_smooth'}


def canonical_split_name(split):
    """Return the canonical public split name for a split or legacy alias."""
    return SPLIT_ALIASES.get(split, split)


def dataset_file_stem(split):
    """Return the on-disk dataset file stem for a canonical split name."""
    canonical = canonical_split_name(split)
    return SPLIT_FILE_STEMS.get(canonical, canonical)


def dataset_file_path(data_dir, split):
    """Return the .npz path for a canonical split name or accepted alias."""
    return os.path.join(data_dir, f'{dataset_file_stem(split)}_dataset.npz')


def get_canonical_splits(include_auxiliary=False):
    """Return the project-plan canonical splits, optionally including auxiliary data."""
    splits = list(CANONICAL_SPLITS)
    if include_auxiliary:
        splits.extend(AUXILIARY_SPLITS)
    return splits


def load_dataset_npz(filename):
    """
    Load dataset from .npz file.

    Parameters:
    -----------
    filename : str
        Path to .npz file

    Returns:
    --------
    data : dict
        Loaded data dictionary
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset file not found: {filename}")
    loaded = np.load(filename, allow_pickle=True)
    required_keys = {'u0', 'c', 'uT', 'labels', 'meta_pickle'}
    missing = required_keys.difference(loaded.keys())
    if missing:
        raise KeyError(f"Dataset file {filename} is missing keys: {sorted(missing)}")
    data = {key: loaded[key] for key in loaded.keys() if key not in ('meta_pickle', 'quality_pickle', 'meta')}
    data['meta'] = pickle.loads(loaded['meta_pickle'])
    if 'quality_pickle' in loaded:
        data['quality_summary'] = pickle.loads(loaded['quality_pickle'])
    return data


def compute_normalization_stats(data_dict):
    """
    Compute normalization statistics from training data.

    Parameters:
    -----------
    data_dict : dict
        Training data dictionary

    Returns:
    --------
    stats : dict
        Normalization statistics
    """
    required = ('u0', 'c', 'uT')
    for key in required:
        if key not in data_dict:
            raise KeyError(f"Missing required data key: {key}")

    u0 = data_dict['u0']
    c = data_dict['c']
    uT = data_dict['uT']

    stats = {
        'u0_mean': np.mean(u0),
        'u0_std': max(float(np.std(u0)), NORMALIZATION_EPS),
        'c_mean': np.mean(c),
        'c_std': max(float(np.std(c)), NORMALIZATION_EPS),
        'uT_mean': np.mean(uT),
        'uT_std': max(float(np.std(uT)), NORMALIZATION_EPS),
    }

    return stats


def apply_normalization(data_dict, stats, normalize=True):
    """
    Apply or reverse normalization to data.

    Parameters:
    -----------
    data_dict : dict
        Data dictionary
    stats : dict
        Normalization statistics
    normalize : bool
        If True, apply normalization; if False, denormalize

    Returns:
    --------
    data_dict : dict
        Normalized/denormalized data
    """
    data_dict = data_dict.copy()
    for key in ('u0', 'c', 'uT'):
        data_dict[key] = np.array(data_dict[key], copy=True)

    if not normalize:
        # Denormalize
        data_dict['u0'] = data_dict['u0'] * stats['u0_std'] + stats['u0_mean']
        data_dict['c'] = data_dict['c'] * stats['c_std'] + stats['c_mean']
        data_dict['uT'] = data_dict['uT'] * stats['uT_std'] + stats['uT_mean']
        return data_dict

    # Normalize
    data_dict['u0'] = (data_dict['u0'] - stats['u0_mean']) / stats['u0_std']
    data_dict['c'] = (data_dict['c'] - stats['c_mean']) / stats['c_std']
    data_dict['uT'] = (data_dict['uT'] - stats['uT_mean']) / stats['uT_std']
    return data_dict


class WaveEquationDataset(Dataset):
    """
    PyTorch Dataset for wave equation data.
    """

    def __init__(self, data_dict, model_type='fno', normalize=False, stats=None):
        """
        Initialize dataset.

        Parameters:
        -----------
        data_dict : dict
            Raw data dictionary
        model_type : str
            'fno' or 'deeponet'
        normalize : bool
            Whether to apply normalization
        stats : dict, optional
            Normalization statistics (required if normalize=True)
        """
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}. Supported types: {sorted(SUPPORTED_MODEL_TYPES)}")
        if normalize and stats is None:
            raise ValueError("stats must be provided when normalize=True")

        self.model_type = model_type
        self.normalize = normalize
        self.stats = stats

        # Apply normalization if requested
        if normalize and stats is not None:
            self.data = apply_normalization(data_dict, stats, normalize=True)
        else:
            self.data = data_dict

        self.N = self.data['u0'].shape[1]
        self.num_samples = self.data['u0'].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        u0 = self.data['u0'][idx]
        c = self.data['c'][idx]
        uT = self.data['uT'][idx]

        if self.model_type == 'fno':
            # FNO: input (N, 2), output (N, 1)
            x = np.stack([u0, c], axis=-1)  # (N, 2)
            y = uT.reshape(-1, 1)  # (N, 1)
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        elif self.model_type == 'deeponet':
            # Fixed-grid DeepONet baseline: branch encodes [u0; c], trunk is
            # the normalized spatial grid shared by all samples.
            branch = np.concatenate([u0, c])  # (2*N,)
            trunk = np.arange(self.N, dtype=np.float32) / (self.N - 1)  # normalized x
            return (torch.tensor(branch, dtype=torch.float32),
                    torch.tensor(trunk, dtype=torch.float32),
                    torch.tensor(uT, dtype=torch.float32))

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


def get_data_loaders(
    data_dir='data',
    batch_size=32,
    model_type='fno',
    normalize=False,
    dataset_types=None,
    include_auxiliary=False,
):
    """
    Get PyTorch DataLoaders for all datasets.

    Parameters:
    -----------
    data_dir : str
        Directory containing dataset files
    batch_size : int
        Batch size for DataLoaders
    model_type : str
        Model type ('fno' or 'deeponet')
    normalize : bool
        Whether to apply normalization
    dataset_types : sequence of str, optional
        Splits to load. Names may be canonical or accepted aliases. When None,
        the canonical project-plan splits are loaded.
    include_auxiliary : bool
        If True and dataset_types is None, also load auxiliary splits such as
        'ood_rough'.

    Returns:
    --------
    loaders : dict
        Dictionary of DataLoaders keyed by canonical split name
    stats : dict
        Normalization statistics (if normalize=True)
    """
    # Load training data for statistics
    train_data = load_dataset_npz(os.path.join(data_dir, 'train_dataset.npz'))
    stats = None
    if normalize:
        stats = compute_normalization_stats(train_data)

    # Create datasets. Public keys are canonical even when existing files use a
    # legacy stem, e.g. ood_smoothness -> ood_smooth_dataset.npz.
    datasets = {}
    if dataset_types is None:
        dataset_types = get_canonical_splits(include_auxiliary=include_auxiliary)

    for dtype in dataset_types:
        canonical = canonical_split_name(dtype)
        valid_splits = set(CANONICAL_SPLITS) | set(AUXILIARY_SPLITS)
        if canonical not in valid_splits:
            raise ValueError(f"Unknown dataset split: {dtype}. Valid splits: {sorted(valid_splits)}")
        data = load_dataset_npz(dataset_file_path(data_dir, canonical))
        datasets[canonical] = WaveEquationDataset(data, model_type=model_type,
                                                  normalize=normalize, stats=stats)

    # Create DataLoaders
    loaders = {}
    for dtype, dataset in datasets.items():
        shuffle = (dtype == 'train')
        loaders[dtype] = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=0)

    return loaders, stats


def get_raw_data(data_dir='data', dataset_type='train'):
    """
    Get raw numpy arrays for a specific dataset.

    Parameters:
    -----------
    data_dir : str
        Directory containing dataset files
    dataset_type : str
        Dataset split. Canonical names are preferred; accepted legacy aliases
        such as 'ood_smooth' are mapped to canonical file locations.

    Returns:
    --------
    u0, c, uT : ndarray
        Raw data arrays
    """
    data = load_dataset_npz(dataset_file_path(data_dir, dataset_type))
    return data['u0'], data['c'], data['uT']


def save_normalization_stats(stats, filename):
    """
    Save normalization statistics to file.

    Parameters:
    -----------
    stats : dict
        Normalization statistics
    filename : str
        Output filename
    """
    if filename.endswith('.json'):
        serializable = {key: float(value) for key, value in stats.items()}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(stats, f)


def load_normalization_stats(filename):
    """
    Load normalization statistics from file.

    Parameters:
    -----------
    filename : str
        Statistics filename

    Returns:
    --------
    stats : dict
        Normalization statistics
    """
    if filename.endswith('.json'):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    with open(filename, 'rb') as f:
        return pickle.load(f)
