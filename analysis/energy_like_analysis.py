"""
Terminal potential-like proxy analysis for wave-equation predictions.

The solver-side ``compute_energy_like`` requires two consecutive time levels to
approximate a velocity term. The learned models in this project predict only a
terminal displacement field, so this analysis deliberately does not claim to
measure total energy or energy conservation.

Instead, it compares the terminal gradient-based proxy

    P_T(u, c) = integral c(x)^2 |u_x(x,T)|^2 dx,

which is the potential-like spatial part of the wave energy. Use it only as an
auxiliary physical-consistency diagnostic.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from data.dataset_loader import get_canonical_splits


ANALYSIS_SPLITS = tuple(get_canonical_splits(include_auxiliary=False)[1:])
MODEL_ORDER = ('FNO', 'DeepONet', 'fno', 'deeponet')


def _as_2d_samples(values, name):
    values = np.asarray(values)
    if values.ndim == 1:
        return values[np.newaxis, :]
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape (N,) or (num_samples,N), got {values.shape}")
    return values


def compute_terminal_potential_proxy(u, c, x):
    """Compute integral c(x)^2 |u_x|^2 dx for one terminal displacement."""
    u = np.asarray(u)
    c = np.asarray(c)
    x = np.asarray(x)
    if u.shape != c.shape or u.shape != x.shape:
        raise ValueError(f"u, c, and x must have the same shape, got {u.shape}, {c.shape}, {x.shape}")
    dx = x[1] - x[0]
    u_x = np.gradient(u, dx)
    return float(np.sum(c**2 * u_x**2) * dx)


def analyze_energy_consistency(u_pred_list, u_true_list, c_list, x, label_list=None):
    """
    Compare terminal potential-like proxy values between predictions and truth.

    This compatibility wrapper keeps the historical function name used by
    ``run_analysis.py``. The returned keys still use ``energy_*`` for backward
    compatibility, but the added ``quantity_name`` field records the precise
    interpretation.
    """
    results = {
        'quantity_name': 'terminal_potential_like_proxy',
        'energy_pred': [],
        'energy_true': [],
        'energy_diff': [],
        'energy_rel_diff': [],
        'labels': [] if label_list is not None else None,
    }

    for idx, (u_pred, u_true, c) in enumerate(zip(u_pred_list, u_true_list, c_list)):
        proxy_pred = compute_terminal_potential_proxy(u_pred, c, x)
        proxy_true = compute_terminal_potential_proxy(u_true, c, x)
        proxy_diff = abs(proxy_pred - proxy_true)
        proxy_rel_diff = proxy_diff / (proxy_true + 1e-10)

        results['energy_pred'].append(proxy_pred)
        results['energy_true'].append(proxy_true)
        results['energy_diff'].append(proxy_diff)
        results['energy_rel_diff'].append(proxy_rel_diff)
        if label_list is not None:
            results['labels'].append(label_list[idx])

    return results


def analyze_split_energy_proxy(split_predictions, x):
    """
    Compute proxy diagnostics for split/model organized predictions.

    Parameters
    ----------
    split_predictions:
        ``{split: {model: (u_pred, u_true, c)}}`` where each array is shaped
        ``(num_samples,N)``.
    x:
        One-dimensional spatial grid.
    """
    split_results = {}
    for split_name, model_data in split_predictions.items():
        if split_name not in ANALYSIS_SPLITS:
            raise ValueError(f"Unknown analysis split '{split_name}'. Expected one of {ANALYSIS_SPLITS}")
        split_results[split_name] = {}
        for model_name, (u_pred, u_true, c_values) in model_data.items():
            u_pred = _as_2d_samples(u_pred, f'{split_name}/{model_name} prediction')
            u_true = _as_2d_samples(u_true, f'{split_name}/{model_name} target')
            c_values = _as_2d_samples(c_values, f'{split_name}/{model_name} wave speed')
            if u_pred.shape != u_true.shape or u_pred.shape != c_values.shape:
                raise ValueError(
                    f"{split_name}/{model_name} arrays must share shape, got "
                    f"{u_pred.shape}, {u_true.shape}, {c_values.shape}"
                )
            split_results[split_name][model_name] = analyze_energy_consistency(
                list(u_pred),
                list(u_true),
                list(c_values),
                x,
            )
    return split_results


def plot_energy_comparison(results, model_names=None, output_dir='results/figures', filename='energy_like_analysis.png'):
    """
    Plot terminal potential-like proxy comparison for one split or aggregate set.
    """
    os.makedirs(output_dir, exist_ok=True)

    if model_names is None:
        model_names = [name for name in MODEL_ORDER if name in results]
        model_names.extend(name for name in results.keys() if name not in model_names)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for model_name in model_names:
        proxy_pred = results[model_name]['energy_pred']
        proxy_true = results[model_name]['energy_true']
        ax.scatter(proxy_true, proxy_pred, alpha=0.6, label=model_name, s=50)

    all_values = []
    for model_name in model_names:
        all_values.extend(results[model_name]['energy_true'])
        all_values.extend(results[model_name]['energy_pred'])
    value_range = [np.min(all_values), np.max(all_values)]
    ax.plot(value_range, value_range, 'k--', linewidth=2, label='Perfect agreement')

    ax.set_xlabel('True terminal proxy')
    ax.set_ylabel('Predicted terminal proxy')
    ax.set_title('Terminal Potential-like Proxy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    positions = []
    errors = []
    labels = []
    for idx, model_name in enumerate(model_names):
        rel_errors = results[model_name]['energy_rel_diff']
        positions.extend([idx] * len(rel_errors))
        errors.extend(rel_errors)
        labels.append(model_name)

    ax.scatter(positions, errors, alpha=0.6, s=50)
    ax.set_ylabel('Relative Proxy Error')
    ax.set_xlabel('Model')
    ax.set_title('Terminal Proxy Relative Error')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved terminal proxy analysis to {out_path}")


def plot_split_energy_proxy(split_results, output_dir='results/figures'):
    """Generate split-wise terminal proxy figures for ID/OOD analysis."""
    ordered_splits = [split for split in ANALYSIS_SPLITS if split in split_results]
    ordered_splits.extend(split for split in split_results.keys() if split not in ordered_splits)
    for split_name in ordered_splits:
        results = split_results[split_name]
        safe_name = split_name.replace('-', '_')
        plot_energy_comparison(
            results,
            output_dir=output_dir,
            filename=f'terminal_potential_proxy_{safe_name}.png',
        )


if __name__ == "__main__":
    print("Terminal potential-like proxy module ready (call functions with prediction data)")
