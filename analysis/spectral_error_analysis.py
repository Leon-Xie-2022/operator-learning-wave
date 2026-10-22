"""
Spectral error analysis for fixed-endpoint wave-equation experiments.

The main analysis uses a discrete sine basis because the PDE and learned
outputs satisfy homogeneous Dirichlet endpoint conditions. FFT-based spectra
are still available as an optional diagnostic, but paper-facing figures should
prefer ``basis="sine"`` and describe the low/high split as a
training-band-inspired heuristic rather than a strict physical frequency
boundary.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset_loader import dataset_file_path, get_canonical_splits, load_dataset_npz
from models.deeponet import DeepONet1D
from models.fno1d import FNO1D


ANALYSIS_SPLITS = tuple(get_canonical_splits(include_auxiliary=False)[1:])
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 16
LEGEND_SIZE = 14
PANEL_LABEL_SIZE = 16


def _as_batch(u):
    u = np.asarray(u)
    if u.ndim == 1:
        return u[np.newaxis, :]
    if u.ndim != 2:
        raise ValueError(f"Expected shape (N,) or (batch,N), got {u.shape}")
    return u


def _smooth_curve(values, window=3):
    """
    Apply a light moving-average smoothing for paper-facing trend plots.

    The default window is intentionally small so the modal-error trend becomes
    easier to read without obscuring where the error rises relative to the
    cutoff.
    """
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size < window:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode='edge')
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode='valid')


def _style_axis(ax):
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.22, color='0.82', linewidth=0.6)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)


def _panel_label(index, label):
    return f"({chr(ord('a') + index)}) {label}"


def _add_panel_text(ax, text):
    ax.text(0.0, 1.08, text, transform=ax.transAxes, ha='left', va='top', fontsize=PANEL_LABEL_SIZE, fontweight='bold')


def sine_coefficients(u):
    """Project fixed-endpoint samples onto the Dirichlet sine basis."""
    u = _as_batch(u)
    n_grid = u.shape[1]
    if n_grid < 3:
        raise ValueError("Need at least three grid points for a Dirichlet sine spectrum")

    interior = u[:, 1:-1]
    j = np.arange(1, n_grid - 1)
    modes = np.arange(1, n_grid - 1)
    basis = np.sin(np.pi * np.outer(j, modes) / (n_grid - 1))
    coeffs = (2.0 / (n_grid - 1)) * interior @ basis
    return modes, coeffs


def compute_error_spectrum(u_pred, u_true, x=None, basis='sine'):
    """
    Compute average prediction-error spectrum.

    Parameters
    ----------
    u_pred, u_true:
        Arrays of shape ``(num_samples,N)`` or ``(N,)``.
    x:
        Spatial grid. Required only for ``basis="fft"``.
    basis:
        ``"sine"`` for Dirichlet-compatible DST-I-style projection, or
        ``"fft"`` for the older periodic-complex-exponential diagnostic.
    """
    u_pred = _as_batch(u_pred)
    u_true = _as_batch(u_true)
    if u_pred.shape != u_true.shape:
        raise ValueError(f"Prediction and target shapes differ: {u_pred.shape} vs {u_true.shape}")

    error = u_pred - u_true

    if basis == 'sine':
        modes, error_coeffs = sine_coefficients(error)
        _, solution_coeffs = sine_coefficients(u_true)
        return modes, np.mean(error_coeffs**2, axis=0), np.mean(solution_coeffs**2, axis=0)

    if basis == 'fft':
        if x is None:
            x = np.linspace(0.0, 1.0, u_pred.shape[1])
        dx = x[1] - x[0]
        error_fft = np.fft.fft(error, axis=-1)
        solution_fft = np.fft.fft(u_true, axis=-1)
        freqs = np.fft.fftfreq(u_pred.shape[1], d=dx)
        positive_idx = freqs > 0
        return (
            freqs[positive_idx],
            np.mean(np.abs(error_fft) ** 2, axis=0)[positive_idx],
            np.mean(np.abs(solution_fft) ** 2, axis=0)[positive_idx],
        )

    raise ValueError(f"Unknown spectral basis: {basis}")


def compute_band_energy_share(u_pred, u_true, low_mode_cutoff=5, basis='sine'):
    """Return low/high error-energy percentages for a heuristic mode split."""
    modes, error_power, _ = compute_error_spectrum(u_pred, u_true, basis=basis)
    low_mask = modes <= low_mode_cutoff
    low_energy = float(np.sum(error_power[low_mask]))
    high_energy = float(np.sum(error_power[~low_mask]))
    total = low_energy + high_energy + 1e-12
    return {
        'low_percent': 100.0 * low_energy / total,
        'high_percent': 100.0 * high_energy / total,
        'low_energy': low_energy,
        'high_energy': high_energy,
    }


def plot_spectral_error_comparison(
    model_results,
    x=None,
    output_dir='results/figures',
    basis='sine',
    filename='spectral_error_comparison.png',
    include_solution_reference=False,
):
    """Overlay model error spectra in the selected basis."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = {'fno': '#4C78A8', 'deeponet': '#E45756'}
    reference_plotted = False
    for model_name, (u_pred, u_true) in model_results.items():
        modes, error_spec, sol_spec = compute_error_spectrum(u_pred, u_true, x=x, basis=basis)
        ax.loglog(modes, error_spec + 1e-18, color=colors.get(model_name, None), linewidth=2, label=f'{model_name} error')
        if include_solution_reference and not reference_plotted:
            ax.loglog(modes, sol_spec + 1e-18, color='k', linewidth=1.5, linestyle=':', label='true solution')
            reference_plotted = True

    ax.set_xlabel('Sine mode k' if basis == 'sine' else 'FFT frequency')
    ax.set_ylabel('Mean squared coefficient')
    ax.set_title('Prediction Error Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_error_frequency_distribution(
    model_results,
    x=None,
    low_mode_cutoff=5,
    output_dir='results/figures',
    basis='sine',
    filename='error_frequency_distribution.png',
):
    """
    Plot low/high spectral error energy shares.

    ``low_mode_cutoff=5`` is a training-band-inspired heuristic because the
    training initial displacements use sine modes 1--5. For variable-coefficient
    terminal states, it is not a strict invariant frequency boundary.
    """
    os.makedirs(output_dir, exist_ok=True)

    labels = list(model_results.keys())
    low_freq_errors = []
    high_freq_errors = []
    for _, (u_pred, u_true) in model_results.items():
        shares = compute_band_energy_share(u_pred, u_true, low_mode_cutoff=low_mode_cutoff, basis=basis)
        low_freq_errors.append(shares['low_percent'])
        high_freq_errors.append(shares['high_percent'])

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(labels))
    width = 0.35
    ax.bar(x_pos - width / 2, low_freq_errors, width, label=f'Modes 1-{low_mode_cutoff}', color='#4C78A8', alpha=0.85)
    ax.bar(x_pos + width / 2, high_freq_errors, width, label=f'Modes >{low_mode_cutoff}', color='#F58518', alpha=0.85)
    ax.set_ylabel('Share of Error Energy (%)')
    ax.set_title('Heuristic Spectral Location of Prediction Error')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend(title='Dirichlet sine band' if basis == 'sine' else 'FFT band')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_split_spectral_analysis(split_model_results, x=None, low_mode_cutoff=5, output_dir='results/figures', basis='sine'):
    """
    Generate one spectral comparison and one band-share chart for each split.

    Parameters
    ----------
    split_model_results:
        ``{split_name: {model_name: (u_pred, u_true)}}``.
    """
    for split_name, model_results in split_model_results.items():
        safe_name = split_name.replace('-', '_')
        plot_spectral_error_comparison(
            model_results,
            x=x,
            output_dir=output_dir,
            basis=basis,
            filename=f'spectral_error_{safe_name}.png',
        )
        plot_error_frequency_distribution(
            model_results,
            x=x,
            low_mode_cutoff=low_mode_cutoff,
            output_dir=output_dir,
            basis=basis,
            filename=f'error_frequency_distribution_{safe_name}.png',
        )


def plot_id_vs_ood_frequency_spectral_error(
    split_model_results,
    x=None,
    output_dir='results/figures',
    basis='sine',
    filename='spectral_error_id_vs_ood_frequency.png',
    retained_modes=16,
    smoothing_window=3,
):
    """
    Create a paper-focused two-panel spectral-error figure comparing FNO and
    DeepONet on ID versus OOD-frequency.

    Parameters
    ----------
    split_model_results:
        ``{split_name: {model_name: (u_pred, u_true)}}`` containing at least
        ``id_test`` and ``ood_frequency``.
    """
    required_splits = ('id_test', 'ood_frequency')
    missing = [split for split in required_splits if split not in split_model_results]
    if missing:
        raise KeyError(f"Missing splits for paper spectral figure: {missing}")

    os.makedirs(output_dir, exist_ok=True)
    colors = {'fno': '#4C78A8', 'deeponet': '#E45756'}
    display_names = {'fno': 'FNO', 'deeponet': 'DeepONet'}
    panel_titles = {'id_test': 'In-distribution (ID)', 'ood_frequency': 'Frequency-shift OOD'}

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True, facecolor='white')

    for panel_idx, (ax, split) in enumerate(zip(axes, required_splits)):
        split_max = 1e-18
        mode_min = None
        mode_max = None
        for model_name in ('fno', 'deeponet'):
            u_pred, u_true = split_model_results[split][model_name]
            modes, error_spec, _ = compute_error_spectrum(u_pred, u_true, x=x, basis=basis)
            plot_spec = _smooth_curve(error_spec, window=smoothing_window)
            split_max = max(split_max, float(np.max(plot_spec)))
            mode_min = float(modes[0])
            mode_max = float(modes[-1])
            ax.plot(
                modes,
                plot_spec + 1e-18,
                color=colors[model_name],
                linewidth=2.5,
                alpha=0.95,
                label=display_names[model_name],
            )

        ax.set_xlim(left=mode_min, right=mode_max)
        ax.axvspan(retained_modes, mode_max, color='0.5', alpha=0.10, zorder=0)
        ax.axvline(
            retained_modes,
            color='0.35',
            linestyle='--',
            linewidth=1.4,
            alpha=0.9,
        )
        ax.text(
            0.13,
            0.93,
            'spectral cutoff\nk = 16',
            transform=ax.transAxes,
            fontsize=14,
            color='0.30',
            ha='left',
            va='top',
        )

        _add_panel_text(ax, _panel_label(panel_idx, panel_titles[split]))
        ax.set_xlabel('Mode index k' if basis == 'sine' else 'Frequency bin', fontsize=AXIS_LABEL_SIZE)
        ax.set_yscale('log')
        _style_axis(ax)

    axes[0].set_ylabel('Mean squared modal error', fontsize=AXIS_LABEL_SIZE)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(
        handles,
        labels,
        loc='upper right',
        # bbox_to_anchor=(0.99, 0.99),
        frameon=False,
        fontsize=LEGEND_SIZE,
    )
    fig.tight_layout(rect=[0, 0, 0.985, 0.96])
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def infer_deeponet_n(checkpoint):
    config = checkpoint['config']
    if 'N' in config:
        return int(config['N'])
    branch_weight = checkpoint['model_state_dict'].get('branch.net.0.weight')
    if branch_weight is None:
        raise KeyError("DeepONet checkpoint is missing config['N'] and branch.net.0.weight")
    return int(branch_weight.shape[1] // 2)


def load_fno(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    if config.get('normalize', False):
        raise ValueError("Standalone spectral analysis currently expects unnormalized checkpoints")
    model = FNO1D(
        modes=config['modes'],
        width=config['width'],
        num_layers=config['num_layers'],
        padding=config.get('padding', 0),
        enforce_dirichlet=config.get('enforce_dirichlet', True),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def load_deeponet(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    if config.get('normalize', False):
        raise ValueError("Standalone spectral analysis currently expects unnormalized checkpoints")
    model = DeepONet1D(
        N=infer_deeponet_n(checkpoint),
        branch_hidden=config['branch_hidden'],
        trunk_hidden=config['trunk_hidden'],
        num_layers=config['num_layers'],
        use_output_bias=config.get('use_output_bias', False),
        enforce_dirichlet=config.get('enforce_dirichlet', False),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_models_for_split(fno, deeponet, data, device):
    u0 = torch.tensor(data['u0'], dtype=torch.float32, device=device)
    c = torch.tensor(data['c'], dtype=torch.float32, device=device)
    x = torch.linspace(0.0, 1.0, u0.shape[1], dtype=torch.float32, device=device).repeat(u0.shape[0], 1)
    branch = torch.cat([u0, c], dim=-1)
    with torch.no_grad():
        fno_pred = fno(torch.stack([u0, c], dim=-1)).squeeze(-1).cpu().numpy()
        deeponet_pred = deeponet.forward_branch_trunk(branch, x).cpu().numpy()
    return {
        'fno': (fno_pred, data['uT']),
        'deeponet': (deeponet_pred, data['uT']),
    }


def run_checkpoint_spectral_analysis(
    fno_checkpoint='results/checkpoints/fno/best_model.pth',
    deeponet_checkpoint='results/checkpoints/deeponet/best_model.pth',
    data_dir='data',
    output_dir='results/figures',
    basis='sine',
):
    """Load checkpoints/data and generate split-wise spectral analysis figures."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fno = load_fno(fno_checkpoint, device)
    deeponet = load_deeponet(deeponet_checkpoint, device)

    split_results = {}
    x = None
    for split in ANALYSIS_SPLITS:
        data = load_dataset_npz(dataset_file_path(data_dir, split))
        if x is None:
            x = np.linspace(0.0, 1.0, data['uT'].shape[1])
        split_results[split] = predict_models_for_split(fno, deeponet, data, device)

    plot_split_spectral_analysis(split_results, x=x, output_dir=output_dir, basis=basis)
    return split_results


def main():
    parser = argparse.ArgumentParser(description='Generate Dirichlet-basis spectral error analysis figures')
    parser.add_argument('--fno_checkpoint', default='results/checkpoints/fno/best_model.pth')
    parser.add_argument('--deeponet_checkpoint', default='results/checkpoints/deeponet/best_model.pth')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default='results/figures')
    parser.add_argument('--basis', choices=['sine', 'fft'], default='sine')
    args = parser.parse_args()

    run_checkpoint_spectral_analysis(
        fno_checkpoint=args.fno_checkpoint,
        deeponet_checkpoint=args.deeponet_checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        basis=args.basis,
    )


if __name__ == "__main__":
    main()
