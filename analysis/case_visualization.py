"""
Paper case visualization for fixed-grid wave-equation experiments.

This module can be used as a plotting utility or as a standalone script. The
standalone path loads trained FNO/DeepONet checkpoints, evaluates the main
splits, selects representative samples by relative L2 error, and writes unified
paper figures for ID, OOD-frequency, and OOD-smoothness cases.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset_loader import dataset_file_path, load_dataset_npz
from models.deeponet import DeepONet1D
from models.fno1d import FNO1D


MODEL_COLORS = {'FNO': '#4C78A8', 'DeepONet': '#E45756'}
SPLITS = {
    'id_test': 'ID',
    'ood_frequency': 'OOD-frequency',
    'ood_smoothness': 'OOD-smoothness',
}
AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 12
PANEL_LABEL_SIZE = 12


def relative_l2(u_pred, u_true):
    return float(np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-12))


def _style_axis(ax):
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.22, color='0.82', linewidth=0.6)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)


def _panel_label(index, label):
    return f"({chr(ord('a') + index)}) {label}"


def _add_panel_text(ax, text):
    ax.text(0.0, 1.1, text, transform=ax.transAxes, ha='left', va='top', fontsize=PANEL_LABEL_SIZE, fontweight='bold')


def plot_sample_inputs_outputs(sample_id, u0, c, uT_true, uT_fno=None, uT_deeponet=None, x=None, output_dir='results/figures'):
    """
    Plot one sample's inputs, terminal predictions, and relative-error curves.
    """
    if x is None:
        x = np.linspace(0, 1, len(u0))

    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, u0, color='#4C78A8', linewidth=2)
    ax1.axhline(0, color='0.2', linewidth=0.6)
    ax1.set_title('Initial displacement', fontsize=20)
    ax1.set_xlabel('x', fontsize=AXIS_LABEL_SIZE)
    ax1.set_ylabel('u0(x)', fontsize=AXIS_LABEL_SIZE)
    _style_axis(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, c, color='#54A24B', linewidth=2)
    ax2.set_title('Wave speed', fontsize=20)
    ax2.set_xlabel('x', fontsize=AXIS_LABEL_SIZE+10)
    ax2.set_ylabel('c(x)', fontsize=AXIS_LABEL_SIZE+10)
    _style_axis(ax2)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(x, uT_true, color='k', linewidth=2.2, label='True', zorder=10)
    if uT_fno is not None:
        ax3.plot(x, uT_fno, color=MODEL_COLORS['FNO'], linestyle='--', linewidth=2, label='FNO')
    if uT_deeponet is not None:
        ax3.plot(x, uT_deeponet, color=MODEL_COLORS['DeepONet'], linestyle='--', linewidth=2, label='DeepONet')
    ax3.set_title('Terminal solution', fontsize=20)
    ax3.set_xlabel('x', fontsize=AXIS_LABEL_SIZE+10)
    ax3.set_ylabel('u(x,T)', fontsize=AXIS_LABEL_SIZE+10)
    ax3.legend(loc='best', fontsize=10)
    _style_axis(ax3)

    if uT_fno is not None:
        ax4 = fig.add_subplot(gs[2, 0])
        error_fno = uT_fno - uT_true
        ax4.plot(x, error_fno, color=MODEL_COLORS['FNO'], linewidth=2)
        ax4.axhline(0, color='0.2', linewidth=0.6)
        ax4.set_title(f'FNO error, rel L2={relative_l2(uT_fno, uT_true):.4f}', fontsize=20)
        ax4.set_xlabel('x', fontsize=AXIS_LABEL_SIZE+10)
        ax4.set_ylabel('error', fontsize=AXIS_LABEL_SIZE+10)
        _style_axis(ax4)

    if uT_deeponet is not None:
        ax5 = fig.add_subplot(gs[2, 1])
        error_deeponet = uT_deeponet - uT_true
        ax5.plot(x, error_deeponet, color=MODEL_COLORS['DeepONet'], linewidth=2)
        ax5.axhline(0, color='0.2', linewidth=0.6)
        ax5.set_title(f'DeepONet error, rel L2={relative_l2(uT_deeponet, uT_true):.4f}', fontsize=20)
        ax5.set_xlabel('x', fontsize=AXIS_LABEL_SIZE+10)
        ax5.set_ylabel('error', fontsize=AXIS_LABEL_SIZE+10)
        _style_axis(ax5)

    fig.suptitle(f'Sample {sample_id}', fontsize=20, fontweight='bold')
    filename = os.path.join(output_dir, f'sample_{sample_id}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def plot_model_comparison_grid(samples_data, x=None, output_dir='results/figures', filename='sample_comparison_grid.png'):
    """Plot representative samples in a stable paper-style comparison grid."""
    if x is None:
        x = np.linspace(0, 1, len(samples_data[0]['u0']))

    os.makedirs(output_dir, exist_ok=True)

    num_samples = len(samples_data)
    fig, axes = plt.subplots(num_samples, 3, figsize=(14, 3.4 * num_samples), squeeze=False)

    for idx, sample in enumerate(samples_data):
        ax = axes[idx, 0]
        ax.plot(x, sample['uT_true'], color='k', linewidth=2.2, label='True')
        ax.plot(x, sample['uT_fno'], color=MODEL_COLORS['FNO'], linestyle='--', linewidth=1.8, label='FNO')
        ax.plot(x, sample['uT_deeponet'], color=MODEL_COLORS['DeepONet'], linestyle='--', linewidth=1.8, label='DeepONet')
        ax.set_ylabel('u(x,T)')
        ax.set_title(f"{sample['type']} #{sample['index']}: terminal solution")
        ax.legend(fontsize=8)
        _style_axis(ax)

        ax = axes[idx, 1]
        error_fno = sample['uT_fno'] - sample['uT_true']
        ax.plot(x, error_fno, color=MODEL_COLORS['FNO'], linewidth=1.8)
        ax.axhline(0, color='0.2', linewidth=0.6)
        ax.set_ylabel('error')
        ax.set_title(f"FNO rel L2={relative_l2(sample['uT_fno'], sample['uT_true']):.4f}")
        _style_axis(ax)

        ax = axes[idx, 2]
        error_deeponet = sample['uT_deeponet'] - sample['uT_true']
        ax.plot(x, error_deeponet, color=MODEL_COLORS['DeepONet'], linewidth=1.8)
        ax.axhline(0, color='0.2', linewidth=0.6)
        ax.set_ylabel('error')
        ax.set_title(f"DeepONet rel L2={relative_l2(sample['uT_deeponet'], sample['uT_true']):.4f}")
        _style_axis(ax)

    for ax in axes[-1, :]:
        ax.set_xlabel('x')

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_representative_cases(samples_data, x=None, output_dir='results/figures', filename='representative_cases_main.png'):
    """
    Plot one paper-ready representative-case figure with three columns:
    ID, OOD-frequency, and OOD-smoothness. Each column shows ground truth,
    FNO, DeepONet, and pointwise absolute errors.
    """
    if x is None:
        x = np.linspace(0, 1, len(samples_data[0]['u0']))

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, len(samples_data), figsize=(15.5, 6.1), squeeze=False, sharex='col', facecolor='white')

    for col, sample in enumerate(samples_data):
        fno_rel = relative_l2(sample['uT_fno'], sample['uT_true'])
        deeponet_rel = relative_l2(sample['uT_deeponet'], sample['uT_true'])
        fno_abs = np.abs(sample['uT_fno'] - sample['uT_true'])
        deeponet_abs = np.abs(sample['uT_deeponet'] - sample['uT_true'])

        ax_top = axes[0, col]
        ax_top.plot(x, sample['uT_true'], color='k', linewidth=2.4, label='Ground truth', zorder=10)
        ax_top.plot(x, sample['uT_fno'], color=MODEL_COLORS['FNO'], linewidth=2.0, linestyle='--', label=f'FNO ({fno_rel:.3f})')
        ax_top.plot(x, sample['uT_deeponet'], color=MODEL_COLORS['DeepONet'], linewidth=2.0, linestyle='-.', label=f'DeepONet ({deeponet_rel:.3f})')
        _add_panel_text(ax_top, _panel_label(col, sample['type']))
        ax_top.set_ylabel('u(x,T)', fontsize=AXIS_LABEL_SIZE)
        _style_axis(ax_top)

        ax_bottom = axes[1, col]
        ax_bottom.plot(x, fno_abs, color=MODEL_COLORS['FNO'], linewidth=2.0, label='|FNO - GT|')
        ax_bottom.plot(x, deeponet_abs, color=MODEL_COLORS['DeepONet'], linewidth=2.0, linestyle='--', label='|DeepONet - GT|')
        ax_bottom.fill_between(x, 0.0, fno_abs, color=MODEL_COLORS['FNO'], alpha=0.08)
        ax_bottom.fill_between(x, 0.0, deeponet_abs, color=MODEL_COLORS['DeepONet'], alpha=0.08)
        ax_bottom.set_xlabel('x', fontsize=AXIS_LABEL_SIZE)
        ax_bottom.set_ylabel('|error|', fontsize=AXIS_LABEL_SIZE)
        _style_axis(ax_bottom)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.955), frameon=False, fontsize=LEGEND_SIZE)
    fig.tight_layout(rect=[0.02, 0.08, 1, 1])
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_local_error_panels(samples_data, x=None, output_dir='results/figures', filename='representative_cases_local_error.png'):
    """
    Plot localized error diagnostics for representative cases. The top row
    highlights the most error-sensitive sub-interval for each case; the bottom
    row shows signed pointwise errors over the full domain.
    """
    if x is None:
        x = np.linspace(0, 1, len(samples_data[0]['u0']))

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, len(samples_data), figsize=(15.5, 5.9), squeeze=False, facecolor='white')
    window_radius = max(8, len(x) // 12)

    for col, sample in enumerate(samples_data):
        fno_err = sample['uT_fno'] - sample['uT_true']
        deeponet_err = sample['uT_deeponet'] - sample['uT_true']
        combined_abs = np.abs(fno_err) + np.abs(deeponet_err)
        center_idx = int(np.argmax(combined_abs))
        start = max(0, center_idx - window_radius)
        end = min(len(x), center_idx + window_radius + 1)
        if end - start < 2 * window_radius:
            start = max(0, end - 2 * window_radius)
            end = min(len(x), start + 2 * window_radius)

        ax_zoom = axes[0, col]
        ax_zoom.plot(x[start:end], sample['uT_true'][start:end], color='k', linewidth=2.2, label='Ground truth', zorder=10)
        ax_zoom.plot(x[start:end], sample['uT_fno'][start:end], color=MODEL_COLORS['FNO'], linewidth=2.0, linestyle='--', label='FNO')
        ax_zoom.plot(x[start:end], sample['uT_deeponet'][start:end], color=MODEL_COLORS['DeepONet'], linewidth=2.0, linestyle='-.', label='DeepONet')
        _add_panel_text(ax_zoom, _panel_label(col, f"{sample['type']} local region"))
        ax_zoom.set_ylabel('u(x,T)', fontsize=AXIS_LABEL_SIZE)
        _style_axis(ax_zoom)

        ax_err = axes[1, col]
        ax_err.plot(x, fno_err, color=MODEL_COLORS['FNO'], linewidth=1.9, label='FNO error')
        ax_err.plot(x, deeponet_err, color=MODEL_COLORS['DeepONet'], linewidth=1.9, linestyle='--', label='DeepONet error')
        ax_err.axhline(0, color='0.2', linewidth=0.7)
        ax_err.axvspan(x[start], x[end - 1], color='0.85', alpha=0.35)
        ax_err.set_xlabel('x', fontsize=AXIS_LABEL_SIZE)
        ax_err.set_ylabel('signed error', fontsize=AXIS_LABEL_SIZE)
        _style_axis(ax_err)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.985, 0.98), frameon=False, fontsize=LEGEND_SIZE)
    fig.tight_layout(rect=[0, 0, 0.98, 0.96])
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_ood_analysis(ood_samples_data, x=None, output_dir='results/figures', filename='ood_analysis.png'):
    """
    Plot OOD-frequency and OOD-smoothness representative samples.
    """
    if x is None:
        x = np.linspace(0, 1, len(ood_samples_data[0]['u0']))

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(len(ood_samples_data), 4, figsize=(16, 3.0 * len(ood_samples_data)), squeeze=False, facecolor='white')
    column_headers = ['input $u_0(x)$', 'wave speed $c(x)$', 'terminal state', 'relative L2']
    for row, sample in enumerate(ood_samples_data):
        ax = axes[row, 0]
        ax.plot(x, sample['u0'], color='#4C78A8', linewidth=1.8)
        if row == 0:
            _add_panel_text(ax, _panel_label(0, column_headers[0]))
        ax.set_ylabel(f"{sample['type']}\n$u_0(x)$", fontsize=AXIS_LABEL_SIZE)
        _style_axis(ax)

        ax = axes[row, 1]
        ax.plot(x, sample['c'], color='#54A24B', linewidth=1.8)
        if row == 0:
            _add_panel_text(ax, _panel_label(1, column_headers[1]))
        ax.set_ylabel('$c(x)$', fontsize=AXIS_LABEL_SIZE)
        _style_axis(ax)

        ax = axes[row, 2]
        ax.plot(x, sample['uT_true'], color='k', linewidth=2.0, label='True')
        ax.plot(x, sample['uT_fno'], color=MODEL_COLORS['FNO'], linestyle='--', linewidth=1.7, label='FNO')
        ax.plot(x, sample['uT_deeponet'], color=MODEL_COLORS['DeepONet'], linestyle='--', linewidth=1.7, label='DeepONet')
        if row == 0:
            _add_panel_text(ax, _panel_label(2, column_headers[2]))
        ax.set_ylabel('u(x,T)', fontsize=AXIS_LABEL_SIZE)
        _style_axis(ax)

        ax = axes[row, 3]
        ax.bar(
            ['FNO', 'DeepONet'],
            [relative_l2(sample['uT_fno'], sample['uT_true']), relative_l2(sample['uT_deeponet'], sample['uT_true'])],
            color=[MODEL_COLORS['FNO'], MODEL_COLORS['DeepONet']],
            alpha=0.85,
        )
        if row == 0:
            _add_panel_text(ax, _panel_label(3, column_headers[3]))
        ax.set_ylabel('error', fontsize=AXIS_LABEL_SIZE)
        _style_axis(ax)

    for ax in axes[-1, :3]:
        ax.set_xlabel('x', fontsize=AXIS_LABEL_SIZE)

    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.985, 0.985), frameon=False, fontsize=LEGEND_SIZE)

    plt.tight_layout(rect=[0, 0, 0.98, 0.96])
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def select_representative_indices(u_true, u_fno, u_deeponet, count=1):
    """
    Select median-difficulty samples using the mean relative L2 of both models.
    """
    errors = []
    for idx in range(len(u_true)):
        mean_error = 0.5 * (relative_l2(u_fno[idx], u_true[idx]) + relative_l2(u_deeponet[idx], u_true[idx]))
        errors.append(mean_error)
    order = np.argsort(np.asarray(errors))
    if count == 1:
        return [int(order[len(order) // 2])]
    positions = np.linspace(0.25, 0.75, count)
    return [int(order[min(len(order) - 1, round(pos * (len(order) - 1)))]) for pos in positions]


def make_sample_dict(dataset_name, display_name, idx, data, fno_pred, deeponet_pred):
    return {
        'id': f'{dataset_name}_{idx}',
        'index': int(idx),
        'type': display_name,
        'u0': data['u0'][idx],
        'c': data['c'][idx],
        'uT_true': data['uT'][idx],
        'uT_fno': fno_pred[idx],
        'uT_deeponet': deeponet_pred[idx],
    }


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
        raise ValueError("Case visualization currently expects unnormalized checkpoints")
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
        raise ValueError("Case visualization currently expects unnormalized checkpoints")
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


def predict_dataset(fno, deeponet, data, device):
    u0 = torch.tensor(data['u0'], dtype=torch.float32, device=device)
    c = torch.tensor(data['c'], dtype=torch.float32, device=device)
    x = torch.linspace(0.0, 1.0, u0.shape[1], dtype=torch.float32, device=device).repeat(u0.shape[0], 1)
    branch = torch.cat([u0, c], dim=-1)
    with torch.no_grad():
        fno_pred = fno(torch.stack([u0, c], dim=-1)).squeeze(-1).cpu().numpy()
        deeponet_pred = deeponet.forward_branch_trunk(branch, x).cpu().numpy()
    return fno_pred, deeponet_pred


def generate_case_figures(
    fno_checkpoint='results/checkpoints/fno/best_model.pth',
    deeponet_checkpoint='results/checkpoints/deeponet/best_model.pth',
    data_dir='data',
    output_dir='results/figures',
    samples_per_split=1,
):
    """Load checkpoints/data, select representative samples, and write paper figures."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fno = load_fno(fno_checkpoint, device)
    deeponet = load_deeponet(deeponet_checkpoint, device)

    selected_samples = []
    ood_samples = []
    x = None
    for dataset_name, display_name in SPLITS.items():
        data = load_dataset_npz(dataset_file_path(data_dir, dataset_name))
        if x is None:
            x = np.linspace(0.0, 1.0, data['u0'].shape[1])
        fno_pred, deeponet_pred = predict_dataset(fno, deeponet, data, device)
        for idx in select_representative_indices(data['uT'], fno_pred, deeponet_pred, count=samples_per_split):
            sample = make_sample_dict(dataset_name, display_name, idx, data, fno_pred, deeponet_pred)
            selected_samples.append(sample)
            if dataset_name != 'id_test':
                ood_samples.append(sample)
            plot_sample_inputs_outputs(sample['id'], sample['u0'], sample['c'], sample['uT_true'], sample['uT_fno'], sample['uT_deeponet'], x=x, output_dir=output_dir)

    plot_model_comparison_grid(selected_samples, x=x, output_dir=output_dir)
    plot_representative_cases(selected_samples, x=x, output_dir=output_dir)
    plot_local_error_panels(selected_samples, x=x, output_dir=output_dir)
    plot_ood_analysis(ood_samples, x=x, output_dir=output_dir)
    return selected_samples


def main():
    parser = argparse.ArgumentParser(description='Generate representative paper case visualizations')
    parser.add_argument('--fno_checkpoint', default='results/checkpoints/fno/best_model.pth')
    parser.add_argument('--deeponet_checkpoint', default='results/checkpoints/deeponet/best_model.pth')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default='results/figures')
    parser.add_argument('--samples_per_split', type=int, default=1)
    args = parser.parse_args()

    generate_case_figures(
        fno_checkpoint=args.fno_checkpoint,
        deeponet_checkpoint=args.deeponet_checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        samples_per_split=args.samples_per_split,
    )


if __name__ == "__main__":
    main()
