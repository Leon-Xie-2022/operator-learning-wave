"""
OOD Results Visualization

Specialized visualization for out-of-distribution test results with focus on
understanding failure modes and generalization patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_ood_frequency_analysis(ood_freq_data, x=None, output_dir='results/figures'):
    """
    Analyze OOD frequency test performance.

    Parameters:
    -----------
    ood_freq_data : dict
        OOD frequency test data with u0, c, uT_true, uT_fno, uT_deeponet
    x : ndarray, optional
        Spatial grid
    output_dir : str
        Output directory
    """
    if x is None:
        x = np.linspace(0, 1, len(ood_freq_data['uT_true']))

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Input (high frequency u0)
    ax = axes[0, 0]
    ax.plot(x, ood_freq_data['u0'], 'b-', linewidth=2)
    ax.fill_between(x, ood_freq_data['u0'], alpha=0.3)
    ax.set_title('OOD Input: High-Frequency u₀(x)', fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('u₀')
    ax.grid(True, alpha=0.3)

    # Plot 2: Wave speed (normal)
    ax = axes[0, 1]
    ax.plot(x, ood_freq_data['c'], 'g-', linewidth=2)
    ax.fill_between(x, ood_freq_data['c'], alpha=0.3, color='g')
    ax.set_title('Wave Speed c(x)', fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('c(x)')
    ax.grid(True, alpha=0.3)

    # Plot 3: Solutions
    ax = axes[1, 0]
    ax.plot(x, ood_freq_data['uT_true'], 'k-', linewidth=2.5, label='True')
    ax.plot(x, ood_freq_data['uT_fno'], 'b--', linewidth=2, label='FNO')
    ax.plot(x, ood_freq_data['uT_deeponet'], 'r--', linewidth=2, label='DeepONet')
    ax.set_title('Terminal Solution u(x, T)', fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, T)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Error comparison
    ax = axes[1, 1]
    error_fno = ood_freq_data['uT_fno'] - ood_freq_data['uT_true']
    error_deeponet = ood_freq_data['uT_deeponet'] - ood_freq_data['uT_true']

    ax.plot(x, error_fno, 'b-', linewidth=2, label='FNO Error')
    ax.plot(x, error_deeponet, 'r-', linewidth=2, label='DeepONet Error')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.fill_between(x, error_fno, alpha=0.2, color='b')
    ax.fill_between(x, error_deeponet, alpha=0.2, color='r')
    ax.set_title('Prediction Errors', fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('Error')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = os.path.join(output_dir, 'ood_frequency_sample.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def plot_ood_smoothness_analysis(ood_smooth_data, ood_rough_data, x=None, output_dir='results/figures'):
    """
    Compare OOD smooth vs rough wave speed performance.

    Parameters:
    -----------
    ood_smooth_data : dict
        OOD smooth test data
    ood_rough_data : dict
        OOD rough test data
    x : ndarray, optional
        Spatial grid
    output_dir : str
        Output directory
    """
    if x is None:
        x = np.linspace(0, 1, len(ood_smooth_data['uT_true']))

    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    for col_idx, (data, title_prefix) in enumerate([(ood_smooth_data, 'OOD Smooth'),
                                                      (ood_rough_data, 'OOD Rough')]):
        # Wave speed
        ax = fig.add_subplot(gs[0, col_idx])
        ax.plot(x, data['c'], 'g-', linewidth=2)
        ax.fill_between(x, data['c'], alpha=0.3, color='g')
        ax.set_title(f'{title_prefix} c(x)', fontsize=11, fontweight='bold')
        ax.set_ylabel('c(x)')
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel('c(x)', fontsize=10)

        # Solutions
        ax = fig.add_subplot(gs[1, col_idx])
        ax.plot(x, data['uT_true'], 'k-', linewidth=2.5, label='True')
        ax.plot(x, data['uT_fno'], 'b--', linewidth=2, label='FNO')
        ax.plot(x, data['uT_deeponet'], 'r--', linewidth=2, label='DeepONet')
        ax.set_title(f'{title_prefix}: Solutions')
        ax.set_ylabel('u(x, T)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Errors
        ax = fig.add_subplot(gs[2, col_idx])
        error_fno = data['uT_fno'] - data['uT_true']
        error_deeponet = data['uT_deeponet'] - data['uT_true']

        l2_fno = np.linalg.norm(error_fno)
        l2_deeponet = np.linalg.norm(error_deeponet)

        ax.plot(x, error_fno, 'b-', linewidth=2, label=f'FNO (L2={l2_fno:.4f})')
        ax.plot(x, error_deeponet, 'r-', linewidth=2, label=f'DeepONet (L2={l2_deeponet:.4f})')
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
        ax.fill_between(x, error_fno, alpha=0.2, color='b')
        ax.fill_between(x, error_deeponet, alpha=0.2, color='r')
        ax.set_title(f'{title_prefix}: Errors')
        ax.set_xlabel('x')
        ax.set_ylabel('Error')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('OOD Wave Speed Comparison: Smooth vs Rough', fontsize=13, fontweight='bold', y=0.995)
    filename = os.path.join(output_dir, 'ood_smoothness_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


if __name__ == "__main__":
    print("OOD results visualization module ready")