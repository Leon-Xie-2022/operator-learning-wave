"""Build the main result table and summary figures used by the paper."""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 18
LEGEND_SIZE = 18
VALUE_LABEL_SIZE = 16


def _style_paper_axis(ax):
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.22, axis='y', color='0.82', linewidth=0.6)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)


MAIN_RESULT_METRICS = ['validation', 'id_test', 'ood_frequency', 'ood_smoothness']
MAIN_RESULT_LABELS = {
    'validation': 'Validation',
    'id_test': 'ID',
    'ood_frequency': 'High-frequency u0',
    'ood_smoothness': 'Wave-speed smoothness OOD',
}

MAIN_RESULT_PLOT_LABELS = {
    'validation': 'Validation',
    'id_test': 'ID',
    'ood_frequency': 'High-frequency\nu0',
    'ood_smoothness': 'Wave-speed\nsmoothness OOD',
}


def _get_result(results, key):
    """Read a result key with compatibility for older smoothness split names."""
    aliases = {
        'ood_smoothness': 'ood_smooth',
    }
    if key in results:
        return results[key]
    alias = aliases.get(key)
    if alias and alias in results:
        return results[alias]
    return None


def _simple_table(rows, headers):
    table_rows = [headers] + rows
    widths = [max(len(str(row[i])) for row in table_rows) for i in range(len(headers))]
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    lines = [fmt.format(*headers), sep]
    lines.extend(fmt.format(*row) for row in rows)
    return "\n".join(lines)


def create_results_table(fno_results, deeponet_results, output_dir='results'):
    """Create the FNO/DeepONet comparison table consumed by the paper."""
    os.makedirs(output_dir, exist_ok=True)

    table_data = []

    metrics_to_show = MAIN_RESULT_METRICS

    for metric_key in metrics_to_show:
        metric_name = MAIN_RESULT_LABELS[metric_key]

        fno_metric = _get_result(fno_results, metric_key)
        deeponet_metric = _get_result(deeponet_results, metric_key)

        fno_rel_l2 = fno_metric['rel_l2_mean'] if fno_metric else 0
        fno_rel_l2_std = fno_metric['rel_l2_std'] if fno_metric else 0
        fno_time = fno_metric.get('inference_time_per_sample', 0) if fno_metric else 0

        deeponet_rel_l2 = deeponet_metric['rel_l2_mean'] if deeponet_metric else 0
        deeponet_rel_l2_std = deeponet_metric['rel_l2_std'] if deeponet_metric else 0
        deeponet_time = deeponet_metric.get('inference_time_per_sample', 0) if deeponet_metric else 0

        table_data.append([
            metric_name,
            f"{fno_rel_l2:.4f} +/- {fno_rel_l2_std:.4f}",
            f"{deeponet_rel_l2:.4f} +/- {deeponet_rel_l2_std:.4f}",
            f"{fno_time*1000:.2f} ms",
            f"{deeponet_time*1000:.2f} ms"
        ])

    headers = ['Dataset', 'FNO Rel L2', 'DeepONet Rel L2', 'FNO Time', 'DeepONet Time']
    table_str = _simple_table(table_data, headers)

    print("\n" + "="*80)
    print("MAIN RESULTS TABLE")
    print("="*80)
    print(table_str)
    print("="*80 + "\n")

    table_file = os.path.join(output_dir, 'main_results_table.txt')
    with open(table_file, 'w') as f:
        f.write(table_str)

    json_table = []
    for i, row in enumerate(table_data):
        json_table.append({
            'dataset': row[0],
            'fno_rel_l2': row[1],
            'deeponet_rel_l2': row[2],
            'fno_time': row[3],
            'deeponet_time': row[4]
        })

    json_file = os.path.join(output_dir, 'main_results_table.json')
    with open(json_file, 'w') as f:
        json.dump(json_table, f, indent=2)

    return table_str


def plot_results_bars(fno_results, deeponet_results, output_dir='results/figures'):
    """Plot relative-L2 results for the main splits."""
    os.makedirs(output_dir, exist_ok=True)

    datasets = MAIN_RESULT_METRICS
    dataset_labels = [MAIN_RESULT_PLOT_LABELS[key] for key in datasets]

    fno_rel_l2 = []
    deeponet_rel_l2 = []

    for dataset in datasets:
        fno_metric = _get_result(fno_results, dataset)
        deeponet_metric = _get_result(deeponet_results, dataset)

        fno_rel_l2.append(fno_metric['rel_l2_mean'] if fno_metric else 0)
        deeponet_rel_l2.append(deeponet_metric['rel_l2_mean'] if deeponet_metric else 0)

    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor='white')

    x_pos = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, fno_rel_l2, width, label='FNO', alpha=0.9, color='#4C78A8', edgecolor='none')
    bars2 = ax.bar(x_pos + width/2, deeponet_rel_l2, width, label='DeepONet', alpha=0.9, color='#E45756', edgecolor='none')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=VALUE_LABEL_SIZE)

    ax.set_ylabel('Relative L2 error', fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel('Dataset', fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dataset_labels)
    ax.set_ylim(top=max(fno_rel_l2 + deeponet_rel_l2) * 1.16)
    ax.legend(fontsize=LEGEND_SIZE, loc='upper right', frameon=False)
    _style_paper_axis(ax)

    plt.tight_layout()
    filename = os.path.join(output_dir, 'main_results_bars.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def plot_ood_degradation(fno_results, deeponet_results, output_dir='results/figures'):
    """Plot OOD error relative to the ID baseline."""
    os.makedirs(output_dir, exist_ok=True)

    id_error_fno = _get_result(fno_results, 'id_test')['rel_l2_mean']
    id_error_deeponet = _get_result(deeponet_results, 'id_test')['rel_l2_mean']

    ood_types = ['ood_frequency', 'ood_smoothness']
    ood_labels = [MAIN_RESULT_LABELS[key] for key in ood_types]

    fno_ood_errors = []
    deeponet_ood_errors = []

    for ood_type in ood_types:
        fno_metric = _get_result(fno_results, ood_type)
        deeponet_metric = _get_result(deeponet_results, ood_type)
        if fno_metric:
            fno_ood_errors.append(fno_metric['rel_l2_mean'])
        if deeponet_metric:
            deeponet_ood_errors.append(deeponet_metric['rel_l2_mean'])

    fno_degradation = [e / id_error_fno for e in fno_ood_errors]
    deeponet_degradation = [e / id_error_deeponet for e in deeponet_ood_errors]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    x_pos = np.arange(len(ood_types))
    width = 0.35

    ax.bar(x_pos - width/2, fno_degradation, width, label='FNO', alpha=0.9, color='#4C78A8', edgecolor='none')
    ax.bar(x_pos + width/2, deeponet_degradation, width, label='DeepONet', alpha=0.9, color='#E45756', edgecolor='none')

    ax.axhline(1.0, color='k', linestyle='--', linewidth=1, label='ID Performance')
    ax.set_ylabel('OOD/ID relative-L2 ratio', fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel('Shift type', fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ood_labels)
    ax.legend(fontsize=LEGEND_SIZE, loc='upper right', frameon=False)
    _style_paper_axis(ax)

    plt.tight_layout()
    filename = os.path.join(output_dir, 'ood_degradation.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


if __name__ == "__main__":
    print("Main results plotting module ready")
