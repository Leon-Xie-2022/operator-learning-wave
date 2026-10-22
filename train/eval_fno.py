"""Evaluate a trained FNO checkpoint on the project splits."""

import torch
import numpy as np
import os
import json
import time
import argparse
import sys
from tqdm import tqdm

# Allow direct execution with `python train/eval_fno.py`.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.fno1d import FNO1D
from data.dataset_loader import get_canonical_splits, get_data_loaders


MODEL_NAME = 'fno'
EVAL_SPLITS = tuple(get_canonical_splits(include_auxiliary=False)[1:])


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_config(config):
    """Reject unsupported normalized Dirichlet evaluation."""
    if config.get('normalize', False) and config.get('enforce_dirichlet', True):
        raise ValueError("normalize=True is not supported with enforce_dirichlet=True in the current pipeline")


def load_model(checkpoint_path, device):
    """Load an FNO checkpoint and its saved training config."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
    model = FNO1D(
        modes=config['modes'],
        width=config['width'],
        num_layers=config['num_layers'],
        padding=config.get('padding', 0),
        enforce_dirichlet=config.get('enforce_dirichlet', True)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def compute_sample_metrics(pred, target, start_index=0):
    """Return per-sample MSE, L2, and relative-L2 metrics for a batch."""
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    diff_flat = pred_flat - target_flat

    mse_samples = np.mean(diff_flat**2, axis=1)
    l2_samples = np.linalg.norm(diff_flat, axis=1)
    rel_l2_samples = l2_samples / (np.linalg.norm(target_flat, axis=1) + 1e-12)

    return [
        {
            'sample_index': int(start_index + idx),
            'mse': float(mse_samples[idx]),
            'rel_l2': float(rel_l2_samples[idx]),
            'l2': float(l2_samples[idx])
        }
        for idx in range(pred_flat.shape[0])
    ]


def evaluate_model(model, data_loader, device):
    """Evaluate one DataLoader and return aggregate, sample, and timing records."""
    model.eval()
    sample_metrics = []
    batch_times = []
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in tqdm(data_loader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            outputs = model(batch_x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.perf_counter() - start_time
            batch_times.append({
                'batch_size': int(len(batch_x)),
                'batch_time': float(inference_time),
                'time_per_sample': float(inference_time / len(batch_x))
            })

            sample_metrics.extend(compute_sample_metrics(outputs, batch_y, start_index=total_samples))
            total_samples += len(batch_x)

    avg_metrics = {}
    for key in ('mse', 'rel_l2', 'l2'):
        values = [m[key] for m in sample_metrics]
        avg_metrics[f'{key}_mean'] = float(np.mean(values))
        avg_metrics[f'{key}_std'] = float(np.std(values))

    total_time = sum(item['batch_time'] for item in batch_times)
    avg_metrics['inference_time_per_sample'] = float(total_time / total_samples)
    avg_metrics['num_samples'] = int(total_samples)

    return avg_metrics, sample_metrics, batch_times


def save_results(results, output_dir, model_name):
    """Write the summary and table JSON files consumed by analysis scripts."""
    os.makedirs(output_dir, exist_ok=True)

    summary_file = os.path.join(output_dir, f'{model_name}_evaluation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    table_data = []
    for dataset, metrics in results.items():
        row = {
            'model': model_name,
            'dataset': dataset,
            'num_samples': metrics['num_samples'],
            'rel_l2_mean': metrics['rel_l2_mean'],
            'rel_l2_std': metrics['rel_l2_std'],
            'mse_mean': metrics['mse_mean'],
            'mse_std': metrics['mse_std'],
            'l2_mean': metrics['l2_mean'],
            'l2_std': metrics['l2_std'],
            'inference_time': metrics['inference_time_per_sample']
        }
        table_data.append(row)

    table_file = os.path.join(output_dir, f'{model_name}_results_table.json')
    with open(table_file, 'w') as f:
        json.dump(table_data, f, indent=2)

    print(f"Results saved to {output_dir}")


def evaluate_fno(checkpoint_path, output_dir='results/eval_fno'):
    """Evaluate an FNO checkpoint on validation, ID, and OOD splits."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, config = load_model(checkpoint_path, device)
    set_seed(config.get('seed', 2026))
    validate_config(config)
    print(f"Loaded model with {model.count_parameters()} parameters")

    loaders, _ = get_data_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        model_type='fno',
        normalize=config['normalize']
    )

    results = {}
    for split in EVAL_SPLITS:
        if split in loaders:
            print(f"\nEvaluating on {split}...")
            avg_metrics, sample_metrics, batch_times = evaluate_model(model, loaders[split], device)
            results[split] = avg_metrics

            sample_metrics_file = os.path.join(output_dir, f'{split}_sample_metrics.json')
            os.makedirs(os.path.dirname(sample_metrics_file), exist_ok=True)
            with open(sample_metrics_file, 'w') as f:
                json.dump(sample_metrics, f, indent=2)

            timing_file = os.path.join(output_dir, f'{split}_batch_timing.json')
            with open(timing_file, 'w') as f:
                json.dump(batch_times, f, indent=2)
        else:
            print(f"Warning: {split} not found in data loaders")

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for dataset, metrics in results.items():
        print(f"{dataset}:")
        print(f"  Rel L2: {metrics['rel_l2_mean']:.4f} +/- {metrics['rel_l2_std']:.4f}")
        print(f"  MSE: {metrics['mse_mean']:.6f} +/- {metrics['mse_std']:.6f}")
        print(f"  Inference/sample: {metrics['inference_time_per_sample']*1000:.4f} ms")
        print()

    save_results(results, output_dir, MODEL_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate FNO model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/eval_fno',
                       help='Output directory')

    args = parser.parse_args()

    evaluate_fno(args.checkpoint, args.output_dir)
