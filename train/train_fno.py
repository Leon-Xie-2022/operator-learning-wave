"""Train the 1D FNO model used in the wave-equation experiments."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import csv
import argparse
from tqdm import tqdm
import sys

# Allow direct execution with `python train/train_fno.py`.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.fno1d import FNO1D
from data.dataset_loader import get_data_loaders, save_normalization_stats


MODEL_NAME = 'fno'


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _parse_config_value(value):
    """Parse a scalar value from the project's flat YAML config files."""
    value = value.strip()
    if value.lower() in {'true', 'false'}:
        return value.lower() == 'true'
    if value.lower() in {'null', 'none'}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip('"').strip("'")


def load_config_file(path):
    """Load a flat JSON or YAML training config."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, 'r') as f:
        if ext == '.json':
            return json.load(f)
        if ext in {'.yaml', '.yml'}:
            config = {}
            for line in f:
                line = line.split('#', 1)[0].strip()
                if not line:
                    continue
                if ':' not in line:
                    raise ValueError(f"Unsupported YAML line in {path}: {line}")
                key, value = line.split(':', 1)
                config[key.strip()] = _parse_config_value(value)
            return config
    raise ValueError(f"Unsupported config extension '{ext}'. Use .json, .yaml, or .yml")


def validate_config(config):
    """Validate training settings shared by the FNO pipeline."""
    if config.get('normalize', False) and config.get('enforce_dirichlet', True):
        raise ValueError("normalize=True is not supported with enforce_dirichlet=True in the current pipeline")


def make_loss(loss_type):
    """Create the configured supervised loss."""
    if loss_type == 'mse':
        return nn.MSELoss()
    if loss_type == 'relative_l2':
        def relative_l2_loss(pred, target):
            pred_flat = pred.reshape(pred.shape[0], -1)
            target_flat = target.reshape(target.shape[0], -1)
            return torch.mean(
                torch.linalg.norm(pred_flat - target_flat, dim=1)
                / (torch.linalg.norm(target_flat, dim=1) + 1e-12)
            )
        return relative_l2_loss
    raise ValueError(f"Unknown loss type: {loss_type}")


def train_fno(config):
    """Train an FNO model from a flat configuration dictionary."""
    set_seed(config.get('seed', 2026))
    validate_config(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    print("Loading data...")
    loaders, stats = get_data_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        model_type='fno',
        normalize=config['normalize']
    )

    if config['normalize'] and stats is not None:
        stats_file = os.path.join(config['output_dir'], 'normalization_stats.json')
        save_normalization_stats(stats, stats_file)

    model = FNO1D(
        modes=config['modes'],
        width=config['width'],
        num_layers=config['num_layers'],
        padding=config.get('padding', 0),
        enforce_dirichlet=config.get('enforce_dirichlet', True)
    ).to(device)

    print(f"Model parameters: {model.count_parameters()}")
    print(f"Model complexity: {model.get_complexity()}")

    criterion = make_loss(config['loss_type'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    if config['use_scheduler']:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step'], gamma=config['scheduler_gamma'])
    else:
        scheduler = None

    best_val_loss = float('inf')
    best_epoch = -1
    patience = config['patience']
    patience_counter = 0

    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_train_loss = 0.0

        for batch_x, batch_y in tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(loaders['train'])
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in loaders['validation']:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(loaders['validation'])
        val_losses.append(epoch_val_loss)

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config['num_epochs']}: "
                  f"Train Loss = {epoch_train_loss:.4f}, "
                  f"Val Loss = {epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0

            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'config': config
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    final_checkpoint_path = os.path.join(config['checkpoint_dir'], 'final_model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': epoch_train_loss,
        'val_loss': epoch_val_loss,
        'config': config
    }, final_checkpoint_path)

    logs = {
        'model': MODEL_NAME,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'final_epoch': epoch,
        'config': config
    }

    logs_path = os.path.join(config['output_dir'], 'training_logs.json')
    with open(logs_path, 'w') as f:
        json.dump(logs, f, indent=2)

    loss_curve_path = os.path.join(config['output_dir'], 'loss_curves.csv')
    with open(loss_curve_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
        writer.writeheader()
        for idx, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            writer.writerow({
                'epoch': idx,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

    best_summary = {
        'model': MODEL_NAME,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'checkpoint_path': os.path.join(config['checkpoint_dir'], 'best_model.pth'),
        'loss_type': config['loss_type'],
        'seed': config.get('seed', 2026),
        'modes': config['modes'],
        'width': config['width'],
        'num_layers': config['num_layers'],
        'padding': config.get('padding', 0),
        'enforce_dirichlet': config.get('enforce_dirichlet', True)
    }
    best_summary_path = os.path.join(config['output_dir'], 'best_checkpoint_summary.json')
    with open(best_summary_path, 'w') as f:
        json.dump(best_summary, f, indent=2)

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Logs saved to {logs_path}")
    print(f"Loss curves saved to {loss_curve_path}")
    print(f"Best checkpoint summary saved to {best_summary_path}")


def get_default_config():
    """Return the training settings used for the main FNO run."""
    return {
        'data_dir': 'data',
        'output_dir': 'results/train_fno',
        'checkpoint_dir': 'results/checkpoints/fno',
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'loss_type': 'relative_l2',  # 'mse' or 'relative_l2'
        'modes': 16,
        'width': 64,
        'num_layers': 4,
        'padding': 0,
        'enforce_dirichlet': True,
        'normalize': False,
        'use_scheduler': False,
        'scheduler_step': 50,
        'scheduler_gamma': 0.5,
        'patience': 10,
        'seed': 2026
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train FNO for wave equation')
    parser.add_argument('--config', type=str, help='Path to config JSON/YAML file')
    parser.add_argument('--modes', type=int, help='Number of Fourier modes')
    parser.add_argument('--width', type=int, help='Model width')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--patience', type=int, help='Early stopping patience')

    args = parser.parse_args()

    config = get_default_config()

    if args.config:
        config.update(load_config_file(args.config))

    if args.modes:
        config['modes'] = args.modes
    if args.width:
        config['width'] = args.width
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.patience:
        config['patience'] = args.patience

    train_fno(config)
