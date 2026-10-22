# Frequency Bias and OOD Generalization in Neural Operators

This repository contains the code for the research project "Frequency Bias and OOD Generalization in Neural Operators under a Variable-Coefficient Wave Equation".

The project studies terminal-state operator learning for the one-dimensional conservative wave equation:

```text
u_tt = d_x(c(x)^2 u_x),  x in [0, 1],  t in [0, T],
u(0, t) = u(1, t) = 0,  u_t(x, 0) = 0.
```

The learned map is `(u0, c) -> u(., T)`. Two representative architectures are compared under a controlled setting: a 1D Fourier Neural Operator (FNO) and a fixed-grid DeepONet.

## Repository Layout

```text
configs/      Experiment and problem configuration files
solver/       Reference finite-difference solver and solver checks
data/         Sampling, dataset generation, and dataset loading code
models/       FNO and DeepONet implementations
train/        Training and evaluation entry points
analysis/     Table and figure generation scripts
run_workflow.py
run_analysis.py
```

The `paper/` and `icml2026paper/` directories are manuscript workspaces and are intentionally excluded from the public code artifact.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA or CPU setup if the default wheel is not appropriate for your machine.

## Reproducing the Pipeline

Generate all datasets:

```bash
python data/generate_dataset.py
```

Train and evaluate the main FNO model:

```bash
python train/train_fno.py --modes 16 --width 64 --batch_size 32 --lr 1e-3
python train/eval_fno.py --checkpoint results/checkpoints/fno/best_model.pth
```

Train and evaluate the DeepONet baseline:

```bash
python train/train_deeponet.py --branch_hidden 128 --trunk_hidden 128 --batch_size 32 --lr 1e-3
python train/eval_deeponet.py --checkpoint results/checkpoints/deeponet/best_model.pth
```

Generate paper tables and figures after the required checkpoints and evaluation summaries exist:

```bash
python run_analysis.py
```

The full workflow, including solver checks, data generation, model training, evaluation, ablations, and figure generation, is available as:

```bash
python run_workflow.py
```

## Configuration Notes

The main paper configuration uses **16 retained Fourier modes** for FNO. `train/train_fno.py`, `configs/train_fno.yaml`, and `run_workflow.py` are aligned to this setting. The 8-mode and 32-mode JSON files are separate ablation configurations.

`configs/problem.yaml` and `configs/data.yaml` document the problem/data specification for reference; the current dataset-generation entry point keeps the corresponding sampling values in the Python generation code rather than loading those two YAML files dynamically.

## Generated Artifacts

Generated datasets (`data/*_dataset.npz`), checkpoints (`results/checkpoints/`), evaluation files, and figures are not committed to GitHub by default. For exact agreement with the reported results, use the frozen artifact bundle associated with the original experiment snapshot. The source code can regenerate the pipeline, but full retraining may not be bitwise identical across hardware, CUDA, PyTorch versions, and timing conditions.

The intended reproducibility levels are:

- Frozen-result reproduction: use the original artifact bundle and verify it with `tools/verify_artifacts.py`.
- Fresh pipeline run: regenerate data, checkpoints, evaluations, and figures from source. This should reproduce the experimental procedure and trends, but training and timing may differ slightly across systems.

To verify a downloaded artifact bundle:

```bash
python tools/verify_artifacts.py artifact_manifest.sha256 --root .
```

To create the manifest from the frozen local artifact bundle:

```bash
python tools/make_artifact_manifest.py --root . --output artifact_manifest.sha256
```

## Main Splits

- `train`: training distribution
- `validation`: validation split from the training distribution
- `id_test`: in-distribution test split
- `ood_frequency`: high-frequency initial-displacement shift
- `ood_smoothness`: wave-speed smoothness shift, stored on disk as `ood_smooth_dataset.npz` for compatibility

## Checks

Run the solver checks:

```bash
python solver/test_solver.py
```

Run the lightweight sampling/model smoke tests:

```bash
python -m unittest tests.test_smoke
```

Run a syntax check over the main code:

```bash
python -m py_compile run_workflow.py run_analysis.py solver/solver_core.py solver/solver_utils.py solver/test_solver.py data/generate_dataset.py data/dataset_loader.py data/sample_u0.py data/sample_c.py models/fno1d.py models/deeponet.py train/train_fno.py train/eval_fno.py train/train_deeponet.py train/eval_deeponet.py
```
