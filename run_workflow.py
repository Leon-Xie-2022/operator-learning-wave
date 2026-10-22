"""Run the complete formal workflow for the wave-equation project."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def run_command(args: list[str], description: str) -> None:
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    print("Running:", " ".join(args))
    result = subprocess.run(args, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with return code {result.returncode}")
    print(f"[OK] {description} completed")


def dataset_files_exist() -> bool:
    required = [
        "train_dataset.npz",
        "validation_dataset.npz",
        "id_test_dataset.npz",
        "ood_frequency_dataset.npz",
        "ood_smooth_dataset.npz",
        "ood_rough_dataset.npz",
    ]
    return all((ROOT / "data" / name).exists() for name in required)


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_result(results: dict, key: str) -> dict:
    if key in results:
        return results[key]
    raise KeyError(f"Missing result key '{key}'")


def print_summary() -> None:
    fno_path = ROOT / "results" / "eval_fno" / "fno_evaluation_summary.json"
    deeponet_path = ROOT / "results" / "eval_deeponet" / "deeponet_evaluation_summary.json"
    if not fno_path.exists() or not deeponet_path.exists():
        return

    fno = load_summary(fno_path)
    deeponet = load_summary(deeponet_path)
    print("\nFORMAL RESULTS SUMMARY")
    print("-" * 70)
    for split in ["validation", "id_test", "ood_frequency", "ood_smoothness"]:
        fno_split = get_result(fno, split)
        deeponet_split = get_result(deeponet, split)
        print(
            f"{split}: "
            f"FNO {fno_split['rel_l2_mean']:.4f} +/- {fno_split['rel_l2_std']:.4f}; "
            f"DeepONet {deeponet_split['rel_l2_mean']:.4f} +/- {deeponet_split['rel_l2_std']:.4f}"
        )


def main() -> None:
    os.chdir(ROOT)
    print("=" * 70)
    print("FORMAL WAVE EQUATION OPERATOR LEARNING WORKFLOW")
    print("=" * 70)

    run_command([PYTHON, "solver/test_solver.py"], "Solver validation")
    run_command([PYTHON, "solver/visualize_solver.py"], "Solver diagnostic figures")

    run_command([PYTHON, "data/generate_dataset.py"], "Dataset generation")

    run_command(
        [PYTHON, "train/train_fno.py", "--modes", "16", "--width", "64", "--batch_size", "32", "--lr", "1e-3"],
        "Formal FNO training, modes=16",
    )
    run_command(
        [PYTHON, "train/eval_fno.py", "--checkpoint", "results/checkpoints/fno/best_model.pth"],
        "Formal FNO evaluation",
    )
    run_command(
        [
            PYTHON,
            "train/eval_fno.py",
            "--checkpoint",
            "results/checkpoints/fno/best_model.pth",
            "--output_dir",
            "results/eval_fno_modes16",
        ],
        "FNO modes=16 ablation evaluation",
    )

    run_command(
        [
            PYTHON,
            "train/train_deeponet.py",
            "--branch_hidden",
            "128",
            "--trunk_hidden",
            "128",
            "--batch_size",
            "32",
            "--lr",
            "1e-3",
        ],
        "Formal DeepONet training",
    )
    run_command(
        [PYTHON, "train/eval_deeponet.py", "--checkpoint", "results/checkpoints/deeponet/best_model.pth"],
        "Formal DeepONet evaluation",
    )

    run_command([PYTHON, "train/train_fno.py", "--config", "configs/fno_modes8.json"], "FNO modes=8 ablation training")
    run_command(
        [
            PYTHON,
            "train/eval_fno.py",
            "--checkpoint",
            "results/checkpoints/fno_modes8/best_model.pth",
            "--output_dir",
            "results/eval_fno_modes8",
        ],
        "FNO modes=8 ablation evaluation",
    )
    run_command([PYTHON, "train/train_fno.py", "--config", "configs/fno_modes32.json"], "FNO modes=32 ablation training")
    run_command(
        [
            PYTHON,
            "train/eval_fno.py",
            "--checkpoint",
            "results/checkpoints/fno_modes32/best_model.pth",
            "--output_dir",
            "results/eval_fno_modes32",
        ],
        "FNO modes=32 ablation evaluation",
    )

    run_command([PYTHON, "run_analysis.py"], "Paper table and figure generation")
    print_summary()

    print("\nGenerated outputs:")
    print("  results/main_results_table.txt")
    print("  results/tables/fno_modes_ablation.csv")
    print("  results/figures/")


if __name__ == "__main__":
    main()
