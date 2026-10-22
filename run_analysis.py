"""Generate paper-ready tables and figures from completed experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from analysis.case_visualization import (
    _panel_label,
    make_sample_dict,
    plot_local_error_panels,
    plot_model_comparison_grid,
    plot_ood_analysis,
    plot_representative_cases,
    plot_sample_inputs_outputs,
    select_representative_indices,
)
from analysis.energy_like_analysis import analyze_energy_consistency, analyze_split_energy_proxy, plot_energy_comparison, plot_split_energy_proxy
from analysis.plot_main_results import create_results_table, plot_ood_degradation, plot_results_bars
from analysis.spectral_error_analysis import (
    plot_error_frequency_distribution,
    plot_id_vs_ood_frequency_spectral_error,
    plot_spectral_error_comparison,
    plot_split_spectral_analysis,
)
from data.dataset_loader import dataset_file_path, load_dataset_npz
from models.deeponet import DeepONet1D
from models.fno1d import FNO1D


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_result(results: dict, key: str) -> dict:
    if key in results:
        return results[key]
    raise KeyError(f"Missing result key '{key}'")


def infer_deeponet_n(checkpoint: dict) -> int:
    config = checkpoint["config"]
    if "N" in config:
        return int(config["N"])
    branch_weight = checkpoint["model_state_dict"].get("branch.net.0.weight")
    if branch_weight is None:
        raise KeyError("DeepONet checkpoint is missing config['N'] and branch.net.0.weight")
    return int(branch_weight.shape[1] // 2)


def load_fno(checkpoint_path: Path, device: torch.device) -> FNO1D:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = FNO1D(
        modes=config["modes"],
        width=config["width"],
        num_layers=config["num_layers"],
        padding=config.get("padding", 0),
        enforce_dirichlet=config.get("enforce_dirichlet", True),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_deeponet(checkpoint_path: Path, device: torch.device) -> DeepONet1D:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = DeepONet1D(
        N=infer_deeponet_n(checkpoint),
        branch_hidden=config["branch_hidden"],
        trunk_hidden=config["trunk_hidden"],
        num_layers=config["num_layers"],
        use_output_bias=config.get("use_output_bias", False),
        enforce_dirichlet=config.get("enforce_dirichlet", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_batch(fno: FNO1D, deeponet: DeepONet1D, data: dict, count: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    u0 = torch.tensor(data["u0"][:count], dtype=torch.float32, device=device)
    c = torch.tensor(data["c"][:count], dtype=torch.float32, device=device)
    x = torch.linspace(0.0, 1.0, u0.shape[1], dtype=torch.float32, device=device).repeat(count, 1)
    branch = torch.cat([u0, c], dim=-1)
    with torch.no_grad():
        fno_pred = fno(torch.stack([u0, c], dim=-1)).squeeze(-1).cpu().numpy()
        deeponet_pred = deeponet.forward_branch_trunk(branch, x).cpu().numpy()
    return fno_pred, deeponet_pred


def write_modes_ablation() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    mode_results = {
        8: load_json(ROOT / "results" / "eval_fno_modes8" / "fno_evaluation_summary.json"),
        16: load_json(ROOT / "results" / "eval_fno_modes16" / "fno_evaluation_summary.json"),
        32: load_json(ROOT / "results" / "eval_fno_modes32" / "fno_evaluation_summary.json"),
    }

    rows = ["mode,validation_rel_l2,id_rel_l2,ood_frequency_rel_l2,ood_smoothness_rel_l2"]
    for mode, results in mode_results.items():
        rows.append(
            ",".join(
                [
                    str(mode),
                    f"{get_result(results, 'validation')['rel_l2_mean']:.6f}",
                    f"{get_result(results, 'id_test')['rel_l2_mean']:.6f}",
                    f"{get_result(results, 'ood_frequency')['rel_l2_mean']:.6f}",
                    f"{get_result(results, 'ood_smoothness')['rel_l2_mean']:.6f}",
                ]
            )
        )
    (TABLE_DIR / "fno_modes_ablation.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    axis_label_size = 16
    tick_label_size = 16
    legend_size = 14

    def style_axis(ax):
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.22, color="0.82", linewidth=0.6)
        ax.tick_params(labelsize=tick_label_size)

    def add_panel_text(ax, index: int, label: str) -> None:
        ax.text(
            0.0,
            1.02,
            f"({chr(ord('a') + index)}) {label}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            clip_on=False,
        )

    modes = np.array(list(mode_results.keys()))
    validation_errors = np.array([get_result(mode_results[m], "validation")["rel_l2_mean"] for m in modes])
    id_errors = np.array([get_result(mode_results[m], "id_test")["rel_l2_mean"] for m in modes])
    ood_freq_errors = np.array([get_result(mode_results[m], "ood_frequency")["rel_l2_mean"] for m in modes])
    ood_smoothness_errors = np.array([get_result(mode_results[m], "ood_smoothness")["rel_l2_mean"] for m in modes])

    plt.figure(figsize=(7, 4.5))
    plt.plot(modes, validation_errors, marker="x", label="Validation")
    plt.plot(modes, id_errors, marker="o", label="ID")
    plt.plot(modes, ood_freq_errors, marker="s", label="High-frequency u0")
    plt.plot(modes, ood_smoothness_errors, marker="^", label="Wave-speed smoothness OOD")
    plt.xlabel("Retained Fourier modes per FNO layer")
    plt.ylabel("Relative L2 error")
    plt.title("Effect of FNO Spectral Capacity")
    plt.xticks(modes)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fno_modes_ablation.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Paper-oriented version: separate lower-error ID/validation curves from
    # the larger OOD-frequency scale so the main comparison remains legible.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True, facecolor="white")

    left_ax, right_ax = axes
    left_ax.plot(modes, validation_errors, marker="o", markersize=6, linewidth=2.2, color="#4C78A8", label="Validation")
    left_ax.plot(modes, id_errors, marker="s", markersize=6, linewidth=2.2, color="#E45756", label="ID")
    left_ax.plot(modes, ood_smoothness_errors, marker="^", markersize=6, linewidth=2.2, color="#54A24B", label="OOD-smoothness")
    add_panel_text(left_ax, 0, "ID and smoothness shift")
    left_ax.set_xlabel("Retained Fourier modes", fontsize=axis_label_size)
    left_ax.set_ylabel("Relative L2 error", fontsize=axis_label_size)
    left_ax.set_xticks(modes)
    style_axis(left_ax)
    legend = left_ax.legend(
        frameon=False,
        fontsize=legend_size,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.6, 1.3),
    )
    legend.set_in_layout(False)

    right_ax.plot(modes, ood_freq_errors, marker="D", markersize=6, linewidth=2.4, color="#F58518")
    add_panel_text(right_ax, 1, "Frequency shift OOD")
    right_ax.set_xlabel("Retained Fourier modes", fontsize=axis_label_size)
    right_ax.set_ylabel("Relative L2 error", fontsize=axis_label_size)
    # The two panels share the same y quantity; keeping only the left label
    # reduces the visual gap between the subplots.
    right_ax.set_xticks(modes)
    style_axis(right_ax)

    best_idx = int(np.argmin(id_errors))
    left_ax.scatter([modes[best_idx]], [id_errors[best_idx]], color="#E45756", s=42, zorder=5)
    #left_ax.annotate(
    #    "best ID",
    #    xy=(modes[best_idx], id_errors[best_idx]),
    #    xytext=(10, -16),
    #    textcoords="offset points",
    #    fontsize=9,
    #    color="#E45756",
    #)

    fig.tight_layout(rect=[0, 0, 1, 0.90], w_pad=0)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(
        FIG_DIR / "fno_modes_ablation_paper.png",
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
    )
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    fno_results = load_json(ROOT / "results" / "eval_fno" / "fno_evaluation_summary.json")
    deeponet_results = load_json(ROOT / "results" / "eval_deeponet" / "deeponet_evaluation_summary.json")
    create_results_table(fno_results, deeponet_results, output_dir=str(ROOT / "results"))
    plot_results_bars(fno_results, deeponet_results, output_dir=str(FIG_DIR))
    plot_ood_degradation(fno_results, deeponet_results, output_dir=str(FIG_DIR))
    write_modes_ablation()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fno = load_fno(ROOT / "results" / "checkpoints" / "fno" / "best_model.pth", device)
    deeponet = load_deeponet(ROOT / "results" / "checkpoints" / "deeponet" / "best_model.pth", device)

    data_dir = ROOT / "data"
    id_data = load_dataset_npz(dataset_file_path(str(data_dir), "id_test"))
    validation_data = load_dataset_npz(dataset_file_path(str(data_dir), "validation"))
    ood_freq_data = load_dataset_npz(dataset_file_path(str(data_dir), "ood_frequency"))
    ood_smooth_data = load_dataset_npz(dataset_file_path(str(data_dir), "ood_smoothness"))
    x = np.linspace(0.0, 1.0, id_data["u0"].shape[1])

    fno_id, deeponet_id = predict_batch(fno, deeponet, id_data, 4, device)
    fno_ood_freq, deeponet_ood_freq = predict_batch(fno, deeponet, ood_freq_data, 2, device)
    fno_ood_smooth, deeponet_ood_smooth = predict_batch(fno, deeponet, ood_smooth_data, 2, device)

    id_index = select_representative_indices(id_data["uT"][:4], fno_id, deeponet_id, count=1)[0]
    ood_freq_index = select_representative_indices(ood_freq_data["uT"][:2], fno_ood_freq, deeponet_ood_freq, count=1)[0]
    ood_smooth_index = select_representative_indices(ood_smooth_data["uT"][:2], fno_ood_smooth, deeponet_ood_smooth, count=1)[0]
    samples = [
        make_sample_dict("id_test", "ID", id_index, id_data, fno_id, deeponet_id),
        make_sample_dict("ood_frequency", "OOD-frequency", ood_freq_index, ood_freq_data, fno_ood_freq, deeponet_ood_freq),
        make_sample_dict("ood_smoothness", "OOD-smoothness", ood_smooth_index, ood_smooth_data, fno_ood_smooth, deeponet_ood_smooth),
    ]
    for sample in samples:
        plot_sample_inputs_outputs(
            sample["id"],
            sample["u0"],
            sample["c"],
            sample["uT_true"],
            sample["uT_fno"],
            sample["uT_deeponet"],
            x=x,
            output_dir=str(FIG_DIR),
        )

    ood_samples = [sample for sample in samples if sample["type"].startswith("OOD")]
    plot_model_comparison_grid(samples, x=x, output_dir=str(FIG_DIR))
    plot_representative_cases(samples, x=x, output_dir=str(FIG_DIR))
    plot_local_error_panels(samples, x=x, output_dir=str(FIG_DIR))
    plot_ood_analysis(ood_samples, x=x, output_dir=str(FIG_DIR))

    spectral_results = {
        "fno": (np.vstack([fno_id, fno_ood_freq, fno_ood_smooth]), np.vstack([id_data["uT"][:4], ood_freq_data["uT"][:2], ood_smooth_data["uT"][:2]])),
        "deeponet": (
            np.vstack([deeponet_id, deeponet_ood_freq, deeponet_ood_smooth]),
            np.vstack([id_data["uT"][:4], ood_freq_data["uT"][:2], ood_smooth_data["uT"][:2]]),
        ),
    }
    plot_spectral_error_comparison(spectral_results, x=x, output_dir=str(FIG_DIR), basis="sine")
    plot_error_frequency_distribution(spectral_results, x=x, output_dir=str(FIG_DIR), basis="sine")

    spectral_count = 32
    fno_val_spec, deeponet_val_spec = predict_batch(fno, deeponet, validation_data, spectral_count, device)
    fno_id_spec, deeponet_id_spec = predict_batch(fno, deeponet, id_data, spectral_count, device)
    fno_ood_freq_spec, deeponet_ood_freq_spec = predict_batch(fno, deeponet, ood_freq_data, spectral_count, device)
    fno_ood_smooth_spec, deeponet_ood_smooth_spec = predict_batch(fno, deeponet, ood_smooth_data, spectral_count, device)
    split_spectral_results = {
        "validation": {
            "fno": (fno_val_spec, validation_data["uT"][:spectral_count]),
            "deeponet": (deeponet_val_spec, validation_data["uT"][:spectral_count]),
        },
        "id_test": {
            "fno": (fno_id_spec, id_data["uT"][:spectral_count]),
            "deeponet": (deeponet_id_spec, id_data["uT"][:spectral_count]),
        },
        "ood_frequency": {
            "fno": (fno_ood_freq_spec, ood_freq_data["uT"][:spectral_count]),
            "deeponet": (deeponet_ood_freq_spec, ood_freq_data["uT"][:spectral_count]),
        },
        "ood_smoothness": {
            "fno": (fno_ood_smooth_spec, ood_smooth_data["uT"][:spectral_count]),
            "deeponet": (deeponet_ood_smooth_spec, ood_smooth_data["uT"][:spectral_count]),
        },
    }
    plot_split_spectral_analysis(split_spectral_results, x=x, output_dir=str(FIG_DIR), basis="sine")
    plot_id_vs_ood_frequency_spectral_error(split_spectral_results, x=x, output_dir=str(FIG_DIR), basis="sine")

    energy_results = {
        "FNO": analyze_energy_consistency(list(fno_id), list(id_data["uT"][:4]), list(id_data["c"][:4]), x),
        "DeepONet": analyze_energy_consistency(list(deeponet_id), list(id_data["uT"][:4]), list(id_data["c"][:4]), x),
    }
    plot_energy_comparison(energy_results, output_dir=str(FIG_DIR))
    split_energy_inputs = {
        "validation": {
            "FNO": (fno_val_spec, validation_data["uT"][:spectral_count], validation_data["c"][:spectral_count]),
            "DeepONet": (deeponet_val_spec, validation_data["uT"][:spectral_count], validation_data["c"][:spectral_count]),
        },
        "id_test": {
            "FNO": (fno_id_spec, id_data["uT"][:spectral_count], id_data["c"][:spectral_count]),
            "DeepONet": (deeponet_id_spec, id_data["uT"][:spectral_count], id_data["c"][:spectral_count]),
        },
        "ood_frequency": {
            "FNO": (fno_ood_freq_spec, ood_freq_data["uT"][:spectral_count], ood_freq_data["c"][:spectral_count]),
            "DeepONet": (deeponet_ood_freq_spec, ood_freq_data["uT"][:spectral_count], ood_freq_data["c"][:spectral_count]),
        },
        "ood_smoothness": {
            "FNO": (fno_ood_smooth_spec, ood_smooth_data["uT"][:spectral_count], ood_smooth_data["c"][:spectral_count]),
            "DeepONet": (deeponet_ood_smooth_spec, ood_smooth_data["uT"][:spectral_count], ood_smooth_data["c"][:spectral_count]),
        },
    }
    plot_split_energy_proxy(analyze_split_energy_proxy(split_energy_inputs, x), output_dir=str(FIG_DIR))
    print("[OK] Analysis tables and figures generated.")


if __name__ == "__main__":
    main()
