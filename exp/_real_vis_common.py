"""
Shared helpers for run_real_exp1_vis.py, run_real_exp2_vis.py,
and run_real_vis_combined.py.

Mirrors the role of _vis_common.py for the simulation experiments.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from flowmatchnn.model import TwoLayerReLU
from flowmatchnn import learning_and_sampling
from flowmatchnn.utils import pca as pca_utils


# ---------------------------------------------------------------------------
# Shared argparse
# ---------------------------------------------------------------------------


def parse_vis_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, required=True,
                   help="Subdir under data/ and results/")
    p.add_argument("--data_root",    type=str, default="data")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--figs_dir",     type=str, default="figs")
    p.add_argument("--n_samples", type=int, default=64,
                   help="Number of images to synthesise for the image grid")
    p.add_argument("--img_h", type=int, default=None,
                   help="Image height override (default: read from config.json)")
    p.add_argument("--img_w", type=int, default=None,
                   help="Image width override (default: read from config.json)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_run_config(run_results_dir: Path) -> dict:
    with open(run_results_dir / "config.json") as f:
        return json.load(f)


def load_pca_model(run_data_dir: Path):
    with open(run_data_dir / "pca_model.pkl", "rb") as f:
        return pickle.load(f)


def load_model(run_data_dir: Path, d: int, m: int,
               device: torch.device) -> TwoLayerReLU:
    model = TwoLayerReLU(
        d=d, m=m, kappa=1.0,
        generator=torch.Generator(device=device).manual_seed(0),
        device=device, dtype=torch.float32,
    )
    state = torch.load(run_data_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_x1_pool(run_data_dir: Path, device: torch.device) -> torch.Tensor:
    with open(run_data_dir / "dataset.pkl", "rb") as f:
        ds = pickle.load(f)
    return torch.from_numpy(ds["x_1"]).to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


def generate_images(
    model: TwoLayerReLU,
    x_1_pool: torch.Tensor,
    pca_model,
    *,
    n_samples: int,
    d: int,
    B1: float,
    T_euler: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """
    Sample from the trained flow, inverse-PCA to pixels.

    Returns float32 array of shape (n_samples, H*W) with values in [0, 1].
    """
    g = torch.Generator(device=device).manual_seed(seed)
    x_hat_1 = learning_and_sampling.sample_algorithm2_real(
        model,
        n_eval=n_samples,
        d=d,
        B1=B1,
        T_euler=T_euler,
        generator=g,
        device=device,
        dtype=torch.float32,
        x_1_pool=x_1_pool,
    )
    z_np = x_hat_1.detach().cpu().numpy()
    return np.clip(pca_utils.inverse_transform(z_np, pca_model), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def read_csv(path: Path) -> dict:
    lines = path.read_text().strip().splitlines()
    header = lines[0].split(",")
    data: dict = {h: [] for h in header}
    for line in lines[1:]:
        for h, v in zip(header, line.split(",")):
            data[h].append(float(v))
    return data


def plot_trace(csv_path: Path, out_path: Path,
               ylabel: str, title: str) -> None:
    data = read_csv(csv_path)
    cols = list(data.keys())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(data[cols[0]], data[cols[1]], lw=1.5)
    ax.set_xlabel("GD iteration $t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_image_grid(
    images: np.ndarray,
    out_path: Path,
    img_h: int,
    img_w: int,
    title: str = "Generated images (flow → inverse PCA)",
    n_cols: int = 8,
) -> None:
    """Plot a grid of greyscale images (float32, shape (N, H*W), values [0,1])."""
    n = images.shape[0]
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    axes = np.array(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i].reshape(img_h, img_w),
                      cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output path helpers (figs/application/<dataset>/<subdir>/)
# ---------------------------------------------------------------------------

def param_suffix(run_name: str, dataset: str) -> str:
    """Strip 'real_<dataset>_' prefix to get the parameter portion of a run name.

    Example: 'real_mnist_m128_eta1e-3_ntr500_nte2000' → 'm128_eta1e-3_ntr500_nte2000'
    """
    prefix = f"real_{dataset}_"
    return run_name[len(prefix):] if run_name.startswith(prefix) else run_name


def app_figs_subdir(figs_dir: Path, dataset: str, subdir: str) -> Path:
    """Return figs_dir/application/<dataset>/<subdir>, creating it if needed."""
    d = figs_dir / "application" / dataset / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Combined multi-curve plot (one curve per (m, eta) setting, x-axis = t)
# ---------------------------------------------------------------------------

_M_LINESTYLES = ["-", "--", ":", "-."]
_M_MARKERS    = ["o",  "s",  "^", "D"]


def _eta_colors(n: int) -> list:
    cmap = cm.get_cmap("viridis")
    return [cmap(0.15 + 0.7 * i / max(n - 1, 1)) for i in range(n)]


def plot_combined_traces(
    *,
    traces: List[Tuple[int, str, List[float], List[float]]],
    m_list: List[int],
    eta_str_list: List[str],
    out: Path,
    ylabel: str,
    title: str,
) -> None:
    """Plot one curve per (m, eta_str) on a shared t-axis.

    traces  — list of (m, eta_str, ts, vals); missing/diverged runs omitted.
    Color encodes eta (viridis, light→dark for large→small).
    Linestyle + marker encode m.
    """
    eta_color = {e: c for e, c in zip(eta_str_list, _eta_colors(len(eta_str_list)))}
    m_style = {m: (_M_LINESTYLES[i], _M_MARKERS[i]) for i, m in enumerate(m_list)}

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)

    for m, eta_str, ts, vals in traces:
        ls, mk = m_style[m]
        ax.plot(
            ts, vals,
            color=eta_color[eta_str],
            linestyle=ls,
            marker=mk,
            markevery=max(1, len(ts) // 20),
            ms=4.5,
            lw=1.5,
            markeredgewidth=0.0,
            label=f"$m={m}$, $\\eta={eta_str}$",
        )

    ax.set_xlabel(r"GD iteration $t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if traces:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)

    if not traces:
        ax.text(0.5, 0.5, "no results found",
                ha="center", va="center", transform=ax.transAxes)
    else:
        leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                        ncol=1, handlelength=2.6, borderaxespad=0.0,
                        fontsize=9)
        leg.get_frame().set_linewidth(0.6)

    fig.savefig(out, dpi=150)
    plt.close(fig)
