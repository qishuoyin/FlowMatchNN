"""
PCA-dimension sweep figures (exp3_real_data). Answers R3 Q4.

Reads results/pca_{dataset}_dpca{k}_n{n}_m{m}_seed{s}/pca_summary.json
over the grid in scripts/exp3_real_data/_pca_grid.sh and produces
exactly two figures and one table:

  Fig a  pca_vel_risk.png   held-out velocity risk RATIO
           E||f - v||_2 / E||v||_2 vs d_pca, one curve per n, one panel
           per dataset, with the trivial-predictor line at 1.0.  This is
           the theorem's own metric (first moment, Theorem 3.3) and the
           only one that is directly comparable across d_pca without
           further normalisation.

  Fig b  pca_pixel_w1.png   pixel-space sliced W_1 of the generated
           samples vs d_pca, one curve per n, one panel per dataset,
           with two overlays:
             - pca_reconstruction_w1 (black dashed): the level PCA
               compression alone commits.  A REFERENCE LEVEL, not a
               floor — sliced W_1 is not monotone under added variance,
               so samples can legitimately score below it.
             - holdout_floor_w1 (dotted, per n): n REAL held-out images
               scored the same way.  This IS the honest floor, and it
               moves with n, which is why it is drawn per curve.

  Table    pca_sweep_table.csv  every metric of every cell, one row per
           (dataset, d_pca, n, m, seed).

The (d_pca, n) grid is ragged (small d_pca is not run at the largest n),
so missing cells are skipped with a warning rather than treated as an
error.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import chdir_to_repo  # noqa: E402


# Columns of pca_sweep_table.csv, in order. Everything the sweep
# measures, so the table stands on its own without the JSONs.
TABLE_FIELDS = [
    "dataset", "d_pca", "n", "m", "seed",
    "model_w1", "pca_reconstruction_w1", "holdout_floor_w1",
    "vel_risk_l2_ratio", "vel_risk_sq_ratio", "train_loss_norm",
    "converged", "converged_ratio", "stopped_at",
    "eta", "lambda_max", "explained_var_ratio", "final_loss",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", type=str, nargs="+",
                   default=["mnist", "fashion_mnist"])
    p.add_argument("--d_pca_list", type=int, nargs="+",
                   default=[2, 4, 8, 16, 32, 64])
    p.add_argument("--n_list", type=int, nargs="+",
                   default=[500, 2000, 8000])
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--figs_dir", type=str, default="figs/exp3_real_data")
    return p.parse_args()


def load_grid(args) -> tuple:
    """
    Read every cell of the grid.

    Returns (cells, rows, missing) where cells maps
    (dataset, n, d_pca) -> list of per-seed summary dicts, rows is the
    flat table, and missing lists the absent pca_summary.json paths.
    """
    root = Path(args.results_root)
    cells, rows, missing = {}, [], []
    for ds in args.datasets:
        for n in args.n_list:
            for k in args.d_pca_list:
                for s in range(args.n_seeds):
                    path = (
                        root / f"pca_{ds}_dpca{k}_n{n}_m{args.m}_seed{s}"
                        / "pca_summary.json"
                    )
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    with open(path) as f:
                        summ = json.load(f)
                    cells.setdefault((ds, n, k), []).append(summ)
                    rows.append({
                        key: summ.get(key) for key in TABLE_FIELDS
                    })
    return cells, rows, missing


def agg(cells, ds, n, ks, key):
    """(present_ks, mean, sd) of `key` over seeds, skipping empty cells."""
    present, mus, sds = [], [], []
    for k in ks:
        vals = [c[key] for c in cells.get((ds, n, k), [])
                if c.get(key) is not None]
        if vals:
            present.append(k)
            mus.append(float(np.mean(vals)))
            sds.append(float(np.std(vals)))
    return present, np.array(mus), np.array(sds)


def style_axis(ax, title: str, ylabel: str) -> None:
    """Shared axis dressing: log2 x in d_pca, legend, labels."""
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"PCA dimension $d_{\mathrm{pca}}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # A dataset can be entirely absent (partial grid); legend() would
    # then warn about having nothing to draw.
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=7.5)



def sample_quality_figure(args, figs, root):
    """
    Fig c — sample quality vs (d_pca, n), the third thing R3 asked to
    see. Each panel tiles four generated images from one cell; rows are
    d_pca, columns are n. Reads the per-cell samples.npy written by
    run_pca_sweep.py (seed 0), so it needs no re-generation.
    """
    for ds in args.datasets:
        panels, present_k = {}, []
        for k in args.d_pca_list:
            row = {}
            for n in args.n_list:
                f = (root / f"pca_{ds}_dpca{k}_n{n}_m{args.m}_seed0"
                     / "samples.npy")
                if f.exists():
                    imgs = np.load(f)[:4]
                    if len(imgs) == 4:
                        h, w = imgs.shape[1:]
                        tile = np.zeros((2 * h, 2 * w), dtype=imgs.dtype)
                        tile[:h, :w], tile[:h, w:] = imgs[0], imgs[1]
                        tile[h:, :w], tile[h:, w:] = imgs[2], imgs[3]
                        row[n] = tile
            if row:
                panels[k] = row
                present_k.append(k)
        if not present_k:
            continue
        nrow, ncol = len(present_k), len(args.n_list)
        fig, axes = plt.subplots(nrow, ncol, figsize=(1.5 * ncol, 1.5 * nrow),
                                 squeeze=False)
        for i, k in enumerate(present_k):
            for j, n in enumerate(args.n_list):
                ax = axes[i][j]
                if n in panels[k]:
                    ax.imshow(panels[k][n], cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"n={n}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"$d_{{pca}}$={k}", fontsize=8)
        fig.suptitle(f"Generated samples — {ds} (seed 0)", fontsize=10)
        fig.tight_layout()
        out = figs / f"pca_samples_{ds}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[run_pca_sweep_vis] wrote {out}")


def main() -> None:
    args = parse_args()
    chdir_to_repo()
    figs = Path(args.figs_dir)
    figs.mkdir(parents=True, exist_ok=True)

    cells, rows, missing = load_grid(args)
    if not rows:
        print("[run_pca_sweep_vis] no cells found under "
              f"{args.results_root}/ — nothing to plot")
        if missing:
            print(f"[run_pca_sweep_vis] WARNING: {len(missing)} missing "
                  f"cells (first: {missing[0]})")
        return

    ks = args.d_pca_list
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(args.n_list)))
    n_ds = len(args.datasets)

    # ---------- Fig a: held-out velocity risk (Theorem 3.3) ----------
    fig, axes = plt.subplots(1, n_ds, figsize=(5.0 * n_ds, 4.2),
                             squeeze=False, constrained_layout=True)
    for ax, ds in zip(axes[0], args.datasets):
        for n, col in zip(args.n_list, colors):
            present, mus, sds = agg(cells, ds, n, ks, "vel_risk_l2_ratio")
            if not present:
                continue
            ax.errorbar(present, mus, yerr=sds, color=col, marker="o", ms=4,
                        lw=1.5, capsize=2, label=f"n={n}")
        # A model with f == 0 scores exactly 1: everything above this
        # line is worse than predicting no motion at all.
        ax.axhline(1.0, color="k", ls="--", lw=1.0,
                   label=r"trivial predictor $f\equiv 0$")
        ax.set_yscale("log")
        style_axis(ax, f"Held-out velocity risk — {ds}",
                   r"$\mathbb{E}\|f_\theta - v\|_2 \,/\, "
                   r"\mathbb{E}\|v\|_2$")
    out = figs / "pca_vel_risk.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[run_pca_sweep_vis] wrote {out}")

    # ---------- Fig b: pixel-space sample quality ----------
    fig, axes = plt.subplots(1, n_ds, figsize=(5.0 * n_ds, 4.2),
                             squeeze=False, constrained_layout=True)
    for ax, ds in zip(axes[0], args.datasets):
        for n, col in zip(args.n_list, colors):
            present, mus, sds = agg(cells, ds, n, ks, "model_w1")
            if not present:
                continue
            ax.errorbar(present, mus, yerr=sds, color=col, marker="o", ms=4,
                        lw=1.5, capsize=2, label=f"model, n={n}")
            fk, fmu, _ = agg(cells, ds, n, ks, "holdout_floor_w1")
            if fk:
                ax.plot(fk, fmu, color=col, ls=":", lw=1.2, marker="^",
                        ms=3, label=f"held-out floor, n={n}")
        # PCA reconstruction level depends only on d_pca, so one curve
        # per panel (read off whichever n produced the cell).
        rk, rmu = [], []
        for k in ks:
            vals = [c["pca_reconstruction_w1"]
                    for n in args.n_list for c in cells.get((ds, n, k), [])]
            if vals:
                rk.append(k)
                rmu.append(float(np.mean(vals)))
        if rk:
            ax.plot(rk, rmu, "k--", marker="s", ms=4, lw=1.2,
                    label="PCA reconstruction level")
        ax.set_yscale("log")
        style_axis(ax, f"Pixel-space sample quality — {ds}",
                   r"pixel sliced $W_1$ vs held-out test set")
    out = figs / "pca_pixel_w1.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[run_pca_sweep_vis] wrote {out}")

    sample_quality_figure(args, figs, Path(args.results_root))

    # ---------- Table ----------
    out = figs / "pca_sweep_table.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[run_pca_sweep_vis] wrote {out} ({len(rows)} rows)")
    if missing:
        print(f"[run_pca_sweep_vis] WARNING: {len(missing)} missing cells "
              f"(expected for the ragged (d_pca, n) grid; "
              f"first: {missing[0]})")


if __name__ == "__main__":
    main()
