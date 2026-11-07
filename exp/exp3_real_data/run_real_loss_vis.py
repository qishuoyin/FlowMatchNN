"""
Exp 1 visualisation (real data): loss vs t + generated image grid.

Reads from data/<run_name>/ and results/<run_name>/; writes to
figs/application/<dataset>/:

  loss/<param_suffix>.png           — Exp 1 training-loss trajectory (log scale).
  reconstruction/<param_suffix>.png — 8×8 grid of images synthesised by the
                                      trained flow: sample from empirical P_0,
                                      run Euler ODE in PCA space,
                                      inverse-PCA → pixels.

Image shape (H, W) is read from results/<run_name>/config.json.

Usage:
    python exp/exp3_real_data/run_real_loss_vis.py --run_name real_mnist_m128_eta1e-3_ntr500_nte2000
    sbatch scripts/exp3_real_data/run_real_loss_vis.sh   (iterates all grid cells)
"""

from __future__ import annotations

from pathlib import Path

import torch

# The shared helpers (_common.py, _vis_common.py, _real_vis_common.py) live
# one level up in exp/. Entry points are executed as plain files, so only
# this subfolder lands on sys.path; add exp/ explicitly before importing.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import chdir_to_repo
from _real_vis_common import (
    app_figs_subdir,
    generate_images,
    load_model,
    load_pca_model,
    load_run_config,
    load_x1_pool,
    param_suffix,
    parse_vis_args,
    plot_image_grid,
    plot_trace,
)


def main() -> None:
    args = parse_vis_args()
    chdir_to_repo()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_data_dir    = Path(args.data_root)    / args.run_name
    run_results_dir = Path(args.results_root) / args.run_name
    figs_dir        = Path(args.figs_dir)

    cfg     = load_run_config(run_results_dir)
    n_pca   = int(cfg["n_pca"])
    m       = int(cfg["m"])
    B1      = float(cfg["B1"])
    T_euler = int(cfg["T_euler"])
    img_h   = args.img_h if args.img_h is not None else int(cfg["img_h"])
    img_w   = args.img_w if args.img_w is not None else int(cfg["img_w"])
    dataset = cfg.get("dataset", args.run_name)
    sfx     = param_suffix(args.run_name, dataset)

    # ---- loss vs t ----
    loss_dir  = app_figs_subdir(figs_dir, dataset, "loss")
    loss_path = loss_dir / f"{sfx}.png"
    plot_trace(
        run_results_dir / "exp1_trace.csv", loss_path,
        ylabel=r"training loss $L(\theta^{(t)})$",
        title=f"Loss vs $t$ — {dataset}  ({sfx})",
    )
    print(f"[run_real_loss_vis] wrote {loss_path}")

    # ---- image grid ----
    pca_model = load_pca_model(run_data_dir)
    model     = load_model(run_data_dir, d=n_pca, m=m, device=device)
    x_1_pool  = load_x1_pool(run_data_dir, device)

    print(f"[run_real_loss_vis] generating {args.n_samples} images …")
    images = generate_images(
        model, x_1_pool, pca_model,
        n_samples=args.n_samples, d=n_pca, B1=B1, T_euler=T_euler,
        device=device, seed=args.seed,
    )

    recon_dir  = app_figs_subdir(figs_dir, dataset, "reconstruction")
    grid_path  = recon_dir / f"{sfx}.png"
    plot_image_grid(
        images, grid_path, img_h=img_h, img_w=img_w,
        title=f"Generated {dataset} images  (m={m}, η={cfg['eta']})",
    )
    print(f"[run_real_loss_vis] wrote {grid_path}")


if __name__ == "__main__":
    main()
