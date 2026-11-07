"""
PCA-dimension sweep on real data (exp3_real_data). Answers R3 Q4.

One cell = one (dataset, d_pca, n, m, seed).  The sweep answers R3's
question: how does the flow-matching pipeline behave as the dimension
of the learned representation grows, and how does that interact with
the training-set size n?

Pipeline
    1. Fit PCA with d_pca components on the FULL training split, project
       train and test images into code space, and STANDARDISE the codes
       to unit per-component variance.
    2. Draw n training codes, build the Algorithm-1 triples
       (tau, x_tau, x_0) with the conditional coupling P(x_0|x_1) =
       N(x_1, I), calibrate the GD step size from the kernel spectrum,
       and train to a loss plateau.
    3. Score the trained model against a FIXED held-out reference in
       PIXEL space, and against the held-out velocity target in code
       space.
    4. Save 16 generated images (npy + 4x4 png) for the sample-quality
       panel.

WHY THE SCRIPT LOOKS LIKE THIS.  Every item below is a correction to a
defect in the first version of this experiment; do not drop any of them
without re-reading this list.

  (1) STANDARDISED CODES.  The coupling adds N(0, I) noise to EVERY
      retained coordinate, but raw PCA code variances decay steeply with
      the component index.  On raw codes the source's noise-to-signal
      ratio therefore grows with d_pca, and a "d_pca sweep" is really a
      sweep in corruption level.  `pca.standardize_codes` removes the
      confound; `scale` is re-applied before every inverse-PCA so pixel
      metrics stay in true pixel units.

  (2) NO EVALUATION LEAK.  `sample_algorithm2_real` draws its source
      points from a pool of real x_1 samples.  That pool MUST be the
      training codes.  The legacy version passed the held-out reference
      as the pool, so the sampler was handed the very set it was scored
      against.  (This is also why the training loop here is
      `train_to_plateau` plus an explicit sampling hook rather than
      `train_and_trace_real`: the latter hard-codes x_1_pool = x_1_ref.)

  (3) SPECTRAL STEP SIZE.  A single hand-picked eta cannot serve a grid
      whose kernel spectrum moves with both d_pca and n — the largest
      cells diverge or the smallest never move.  eta is calibrated per
      cell as eta = safety / lambda_max(H_tilde), converted to the units
      of the mean-squared training loss (`ntk.eta_spectral`).  Both eta
      and lambda_max are recorded in config.json.

  (4) TRAINED TO CONVERGENCE.  A fixed small iteration budget confounds
      "this representation is hard to learn" with "this cell got less
      optimisation".  Training stops when the relative training-loss
      improvement over the last --plateau_window recorded evals drops
      below --plateau_tol, capped at --T.  The achieved ratio is written
      to pca_summary.json as `converged_ratio`, so convergence is
      auditable per cell rather than assumed.

  (5) ONE FIXED DIRECTION SET.  Every pixel-space sliced-W_1 number in
      the sweep uses the SAME --n_proj_pixel directions, drawn from
      --pixel_dir_seed, and the SAME held-out reference (--ref_seed).
      Code-space W_1 is not comparable across d_pca and is kept only as
      a diagnostic column of pixel_trace.csv.

  (6) METRICS THAT ARE DELIBERATELY ABSENT.  The legacy summary reported
      `identity_w1` (sliced W_1 of the raw noisy source, "no model") and
      `gap_closed = (identity_w1 - model_w1) / (identity_w1 - floor)`.
      Both are invalid and are NOT recomputed here: sliced W_1 along
      random pixel directions is not monotone under added variance, so a
      generator that is *better* than the reference-vs-reference floor
      scores a negative numerator, and `gap_closed` can exceed 1 or go
      negative for a perfect generator.  Use `holdout_floor_w1` as the
      honest reference level instead.

Reported metrics (pca_summary.json)
    model_w1               pixel sliced W_1 of n generated samples vs the
                           held-out reference (n_gen = n).
    pca_reconstruction_w1  same metric for PCA-reconstructed test images:
                           a REFERENCE LEVEL, not a floor (see the
                           docstring of pca.pca_reconstruction_w1).
    holdout_floor_w1       honest finite-sample floor: n real held-out
                           images vs the reference, same n, same
                           directions.
    vel_risk_l2_ratio      held-out E||f - v||_2 / E||v||_2.  PRIMARY:
                           this is the first moment Theorem 3.3 bounds.
    vel_risk_sq_ratio      held-out E||f - v||^2 / E||v||^2.  Reference
                           only; > 1 means worse than predicting 0.
    train_loss_norm        final training loss / E||v||^2 on the training
                           triples — how much of the target the model
                           actually fit.

Outputs (results/pca_{dataset}_dpca{k}_n{n}_m{m}_seed{s}/)
    config.json        resolved hyperparameters + eta, lambda_max
    pixel_trace.csv    t, loss, pca_w1, pixel_w1
    pca_summary.json   config + every metric above
    samples.npy        (16, H, W) generated images
    samples_grid.png   4x4 grid of the same
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# The shared helpers (_common.py, ...) live one level up in exp/.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import chdir_to_repo, require_cuda, write_csv  # noqa: E402

from flowmatchnn import learning_and_sampling as LS  # noqa: E402
from flowmatchnn.utils import data as data_utils  # noqa: E402
from flowmatchnn.utils import metrics as metric_utils  # noqa: E402
from flowmatchnn.utils import ntk as ntk_utils  # noqa: E402
from flowmatchnn.utils import pca as pca_utils  # noqa: E402
from flowmatchnn.utils import realdata  # noqa: E402
from flowmatchnn.utils import velocity as velocity_utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, required=True,
                   choices=["mnist", "fashion_mnist"])
    p.add_argument("--d_pca", type=int, required=True)
    p.add_argument("--n", type=int, default=500,
                   help="Training-set size (the sweep varies it).")
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--kappa", type=float, default=1.0)

    # Optimisation. eta is NOT a flag: it is derived from the kernel
    # spectrum of this cell (see --eta_safety).
    p.add_argument("--T", type=int, default=2000,
                   help="Iteration cap; the plateau rule usually stops "
                        "earlier.")
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--plateau_window", type=int, default=10,
                   help="Stop when the relative loss improvement over "
                        "this many recorded evals falls below "
                        "--plateau_tol.")
    p.add_argument("--plateau_tol", type=float, default=0.01)
    p.add_argument("--eta_safety", type=float, default=0.5,
                   help="eta = safety / lambda_max(H_tilde), in "
                        "mean-squared-loss units (ntk.eta_spectral).")
    p.add_argument("--lambda_n_sub", type=int, default=1000,
                   help="Subsample size for the lambda_max estimate; the "
                        "estimate is rescaled by n / n_sub.")

    # Sampling.
    p.add_argument("--T_euler", type=int, default=200)
    p.add_argument("--B1", type=float, default=10.0)

    # Evaluation. The reference set and the pixel directions are fixed
    # by their own seeds, NOT by --seed, so every cell of the sweep is
    # scored against exactly the same target with the same slices.
    p.add_argument("--n_test", type=int, default=10000,
                   help="Held-out reference size (clipped to the test "
                        "split; the default takes the whole split).")
    p.add_argument("--ref_seed", type=int, default=12345,
                   help="FIXED across cells: selects the reference set.")
    p.add_argument("--n_proj_pixel", type=int, default=128)
    p.add_argument("--pixel_dir_seed", type=int, default=777,
                   help="FIXED across cells so pixel W_1 is comparable.")
    p.add_argument("--n_proj_pca", type=int, default=128,
                   help="Directions for the code-space W_1 diagnostic "
                        "column (not comparable across d_pca).")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_root", type=str, default="data/application")
    p.add_argument("--image_col", type=str, default="image")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def heldout_velocity_risk(model, z_ref, *, generator, dtype):
    """
    Held-out velocity risk on real data (Theorem 3.3's quantity).

    `learning_and_sampling.evaluate_velocity_risk` draws fresh triples
    from the GMM simulation model, so it cannot be used here; we build
    the same Algorithm-1 triples from held-out CODES instead — the exact
    real-data analogue.  Both moments are returned, normalised by the
    corresponding moment of the target v, so the numbers are comparable
    across d_pca (||v|| itself grows with dimension):

        risk_l2_ratio = E||f - v||_2 / E||v||_2      <- primary
        risk_sq_ratio = E||f - v||^2  / E||v||^2     <- reference; a
            value >= 1 means the model is no better than f == 0.
    """
    model.eval()
    tau, x_tau, x_0 = data_utils.make_dataset(
        z_ref, generator=generator, dtype=dtype
    )
    v = velocity_utils.velocity_on_path(x_0, z_ref, tau)
    err_sq = ((model(x_tau, tau, x_0) - v) ** 2).sum(dim=-1)
    v_sq = (v ** 2).sum(dim=-1)
    model.train()
    return {
        "vel_risk_l2": float(err_sq.sqrt().mean()),
        "vel_risk_sq": float(err_sq.mean()),
        "vel_target_l2": float(v_sq.sqrt().mean()),
        "vel_target_sq": float(v_sq.mean()),
    }


def main() -> None:
    args = parse_args()
    chdir_to_repo()
    device = require_cuda()
    dtype = torch.float32

    run_name = args.run_name or (
        f"pca_{args.dataset}_dpca{args.d_pca}_n{args.n}"
        f"_m{args.m}_seed{args.seed}"
    )
    out_dir = Path(args.results_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. images -> PCA -> STANDARDISED codes (correction 1) ----
    data_dir = Path(args.data_root) / args.dataset
    x_tr, (img_h, img_w) = realdata.load_split(
        data_dir / "train.parquet", args.image_col)
    x_te, _ = realdata.load_split(data_dir / "test.parquet", args.image_col)
    d_flat = x_tr.shape[1]

    pca_model = pca_utils.fit(x_tr, args.d_pca)
    total_var = float(x_tr.var(axis=0, ddof=1).sum())
    evr = float(pca_model.explained_var.sum()) / max(total_var, 1e-12)

    # `scale` is computed from the FULL training split, so it is a
    # property of the representation and does not drift with n.
    scale, z_tr_all, z_te_all = pca_utils.standardize_codes(
        pca_utils.transform(x_tr, pca_model),
        pca_utils.transform(x_te, pca_model),
    )
    scale_t = torch.from_numpy(np.asarray(scale)).to(device=device,
                                                     dtype=dtype)

    # ---- 2. fixed held-out reference + fixed slices (correction 5) ----
    n_test = min(args.n_test, len(x_te))
    rng_ref = np.random.default_rng(args.ref_seed)
    idx_te = rng_ref.permutation(len(x_te))[:n_test]
    x_ref_pixel = torch.from_numpy(
        np.ascontiguousarray(x_te[idx_te])).to(device=device, dtype=dtype)
    z_ref = torch.from_numpy(
        np.ascontiguousarray(z_te_all[idx_te])).to(device=device, dtype=dtype)

    g_dir = torch.Generator(device=device).manual_seed(args.pixel_dir_seed)
    pixel_dirs = metric_utils.random_unit_directions(
        args.n_proj_pixel, d_flat, generator=g_dir, device=device,
        dtype=dtype,
    )
    g_proj = torch.Generator(device=device).manual_seed(
        args.seed * 1_000_003 + 4)
    pca_dirs = metric_utils.random_unit_directions(
        args.n_proj_pca, args.d_pca, generator=g_proj, device=device,
        dtype=dtype,
    )

    def pixel_w1_of(samples_std: torch.Tensor) -> float:
        """
        Pixel-space sliced W_1 of standardised code samples.

        pca_utils.pixel_sliced_w1 inverse-transforms its first argument,
        so the standardisation of correction (1) must be undone FIRST —
        forgetting `* scale_t` silently shrinks every image toward the
        PCA mean.
        """
        return pca_utils.pixel_sliced_w1(
            samples_std * scale_t, pca_model, x_ref_pixel, pixel_dirs,
        )

    # ---- 3. training codes + Algorithm-1 triples ----
    rng = np.random.default_rng(args.seed * 1_000_003 + 17)
    n_train = min(args.n, len(z_tr_all))
    idx_tr = rng.choice(len(z_tr_all), size=n_train, replace=False)
    x_1 = torch.from_numpy(
        np.ascontiguousarray(z_tr_all[idx_tr])).to(device=device, dtype=dtype)

    g_data = torch.Generator(device=device).manual_seed(
        args.seed * 1_000_003 + 19)
    tau, x_tau, x_0 = data_utils.make_dataset(x_1, generator=g_data,
                                              dtype=dtype)
    dataset = {"tau": tau, "x_tau": x_tau, "x_0": x_0, "x_1": x_1}
    v_train_sq = float(
        (velocity_utils.velocity_on_path(x_0, x_1, tau) ** 2).sum(-1).mean())

    # ---- 4. step size from this cell's kernel spectrum (correction 3) --
    g_lam = torch.Generator(device=device).manual_seed(
        args.seed * 1_000_003 + 5)
    lambda_max = ntk_utils.lambda_max_subsampled(
        x_tau, tau, x_0, kappa=args.kappa, n_sub=args.lambda_n_sub,
        generator=g_lam,
    )
    eta = ntk_utils.eta_spectral(
        lambda_max, n=n_train, safety=args.eta_safety, mean_loss=True)
    print(f"[run_pca_sweep] {run_name}: d_flat={d_flat} evr={evr:.4f} "
          f"lambda_max={lambda_max:.4g} eta={eta:.4g}")

    # ---- 5. train to plateau, tracing pixel W_1 (corrections 2, 4) ----
    cfg = LS.TrainConfig(
        n_train=n_train, d=args.d_pca, m=args.m, kappa=args.kappa,
        T=args.T, eta=eta, B1=args.B1, T_euler=args.T_euler,
        n_test=n_test, eval_every=args.eval_every, seed=args.seed,
        n_proj=args.n_proj_pca,
    )
    g_trace = torch.Generator(device=device).manual_seed(
        args.seed * 1_000_003 + 2)
    pca_w1_trace: list = []

    def eval_hook(model, t: int) -> float:
        """Sample, record code-space W_1, return pixel W_1."""
        samples = LS.sample_algorithm2_real(
            model, n_eval=n_train, d=args.d_pca, B1=args.B1,
            T_euler=args.T_euler, generator=g_trace, device=device,
            dtype=dtype,
            # Correction (2): source pool = TRAINING codes, never x_ref.
            x_1_pool=x_1,
        )
        pca_w1_trace.append(float(metric_utils.sliced_wasserstein_1(
            samples, z_ref, directions=pca_dirs)))
        return pixel_w1_of(samples)

    res = LS.train_to_plateau(
        cfg, dataset=dataset, device=device, dtype=dtype,
        plateau_window=args.plateau_window, plateau_tol=args.plateau_tol,
        eval_hook=eval_hook,
    )
    model = res["model"]
    losses = res["loss"]

    # Audit trail for correction (4): the exact quantity the stopping
    # rule tests, whether or not it fired.
    window = args.plateau_window
    if len(losses) > window and losses[-1 - window] > 0:
        past = losses[-1 - window]
        converged_ratio = (past - losses[-1]) / past
    else:
        converged_ratio = float("nan")
    converged = bool(converged_ratio < args.plateau_tol)

    # ---- 6. final metrics ----
    g_final = torch.Generator(device=device).manual_seed(
        args.seed * 1_000_003 + 23)
    samples_std = LS.sample_algorithm2_real(
        model, n_eval=n_train, d=args.d_pca, B1=args.B1,
        T_euler=args.T_euler, generator=g_final, device=device,
        dtype=dtype, x_1_pool=x_1,
    )
    model_w1 = pixel_w1_of(samples_std)
    reconstruction_w1 = pca_utils.pca_reconstruction_w1(
        pca_model, x_ref_pixel, pixel_dirs)
    g_floor = torch.Generator(device=device).manual_seed(
        args.ref_seed + 1)
    # NOTE: holdout_floor_w1 draws its n images FROM the reference, so
    # the floor carries a finite-population factor ~ sqrt(1 - n/n_test);
    # it is exact for n << n_test and mildly optimistic as n approaches
    # it (n_test and n are both recorded so this is checkable).
    floor_w1 = pca_utils.holdout_floor_w1(
        min(n_train, n_test), x_ref_pixel, pixel_dirs, generator=g_floor)

    g_vel = torch.Generator(device=device).manual_seed(
        args.seed * 1_000_003 + 31)
    vel = heldout_velocity_risk(model, z_ref, generator=g_vel, dtype=dtype)

    # ---- 7. sample images ----
    imgs = pca_utils.reconstruct_clipped(
        (samples_std[:16] * scale_t).detach().cpu().numpy(), pca_model
    ).reshape(-1, img_h, img_w)
    np.save(out_dir / "samples.npy", imgs)
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for ax, im in zip(axes.ravel(), imgs):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(f"{args.dataset}  d_pca={args.d_pca}  n={n_train}",
                 fontsize=9)
    fig.savefig(out_dir / "samples_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 8. outputs ----
    config = {
        **vars(args), "run_name": run_name, "n_train": n_train,
        "n_test": n_test, "d_flat": d_flat, "img_h": img_h,
        "img_w": img_w, "eta": eta, "lambda_max": lambda_max,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    write_csv(out_dir / "pixel_trace.csv", "t,loss,pca_w1,pixel_w1",
              zip(res["t"], losses, pca_w1_trace, res["hook"]))

    summary = {
        **config,
        "explained_var_ratio": evr,
        "stopped_at": res["stopped_at"],
        "converged": converged,
        "converged_ratio": converged_ratio,
        "final_loss": losses[-1],
        "train_loss_norm": losses[-1] / max(v_train_sq, 1e-12),
        "model_w1": model_w1,
        "pca_reconstruction_w1": reconstruction_w1,
        "holdout_floor_w1": floor_w1,
        "vel_risk_l2_ratio": vel["vel_risk_l2"] / max(vel["vel_target_l2"],
                                                      1e-12),
        "vel_risk_sq_ratio": vel["vel_risk_sq"] / max(vel["vel_target_sq"],
                                                      1e-12),
        **vel,
        "v_train_sq": v_train_sq,
        # `identity_w1` / `gap_closed` are intentionally NOT emitted; see
        # correction (6) in the module docstring.
    }
    with open(out_dir / "pca_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[run_pca_sweep] wrote {out_dir}/  stopped_at={res['stopped_at']} "
          f"converged_ratio={converged_ratio:.4f} "
          f"train_loss_norm={summary['train_loss_norm']:.4f} "
          f"vel_risk_l2_ratio={summary['vel_risk_l2_ratio']:.4f} "
          f"vel_risk_sq_ratio={summary['vel_risk_sq_ratio']:.4f} "
          f"model_w1={model_w1:.5f} (recon {reconstruction_w1:.5f}, "
          f"floor {floor_w1:.5f})")


if __name__ == "__main__":
    main()
