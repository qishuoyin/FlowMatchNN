"""
Coupling comparison (exp2_coupling). Answers R3 Q1.

Answers Reviewer 3: "compare the conditional coupling used in the
theory with a more standard coupling ... show how large the gap is
between the theoretical setting and common training practice."

-------------------------------------------------------------------
Design: a 2 x 2 (plus one extra coupling) sharing ONE data draw
-------------------------------------------------------------------
Per (d, n, seed) we draw the training data EXACTLY ONCE:

    x_1 ~ P_1 (GMM, centers +- c e_1, identity cov),  x_0 ~ N(x_1, I),
    tau ~ U[0, 1]

and then vary only two factors:

  --coupling {conditional, permuted, ot}   how (x_0, x_1) are PAIRED
      conditional : the paper's pairing — x_0 as drawn, so
                    x_{0,i} = x_{1,i} + eps_i (Algorithm 1).
      permuted    : data_utils.permute_pairing(x_0) — the SAME x_0
                    rows, randomly re-paired. This realises an
                    INDEPENDENT coupling while keeping the source
                    marginal, the target marginal and the sample size
                    BIT-IDENTICAL to the conditional arm.
      ot          : data_utils.minibatch_ot_pairing(x_0, x_1) — the
                    same rows again, re-paired by an exact OT plan.

  --net {conditional, unconditional}       how the network is PARAMETERISED
      conditional   : TwoLayerReLU        f(x_tau, tau, x_0)
      unconditional : UncondTwoLayerReLU  f(x_tau, tau)

3 couplings x 2 nets = 6 arms. "Standard flow-matching practice" is
(permuted, unconditional); the paper's setting is (conditional,
conditional). Because the ONLY things that vary across the 6 arms are
the pairing and the conditioning — same x_1, same x_0 rows, same tau,
same n, same m, same eval protocol — the 2 x 2 ISOLATES THE COUPLING
EFFECT FROM THE ARCHITECTURE EFFECT. Reading down a column gives the
pure coupling effect at fixed architecture; reading across a row gives
the pure architecture effect at fixed coupling.

The previous version of this experiment was confounded four ways: it
changed the coupling, the architecture, the source marginal
(N(x_1, I) vs N(0, I)) and — through a shared hard-coded eta acting on
different Gram matrices — the effective step size, all at once. No
observed difference could be attributed to the coupling. Every control
below removes one of those confounds and stays.

-------------------------------------------------------------------
Controls
-------------------------------------------------------------------
1. SAME SOURCE POOL FOR SAMPLING, FROM TRAINING DATA ONLY.
   The conditional net samples via LS.sample_algorithm2_real with
   x_1_pool = the TRAINING x_1 (so x_hat_0 = x_1' + N(0, I), x_1' drawn
   from the training targets); the unconditional net resamples
   x_hat_0 from the TRAINING x_0 pool. Both are draws from the same
   empirical source P_0, and the held-out reference set is NEVER shown
   to the sampler. (The previous version passed the evaluation
   reference in as the sampler's pool, i.e. it leaked the test set.)

2. PER-ARM STEP SIZE, calibrated to that arm's own kernel:
       lam_max = ntk.lambda_max_subsampled(x_tau, tau, z0, kappa=...)
       eta     = ntk.eta_spectral(lam_max, n=n, safety=..., mean_loss=True)
   Re-pairing changes x_tau and therefore the Gram matrix, so ONE
   shared eta silently gives the arms different effective step sizes —
   that was one of the four confounds. z0 is the arm's own conditioning
   block (x_0 for the conditional net; an empty block for the
   unconditional net, whose inputs are z = [x_tau; tau]), so each arm's
   eta comes from the kernel it actually trains under. eta and
   lambda_max are recorded per run.

3. TRAIN TO CONVERGENCE, not a fixed short budget. GD stops when the
   relative training-loss improvement over the last --plateau_window
   recorded evals falls below --plateau_tol (1 %), capped at --T
   (default 50000). `stopped_at` and `converged_ratio` are recorded so
   a reader can check no arm was cut off mid-descent — a fixed short
   budget would confound "worse coupling" with "not yet trained".

4. TWO REFERENCE LINES, so the numbers can be read in absolute terms.
       donothing_w1 : sliced W1 of the RAW source draws x_0 vs the
                      reference. Here P_0 = P_1 * N(0, I) is genuinely
                      close to P_1, so a trained flow MUST beat this;
                      reporting it shows that it does.
       floor_w1     : sliced W1 between n FRESH true P_1 samples and
                      the reference — the honest finite-sample floor at
                      this n. No arm can do better.
   Both baselines use n rows against the same 4000-row reference with
   the same 128 fixed directions as the model samples, so the three are
   directly comparable — and both are also reported under the exact W1
   whenever the model's exact W1 is, since sliced and exact W1 live on
   different scales and must never be compared to each other. (These
   baselines are legitimate here because the simulation's P_1 can be
   sampled exactly — unlike in the pixel-space PCA experiment.)

Data: simulation GMM only, d in {3, 5, 10}. No real-data arm: the point
is the coupling, and only the simulation gives an exactly-samplable P_1
for the two baselines above.

Metrics: sliced-W1 trace against a fixed held-out reference of
--n_test fresh P_1 samples with 128 fixed directions, the final sliced
W1, the exact W1 (metrics.wasserstein1_exact) when d <= --d_exact_max,
the training-loss trace, and the two baselines.

Output (results/coup_{coupling}_{net}_d{d}_n{n}_m{m}_seed{seed}/):
    config.json         resolved hyperparameters (+ eta, lambda_max)
    trace.csv           t, loss, w1
    coup_summary.json   final_w1, exact_w1 (nan when d > d_exact_max),
                        donothing_w1, floor_w1, their exact counterparts
                        donothing_exact_w1 / floor_exact_w1, eta,
                        lambda_max, stopped_at, converged_ratio,
                        final_loss

NOTE ON THE TRAINING LOOP. The GD loop lives here rather than in
LS.train_and_trace_real / LS.train_and_trace_uncond because those
wrappers hard-code one of the things this experiment must control:
train_and_trace_real samples from its W1 reference set (control 1) and
neither supports the plateau stopping rule (control 3). The loop below
is the repo's standard update — mean-squared `losses.empirical_loss`,
theta <- theta - eta * grad L, full batch — driven through one code
path for both networks, so the two architectures differ ONLY in the
forward signature and the sampler. That single shared path is what
makes the 2 x 2 a clean factorial.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, List

import torch

# The shared helpers (_common.py, ...) live one level up in exp/.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import chdir_to_repo, require_cuda, write_csv  # noqa: E402

from flowmatchnn import learning_and_sampling as LS  # noqa: E402
from flowmatchnn.model import TwoLayerReLU, UncondTwoLayerReLU  # noqa: E402
from flowmatchnn.utils import data as data_utils  # noqa: E402
from flowmatchnn.utils import losses as loss_utils  # noqa: E402
from flowmatchnn.utils import metrics as metric_utils  # noqa: E402
from flowmatchnn.utils import ntk  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coupling", choices=["conditional", "permuted", "ot"],
                   required=True,
                   help="How (x_0, x_1) are paired. The x_0 ROWS are the "
                        "same in all three cases.")
    p.add_argument("--net", choices=["conditional", "unconditional"],
                   required=True,
                   help="f(x_tau, tau, x_0) vs f(x_tau, tau).")
    p.add_argument("--d", type=int, default=5)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--T", type=int, default=50000,
                   help="Iteration cap; the plateau rule usually stops "
                        "earlier (control 3).")
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--plateau_window", type=int, default=10,
                   help="# recorded evals over which relative loss "
                        "improvement is measured.")
    p.add_argument("--plateau_tol", type=float, default=0.01,
                   help="Stop when that relative improvement < tol.")
    p.add_argument("--safety", type=float, default=0.5,
                   help="eta = safety / lambda_max (in theorem units).")
    p.add_argument("--n_sub", type=int, default=1000,
                   help="Subsample size for the lambda_max estimate.")
    p.add_argument("--n_test", type=int, default=4000,
                   help="Held-out reference: fresh P_1 samples. Never "
                        "given to the sampler (control 1).")
    p.add_argument("--n_proj", type=int, default=128)
    p.add_argument("--T_euler", type=int, default=200)
    p.add_argument("--B1", type=float, default=10.0)
    p.add_argument("--d_exact_max", type=int, default=5,
                   help="Exact W1 (POT LP) only for d <= this.")
    p.add_argument("--n_exact", type=int, default=2000,
                   help="Cap on each side of the exact-W1 LP.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target_center_norm", type=float, default=2.0)
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sampling for the unconditional arm
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_uncond(
    model: UncondTwoLayerReLU,
    *,
    n_eval: int,
    x_0_pool: torch.Tensor,
    B1: float,
    T_euler: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Algorithm 2 for the unconditional parameterisation.

    Mirrors LS.sample_algorithm2_real step for step (same Euler solver,
    same 2*sqrt(d)*B_1 truncation) except that the velocity call drops
    the x_0 argument and the source x_hat_0 is resampled (with
    replacement) from `x_0_pool` — the TRAINING source draws, never the
    evaluation reference (control 1).
    """
    d = x_0_pool.shape[1]
    idx = torch.randint(
        x_0_pool.shape[0], (n_eval,), generator=generator, device=device
    )
    x = x_0_pool[idx].clone()
    dt = 1.0 / T_euler
    for k in range(T_euler):
        tau_batch = torch.full((n_eval,), k * dt, device=device, dtype=dtype)
        x = x + dt * model(x, tau_batch)
    R = 2.0 * math.sqrt(d) * B1
    norms = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    inside = (norms <= R).to(dtype)
    return inside * x + (1.0 - inside) * (x * (R / norms))


# ---------------------------------------------------------------------------
# Training (one code path for both networks — see module docstring)
# ---------------------------------------------------------------------------


def train_arm(
    *,
    model: torch.nn.Module,
    forward: Callable[[], torch.Tensor],
    sample_fn: Callable[[int], torch.Tensor],
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    tau: torch.Tensor,
    x_1_ref: torch.Tensor,
    directions: torch.Tensor,
    n_gen: int,
    eta: float,
    T: int,
    eval_every: int,
    plateau_window: int,
    plateau_tol: float,
) -> Dict[str, object]:
    """
    Full-batch GD with the plateau stopping rule and a sliced-W1 trace.

    `forward` closes over the arm's training tensors and returns the
    (n, d) prediction; `sample_fn(k)` returns k fresh model samples.
    Everything else — the loss, the update, the eval cadence, the
    stopping rule — is shared by both architectures.

    Returns {"t", "loss", "w1", "stopped_at", "converged_ratio",
             "samples"} where "samples" are the generated samples of the
    LAST evaluation (so the exact W1 refers to the same sample set as
    the reported final sliced W1).
    """
    ts: List[int] = []
    losses: List[float] = []
    w1s: List[float] = []
    last_samples = None

    def record(t: int) -> None:
        nonlocal last_samples
        model.eval()
        with torch.no_grad():
            loss_val = loss_utils.empirical_loss(
                forward(), x_0, x_1, tau
            ).item()
        x_hat_1 = sample_fn(n_gen)
        w1_val = metric_utils.sliced_wasserstein_1(
            x_hat_1, x_1_ref, directions=directions,
        ).item()
        ts.append(t)
        losses.append(loss_val)
        w1s.append(w1_val)
        last_samples = x_hat_1
        model.train()

    def rel_improvement() -> float:
        """Relative loss improvement over the last `plateau_window` evals."""
        if len(losses) <= plateau_window:
            return float("nan")
        past = losses[-1 - plateau_window]
        if not math.isfinite(past) or past <= 0.0:
            return float("nan")
        return (past - losses[-1]) / past

    record(0)
    stopped_at = T
    for t in range(1, T + 1):
        model.train()
        if model.W.grad is not None:
            model.W.grad = None
        if model.A.grad is not None:
            model.A.grad = None
        loss = loss_utils.empirical_loss(forward(), x_0, x_1, tau)
        loss.backward()
        with torch.no_grad():
            model.W.add_(model.W.grad, alpha=-eta)
            model.A.add_(model.A.grad, alpha=-eta)
        if t % eval_every == 0:
            record(t)
            if not math.isfinite(losses[-1]):
                # Diverged: stop rather than fill the trace with NaNs.
                stopped_at = t
                break
            ratio = rel_improvement()
            if math.isfinite(ratio) and ratio < plateau_tol:
                stopped_at = t
                break

    return {
        "t": ts, "loss": losses, "w1": w1s,
        "stopped_at": stopped_at,
        "converged_ratio": rel_improvement(),
        "samples": last_samples,
    }


# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    chdir_to_repo()
    device = require_cuda()
    dtype = torch.float32
    d, n = args.d, args.n

    run_name = args.run_name or (
        f"coup_{args.coupling}_{args.net}_d{d}_n{n}_m{args.m}"
        f"_seed{args.seed}"
    )
    out_dir = Path(args.results_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- ONE data draw, shared by all 6 arms of this (d, n, seed) ----
    # g_data depends only on the seed, and x_1 / x_0 / tau are drawn
    # from it BEFORE any arm-specific branching, so the three tensors
    # are bit-identical across couplings and networks.
    g_data = torch.Generator(device=device).manual_seed(
        args.seed * 1_000_003 + 17
    )
    x_1 = data_utils.sample_p1_gmm(
        n, d, center_norm=args.target_center_norm,
        generator=g_data, device=device, dtype=dtype,
    )
    x_0 = data_utils.sample_p_x0_given_x1(
        x_1, generator=g_data, dtype=dtype,
    )
    tau = data_utils.sample_taus(
        n, generator=g_data, device=device, dtype=dtype,
    )

    # ---- the ONLY difference between couplings: the pairing ----
    if args.coupling == "conditional":
        x_0_paired = x_0
    elif args.coupling == "permuted":
        g_pair = torch.Generator(device=device).manual_seed(
            args.seed * 1_000_003 + 23
        )
        x_0_paired = data_utils.permute_pairing(x_0, generator=g_pair)
    else:  # "ot"
        x_0_paired = data_utils.minibatch_ot_pairing(x_0, x_1)
    x_tau = data_utils.interpolate(x_0_paired, x_1, tau)

    # ---- held-out reference + fixed slicing directions ----
    g_ref = torch.Generator(device=device).manual_seed(
        args.seed * 1_000_003 + 3
    )
    x_1_ref = data_utils.sample_p1_gmm(
        args.n_test, d, center_norm=args.target_center_norm,
        generator=g_ref, device=device, dtype=dtype,
    )
    g_proj = torch.Generator(device=device).manual_seed(
        args.seed * 1_000_003 + 4
    )
    directions = metric_utils.random_unit_directions(
        args.n_proj, d, generator=g_proj, device=device, dtype=dtype,
    )

    # ---- control 2: per-arm step size from THIS arm's kernel ----
    # The conditional net sees z = [x_tau; tau; x_0]; the unconditional
    # net sees z = [x_tau; tau]. Passing an empty conditioning block
    # gives ntk.augment exactly the latter, so gram_htilde returns the
    # correct H_tilde for each architecture. x_tau depends on the
    # pairing, so lambda_max (and hence eta) is arm-specific.
    z0 = x_0_paired if args.net == "conditional" else x_0_paired[:, :0]
    g_ntk = torch.Generator(device=device).manual_seed(
        args.seed * 1_000_003 + 5
    )
    lambda_max = ntk.lambda_max_subsampled(
        x_tau, tau, z0, kappa=args.kappa, n_sub=args.n_sub,
        generator=g_ntk,
    )
    eta = ntk.eta_spectral(
        lambda_max, n=n, safety=args.safety, mean_loss=True,
    )

    # ---- build the arm ----
    g_init = torch.Generator(device=device).manual_seed(
        args.seed * 1_000_003 + 1
    )
    g_sample = torch.Generator(device=device).manual_seed(
        args.seed * 1_000_003 + 2
    )
    if args.net == "conditional":
        model = TwoLayerReLU(
            d=d, m=args.m, kappa=args.kappa,
            generator=g_init, device=device, dtype=dtype,
        )

        def forward() -> torch.Tensor:
            return model(x_tau, tau, x_0_paired)

        def sample_fn(k: int) -> torch.Tensor:
            # x_1_pool = TRAINING targets, so x_hat_0 = x_1' + N(0, I)
            # is a draw from the empirical source. The held-out
            # reference is never passed here (control 1).
            return LS.sample_algorithm2_real(
                model, n_eval=k, d=d, B1=args.B1, T_euler=args.T_euler,
                generator=g_sample, device=device, dtype=dtype,
                x_1_pool=x_1,
            )
    else:
        model = UncondTwoLayerReLU(
            d=d, m=args.m, kappa=args.kappa,
            generator=g_init, device=device, dtype=dtype,
        )

        def forward() -> torch.Tensor:
            return model(x_tau, tau)

        def sample_fn(k: int) -> torch.Tensor:
            # x_0_pool = TRAINING source draws (same rows as every
            # other arm), so the source marginal is matched exactly.
            return sample_uncond(
                model, n_eval=k, x_0_pool=x_0, B1=args.B1,
                T_euler=args.T_euler, generator=g_sample,
                device=device, dtype=dtype,
            )

    print(f"[run_coupling] {run_name}: coupling={args.coupling} "
          f"net={args.net} d={d} n={n} m={args.m} "
          f"lambda_max={lambda_max:.4g} eta={eta:.4g} (cap T={args.T})")

    res = train_arm(
        model=model, forward=forward, sample_fn=sample_fn,
        x_0=x_0_paired, x_1=x_1, tau=tau,
        x_1_ref=x_1_ref, directions=directions, n_gen=n,
        eta=eta, T=args.T, eval_every=args.eval_every,
        plateau_window=args.plateau_window,
        plateau_tol=args.plateau_tol,
    )

    # ---- control 4: the two reference lines ----
    # Both use n rows against the same reference with the same
    # directions as the model samples, so all three are comparable.
    donothing_w1 = float(metric_utils.sliced_wasserstein_1(
        x_0, x_1_ref, directions=directions,
    ))
    g_floor = torch.Generator(device=device).manual_seed(
        args.seed * 1_000_003 + 7
    )
    floor_samples = data_utils.sample_p1_gmm(
        n, d, center_norm=args.target_center_norm,
        generator=g_floor, device=device, dtype=dtype,
    )
    floor_w1 = float(metric_utils.sliced_wasserstein_1(
        floor_samples, x_1_ref, directions=directions,
    ))

    # ---- exact W1 in low dimension ----
    # Applied to all three sample sets (model, source, fresh P_1)
    # against the same reference, because sliced and exact W1 are on
    # different scales: an exact model W1 is only readable next to
    # exact baselines.
    def exact_w1_of(samples) -> float:
        if d > args.d_exact_max or samples is None:
            return float("nan")
        k = args.n_exact
        return metric_utils.wasserstein1_exact(
            samples[:k], x_1_ref[:k], max_points=k,
        )

    exact_w1 = exact_w1_of(res["samples"])
    donothing_exact_w1 = exact_w1_of(x_0)
    floor_exact_w1 = exact_w1_of(floor_samples)

    # ---- IO ----
    config = {
        **vars(args), "run_name": run_name,
        "eta": eta, "lambda_max": lambda_max,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    write_csv(out_dir / "trace.csv", "t,loss,w1",
              zip(res["t"], res["loss"], res["w1"]))
    summary = {
        **config,
        "final_w1": res["w1"][-1],
        "exact_w1": exact_w1,
        "donothing_w1": donothing_w1,
        "floor_w1": floor_w1,
        "donothing_exact_w1": donothing_exact_w1,
        "floor_exact_w1": floor_exact_w1,
        "stopped_at": res["stopped_at"],
        "converged_ratio": res["converged_ratio"],
        "final_loss": res["loss"][-1],
    }
    with open(out_dir / "coup_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[run_coupling] wrote {out_dir}/  "
          f"final_w1={summary['final_w1']:.4f} "
          f"exact_w1={exact_w1:.4f} "
          f"donothing_w1={donothing_w1:.4f} (exact {donothing_exact_w1:.4f}) "
          f"floor_w1={floor_w1:.4f} (exact {floor_exact_w1:.4f}) "
          f"eta={eta:.4g} lambda_max={lambda_max:.4g} "
          f"stopped_at={res['stopped_at']} "
          f"converged_ratio={res['converged_ratio']:.4f} "
          f"final_loss={summary['final_loss']:.4f}")


if __name__ == "__main__":
    main()
