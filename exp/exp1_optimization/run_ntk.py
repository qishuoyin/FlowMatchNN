"""
NTK convergence validation (exp1_optimization). Answers R1's major
weakness and R1 Q2.

One cell = one (d, n, m, eta_mult, seed). Pipeline:

  1. Draw the training set (Algorithm 1's law, linear interpolant).
  2. Compute the closed-form H_tilde (H^(inf) = H_tilde (x) I_d), its
     spectrum, and lambda_min via `flowmatchnn.utils.ntk`.
  3. Pick kappa and the GD step size eta (see below), train with
     per-iteration diagnostics (train_ntk_trace), and record the exact
     kernel prediction ||(I - eta H)^t V|| alongside the measured
     residual.
  4. Optionally compute the finite-m empirical NTK at initialization
     (small cells only).

Step-size modes
    --eta_mode spectral (default): eta = eta_mult / lambda_max(H_tilde).
        The theorem's admissible step eta < 2 lambda_min / (n^2 d^2) is
        far too small to visualize convergence in finite T (the
        per-step contraction 1 - eta lambda_min / 2 would be
        indistinguishable from 1). The spectral scale keeps GD in the
        provably convergent kernel regime while making the contraction
        visible; the theoretical bound for the SAME eta is overlaid by
        the vis script, so Theorem 3.2's guarantee is still what the
        figure tests.
    --eta_mode theory: eta = eta_mult * 2 lambda_min / (n^2 d^2),
        the literal theorem condition (kept for reference runs).

kappa: the theorem wants kappa = o(lambda_min delta / (n sqrt(d))).
    Since B_tilde ∝ kappa^2 while C_tilde is kappa-free and PSD,
    lambda_min(C_tilde) <= lambda_min(H_tilde) for any kappa; we set
    kappa = kappa_scale * lambda_min(C_tilde) / (n sqrt(d)) which
    satisfies the condition without the circular dependence on H(kappa).

Outputs (results/<run_name>/):
    config.json       resolved hyperparameters + spectral summary
    ntk_trace.csv     t, delta_sq, kernel_pred_sq, err_norm, w_move, a_move
    ntk_summary.json  lambda_min_inf, lambda_max_inf, lambda_min_emp,
                      kappa, eta, eta_theory_bound, contraction_bound,
                      contraction_kernel, contraction_fit
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

# The shared helpers (_common.py, ...) live one level up in exp/.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import chdir_to_repo, require_cuda, write_csv  # noqa: E402

from flowmatchnn import learning_and_sampling as LS  # noqa: E402
from flowmatchnn.model import TwoLayerReLU  # noqa: E402
from flowmatchnn.utils import data as data_utils  # noqa: E402
from flowmatchnn.utils import ntk  # noqa: E402
from flowmatchnn.utils import velocity as velocity_utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--eta_mult", type=float, default=1.0)
    p.add_argument("--eta_mode", choices=["spectral", "theory"],
                   default="spectral")
    p.add_argument("--kappa_scale", type=float, default=1.0)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target_center_norm", type=float, default=2.0)
    p.add_argument("--emp_ntk", choices=["auto", "on", "off"],
                   default="auto",
                   help="Finite-m empirical NTK at init. 'auto' runs it "
                        "when n*d <= 4096.")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    chdir_to_repo()
    device = require_cuda()
    dtype = torch.float32

    mode_tag = "" if args.eta_mode == "spectral" else "_theory"
    run_name = args.run_name or (
        f"ntk_d{args.d}_n{args.n}_m{args.m}"
        f"_em{args.eta_mult:g}{mode_tag}_seed{args.seed}"
    )
    out_dir = Path(args.results_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. data (Algorithm 1) ----
    g_data = torch.Generator(device=device).manual_seed(
        args.seed * 1_000_003 + 17
    )
    x_1 = data_utils.sample_p1_gmm(
        args.n, args.d, center_norm=args.target_center_norm,
        generator=g_data, device=device, dtype=dtype,
    )
    tau, x_tau, x_0 = data_utils.make_dataset(
        x_1, generator=g_data, dtype=dtype,
    )
    dataset = {"tau": tau, "x_tau": x_tau, "x_0": x_0}
    targets = velocity_utils.velocity_on_path(x_0, x_1, tau)  # V, (n, d)

    # ---- 2. closed-form kernel + kappa + eta ----
    # C_tilde = H_tilde at kappa = 0 (B vanishes); kappa-free PSD lower
    # block used to set kappa without circularity.
    C = ntk.gram_htilde(x_tau, tau, x_0, kappa=0.0)
    lam_min_C = ntk.lambda_min_htilde(C)
    # Guard against a numerically zero lambda_min(C): kappa must stay
    # strictly positive or the ReLU units initialize dead.
    kappa = max(
        args.kappa_scale * lam_min_C / (args.n * math.sqrt(args.d)),
        1e-8,
    )

    H = ntk.gram_htilde(x_tau, tau, x_0, kappa=kappa)
    evals, U = ntk.eig_htilde(H)
    lam_min = float(evals[0])
    lam_max = float(evals[-1])
    eta_theory_bound = 2.0 * lam_min / (args.n ** 2 * args.d ** 2)
    if args.eta_mode == "spectral":
        eta = args.eta_mult / lam_max
    else:
        eta = args.eta_mult * eta_theory_bound

    print(f"[run_ntk] {run_name}: lambda_min={lam_min:.4e} "
          f"lambda_max={lam_max:.4e} kappa={kappa:.3e} eta={eta:.3e} "
          f"(theory bound {eta_theory_bound:.3e})")

    # ---- 3. train with diagnostics ----
    g_init = torch.Generator(device=device).manual_seed(
        args.seed * 1_000_003 + 1
    )
    model = TwoLayerReLU(
        d=args.d, m=args.m, kappa=kappa,
        generator=g_init, device=device, dtype=dtype,
    )

    # Empirical NTK BEFORE training (t = 0 parameters).
    lam_min_emp = None
    want_emp = args.emp_ntk == "on" or (
        args.emp_ntk == "auto" and args.n * args.d <= 4096
    )
    if want_emp:
        lam_min_emp = ntk.lambda_min_empirical(model, x_tau, tau, x_0)
        print(f"[run_ntk] empirical NTK lambda_min = {lam_min_emp:.4e}")

    step = ntk.kernel_residual_stepper(evals, U, targets.double(), eta)
    trace = LS.train_ntk_trace(
        model=model, dataset=dataset, targets=targets,
        eta=eta, T=args.T, kernel_step=step,
    )

    # ---- 4. summaries ----
    # Fitted per-step contraction of delta_sq from the last half of the
    # trace (geometric fit via the endpoints of a log-linear regression).
    # Theorem 3.2's actual guarantee (as derived in the proof) is
    #     ||Delta(t)||^2 <= (1 - eta*lambda_min/2)^t ||Delta(0)||^2 .
    # The right statistic is therefore the WORST-CASE ratio of the
    # measured residual to that bound over the whole trajectory: the
    # theorem's conclusion holds for this run iff thm_ratio_max <= 1.
    # (An earlier version fitted a geometric decay rate and compared
    # rates; that tests a different, weaker statement.)
    rho = 1.0 - eta * lam_min / 2.0
    d0 = trace["delta_sq"][0]
    thm_ratio_max = 0.0
    if d0 > 0 and rho > 0:
        # Skip t = 0, where the bound is trivially tight (ratio == 1)
        # and would mask the informative part of the trajectory.
        for t_i, dsq in zip(trace["t"][1:], trace["delta_sq"][1:]):
            bound = d0 * (rho ** t_i)
            if bound > 0:
                thm_ratio_max = max(thm_ratio_max, dsq / bound)

    half = len(trace["t"]) // 2
    ts = torch.tensor(trace["t"][half:], dtype=torch.float64)
    ys = torch.tensor(trace["delta_sq"][half:], dtype=torch.float64)
    contraction_fit = None
    if (ys > 0).all() and len(ys) > 2:
        logy = torch.log(ys)
        tbar, lbar = ts.mean(), logy.mean()
        slope = ((ts - tbar) * (logy - lbar)).sum() / ((ts - tbar) ** 2).sum()
        contraction_fit = float(torch.exp(slope))

    summary = {
        "run_name": run_name,
        "d": args.d, "n": args.n, "m": args.m, "seed": args.seed,
        "kappa": kappa, "kappa_scale": args.kappa_scale,
        "eta": eta, "eta_mode": args.eta_mode, "eta_mult": args.eta_mult,
        "eta_theory_bound": eta_theory_bound,
        "lambda_min_inf": lam_min, "lambda_max_inf": lam_max,
        "lambda_min_C": lam_min_C,
        "lambda_min_emp": lam_min_emp,
        # Theorem 3.2 bound: ||Delta(t)||^2 <= (1 - eta lam_min / 2)^t ...
        "contraction_bound": 1.0 - eta * lam_min / 2.0,
        # exact-kernel slow mode: ||Delta|| ~ (1 - eta lam_min)^t
        "contraction_kernel": (1.0 - eta * lam_min) ** 2,
        "contraction_fit": contraction_fit,
        # <= 1 iff the theorem's inequality held at EVERY iteration.
        "thm_ratio_max": thm_ratio_max,
        "thm_holds": bool(thm_ratio_max <= 1.0),
        # Sharp form of the theorem's step-size condition: recombining
        # the T1/T2 terms of the proof gives the same contraction factor
        # for any eta <= 1/lambda_max, which is what we actually use.
        "eta_spectral_bound": 1.0 / lam_max,
        "satisfies_spectral_condition": bool(eta <= 1.0 / lam_max),
        "T": args.T,
        "delta_sq_init": trace["delta_sq"][0],
        "delta_sq_final": trace["delta_sq"][-1],
        "err_norm_final": trace["err_norm"][-1],
        "w_move_final": trace["w_move"][-1],
        "a_move_final": trace["a_move"][-1],
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "ntk_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    write_csv(
        out_dir / "ntk_trace.csv",
        "t,delta_sq,kernel_pred_sq,err_norm,w_move,a_move",
        zip(trace["t"], trace["delta_sq"], trace["kernel_pred_sq"],
            trace["err_norm"], trace["w_move"], trace["a_move"]),
    )
    print(f"[run_ntk] wrote {out_dir}/ntk_trace.csv "
          f"(delta_sq {trace['delta_sq'][0]:.3e} -> "
          f"{trace['delta_sq'][-1]:.3e}, "
          f"err_norm(T) = {trace['err_norm'][-1]:.3e}, "
          f"thm_ratio_max = {thm_ratio_max:.3f})")


if __name__ == "__main__":
    main()
