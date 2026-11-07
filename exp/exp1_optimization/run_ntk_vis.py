"""
NTK convergence figures (exp1_optimization).

Reads the results/<run>/ntk_trace.csv + ntk_summary.json files written
by run_ntk.py over the grid in scripts/exp1_optimization/_ntk_grid.sh
and produces, under --figs_dir:

  Fig 1a  ntk_rate_d{d}_n{n}.png
           semi-log ||Delta(t)||^2 vs t at the LARGEST width, one color
           per eta_mult; solid = measured, dashed = exact kernel
           prediction, dotted = Theorem 3.2 bound
           (1 - eta lambda_min/2)^t ||Delta(0)||^2.
  Fig 1b  ntk_width_err_d{d}_n{n}.png
           log-log ||E(T)|| vs m (mean +/- sd over seeds), one line per
           eta_mult.
  Fig 1c  ntk_lambda_d{d}_n{n}.png
           lambda_min(empirical NTK at init) vs m with the closed-form
           lambda_min(H^(inf)) asymptote; second panel: relative
           parameter movement vs m.
  Summary table ntk_table.csv
           per (d, n): lambda_min, lambda_max, eta, contraction bound /
           kernel / fitted.

Missing runs are skipped with a warning so partial sweeps still plot.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d_list", type=int, nargs="+", required=True)
    p.add_argument("--n_list", type=int, nargs="+", required=True)
    p.add_argument("--m_list", type=int, nargs="+", required=True)
    p.add_argument("--eta_mult_list", type=float, nargs="+", required=True)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--figs_dir", type=str, default="figs/exp1_optimization")
    return p.parse_args()


def run_name(d, n, m, em, seed):
    return f"ntk_d{d}_n{n}_m{m}_em{em:g}_seed{seed}"


def load_cell(root: Path, name: str):
    tdir = root / name
    if not (tdir / "ntk_trace.csv").exists():
        return None
    with open(tdir / "ntk_summary.json") as f:
        summ = json.load(f)
    rows = np.genfromtxt(tdir / "ntk_trace.csv", delimiter=",", names=True)
    return summ, rows


def main() -> None:
    args = parse_args()
    chdir_to_repo()
    root = Path(args.results_root)
    figs = Path(args.figs_dir)
    figs.mkdir(parents=True, exist_ok=True)
    m_big = max(args.m_list)
    missing = []
    table_rows = []

    for d in args.d_list:
        for n in args.n_list:
            # ---------- Fig 1a: rate overlay at largest width ----------
            fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
            colors = plt.cm.viridis(np.linspace(0.15, 0.85,
                                                len(args.eta_mult_list)))
            plotted = False
            for em, col in zip(args.eta_mult_list, colors):
                traces, summ0, cells_all = [], None, []
                for s in range(args.n_seeds):
                    cell = load_cell(root, run_name(d, n,
                                                    m_big, em, s))
                    if cell is None:
                        missing.append(run_name(d, n,
                                                m_big, em, s))
                        continue
                    summ0 = summ0 or cell[0]
                    cells_all.append(cell)
                    traces.append(cell[1])
                if not traces:
                    continue
                plotted = True
                t = traces[0]["t"]
                dsq = np.mean([tr["delta_sq"] for tr in traces], axis=0)
                ksq = np.mean([tr["kernel_pred_sq"] for tr in traces], axis=0)
                ax.semilogy(t, dsq, color=col, lw=1.6,
                            label=rf"measured, $\eta$={summ0['eta']:.2e}")
                ax.semilogy(t, ksq, color=col, lw=1.2, ls="--",
                            label="kernel pred.")
                bound = dsq[0] * (summ0["contraction_bound"] ** t)
                ax.semilogy(t, bound, color=col, lw=1.0, ls=":",
                            label="Thm 3.2 bound")
            if plotted:
                ax.set_xlabel("GD iteration $t$")
                ax.set_ylabel(r"$\|\Delta(t)\|_2^2$")
                ax.set_title(f"Convergence vs kernel prediction "
                             f"(d={d}, n={n}, m={m_big})")
                ax.legend(fontsize=7)
                out = figs / f"ntk_rate_d{d}_n{n}.png"
                fig.savefig(out, dpi=200)
                print(f"[run_ntk_vis] wrote {out}")
            plt.close(fig)

            # ---------- Fig 1b: ||E(T)|| vs m ----------
            fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
            plotted = False
            for em, col in zip(args.eta_mult_list, colors):
                ms, mus, sds = [], [], []
                for m in args.m_list:
                    vals = []
                    for s in range(args.n_seeds):
                        cell = load_cell(root, run_name(d, n,
                                                        m, em, s))
                        if cell is not None:
                            vals.append(cell[0]["err_norm_final"])
                    if vals:
                        ms.append(m)
                        mus.append(np.mean(vals))
                        sds.append(np.std(vals))
                if ms:
                    plotted = True
                    ax.errorbar(ms, mus, yerr=sds, color=col, marker="o",
                                ms=4, lw=1.4, capsize=2,
                                label=rf"$\eta$-mult {em:g}")
            if plotted:
                ax.set_xscale("log", base=2)
                ax.set_yscale("log")
                ax.set_xlabel("width $m$")
                ax.set_ylabel(r"$\|E(T)\|_2$")
                ax.set_title(f"Finite-width error vs m (d={d}, n={n})")
                ax.legend(fontsize=8)
                out = figs / f"ntk_width_err_d{d}_n{n}.png"
                fig.savefig(out, dpi=200)
                print(f"[run_ntk_vis] wrote {out}")
            plt.close(fig)

            # ---------- Fig 1c: lambda_min(emp) and movement vs m ----------
            em0 = args.eta_mult_list[0]
            ms, lam_emp, wmove = [], [], []
            lam_inf = None
            for m in args.m_list:
                vals_l, vals_w = [], []
                for s in range(args.n_seeds):
                    cell = load_cell(root, run_name(d, n,
                                                    m, em0, s))
                    if cell is None:
                        continue
                    lam_inf = cell[0]["lambda_min_inf"]
                    if cell[0]["lambda_min_emp"] is not None:
                        vals_l.append(cell[0]["lambda_min_emp"])
                    vals_w.append(cell[0]["w_move_final"])
                if vals_w:
                    ms.append(m)
                    lam_emp.append(np.mean(vals_l) if vals_l else np.nan)
                    wmove.append(np.mean(vals_w))
            if ms:
                fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8),
                                         constrained_layout=True)
                axes[0].semilogx(ms, lam_emp, "o-", base=2, color="tab:blue",
                                 label=r"$\lambda_{\min}$(empirical NTK)")
                if lam_inf is not None:
                    axes[0].axhline(lam_inf, color="k", ls="--", lw=1,
                                    label=r"$\lambda_{\min}(H^{(\infty)})$")
                axes[0].set_xlabel("width $m$")
                axes[0].legend(fontsize=8)
                axes[1].loglog(ms, wmove, "o-", color="tab:red")
                axes[1].set_xlabel("width $m$")
                axes[1].set_ylabel(r"$\|W-W_0\|_F / \|W_0\|_F$")
                axes[1].set_title("lazy-training check")
                out = figs / f"ntk_lambda_d{d}_n{n}.png"
                fig.savefig(out, dpi=200)
                print(f"[run_ntk_vis] wrote {out}")
                plt.close(fig)

    # ---------- Summary table: EVERY cell, not just the widest ----------
    # The headline claim ("the Theorem 3.2 inequality holds in N/N runs")
    # has to be auditable from this file, so the table spans the whole
    # (d, n, m, eta_mult) grid rather than the slice that Fig 1a plots.
    for d in args.d_list:
        for n in args.n_list:
            for m in args.m_list:
                for em in args.eta_mult_list:
                    summs = []
                    for s in range(args.n_seeds):
                        cell = load_cell(root, run_name(d, n,
                                                        m, em, s))
                        if cell is not None:
                            summs.append(cell[0])
                    if not summs:
                        continue

                    def _ms(key, ss=summs):
                        vals = [x[key] for x in ss if x.get(key) is not None]
                        return ((float(np.mean(vals)), float(np.std(vals)))
                                if vals else (float("nan"), float("nan")))

                    lmin_mu, lmin_sd = _ms("lambda_min_inf")
                    lmax_mu, _ = _ms("lambda_max_inf")
                    eta_mu, _ = _ms("eta")
                    tr_mu, tr_sd = _ms("thm_ratio_max")
                    n_hold = sum(1 for x in summs if x.get("thm_holds"))
                    n_spec = sum(1 for x in summs
                                 if x.get("satisfies_spectral_condition"))
                    table_rows.append({
                        "d": d, "n": n, "m": m, "eta_mult": em,
                        "n_seeds": len(summs),
                        "lambda_min_mean": lmin_mu, "lambda_min_sd": lmin_sd,
                        "lambda_max_mean": lmax_mu, "eta_mean": eta_mu,
                        "thm_ratio_max_mean": tr_mu,
                        "thm_ratio_max_sd": tr_sd,
                        "thm_holds": f"{n_hold}/{len(summs)}",
                        "eta_le_inv_lambda_max": f"{n_spec}/{len(summs)}",
                        "contraction_bound": summs[0]["contraction_bound"],
                    })

    # ---------- Summary table ----------
    if table_rows:
        out = figs / f"ntk_table.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
            w.writeheader()
            w.writerows(table_rows)
        print(f"[run_ntk_vis] wrote {out} ({len(table_rows)} rows)")
    if missing:
        print(f"[run_ntk_vis] WARNING: {len(missing)} missing cells "
              f"(first: {missing[0]})")


if __name__ == "__main__":
    main()
