"""
Coupling comparison figures (exp2_coupling).

Reads results/coup_{coupling}_{net}_d{d}_n{n}_m{m}_seed{s}/coup_summary.json
over the 2 x 2 (+ OT) grid of run_coupling.py and produces:

  Fig      coupling_w1.png    final sliced W1 (mean +- sd over seeds) as
                             a grouped comparison across the 6 arms,
                             one panel per d. Color encodes the
                             COUPLING, hatch encodes the NETWORK, so
                             the coupling effect (compare bars of
                             different color at the same hatch) and the
                             architecture effect (compare hatches
                             within one color) can be read off
                             separately. Two horizontal reference lines
                             per panel: the do-nothing baseline (raw
                             source draws vs the reference) and the
                             finite-sample floor (n fresh P_1 samples
                             vs the reference).
  Table   coupling_table.csv  every metric, mean and sd over seeds, one
                              row per (d, coupling, net).

The paper's setting is (conditional, conditional); standard
flow-matching practice is (permuted, unconditional). Both are
annotated on the figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import chdir_to_repo  # noqa: E402

# Fixed categorical order, assigned once and never cycled. Okabe-Ito
# blue / orange / bluish-green: distinguishable under all three common
# CVD types as well as in greyscale print.
COUPLING_COLORS = {
    "conditional": "#0072B2",
    "permuted": "#E69F00",
    "ot": "#009E73",
}
# The second factor is encoded WITHOUT color, so identity never rests
# on hue alone.
NET_HATCH = {"conditional": "", "unconditional": "//"}
NET_LABEL = {"conditional": r"cond. net $f(x_\tau,\tau,x_0)$",
             "unconditional": r"uncond. net $f(x_\tau,\tau)$"}

# The two arms the reviewer's question is about.
PAPER_ARM = ("conditional", "conditional")
PRACTICE_ARM = ("permuted", "unconditional")

# Metrics carried into coupling_table.csv (mean + sd over seeds).
METRICS = [
    "final_w1", "exact_w1",
    "donothing_w1", "floor_w1",
    "donothing_exact_w1", "floor_exact_w1",
    "eta", "lambda_max", "stopped_at", "converged_ratio", "final_loss",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--couplings", type=str, nargs="+",
                   default=["conditional", "permuted", "ot"])
    p.add_argument("--nets", type=str, nargs="+",
                   default=["conditional", "unconditional"])
    p.add_argument("--d_list", type=int, nargs="+", default=[3, 5, 10])
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--figs_dir", type=str, default="figs/exp2_coupling")
    return p.parse_args()


def mean_sd(values: List[float]) -> tuple:
    """Mean and (population) sd over seeds; NaN-safe reporting."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return float("nan"), float("nan")
    arr = np.asarray(finite, dtype=float)
    return float(arr.mean()), float(arr.std())


def main() -> None:
    args = parse_args()
    chdir_to_repo()
    root = Path(args.results_root)
    figs = Path(args.figs_dir)
    figs.mkdir(parents=True, exist_ok=True)

    # cells[(d, coupling, net)] = {metric: [per-seed values]}
    cells: Dict[tuple, Dict[str, List[float]]] = {}
    missing: List[str] = []
    for d in args.d_list:
        for coupling in args.couplings:
            for net in args.nets:
                acc: Dict[str, List[float]] = {k: [] for k in METRICS}
                for s in range(args.n_seeds):
                    path = (
                        root
                        / f"coup_{coupling}_{net}_d{d}_n{args.n}"
                          f"_m{args.m}_seed{s}"
                        / "coup_summary.json"
                    )
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    with open(path) as f:
                        row = json.load(f)
                    for k in METRICS:
                        val = row.get(k, float("nan"))
                        acc[k].append(
                            float("nan") if val is None else float(val)
                        )
                cells[(d, coupling, net)] = acc

    # ------------------------------------------------------------------
    # One panel per d, grouped bars
    # ------------------------------------------------------------------
    n_panels = len(args.d_list)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.9 * n_panels, 4.3),
        constrained_layout=True, squeeze=False, sharey=False,
    )
    n_nets = len(args.nets)
    bar_w = 0.8 / n_nets
    any_plotted = False

    for ax, d in zip(axes[0], args.d_list):
        centers = np.arange(len(args.couplings), dtype=float)
        heights_seen: List[float] = []
        for j, net in enumerate(args.nets):
            offs = (j - (n_nets - 1) / 2.0) * bar_w
            for i, coupling in enumerate(args.couplings):
                acc = cells[(d, coupling, net)]
                mu, sd = mean_sd(acc["final_w1"])
                if not math.isfinite(mu):
                    continue
                any_plotted = True
                heights_seen.append(mu + (sd if math.isfinite(sd) else 0.0))
                ax.bar(
                    centers[i] + offs, mu, width=bar_w * 0.9,
                    yerr=(sd if math.isfinite(sd) else None),
                    color=COUPLING_COLORS.get(coupling, "#666666"),
                    hatch=NET_HATCH.get(net, ""),
                    edgecolor="white", linewidth=1.0,
                    error_kw={"ecolor": "#333333", "capsize": 3,
                              "lw": 1.0},
                    zorder=3,
                )
                # Value label sits clear of the error-bar cap.
                top = mu + (sd if math.isfinite(sd) else 0.0)
                ax.annotate(
                    f"{mu:.3f}",
                    (centers[i] + offs, top),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontsize=7.5,
                    color="#333333", zorder=4,
                )
                tag = None
                if (coupling, net) == PAPER_ARM:
                    tag = "paper"
                elif (coupling, net) == PRACTICE_ARM:
                    tag = "practice"
                if tag is not None:
                    # Rotated so the tag fits inside a narrow bar.
                    ax.annotate(
                        tag, (centers[i] + offs, 0.0),
                        textcoords="offset points", xytext=(0, 5),
                        ha="center", va="bottom", fontsize=7.5,
                        rotation=90, style="italic", color="white",
                        zorder=5,
                    )

        # Reference lines: shared by construction across arms (one data
        # draw per (d, seed)), so pool every cell of this panel.
        dn_mu, _ = mean_sd(
            [v for c in args.couplings for nt in args.nets
             for v in cells[(d, c, nt)]["donothing_w1"]]
        )
        fl_mu, _ = mean_sd(
            [v for c in args.couplings for nt in args.nets
             for v in cells[(d, c, nt)]["floor_w1"]]
        )
        if math.isfinite(dn_mu):
            ax.axhline(dn_mu, color="#555555", ls="--", lw=1.2, zorder=2,
                       label="do-nothing (source vs ref.)")
            heights_seen.append(dn_mu)
        if math.isfinite(fl_mu):
            ax.axhline(fl_mu, color="#555555", ls=":", lw=1.2, zorder=2,
                       label=f"finite-sample floor ($n={args.n}$)")
            heights_seen.append(fl_mu)

        ax.set_xticks(centers)
        ax.set_xticklabels(args.couplings)
        ax.set_xlabel("coupling")
        ax.set_ylabel(r"final sliced $W_1$")
        ax.set_title(f"$d = {d}$")
        ax.grid(True, axis="y", alpha=0.25, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        if heights_seen:
            ax.set_ylim(0.0, 1.18 * max(heights_seen))
        else:
            # Labelled placeholder so a gap in the grid is visible by
            # inspection rather than as a silently blank panel.
            ax.text(0.5, 0.5, "no runs for this $d$", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9,
                    color="#777777")

    if any_plotted:
        handles = [
            Patch(facecolor="#BBBBBB", edgecolor="white",
                  hatch=NET_HATCH[net], label=NET_LABEL.get(net, net))
            for net in args.nets
        ]
        handles += [
            Patch(facecolor=COUPLING_COLORS.get(c, "#666666"),
                  edgecolor="white", label=f"coupling: {c}")
            for c in args.couplings
        ]
        h_ref, l_ref = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles=handles + list(h_ref),
            loc="outside lower center", ncol=4, fontsize=8,
            frameon=True, framealpha=0.9,
        )
        out = figs / "coupling_w1.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[run_coupling_vis] wrote {out}")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    table_rows = []
    for d in args.d_list:
        for coupling in args.couplings:
            for net in args.nets:
                acc = cells[(d, coupling, net)]
                n_ok = len([v for v in acc["final_w1"] if math.isfinite(v)])
                row = {"d": d, "coupling": coupling, "net": net,
                       "n": args.n, "m": args.m, "n_seeds": n_ok}
                if (coupling, net) == PAPER_ARM:
                    row["arm"] = "paper"
                elif (coupling, net) == PRACTICE_ARM:
                    row["arm"] = "practice"
                else:
                    row["arm"] = ""
                for k in METRICS:
                    mu, sd = mean_sd(acc[k])
                    row[f"{k}_mean"] = mu
                    row[f"{k}_sd"] = sd
                table_rows.append(row)

    if table_rows:
        out = figs / "coupling_table.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
            w.writeheader()
            w.writerows(table_rows)
        print(f"[run_coupling_vis] wrote {out} ({len(table_rows)} rows)")

    if missing:
        print(f"[run_coupling_vis] WARNING: {len(missing)} missing cells "
              f"(first: {missing[0]})")


if __name__ == "__main__":
    main()
