"""
Combined visualisation (real data): overlay all (m, eta) curves on one plot.

For a given dataset, reads every (m, eta, n_train, n_test) cell's CSV and
produces two summary figures in figs/application/<dataset>/:

  loss_vs_t.png   — training loss L(θ^(t)) vs t, one curve per (m, η).
  w1_vs_t.png     — sliced W_1 vs t, one curve per (m, η).

Color encodes η (viridis, light→dark for large→small η).
Linestyle + marker encode m.

Usage:
    python exp/run_real_vis_combined.py --dataset mnist
    bash scripts/run_real_vis_combined_mnist.sh
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# The shared helpers (_common.py, _vis_common.py, _real_vis_common.py) live
# one level up in exp/. Entry points are executed as plain files, so only
# this subfolder lands on sys.path; add exp/ explicitly before importing.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import chdir_to_repo
from _real_vis_common import app_figs_subdir, plot_combined_traces, read_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset",      type=str, required=True,
                   help="Dataset name (e.g. mnist, fashion_mnist).")
    p.add_argument("--m_list",       type=str, default="128,256,512,1024",
                   help="Comma-separated m values (default: 128,256,512,1024).")
    p.add_argument("--eta_list",     type=str, default="1e-4,1e-3,1e-2",
                   help="Comma-separated eta strings (default: 1e-4,1e-3,1e-2).")
    p.add_argument("--n_train",      type=int, default=500)
    p.add_argument("--n_test",       type=int, default=2000)
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--figs_dir",     type=str, default="figs")
    return p.parse_args()


def _load_trace(csv_path: Path):
    """Return (ts, vals) or (None, None) if missing / non-finite."""
    if not csv_path.exists():
        return None, None
    data = read_csv(csv_path)
    cols = list(data.keys())
    ts, vals = data[cols[0]], data[cols[1]]
    if not ts or any(not math.isfinite(v) for v in vals):
        return None, None
    return ts, vals


def main() -> None:
    args = parse_args()
    chdir_to_repo()

    m_list       = [int(x) for x in args.m_list.split(",") if x.strip()]
    eta_str_list = [x.strip() for x in args.eta_list.split(",") if x.strip()]
    results_root = Path(args.results_root)
    figs_dir     = Path(args.figs_dir)

    loss_traces = []
    w1_traces   = []

    for m in m_list:
        for eta_str in eta_str_list:
            run_name = (f"real_{args.dataset}_m{m}_eta{eta_str}"
                        f"_ntr{args.n_train}_nte{args.n_test}")
            run_dir = results_root / run_name

            ts_l, vals_l = _load_trace(run_dir / "exp1_trace.csv")
            if ts_l is not None:
                loss_traces.append((m, eta_str, ts_l, vals_l))
            else:
                print(f"[run_real_vis_combined] missing/diverged: {run_name}/exp1_trace.csv")

            ts_w, vals_w = _load_trace(run_dir / "exp2_trace.csv")
            if ts_w is not None:
                w1_traces.append((m, eta_str, ts_w, vals_w))
            else:
                print(f"[run_real_vis_combined] missing/diverged: {run_name}/exp2_trace.csv")

    dataset_figs = figs_dir / "application" / args.dataset
    dataset_figs.mkdir(parents=True, exist_ok=True)

    loss_out = dataset_figs / "loss_vs_t.png"
    plot_combined_traces(
        traces=loss_traces,
        m_list=m_list,
        eta_str_list=eta_str_list,
        out=loss_out,
        ylabel=r"training loss $L(\theta^{(t)})$",
        title=f"Loss vs $t$ — {args.dataset}",
    )
    print(f"[run_real_vis_combined] wrote {loss_out}")

    w1_out = dataset_figs / "w1_vs_t.png"
    plot_combined_traces(
        traces=w1_traces,
        m_list=m_list,
        eta_str_list=eta_str_list,
        out=w1_out,
        ylabel=r"sliced $W_1$",
        title=f"$W_1$ vs $t$ — {args.dataset}",
    )
    print(f"[run_real_vis_combined] wrote {w1_out}")


if __name__ == "__main__":
    main()
