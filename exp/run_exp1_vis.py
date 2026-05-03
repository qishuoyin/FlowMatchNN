"""
Exp 1 visualisation. Two modes selected by --mode:

  --mode setting (default)  loss vs t for one (m, eta_str, n_train,
                            n_test) setting; every d in --d_list is
                            plotted as a separate curve, colored by d.
                            -> figs/loss_vs_t__<setting>.png

  --mode global             tail-mean(loss) vs d for one
                            (n_train, n_test) pair, with every (m, eta)
                            in (--m_list, --eta_str_list) overlaid.
                            -> figs/loss_vs_d__ntr<ntr>_nte<nte>.png

Both modes read exp1_trace.csv files written by exp/run_exp1.py.
"""

from __future__ import annotations

from pathlib import Path

from _common import chdir_to_repo, parse_args
from _vis_common import (
    collect_global_curves,
    collect_setting_traces,
    parse_d_list,
    parse_int_list,
    parse_str_list,
    plot_global_d,
    plot_setting_curves,
    setting_name,
)


CSV_FILENAME = "exp1_trace.csv"
YLABEL_T = r"training loss  $L(\theta^{(t)})$"
YLABEL_D = r"training loss  $\overline{L(\theta^{(t)})}$"


# ---------------------------------------------------------------------------
# Mode: setting (one PNG per (m, eta, n_train, n_test); 10 d-curves)
# ---------------------------------------------------------------------------


def _main_setting(args) -> None:
    """
    Render figs/loss_vs_t__<setting>.png by overlaying one curve per
    d in --d_list; colour encodes d via the viridis colormap.
    """
    if args.eta_str is None:
        raise SystemExit(
            "[run_exp1_vis] --mode setting requires --eta_str "
            "(the literal eta token used by scripts/_grid.sh, "
            "e.g. '1e-3'; not the float '--eta')."
        )
    if args.d_list is None:
        raise SystemExit(
            "[run_exp1_vis] --mode setting requires --d_list "
            "(comma-separated, e.g. '5,10,...,50'; the shell driver "
            "passes scripts/_grid.sh's D_LIST)."
        )
    d_list = parse_d_list(args.d_list)

    figs_dir = Path(args.figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_root)

    setting = setting_name(
        args.m, args.eta_str, args.n_train, args.n_test,
    )
    out = figs_dir / f"loss_vs_t__{setting}.png"

    traces, missing = collect_setting_traces(
        m=args.m, eta_str=args.eta_str,
        n_train=args.n_train, n_test=args.n_test,
        d_list=d_list,
        results_root=results_root,
        csv_filename=CSV_FILENAME,
    )

    plot_setting_curves(
        traces=traces,
        d_range=(min(d_list), max(d_list)),
        out=out,
        ylabel=YLABEL_T,
        yscale="log",
        colorbar_ticks=[d_list[0], d_list[len(d_list) // 2], d_list[-1]],
    )
    print(
        f"[run_exp1_vis] wrote {out}  "
        f"(setting={setting}, ok={len(traces)}/{len(d_list)}, "
        f"missing={missing if missing else '[]'})"
    )


# ---------------------------------------------------------------------------
# Mode: global (single PNG per (n_train, n_test); 15 (m, eta) curves)
# ---------------------------------------------------------------------------


def _main_global(args) -> None:
    """
    Render figs/loss_vs_d__ntr<ntr>_nte<nte>.png by overlaying one
    curve per (m, eta) in (--m_list x --eta_str_list); each curve's
    y-value is the tail mean of its cell trace.
    """
    if args.m_list is None or args.eta_str_list is None:
        raise SystemExit(
            "[run_exp1_vis] --mode global requires --m_list and "
            "--eta_str_list (comma-separated, matching M_LIST and "
            "ETA_LIST in scripts/_grid.sh)."
        )
    if args.d_list is None:
        raise SystemExit(
            "[run_exp1_vis] --mode global requires --d_list "
            "(comma-separated, e.g. '5,10,...,50'; the shell driver "
            "passes scripts/_grid.sh's D_LIST)."
        )
    m_list = parse_int_list(args.m_list)
    eta_str_list = parse_str_list(args.eta_str_list)
    d_list = parse_d_list(args.d_list)

    figs_dir = Path(args.figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_root)

    out = figs_dir / (
        f"loss_vs_d__ntr{args.n_train}_nte{args.n_test}.png"
    )

    curves, empty = collect_global_curves(
        m_list=m_list, eta_str_list=eta_str_list,
        n_train=args.n_train, n_test=args.n_test,
        d_list=d_list, tail=args.tail,
        results_root=results_root,
        csv_filename=CSV_FILENAME,
    )
    n_total = len(m_list) * len(eta_str_list)

    plot_global_d(
        curves=curves,
        m_list=m_list, eta_str_list=eta_str_list,
        out=out,
        ylabel=YLABEL_D,
        yscale="log",
    )
    print(
        f"[run_exp1_vis] wrote {out}  "
        f"(ntr={args.n_train}, nte={args.n_test}, "
        f"ok={len(curves)}/{n_total}, "
        f"empty={empty if empty else '[]'})"
    )


def main() -> None:
    args = parse_args()
    chdir_to_repo()
    if args.mode == "setting":
        _main_setting(args)
    elif args.mode == "global":
        _main_global(args)
    else:
        raise SystemExit(f"unknown --mode: {args.mode!r}")


if __name__ == "__main__":
    main()
