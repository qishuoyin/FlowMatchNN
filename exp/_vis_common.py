"""
Helpers shared between exp/run_exp1_vis.py and exp/run_exp2_vis.py.

Both scripts now operate in two modes:

  --mode setting  per-setting trace plot. One PNG per
                  (m, eta_str, n_train, n_test); on each PNG, every d
                  in --d_list is overlaid as a separate curve, colored
                  by d (viridis). Y-axis: training loss / sliced W_1.

  --mode global   one PNG per (n_train, n_test); every (m, eta) pair
                  in (--m_list, --eta_str_list) is overlaid as a
                  separate curve. X-axis: d. Y-axis: tail-mean of
                  training loss / sliced W_1 over the last --tail evals.

The two modes share CSV / naming utilities (`read_trace`, `tail_mean`,
`cell_run_name`, `setting_name`) and a `collect_tail_means` helper.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------------
# Global matplotlib style. Applied once on import so every figure inherits
# the same font sizes, line widths, and grid behaviour without each plot
# function having to re-set the rcParams.
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "#888888",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.0,
})


# ---------------------------------------------------------------------------
# CSV / trace helpers
# ---------------------------------------------------------------------------


def read_trace(path: Path) -> Tuple[List[int], List[float]]:
    """
    Parse a 2-column CSV produced by exp/_common.write_csv.

    Returns (xs, ys); used for both exp1_trace.csv (t, loss) and
    exp2_trace.csv (t, w1).
    """
    ts, vals = [], []
    with open(path) as f:
        next(f)
        for line in f:
            t_str, val_str = line.strip().split(",")
            ts.append(int(t_str))
            vals.append(float(val_str))
    return ts, vals


def tail_mean(values: List[float], tail: int) -> float:
    """
    Mean of the last `tail` entries; returns NaN if any are non-finite
    or the trace has fewer than `tail` entries.

    Diverged runs (NaN tail) therefore propagate as NaN rather than
    silently averaging to a misleading finite number.
    """
    if len(values) < tail:
        return float("nan")
    last = values[-tail:]
    if any(not math.isfinite(v) for v in last):
        return float("nan")
    return sum(last) / tail


# ---------------------------------------------------------------------------
# Naming (must match scripts/_grid.sh byte-for-byte; see comment in
# exp/run_exp1_vis.py for the eta-string rationale)
# ---------------------------------------------------------------------------


def cell_run_name(
    m: int, eta_str: str, d: int, n_train: int, n_test: int,
) -> str:
    """
    Cell directory name `m{m}_eta{eta}_d{d}_ntr{ntr}_nte{nte}`, matching
    scripts/_grid.sh's RUN_NAME byte-for-byte. Used to look up
    results/<cell>/exp{1,2}_trace.csv from setting / global mode.
    """
    return f"m{m}_eta{eta_str}_d{d}_ntr{n_train}_nte{n_test}"


def setting_name(
    m: int, eta_str: str, n_train: int, n_test: int,
) -> str:
    """
    Setting directory-key (no d), matching scripts/_grid.sh's
    SETTING_NAME byte-for-byte.
    """
    return f"m{m}_eta{eta_str}_ntr{n_train}_nte{n_test}"


def parse_d_list(spec: str) -> List[int]:
    """
    Comma-separated string -> list of int. Used as argparse `type` for
    the --d_list flag.
    """
    return [int(x) for x in spec.split(",") if x.strip()]


def parse_int_list(spec: str) -> List[int]:
    """Comma-separated string -> list of int (alias of parse_d_list)."""
    return [int(x) for x in spec.split(",") if x.strip()]


def parse_str_list(spec: str) -> List[str]:
    """
    Comma-separated string -> list of str. Used for --eta_str_list so
    we keep the literal eta tokens ("1e-3") rather than reformatting
    via Python.
    """
    return [x.strip() for x in spec.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Aggregation helpers (read many cell CSVs, compute summaries)
# ---------------------------------------------------------------------------


def collect_tail_means(
    *,
    m: int,
    eta_str: str,
    n_train: int,
    n_test: int,
    d_list: List[int],
    tail: int,
    results_root: Path,
    csv_filename: str,
) -> Tuple[List[int], List[float], List[int]]:
    """
    For each d in d_list, read results/<cell>/<csv_filename> and compute
    tail_mean of column 1.

    Returns (ds, ys, missing) where ds, ys are parallel lists for
    surviving cells and missing is the list of d's whose CSV was absent
    or had a non-finite tail (so the caller can surface them in the
    title or print).
    """
    ds: List[int] = []
    ys: List[float] = []
    missing: List[int] = []
    for d in d_list:
        run = cell_run_name(m, eta_str, d, n_train, n_test)
        csv_path = results_root / run / csv_filename
        if not csv_path.exists():
            missing.append(d)
            continue
        _, values = read_trace(csv_path)
        m_ = tail_mean(values, tail)
        if not math.isfinite(m_):
            missing.append(d)
            continue
        ds.append(d)
        ys.append(m_)
    return ds, ys, missing


# ---------------------------------------------------------------------------
# Mode "setting" : trace plot, all d on one figure
# ---------------------------------------------------------------------------


def collect_setting_traces(
    *,
    m: int,
    eta_str: str,
    n_train: int,
    n_test: int,
    d_list: List[int],
    results_root: Path,
    csv_filename: str,
) -> Tuple[List[Tuple[int, List[int], List[float]]], List[int]]:
    """
    Read every d's trace CSV under results_root.

    Returns (traces, missing) where traces is a list of
    (d, ts, vals) for d's whose CSV exists and whose values are all
    finite, and missing is the parallel list of skipped d's.
    """
    traces: List[Tuple[int, List[int], List[float]]] = []
    missing: List[int] = []
    for d in d_list:
        run = cell_run_name(m, eta_str, d, n_train, n_test)
        csv_path = results_root / run / csv_filename
        if not csv_path.exists():
            missing.append(d)
            continue
        ts, vals = read_trace(csv_path)
        if not ts or any(not math.isfinite(v) for v in vals):
            missing.append(d)
            continue
        traces.append((d, ts, vals))
    return traces, missing


def plot_setting_curves(
    *,
    traces: List[Tuple[int, List[int], List[float]]],
    d_range: Tuple[int, int],
    out: Path,
    ylabel: str,
    yscale: str,
    cmap_name: str = "viridis",
    colorbar_ticks: List[int] | None = None,
) -> None:
    """
    Render one PNG with one curve per (d, ts, vals) entry in `traces`,
    colored by d (`d_range = (d_min, d_max)` defines the colormap
    normalization).

    If `traces` is empty a labelled placeholder is written so the gap
    is visible by inspection rather than by an absent file.
    """
    cmap = cm.get_cmap(cmap_name)
    d_min, d_max = d_range
    norm = Normalize(vmin=d_min, vmax=d_max)

    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)

    for d, ts, vals in traces:
        ax.plot(
            ts, vals,
            color=cmap(norm(d)),
            lw=1.0,
            alpha=0.9,
            solid_capstyle="round",
        )

    ax.set_xlabel(r"GD iteration $t$")
    ax.set_ylabel(ylabel)
    if yscale == "log" and traces:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)

    if not traces:
        ax.text(
            0.5, 0.5, "no finite trace for any d",
            ha="center", va="center", transform=ax.transAxes,
        )
    else:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.015, aspect=28)
        cbar.set_label(r"ambient dimension $d$")
        if colorbar_ticks:
            cbar.set_ticks(colorbar_ticks)

    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mode "global" : single (n_train, n_test) plot, all (m, eta) overlaid
# ---------------------------------------------------------------------------


# Stable per-axis encodings; chosen so that nearby values get
# nearby visual properties (warm to cool eta, simple to fancy m).
# The lists must be at least as long as `m_list` so that no two m's
# share the same (linestyle, marker) pair -- otherwise curves with
# the same eta but different m would render with identical legend
# handles. The current grid has |M_LIST|=5, so 5 distinct entries.
_M_LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))]
_M_MARKERS = ["o", "s", "^", "D", "v"]


def _eta_palette(n: int):
    """
    Pick `n` distinct colors for eta. Uses a perceptually-uniform
    slice of viridis so going from large to small eta has a readable
    light->dark progression.
    """
    cmap = cm.get_cmap("viridis")
    if n == 1:
        return [cmap(0.5)]
    return [cmap(0.15 + 0.7 * i / (n - 1)) for i in range(n)]


def collect_global_curves(
    *,
    m_list: List[int],
    eta_str_list: List[str],
    n_train: int,
    n_test: int,
    d_list: List[int],
    tail: int,
    results_root: Path,
    csv_filename: str,
) -> Tuple[
    List[Tuple[int, str, List[int], List[float]]],
    List[Tuple[int, str]],
]:
    """
    Read every (m, eta_str) cell's tail mean across d_list.

    Returns (curves, empty) where curves is a list of
    (m, eta_str, ds, ys) for pairs that had at least one finite
    tail-mean, and empty is the list of (m, eta_str) pairs that were
    completely missing / diverged.
    """
    curves: List[Tuple[int, str, List[int], List[float]]] = []
    empty: List[Tuple[int, str]] = []
    for m in m_list:
        for eta_str in eta_str_list:
            ds, ys, _ = collect_tail_means(
                m=m, eta_str=eta_str,
                n_train=n_train, n_test=n_test,
                d_list=d_list, tail=tail,
                results_root=results_root,
                csv_filename=csv_filename,
            )
            if not ds:
                empty.append((m, eta_str))
                continue
            curves.append((m, eta_str, ds, ys))
    return curves, empty


def plot_global_d(
    *,
    curves: List[Tuple[int, str, List[int], List[float]]],
    m_list: List[int],
    eta_str_list: List[str],
    out: Path,
    ylabel: str,
    yscale: str,
) -> None:
    """
    Render one PNG with one curve per entry in `curves`. X-axis is d.

    Encoding (so every (m, eta) curve gets a unique legend handle):
      color     <- eta_str (light->dark for large->small eta)
      marker    <- m       (one distinct marker per m)
      linestyle <- m       (one distinct linestyle per m)

    `m_list` and `eta_str_list` define the palette/style mapping so
    runs with the same canonical grid produce visually identical
    encodings even if some curves are missing. We require
    len(m_list) <= len(_M_LINESTYLES) (and same for markers) so no
    two m's collapse to the same legend handle; the lists are sized
    for the project's |M_LIST|=5 grid.
    """
    if len(m_list) > len(_M_LINESTYLES) \
            or len(m_list) > len(_M_MARKERS):
        raise ValueError(
            f"plot_global_d: m_list has {len(m_list)} entries but only "
            f"{len(_M_LINESTYLES)} linestyles / {len(_M_MARKERS)} "
            "markers are defined; extend _M_LINESTYLES / _M_MARKERS "
            "in exp/_vis_common.py to keep every legend entry unique."
        )

    eta_colors = _eta_palette(len(eta_str_list))
    eta_color = {e: c for e, c in zip(eta_str_list, eta_colors)}
    m_style = {
        m: (_M_LINESTYLES[i], _M_MARKERS[i])
        for i, m in enumerate(m_list)
    }

    # `constrained_layout` (vs tight_layout) plays nicely with an
    # external legend: the data axes stay full-height while the legend
    # gets its own slot on the right.
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)

    for m, eta_str, ds, ys in curves:
        ls, mk = m_style[m]
        ax.plot(
            ds, ys,
            color=eta_color[eta_str],
            linestyle=ls,
            marker=mk,
            lw=1.6,
            ms=5.0,
            markeredgewidth=0.0,
            label=f"$m={m}$, $\\eta={eta_str}$",
        )

    ax.set_xlabel(r"ambient dimension $d$")
    ax.set_ylabel(ylabel)
    if yscale == "log" and curves:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)

    if not curves:
        ax.text(
            0.5, 0.5, "no finite tail mean for any (m, eta)",
            ha="center", va="center", transform=ax.transAxes,
        )
    else:
        # Legend just outside the right edge so 9+ entries don't crowd
        # the data area; constrained_layout reserves the slot.
        leg = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            handlelength=2.6,
            borderaxespad=0.0,
        )
        leg.get_frame().set_linewidth(0.6)

    fig.savefig(out)
    plt.close(fig)


