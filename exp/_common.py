"""
Helpers shared across exp/exp*/run_*.py: argparse, IO, CUDA check,
path resolution.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def parse_args():
    """
    Parse command-line arguments for every entry point under exp/.

    Single shared parser so all run_*.py / vis scripts accept the
    same flags. Vis-only scripts simply ignore the training fields.
    """
    args = argparse.ArgumentParser()

    # Dataset construction
    args.add_argument('--n_train', type=int, default=500,
                      help='Training set size n')
    args.add_argument('--d', type=int, default=5,
                      help='Ambient dimension d')
    args.add_argument('--target_center_norm', type=float, default=2.0,
                      help='GMM center norm; centers at +- c * e_1')

    # Network and gradient descent
    args.add_argument('--m', type=int, default=128,
                      help='Two-layer ReLU width m')
    args.add_argument('--kappa', type=float, default=1.0,
                      help='Init scale of the first-layer weights '
                           '(w_r ~ N(0, kappa^2 * I))')
    args.add_argument('--T', type=int, default=500,
                      help='Number of full-batch GD iterations')
    args.add_argument('--eta', type=float, default=1e-3,
                      help='GD step size eta')

    # Sampling (Euler ODE solve + truncation)
    args.add_argument('--T_euler', type=int, default=200,
                      help='Number of Euler steps for the ODE solve at '
                           'sampling time')
    args.add_argument('--B1', type=float, default=10.0,
                      help='Truncation radius factor: cap samples at '
                           '2 * sqrt(d) * B_1')

    # Evaluation
    # The model side of the W_1 estimate always draws n_train samples
    # (matched against the n_test reference set); there is intentionally
    # no separate --n_eval knob.
    args.add_argument('--n_test', type=int, default=5000,
                      help='# samples drawn from P_1 as the W_1 reference '
                           '(fixed across all evaluations of one run)')
    args.add_argument('--n_proj', type=int, default=128,
                      help='# random unit directions K used by the sliced '
                           'Wasserstein-1 estimator')
    args.add_argument('--eval_every', type=int, default=10,
                      help='Record loss/W_1 every k GD steps')

    # Reproducibility
    args.add_argument('--seed', type=int, default=0,
                      help='Master RNG seed')

    # IO
    args.add_argument('--run_name', type=str, default=None,
                      help='Subdir under data/ and results/. '
                           'Default: m{m}_eta{eta:g}')
    args.add_argument('--data_root', type=str, default='data',
                      help='Root dir for data/<run_name>/')
    args.add_argument('--results_root', type=str, default='results',
                      help='Root dir for results/<run_name>/')
    args.add_argument('--figs_dir', type=str, default='figs',
                      help='Output dir for the PNG figures (vis scripts)')

    # ----- vis-only flags -----
    # Both run_loss_vis.py and run_w1_vis.py support two modes:
    #   setting : per-setting trace plot, x = t, one curve per d in
    #             --d_list. Fixes (m, eta_str, n_train, n_test).  [default]
    #             -> figs/{loss,w1}_vs_t__<setting>.png
    #   global  : single overlay plot, x = d, one curve per (m, eta)
    #             pair in (--m_list, --eta_str_list). Fixes (n_train,
    #             n_test); each curve is a tail-mean(loss / W_1) trace.
    #             -> figs/{loss,w1}_vs_d__ntr<ntr>_nte<nte>.png
    #
    # Each mode ignores the flags it does not need; they are all parsed
    # here so the shared parser doesn't need to be split.
    args.add_argument('--mode', type=str, default='setting',
                      choices=['setting', 'global'],
                      help='Vis mode: "setting" overlays all d curves on '
                           'one (m, eta, n_train, n_test) trace plot '
                           '(default); "global" overlays all (m, eta) '
                           'tail-mean curves on one (n_train, n_test) '
                           '"vs d" plot.')
    args.add_argument('--eta_str', type=str, default=None,
                      help='[setting only] eta as a string, matching the '
                           'literal token in scripts/exp1_optimization/_grid.sh (e.g. "1e-4", '
                           '"1e-3"). Required because cell directory names '
                           'are built by bash from this literal, not from '
                           'Python\'s "{eta:g}" format.')
    args.add_argument('--m_list', type=str, default=None,
                      help='[global only] comma-separated list of widths '
                           'm, e.g. "16,32,64,128,256". Defaults to M_LIST '
                           'in scripts/exp1_optimization/_grid.sh; the shell driver always '
                           'passes it explicitly.')
    args.add_argument('--eta_str_list', type=str, default=None,
                      help='[global only] comma-separated list of eta '
                           'literals, e.g. "1e-4,1e-3,1e-2". Defaults to '
                           'ETA_LIST in scripts/exp1_optimization/_grid.sh; the shell '
                           'driver always passes it explicitly.')
    args.add_argument('--tail', type=int, default=5,
                      help='# of trailing evals to average over for each '
                           'cell in "global" mode. The shell driver always '
                           'passes this explicitly from TAIL in '
                           'scripts/exp1_optimization/_grid.sh (the source of truth), so this '
                           'argparse default only applies when '
                           'run_exp{1,2}_vis.py is invoked directly.')
    args.add_argument('--n_seeds', type=int, default=10,
                      help='[global only] number of replications (seeds '
                           '0..n_seeds-1) to aggregate over when computing '
                           'error bars in the loss/W_1 vs d plot. The shell '
                           'driver always passes this explicitly from '
                           'N_SEEDS in scripts/exp1_optimization/_grid.sh.')
    args.add_argument('--d_list', type=str, default=None,
                      help='Comma-separated list of d values to read CSVs '
                           'for, e.g. "5,10,15,...,100". Defaults to '
                           'D_LIST in scripts/exp1_optimization/_grid.sh; the shell driver '
                           'always passes it explicitly. Used as the curve '
                           'set in "setting" mode and the x-axis in '
                           '"global" mode.')

    args = args.parse_args()
    return args


# ---------------------------------------------------------------------------
# Device / IO helpers
# ---------------------------------------------------------------------------


def require_cuda() -> torch.device:
    """
    Hard-fail if CUDA is unavailable.

    Per project policy, every entry point must run on a GPU node;
    we do not silently fall back to CPU.

    """
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required (torch.cuda.is_available() == False). "
            "Aborting per project policy. "
            "Re-run on a GPU node (sbatch --gres=gpu:1)."
        )
    return torch.device("cuda")


def default_run_name(
    m: int, eta: float, d: int, n_train: int, n_test: int
) -> str:
    """
    Default run_name when --run_name is not supplied.

    Convention: "m{m}_eta{eta:g}_d{d}_ntr{n_train}_nte{n_test}" so each
    (m, eta, d, n_train, n_test) cell of the grid maps to a distinct
    subdir. Mirrors the RUN_NAME constructed by scripts/exp1_optimization/_grid.sh.
    """
    return f"m{m}_eta{eta:g}_d{d}_ntr{n_train}_nte{n_test}"


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    """
    Compute and create the per-run paths used by every script.

    Returns a dict with keys
        run_name, data_dir, results_dir,
        dataset_pkl, config_json, exp1_csv, exp2_csv.
    """
    run = args.run_name or default_run_name(
        args.m, args.eta, args.d, args.n_train, args.n_test
    )
    data_dir = Path(args.data_root) / run
    results_dir = Path(args.results_root) / run
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_name": run,
        "data_dir": data_dir,
        "results_dir": results_dir,
        "dataset_pkl": data_dir / "dataset.pkl",
        "config_json": results_dir / "config.json",
        "exp1_csv": results_dir / "exp1_trace.csv",
        "exp2_csv": results_dir / "exp2_trace.csv",
    }


def write_config(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    """
    Snapshot the resolved CLI args next to the trace files.

    Purely informational (nothing in the codebase reads it back);
    written so that any results/<run_name>/ folder is self-describing
    months later.
    """
    payload: Dict[str, Any] = vars(args).copy()
    payload["_paths"] = {k: str(v) for k, v in paths.items()}
    with open(paths["config_json"], "w") as f:
        json.dump(payload, f, indent=2, default=str)


def write_csv(path: Path, header: str, rows: list) -> None:
    """
    Minimal 2-column CSV writer to avoid a pandas dependency.

    Writes `header` as the first line, then comma-joins each tuple
    in `rows`. Used for exp1_trace.csv (t, loss) and exp2_trace.csv
    (t, w1).
    """
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def parent_of_repo() -> Path:
    """
    Return the repo root (the directory holding pyproject.toml).

    Used so that relative `data/`, `results/`, `figs/` paths resolve
    consistently regardless of where the script is invoked from.
    """
    return Path(__file__).resolve().parent.parent


def chdir_to_repo() -> None:
    """
    Chdir into the repo root so relative path defaults resolve correctly.
    """
    os.chdir(parent_of_repo())
