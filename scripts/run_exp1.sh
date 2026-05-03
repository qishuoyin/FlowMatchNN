#!/usr/bin/env bash
#SBATCH --job-name=fmnn-exp1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=jqfan
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qy1448@princeton.edu

# Exp 1: trace L(theta^(t)) vs t for every (m, eta, d, n_train, n_test) cell
# of the grid (default: 150 cells; see scripts/_grid.sh).
#
# The default header `--array=0-9` shards the sweep across 10 SLURM array
# tasks (15 cells per task, stride-10 round robin). See "Compute budget"
# in README §3 for per-cell timing notes.
#
#   sbatch scripts/run_exp1.sh                      # full sweep, 10 shards
#   sbatch --array=0-19 scripts/run_exp1.sh         # 20 shards (~7-8 cells each)
#   sbatch --array=0-9%5 scripts/run_exp1.sh        # cap concurrency at 5
#   sbatch --array=42 scripts/run_exp1.sh           # single-cell sbatch
#   IDX=42 bash scripts/run_exp1.sh                 # local single cell
#
# Note: run_exp1.py also writes results/<run_name>/exp2_trace.csv, so this
# single script populates both Exp 1 and Exp 2 traces; you only need to
# also submit run_exp2.sh if you want a separate trajectory.

set -euo pipefail
# See run_all_exp.sh for why we prefer SLURM_SUBMIT_DIR over $0.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# Activate the `flowmatchnn` conda env (no-op if already active).
source scripts/_env.sh

mkdir -p logs data results figs

# ---------------------------------------------------------------------------
# Editable python args. Edit a value here, OR override at submit time:
#     T=500 KAPPA=0.5 sbatch scripts/run_exp1.sh
#     sbatch scripts/run_exp1.sh --T 500 --kappa 0.5    # also works (last-wins)
# (m, eta, d, n_train, n_test, run_name) come from the grid loop below and
# are NOT editable here; edit scripts/_grid.sh to change the grid.
# Note: the model side of the W_1 estimate always draws n_train samples
# (matched against the n_test reference set), so there is no separate
# N_EVAL knob.
# ---------------------------------------------------------------------------
KAPPA="${KAPPA:-1.0}"
T="${T:-500}"
T_EULER="${T_EULER:-200}"
B1="${B1:-10.0}"
EVAL_EVERY="${EVAL_EVERY:-10}"
SEED="${SEED:-0}"
TARGET_CENTER_NORM="${TARGET_CENTER_NORM:-2.0}"
DATA_ROOT="${DATA_ROOT:-data}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"

source scripts/_grid.sh

for IDX in $(cell_indices); do
  resolve_cell "$IDX"
  echo "[run_exp1] cell idx=$IDX m=$M eta=$ETA d=$D n_train=$N_TRAIN n_test=$N_TEST run=$RUN_NAME"
  python exp/run_exp1.py \
    --m "$M" --eta "$ETA" --run_name "$RUN_NAME" \
    --d "$D" --n_train "$N_TRAIN" --n_test "$N_TEST" \
    --kappa "$KAPPA" --T "$T" \
    --T_euler "$T_EULER" --B1 "$B1" \
    --eval_every "$EVAL_EVERY" --seed "$SEED" \
    --target_center_norm "$TARGET_CENTER_NORM" \
    --data_root "$DATA_ROOT" --results_root "$RESULTS_ROOT" \
    "$@"
done
