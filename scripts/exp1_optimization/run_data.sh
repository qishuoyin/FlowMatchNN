#!/usr/bin/env bash
#SBATCH --job-name=fmnn-data
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

# Generate the regression dataset for every (m, eta, d, n_train, n_test) cell
# of the grid (default: 150 cells; see scripts/exp1_optimization/_grid.sh).
#
# The default header `--array=0-9` shards the sweep across 10 SLURM array
# tasks (15 cells per task, stride-10 round robin). Submit as
#
#   sbatch scripts/exp1_optimization/run_data.sh                      # full sweep, 10 shards
#   sbatch --array=0-19 scripts/exp1_optimization/run_data.sh         # 20 shards (~7-8 cells each)
#   sbatch --array=42   scripts/exp1_optimization/run_data.sh         # single-cell sbatch
#   IDX=42 bash scripts/exp1_optimization/run_data.sh                 # local single cell
#   bash scripts/exp1_optimization/run_data.sh                        # local sequential (slow)
#
# Note: the dataset depends only on (n_train, d, seed, target_center_norm)
# and is independent of (m, eta, n_test); generating per cell is wasteful
# but cheap (n_train = 500). Pass --run_name to share a single pickle
# across cells if you need to dedupe.

set -euo pipefail
# See run_all.sh for why we prefer SLURM_SUBMIT_DIR over $0.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Activate the `flowmatchnn` conda env (no-op if already active).
source scripts/_env.sh

mkdir -p logs data

# ---------------------------------------------------------------------------
# Editable python args. Edit a value here, OR override at submit time:
#     SEED=1 sbatch scripts/exp1_optimization/run_data.sh
#     sbatch scripts/exp1_optimization/run_data.sh --seed 1            # also works (last-wins)
# (m, eta, d, n_train, n_test, run_name) come from the grid loop below and
# are NOT editable here; edit scripts/exp1_optimization/_grid.sh to change the grid.
# ---------------------------------------------------------------------------
SEED="${SEED:-0}"
TARGET_CENTER_NORM="${TARGET_CENTER_NORM:-2.0}"
DATA_ROOT="${DATA_ROOT:-data}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"

source scripts/exp1_optimization/_grid.sh

for IDX in $(cell_indices); do
  resolve_cell "$IDX"
  echo "[run_data] cell idx=$IDX m=$M eta=$ETA d=$D n_train=$N_TRAIN n_test=$N_TEST run=$RUN_NAME"
  python exp/exp1_optimization/run_data.py \
    --m "$M" --eta "$ETA" --run_name "$RUN_NAME" \
    --d "$D" --n_train "$N_TRAIN" --n_test "$N_TEST" \
    --seed "$SEED" --target_center_norm "$TARGET_CENTER_NORM" \
    --data_root "$DATA_ROOT" --results_root "$RESULTS_ROOT" \
    "$@"
done
