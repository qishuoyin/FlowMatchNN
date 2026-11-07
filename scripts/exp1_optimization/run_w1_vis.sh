#!/usr/bin/env bash
#SBATCH --job-name=fmnn-exp2-vis
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --account=jqfan
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qy1448@princeton.edu

# Visualise Exp 2. Two passes per shard:
#   1. setting : per-(m, eta, n_train, n_test) W_1 vs t plot, all
#                d's overlaid in one figure (color = d via viridis).
#                -> figs/w1_vs_t__<setting>.png
#   2. global  : single overall plot per (n_train, n_test); every
#                (m, eta) pair from the grid is one curve, y is the
#                tail mean of W_1, x is d.
#                -> figs/w1_vs_d__ntr<ntr>_nte<nte>.png
#
# Same sharding contract as scripts/exp1_optimization/run_loss_vis.sh; see that file for usage
# notes, submit examples, and the rationale for gating the global
# pass to a single shard.

set -euo pipefail
# See scripts/exp1_optimization/run_all.sh for why we prefer
# SLURM_SUBMIT_DIR over $0.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Activate the `flowmatchnn` conda env (no-op if already active).
source scripts/_env.sh

mkdir -p logs figs

# ---------------------------------------------------------------------------
# Editable python args. Edit a value here, OR override at submit time:
#     FIGS_DIR=figs/v2 bash scripts/exp1_optimization/run_w1_vis.sh
#     bash scripts/exp1_optimization/run_w1_vis.sh --figs_dir figs/v2    # also works
# (m, eta, n_train, n_test) come from the grid loop below; edit
# scripts/exp1_optimization/_grid.sh to change the grid.
# ---------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-data}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
FIGS_DIR="${FIGS_DIR:-figs/exp1_optimization}"

# TAIL (tail-window for the global w1-vs-d plot) is defined in
# scripts/exp1_optimization/_grid.sh. Override at submit time via `TAIL=10 sbatch ...`.
source scripts/exp1_optimization/_grid.sh

D_LIST_CSV=$(IFS=, ; echo "${D_LIST[*]}")
M_LIST_CSV=$(IFS=, ; echo "${M_LIST[*]}")
ETA_LIST_CSV=$(IFS=, ; echo "${ETA_LIST[*]}")

# Pass 1: per-setting W_1 vs t (all d's overlaid).
for SIDX in $(setting_indices); do
  resolve_setting "$SIDX"
  echo "[run_w1_vis] setting sidx=$SIDX setting=$SETTING_NAME"
  python exp/exp1_optimization/run_w1_vis.py \
    --mode setting \
    --m "$M" --eta_str "$ETA" \
    --n_train "$N_TRAIN" --n_test "$N_TEST" \
    --d_list "$D_LIST_CSV" \
    --data_root "$DATA_ROOT" --results_root "$RESULTS_ROOT" \
    --figs_dir "$FIGS_DIR" \
    "$@"
done

# Pass 2: global tail-mean(W_1) vs d. Gated to a single shard; see
# scripts/exp1_optimization/run_loss_vis.sh for the rationale.
GLOBAL_TASK_ID="${SLURM_ARRAY_TASK_MIN:-0}"
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] \
   || [[ "${SLURM_ARRAY_TASK_ID}" == "${GLOBAL_TASK_ID}" ]]; then
  for ntr in "${N_TRAIN_LIST[@]}"; do
    for nte in "${N_TEST_LIST[@]}"; do
      echo "[run_w1_vis] global ntr=$ntr nte=$nte (m_list=$M_LIST_CSV, eta_list=$ETA_LIST_CSV, tail=$TAIL, n_seeds=$N_SEEDS)"
      python exp/exp1_optimization/run_w1_vis.py \
        --mode global \
        --n_train "$ntr" --n_test "$nte" \
        --m_list "$M_LIST_CSV" --eta_str_list "$ETA_LIST_CSV" \
        --d_list "$D_LIST_CSV" --tail "$TAIL" --n_seeds "$N_SEEDS" \
        --data_root "$DATA_ROOT" --results_root "$RESULTS_ROOT" \
        --figs_dir "$FIGS_DIR" \
        "$@"
    done
  done
else
  echo "[run_w1_vis] skipping global pass on shard ${SLURM_ARRAY_TASK_ID} (only shard ${GLOBAL_TASK_ID} runs it)"
fi
