#!/usr/bin/env bash
#SBATCH --job-name=fmnn-exp1-vis
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

# Visualise Exp 1. Two passes per shard:
#   1. setting : per-(m, eta, n_train, n_test) loss vs t plot, all
#                d's overlaid in one figure (color = d via viridis).
#                -> figs/loss_vs_t__<setting>.png
#                Iterates `setting_indices` from scripts/_grid.sh
#                (default: 15 settings; sharded stride-N like cells).
#   2. global  : single overall plot per (n_train, n_test); every
#                (m, eta) pair from the grid is one curve, y is the
#                tail mean of loss, x is d.
#                -> figs/loss_vs_d__ntr<ntr>_nte<nte>.png
#                Cheap; gated to one shard (the one whose
#                SLURM_ARRAY_TASK_ID equals SLURM_ARRAY_TASK_MIN, or
#                run unconditionally for unsharded local invocations)
#                so concurrent shards don't race on the same file.
#
# Cheap; default header `--array=0-9` shards setting-pass stride-10
# (1-2 settings per task on the current 15-setting grid). Both passes
# only read CSVs written by run_exp1.py, so they require no GPU and
# do not wait on each other.
#
#   sbatch scripts/run_exp1_vis.sh                 # full sweep, 10 shards
#   bash   scripts/run_exp1_vis.sh                 # local sequential
#   SIDX=4 bash scripts/run_exp1_vis.sh            # local single setting
#   TAIL=10 bash scripts/run_exp1_vis.sh           # change avg window

set -euo pipefail
# See run_all_exp.sh for why we prefer SLURM_SUBMIT_DIR over $0.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# Activate the `flowmatchnn` conda env (no-op if already active).
source scripts/_env.sh

mkdir -p logs figs

# ---------------------------------------------------------------------------
# Editable python args. Edit a value here, OR override at submit time:
#     FIGS_DIR=figs/v2 bash scripts/run_exp1_vis.sh
#     bash scripts/run_exp1_vis.sh --figs_dir figs/v2    # also works
# (m, eta, n_train, n_test) come from the grid loop below; edit
# scripts/_grid.sh to change the grid.
# ---------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-data}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
FIGS_DIR="${FIGS_DIR:-figs}"

# TAIL (tail-window for the global loss-vs-d plot) is defined in
# scripts/_grid.sh. Override at submit time via `TAIL=10 sbatch ...`.
source scripts/_grid.sh

# d_list / m_list / eta_str_list are passed as CSV strings so the
# python script can iterate the same axes the bash grid defines.
D_LIST_CSV=$(IFS=, ; echo "${D_LIST[*]}")
M_LIST_CSV=$(IFS=, ; echo "${M_LIST[*]}")
ETA_LIST_CSV=$(IFS=, ; echo "${ETA_LIST[*]}")

# Pass 1: per-setting loss vs t (all d's overlaid).
for SIDX in $(setting_indices); do
  resolve_setting "$SIDX"
  echo "[run_exp1_vis] setting sidx=$SIDX setting=$SETTING_NAME"
  python exp/run_exp1_vis.py \
    --mode setting \
    --m "$M" --eta_str "$ETA" \
    --n_train "$N_TRAIN" --n_test "$N_TEST" \
    --d_list "$D_LIST_CSV" \
    --data_root "$DATA_ROOT" --results_root "$RESULTS_ROOT" \
    --figs_dir "$FIGS_DIR" \
    "$@"
done

# Pass 2: global tail-mean(loss) vs d. One PNG per (n_train, n_test);
# we want exactly one shard to produce it to avoid two array tasks
# overwriting the same file simultaneously. The shard whose
# SLURM_ARRAY_TASK_ID equals SLURM_ARRAY_TASK_MIN (or the unsharded
# local run) wins.
GLOBAL_TASK_ID="${SLURM_ARRAY_TASK_MIN:-0}"
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] \
   || [[ "${SLURM_ARRAY_TASK_ID}" == "${GLOBAL_TASK_ID}" ]]; then
  for ntr in "${N_TRAIN_LIST[@]}"; do
    for nte in "${N_TEST_LIST[@]}"; do
      echo "[run_exp1_vis] global ntr=$ntr nte=$nte (m_list=$M_LIST_CSV, eta_list=$ETA_LIST_CSV, tail=$TAIL, n_seeds=$N_SEEDS)"
      python exp/run_exp1_vis.py \
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
  echo "[run_exp1_vis] skipping global pass on shard ${SLURM_ARRAY_TASK_ID} (only shard ${GLOBAL_TASK_ID} runs it)"
fi
