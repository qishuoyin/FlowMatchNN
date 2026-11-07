#!/usr/bin/env bash
#SBATCH --job-name=fmnn-all
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

# Top-level pipeline. One sbatch -> the whole experiment.
#
#   sbatch scripts/exp1_optimization/run_all.sh              # full grid, 10 shards
#
# Steps in order:
#
#   1. run_train.sh        -- train each cell for every seed in
#                            0..N_SEEDS-1 (default 10 replications).
#                            run_train.py generates the dataset on demand
#                            under data/<cell>_seed<S>/dataset.pkl, so
#                            run_data.sh is no longer needed (it used
#                            the old seedless directory format and would
#                            produce unused files). run_train.py writes
#                            BOTH results/<cell>_seed<S>/exp1_trace.csv
#                            (loss) AND exp2_trace.csv (W_1) from the
#                            same trajectory. Runs inline in the array.
#   2. run_loss_vis.sh    -- per-setting loss-vs-t (all d overlaid,
#                            seed-0 traces only) + global loss-vs-d
#                            (mean ± 1 std across all seeds).
#                            Submitted as a SEPARATE sbatch with
#                            --dependency=afterany:$SLURM_ARRAY_JOB_ID
#                            (see "vis race" comment below).
#   3. scripts/exp1_optimization/run_w1_vis.sh
#                         -- per-setting W_1-vs-t (seed-0 only) +
#                            global W_1-vs-d (mean ± 1 std).
#                            Same dependency strategy as step 2.
#
# Vis race (job 7617946): the previous version called both vis scripts
# inline at the end of every shard's allocation. But each shard had only
# computed 1/N of the d-values for its own setting -- the rest were
# being computed concurrently in peer shards. A shard that finished
# training early therefore rendered figs/loss_vs_t__<setting>.png from
# the partial CSVs that happened to exist at that instant
# (e.g. ok=4/10). Worse, the global plot is gated on
# SLURM_ARRAY_TASK_MIN, i.e. the FASTEST shard, so the global PNG was
# always the most-incomplete snapshot. The fix is to defer vis until
# after the entire training array is done, by having one shard submit
# vis as a dependent sbatch job. afterany (rather than afterok) so vis
# still runs even if a few cells diverge -- those just appear in the
# subtitle's "missing" list.

set -euo pipefail
# Under sbatch, $0 points at SLURM's spool copy of this script, so the
# usual "$(dirname "$0")/../.." trick lands in the spool dir. Prefer
# SLURM_SUBMIT_DIR (where `sbatch` was invoked) when present.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Queue vis as a dependent sbatch (one per experiment) BEFORE training
# starts, so the dependency is registered even if training fails partway
# through. Only the lowest-index shard submits, so we don't get N copies
# of vis. Skipped in local mode (no SLURM_ARRAY_JOB_ID) -- there, vis
# falls back to an inline run after training, see below.
GLOBAL_TASK_ID="${SLURM_ARRAY_TASK_MIN:-0}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]] \
   && [[ "${SLURM_ARRAY_TASK_ID:-}" == "${GLOBAL_TASK_ID}" ]]; then
  VIS1_JID=$(sbatch --parsable \
    --dependency=afterany:"${SLURM_ARRAY_JOB_ID}" \
    scripts/exp1_optimization/run_loss_vis.sh "$@")
  echo "[run_all] queued loss_vis as $VIS1_JID  (after array ${SLURM_ARRAY_JOB_ID})"
  VIS2_JID=$(sbatch --parsable \
    --dependency=afterany:"${SLURM_ARRAY_JOB_ID}" \
    scripts/exp1_optimization/run_w1_vis.sh "$@")
  echo "[run_all] queued w1_vis as $VIS2_JID  (after array ${SLURM_ARRAY_JOB_ID})"
fi

bash scripts/exp1_optimization/run_train.sh "$@"

# Local mode (no SLURM): run vis inline. This is the IDX=42 / unsharded
# loop path. There's nothing concurrent to race against, so inline is
# safe -- and it's the only way the user gets figures without SLURM.
if [[ -z "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  bash scripts/exp1_optimization/run_loss_vis.sh "$@"
  bash scripts/exp1_optimization/run_w1_vis.sh "$@"
fi
