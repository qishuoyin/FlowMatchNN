#!/usr/bin/env bash
#SBATCH --job-name=fmnn-real-all
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=jqfan
#SBATCH --time=6:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qy1448@princeton.edu

# One-for-all entry point for real-data experiments.
#
# One sbatch submission runs the complete pipeline:
#
#   Step 1  run_real_exp1.sh  (this script, array job)
#           For every cell in the (DATASET x m x eta x n_train x n_test)
#           grid: load images, fit PCA, build dataset, train velocity
#           field, write exp1_trace.csv + exp2_trace.csv + model artefacts.
#
#   Step 2  run_real_exp1_vis.sh  (dependent SLURM job)
#           loss_vs_t.png + generated image grid per cell.
#
#   Step 3  run_real_exp2_vis.sh  (dependent SLURM job, no GPU)
#           w1_vs_t.png per cell.
#
# Usage:
#   sbatch scripts/run_all_real_exp.sh           # full grid, 5 shards
#   sbatch --array=0-14 scripts/run_all_real_exp.sh  # 15 shards (1 each)
#   bash   scripts/run_all_real_exp.sh           # local sequential
#   RIDX=2 bash scripts/run_all_real_exp.sh      # local single cell
#
# To add a new dataset:
#   1. Place parquet files at data/application/<name>/{train,test}.parquet
#      (use scripts/download_<name>.sh or scripts/download_hf_dataset.sh).
#   2. Append <name> to DATASET_LIST in scripts/_real_grid.sh.
#   3. Resubmit — all new cells are picked up automatically.
#
# Vis-race rationale (mirrors run_all_exp.sh):
#   Vis is submitted as a SEPARATE sbatch with --dependency=afterany on
#   the training array, so it only runs after ALL cells are done. Only
#   the lowest-index shard submits the vis jobs (prevents duplicates).
#   In local mode (no SLURM_ARRAY_JOB_ID) we run vis inline instead.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source scripts/_env.sh
mkdir -p logs data results figs

GLOBAL_TASK_ID="${SLURM_ARRAY_TASK_MIN:-0}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]] \
   && [[ "${SLURM_ARRAY_TASK_ID:-}" == "${GLOBAL_TASK_ID}" ]]; then
  VIS1_JID=$(sbatch --parsable \
    --dependency=afterany:"${SLURM_ARRAY_JOB_ID}" \
    scripts/run_real_exp1_vis.sh "$@") \
    && echo "[run_all_real_exp] queued exp1_vis as $VIS1_JID  (after array ${SLURM_ARRAY_JOB_ID})" \
    || echo "[run_all_real_exp] WARNING: exp1_vis submission failed — submit manually after training."
  VIS2_JID=$(sbatch --parsable \
    --dependency=afterany:"${SLURM_ARRAY_JOB_ID}" \
    scripts/run_real_exp2_vis.sh "$@") \
    && echo "[run_all_real_exp] queued exp2_vis as $VIS2_JID  (after array ${SLURM_ARRAY_JOB_ID})" \
    || echo "[run_all_real_exp] WARNING: exp2_vis submission failed — submit manually after training."
fi

bash scripts/run_real_exp1.sh "$@"

if [[ -z "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  bash scripts/run_real_exp1_vis.sh "$@"
  bash scripts/run_real_exp2_vis.sh "$@"
fi
