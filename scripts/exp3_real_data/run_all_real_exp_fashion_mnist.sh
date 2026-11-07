#!/usr/bin/env bash
#SBATCH --job-name=fmnn-real-all-fashion_mnist
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

# One-for-all pipeline for fashion_mnist:
#   Step 1  run_real_train_fashion_mnist.sh          (this script, array job)
#   Step 2  run_real_loss_vis_fashion_mnist.sh      (dependent, GPU)
#   Step 3  run_real_w1_vis_fashion_mnist.sh      (dependent, no GPU)
#   Step 4  run_real_vis_combined_fashion_mnist.sh  (dependent, no GPU)
#
# Usage:
#   sbatch scripts/exp3_real_data/run_all_real_exp_fashion_mnist.sh
#   bash   scripts/exp3_real_data/run_all_real_exp_fashion_mnist.sh    # local sequential

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
source scripts/_env.sh
mkdir -p logs data results figs

GLOBAL_TASK_ID="${SLURM_ARRAY_TASK_MIN:-0}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]] \
   && [[ "${SLURM_ARRAY_TASK_ID:-}" == "$GLOBAL_TASK_ID" ]]; then
  VIS1_JID=$(sbatch --parsable \
    --dependency=afterany:"${SLURM_ARRAY_JOB_ID}" \
    scripts/exp3_real_data/run_real_loss_vis_fashion_mnist.sh "$@") \
    && echo "[run_all_real_exp_fashion_mnist] queued exp1_vis as $VIS1_JID" \
    || echo "[run_all_real_exp_fashion_mnist] WARNING: exp1_vis submission failed — submit manually after training."
  VIS2_JID=$(sbatch --parsable \
    --dependency=afterany:"${SLURM_ARRAY_JOB_ID}" \
    scripts/exp3_real_data/run_real_w1_vis_fashion_mnist.sh "$@") \
    && echo "[run_all_real_exp_fashion_mnist] queued exp2_vis as $VIS2_JID" \
    || echo "[run_all_real_exp_fashion_mnist] WARNING: exp2_vis submission failed — submit manually after training."
  COMB_JID=$(sbatch --parsable \
    --dependency=afterany:"${SLURM_ARRAY_JOB_ID}" \
    scripts/exp3_real_data/run_real_vis_combined_fashion_mnist.sh "$@") \
    && echo "[run_all_real_exp_fashion_mnist] queued vis_combined as $COMB_JID" \
    || echo "[run_all_real_exp_fashion_mnist] WARNING: vis_combined submission failed — submit manually after training."
fi

bash scripts/exp3_real_data/run_real_train_fashion_mnist.sh "$@"

if [[ -z "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  bash scripts/exp3_real_data/run_real_loss_vis_fashion_mnist.sh "$@"
  bash scripts/exp3_real_data/run_real_w1_vis_fashion_mnist.sh "$@"
  bash scripts/exp3_real_data/run_real_vis_combined_fashion_mnist.sh "$@"
fi
