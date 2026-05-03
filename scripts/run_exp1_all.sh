#!/usr/bin/env bash
#SBATCH --job-name=fmnn-exp1-all
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

# End-to-end Exp 1 driver: training+eval -> figures. Default header
# `--array=0-9` shards the 150-cell grid across 10 SLURM array tasks;
# run_exp1.sh inherits SLURM_ARRAY_TASK_{ID,COUNT} and therefore
# processes the same shard.
#
# Dataset generation is *not* called here -- the dataset depends only
# on (n_train, d, seed, target_center_norm) and is built lazily by
# run_exp1.py the first time a cell is touched. To pre-generate
# explicitly (e.g. to inspect the dataset before training), submit
# `sbatch scripts/run_data.sh` separately.
#
# Vis runs as a SEPARATE sbatch with --dependency=afterany on this
# array. Inline-per-shard would race -- see the "vis race" comment in
# scripts/run_all_exp.sh.

set -euo pipefail
# See run_all_exp.sh for why we prefer SLURM_SUBMIT_DIR over $0.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

GLOBAL_TASK_ID="${SLURM_ARRAY_TASK_MIN:-0}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]] \
   && [[ "${SLURM_ARRAY_TASK_ID:-}" == "${GLOBAL_TASK_ID}" ]]; then
  VIS_JID=$(sbatch --parsable \
    --dependency=afterany:"${SLURM_ARRAY_JOB_ID}" \
    scripts/run_exp1_vis.sh "$@")
  echo "[run_exp1_all] queued exp1_vis as $VIS_JID  (after array ${SLURM_ARRAY_JOB_ID})"
fi

bash scripts/run_exp1.sh "$@"

if [[ -z "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  bash scripts/run_exp1_vis.sh "$@"
fi
