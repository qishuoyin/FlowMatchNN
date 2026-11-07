#!/usr/bin/env bash
#SBATCH --job-name=fmnn-pca
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=jqfan
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qy1448@princeton.edu

# PCA-dimension sweep over the
# (dataset, d_pca, n, seed) grid in _pca_grid.sh — 96 cells in 8 shards
# of 12 (stride sharding, so every shard sees a mix of cheap and
# expensive cells).
#
# Each cell standardises the PCA codes, calibrates its GD step size from
# the kernel spectrum, trains to a loss plateau, and scores the model in
# PIXEL space against one fixed held-out reference. The sampler's source
# pool is the TRAINING codes -- the held-out reference is never fed to
# it. See the module docstring of exp/exp3_real_data/run_pca_sweep.py.
#
# Prerequisites:
#   bash scripts/exp3_real_data/download_mnist.sh
#   bash scripts/exp3_real_data/download_fashion_mnist.sh
#
# Submit options:
#   sbatch scripts/exp3_real_data/run_pca_sweep.sh       # full grid
#   IDX=0 bash scripts/exp3_real_data/run_pca_sweep.sh   # single cell

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

source scripts/_env.sh
mkdir -p logs results

source scripts/exp3_real_data/_pca_grid.sh

for IDXC in $(pca_cell_indices); do
  resolve_pca_cell "$IDXC"
  echo "[run_pca_sweep] cell idx=$IDXC dataset=$DATASET d_pca=$DPCA" \
       "n=$N_PCA m=$M_PCA T=$T_PCA seed=$SEED -> $PCA_RUN_NAME"
  python exp/exp3_real_data/run_pca_sweep.py \
    --dataset "$DATASET" --d_pca "$DPCA" --n "$N_PCA" \
    --m "$M_PCA" --T "$T_PCA" --seed "$SEED" \
    "$@"
done
