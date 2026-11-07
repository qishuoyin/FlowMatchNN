#!/usr/bin/env bash
#SBATCH --job-name=fmnn-pca-vis
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=ailab
#SBATCH --account=jqfan
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qy1448@princeton.edu

# PCA-sweep figures: pca_vel_risk.png, pca_pixel_w1.png, pca_sweep_table.csv.
# Reads only the per-cell pca_summary.json files, so it is CPU-only and
# can be run locally after run_pca_sweep.sh finishes:
#   bash scripts/exp3_real_data/run_pca_sweep_vis.sh
#
# The (d_pca, n) grid is ragged, so a "missing cells" warning listing
# the d_pca in {2,4} x n = 8000 combinations is expected, not an error.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

source scripts/_env.sh
mkdir -p logs

source scripts/exp3_real_data/_pca_grid.sh
FIGS_DIR="${FIGS_DIR:-figs/exp3_real_data}"

python exp/exp3_real_data/run_pca_sweep_vis.py \
  --datasets   "${DATASET_LIST_PCA[@]}" \
  --d_pca_list "${DPCA_LIST_PCA[@]}" \
  --n_list     "${N_LIST_PCA[@]}" \
  --m          "$M_PCA" \
  --n_seeds    "$N_SEEDS_PCA" \
  --figs_dir   "$FIGS_DIR" \
  "$@"
