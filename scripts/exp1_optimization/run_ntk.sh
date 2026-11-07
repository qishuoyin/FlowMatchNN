#!/usr/bin/env bash
#SBATCH --job-name=fmnn-ntk
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=jqfan
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qy1448@princeton.edu

# NTK convergence validation over the
# (d, n, m, eta_mult, seed) grid in _ntk_grid.sh (720 cells).
# The default header shards them across 8 array tasks (~90 cells each).
#
# Submit options:
#   sbatch scripts/exp1_optimization/run_ntk.sh          # full grid, 8 shards
#   IDX=0  bash scripts/exp1_optimization/run_ntk.sh     # single cell, local

set -euo pipefail
# See scripts/exp1_optimization/run_all.sh for why we prefer
# SLURM_SUBMIT_DIR over $0.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

source scripts/_env.sh
mkdir -p logs results

source scripts/exp1_optimization/_ntk_grid.sh

for IDXC in $(ntk_cell_indices); do
  resolve_ntk_cell "$IDXC"
  echo "[run_ntk] cell idx=$IDXC d=$D n=$N m=$M eta_mult=$EM seed=$SEED"
  python exp/exp1_optimization/run_ntk.py \
    --d "$D" --n "$N" --m "$M" \
    --eta_mult "$EM" --T "$T_NTK" --seed "$SEED" \
    "$@"
done
