#!/usr/bin/env bash
#SBATCH --job-name=fmnn-real-exp1-vis
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=jqfan
#SBATCH --time=1:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qy1448@princeton.edu

# Exp 1 visualisation (real data): loss vs t + generated image grid.
#
# For each cell in the grid, runs run_real_loss_vis.py which produces:
#   figs/<run_name>/loss_vs_t.png        — Exp 1 loss curve
#   figs/<run_name>/generated_grid.png   — 8x8 synthesised image grid
#
# Requires a GPU (model sampling for the image grid).
# Typically submitted as a dependent job after run_real_train.sh:
#   EXP_JID=$(sbatch --parsable scripts/exp3_real_data/run_real_train.sh)
#   sbatch --dependency=afterany:$EXP_JID scripts/exp3_real_data/run_real_loss_vis.sh
#
# Submit options:
#   sbatch scripts/exp3_real_data/run_real_loss_vis.sh          # full grid, 5 shards
#   RIDX=3 bash scripts/exp3_real_data/run_real_loss_vis.sh     # single cell, local

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
source scripts/_env.sh
mkdir -p logs figs

N_SAMPLES="${N_SAMPLES:-64}"
DATA_ROOT="${DATA_ROOT:-data}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
FIGS_DIR="${FIGS_DIR:-figs/exp3_real_data}"
SEED="${SEED:-42}"

source scripts/exp3_real_data/_real_grid.sh

for RIDX in $(real_cell_indices); do
  resolve_real_cell "$RIDX"
  echo "[run_real_loss_vis] ridx=$RIDX  run=$RUN_NAME"

  python exp/exp3_real_data/run_real_loss_vis.py \
    --run_name     "$RUN_NAME"     \
    --n_samples    "$N_SAMPLES"    \
    --data_root    "$DATA_ROOT"    \
    --results_root "$RESULTS_ROOT" \
    --figs_dir     "$FIGS_DIR"     \
    --seed         "$SEED"         \
    "$@"
done
