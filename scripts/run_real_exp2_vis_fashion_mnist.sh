#!/usr/bin/env bash
#SBATCH --job-name=fmnn-real-exp2-vis-fashion-mnist
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=jqfan
#SBATCH --time=0:30:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qy1448@princeton.edu

# Exp 2 vis on fashion_mnist: w1_vs_t.png per cell. No GPU required.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source scripts/_env.sh
mkdir -p logs figs

DATASET_LIST=(fashion_mnist)
DATA_ROOT="${DATA_ROOT:-data}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
FIGS_DIR="${FIGS_DIR:-figs}"

source scripts/_real_grid.sh

for RIDX in $(real_cell_indices); do
  resolve_real_cell "$RIDX"
  echo "[run_real_exp2_vis_fashion_mnist] ridx=$RIDX  run=$RUN_NAME"
  python exp/run_real_exp2_vis.py \
    --run_name "$RUN_NAME" \
    --data_root "$DATA_ROOT" --results_root "$RESULTS_ROOT" \
    --figs_dir "$FIGS_DIR" "$@"
done
