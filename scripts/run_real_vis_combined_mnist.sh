#!/usr/bin/env bash
#SBATCH --job-name=fmnn-real-vis-combined-mnist
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=jqfan
#SBATCH --time=0:15:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qy1448@princeton.edu

# Combined visualisation for mnist: overlay all (m, eta) curves.
# Produces figs/application/mnist/loss_vs_t.png and w1_vs_t.png.
# No GPU required.  Run after training (exp1/exp2) has finished.
#
# Usage:
#   sbatch scripts/run_real_vis_combined_mnist.sh
#   bash   scripts/run_real_vis_combined_mnist.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source scripts/_env.sh
mkdir -p logs figs/application

DATASET_LIST=(mnist)
source scripts/_real_grid.sh

M_STR=$(IFS=,; echo "${M_LIST[*]}")
ETA_STR=$(IFS=,; echo "${ETA_LIST[*]}")

python exp/run_real_vis_combined.py \
  --dataset      mnist \
  --m_list       "$M_STR" \
  --eta_list     "$ETA_STR" \
  --n_train      "${N_TRAIN_LIST[0]}" \
  --n_test       "${N_TEST_LIST[0]}" \
  --results_root "${RESULTS_ROOT:-results}" \
  --figs_dir     "${FIGS_DIR:-figs}" \
  "$@"
