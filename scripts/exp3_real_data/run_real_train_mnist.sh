#!/usr/bin/env bash
#SBATCH --job-name=fmnn-real-exp1-mnist
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

# Exp 1 on mnist: loss vs t, all (m, eta, n_train, n_test) cells.
# Writes exp1_trace.csv + exp2_trace.csv + model artefacts per cell.
# Submit options:
#   sbatch scripts/exp3_real_data/run_real_train_mnist.sh          # 5 shards, 3 cells each
#   RIDX=2 bash scripts/exp3_real_data/run_real_train_mnist.sh     # single cell, local

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
source scripts/_env.sh
mkdir -p logs data results

DATASET_LIST=(mnist)   # restrict grid to this dataset
N_PCA="${N_PCA:-}"
KAPPA="${KAPPA:-1.0}"
T="${T:-500}"
T_EULER="${T_EULER:-200}"
B1="${B1:-10.0}"
EVAL_EVERY="${EVAL_EVERY:-10}"
SEED="${SEED:-0}"
N_PROJ="${N_PROJ:-128}"
DATA_ROOT="${DATA_ROOT:-data}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"

source scripts/exp3_real_data/_real_grid.sh

N_PCA_FLAG=()
[[ -n "${N_PCA}" ]] && N_PCA_FLAG=(--n_pca "${N_PCA}")

for RIDX in $(real_cell_indices); do
  resolve_real_cell "$RIDX"
  echo "[run_real_train_mnist] ridx=$RIDX  m=$M  eta=$ETA  run=$RUN_NAME"
  python exp/exp3_real_data/run_real_train.py \
    --dataset "$DATASET" --data_dir "$DATA_DIR" \
    --n_train "$N_TRAIN" --n_test "$N_TEST" \
    --m "$M" --kappa "$KAPPA" --T "$T" --eta "$ETA" \
    --T_euler "$T_EULER" --B1 "$B1" \
    --eval_every "$EVAL_EVERY" --seed "$SEED" --n_proj "$N_PROJ" \
    --run_name "$RUN_NAME" \
    --data_root "$DATA_ROOT" --results_root "$RESULTS_ROOT" \
    "${N_PCA_FLAG[@]}" "$@"
done
