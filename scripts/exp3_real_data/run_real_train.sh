#!/usr/bin/env bash
#SBATCH --job-name=fmnn-real-exp1
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

# Exp 1 (real data): trace L(theta^(t)) vs t for every cell of the
# (DATASET x m x eta x n_train x n_test) grid in scripts/exp3_real_data/_real_grid.sh.
#
# Default header --array=0-3 shards 12 cells across 4 tasks (3 per task,
# stride-4 round-robin). run_real_train.py writes BOTH exp1_trace.csv
# (loss) AND exp2_trace.csv (W_1) from the same training run, so this
# single script fully populates both trace files.
#
# Submit options:
#   sbatch scripts/exp3_real_data/run_real_train.sh               # full grid, 4 shards
#   sbatch --array=0-11 scripts/exp3_real_data/run_real_train.sh  # 12 shards (1 cell each)
#   RIDX=7 bash scripts/exp3_real_data/run_real_train.sh          # single cell, local
#
# To add a dataset: add its name to DATASET_LIST in scripts/exp3_real_data/_real_grid.sh.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
source scripts/_env.sh
mkdir -p logs data results

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
  echo "[run_real_train] ridx=$RIDX  dataset=$DATASET  m=$M  eta=$ETA  " \
       "n_train=$N_TRAIN  n_test=$N_TEST  run=$RUN_NAME"

  python exp/exp3_real_data/run_real_train.py \
    --dataset      "$DATASET"  \
    --data_dir     "$DATA_DIR" \
    --n_train      "$N_TRAIN"  \
    --n_test       "$N_TEST"   \
    --m            "$M"        \
    --kappa        "$KAPPA"    \
    --T            "$T"        \
    --eta          "$ETA"      \
    --T_euler      "$T_EULER"  \
    --B1           "$B1"       \
    --eval_every   "$EVAL_EVERY" \
    --seed         "$SEED"     \
    --n_proj       "$N_PROJ"   \
    --run_name     "$RUN_NAME" \
    --data_root    "$DATA_ROOT" \
    --results_root "$RESULTS_ROOT" \
    "${N_PCA_FLAG[@]}" \
    "$@"
done
