#!/usr/bin/env bash
# Grid definition for real-data (image + PCA) experiments.
# Sourced by run_real_exp.sh, run_real_vis.sh, and run_all_real_exp.sh.
#
# Axes:
#   DATASET_LIST — image datasets in data/application/<name>/
#   M_LIST       — network widths
#   ETA_LIST     — GD step sizes
#   N_TRAIN_LIST — training set sizes
#   N_TEST_LIST  — W_1 reference set sizes
#
# Index mapping (lexicographic; DATASET outermost, N_TEST innermost):
#
#   idx = ds_i  * (N_M * N_ETA * N_NTR * N_NTE)
#       + m_i   * (N_ETA * N_NTR * N_NTE)
#       + e_i   * (N_NTR * N_NTE)
#       + ntr_i * N_NTE
#       + nte_i
#
# To add a new dataset: append its name to DATASET_LIST and make sure
# its parquet files live at data/application/<name>/{train,test}.parquet.
#
# Default: 1 * 5 * 3 * 1 * 1 = 15 cells.

# Default when not pre-set by a per-dataset script. `[*]:-` keeps the
# check safe under `set -u` when DATASET_LIST is entirely unset (the
# generic run_real_*.sh drivers source this file without pre-setting it).
[[ -z "${DATASET_LIST[*]:-}" ]] && DATASET_LIST=(mnist)
M_LIST=(128 256 512 1024)
ETA_LIST=(1e-4 1e-3 1e-2)
N_TRAIN_LIST=(500)
N_TEST_LIST=(2000)

# Root directory where application datasets are stored.
# Each dataset lives at ${DATA_APP_ROOT}/<dataset_name>/.
DATA_APP_ROOT="${DATA_APP_ROOT:-data/application}"

# Grid sizes
N_DS=${#DATASET_LIST[@]}
N_M=${#M_LIST[@]}
N_ETA=${#ETA_LIST[@]}
N_NTR=${#N_TRAIN_LIST[@]}
N_NTE=${#N_TEST_LIST[@]}
REAL_GRID_SIZE=$(( N_DS * N_M * N_ETA * N_NTR * N_NTE ))

# Resolve a cell index into (DATASET, M, ETA, N_TRAIN, N_TEST, RUN_NAME).
# run_name is stable: real_{dataset}_m{m}_eta{eta}_ntr{n_train}_nte{n_test}.
resolve_real_cell() {
  local idx=$1
  local nte_i=$(( idx % N_NTE )); idx=$(( idx / N_NTE ))
  local ntr_i=$(( idx % N_NTR )); idx=$(( idx / N_NTR ))
  local e_i=$((  idx % N_ETA  )); idx=$(( idx / N_ETA  ))
  local m_i=$((  idx % N_M    )); idx=$(( idx / N_M    ))
  local ds_i=$idx
  DATASET=${DATASET_LIST[$ds_i]}
  M=${M_LIST[$m_i]}
  ETA=${ETA_LIST[$e_i]}
  N_TRAIN=${N_TRAIN_LIST[$ntr_i]}
  N_TEST=${N_TEST_LIST[$nte_i]}
  RUN_NAME="real_${DATASET}_m${M}_eta${ETA}_ntr${N_TRAIN}_nte${N_TEST}"
  DATA_DIR="${DATA_APP_ROOT}/${DATASET}"
}

# Cell indices for this invocation. Same dispatch as _grid.sh::cell_indices:
#   1. RIDX env var    — single-cell mode for interactive testing
#   2. SLURM_ARRAY_TASK_ID + SLURM_ARRAY_TASK_COUNT — stride-N sharding
#   3. otherwise       — iterate the entire grid sequentially
real_cell_indices() {
  if [[ -n "${RIDX:-}" ]]; then
    echo "$RIDX"
  elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    local n=${SLURM_ARRAY_TASK_COUNT:-1}
    local k=$SLURM_ARRAY_TASK_ID
    if (( n <= 1 )); then
      echo "$k"
    else
      seq "$k" "$n" $(( REAL_GRID_SIZE - 1 ))
    fi
  else
    seq 0 $(( REAL_GRID_SIZE - 1 ))
  fi
}
