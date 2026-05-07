#!/usr/bin/env bash
# Shared (m, eta, d, n_train, n_test) grid definition. Sourced by all SLURM
# array scripts so the grid lives in exactly one place.
#
#   M_LIST, ETA_LIST, D_LIST, N_TRAIN_LIST, N_TEST_LIST
#   N_M, N_ETA, N_D, N_NTR, N_NTE - sizes
#   GRID_SIZE                     - total number of cells
#
# Index mapping (lexicographic, m outermost / n_test innermost):
#
#   idx = m_i * (N_ETA * N_D * N_NTR * N_NTE)
#       + e_i * (N_D   * N_NTR * N_NTE)
#       + d_i * (N_NTR * N_NTE)
#       + ntr_i * N_NTE
#       + nte_i
#
# Default: 5 * 3 * 10 * 1 * 1 = 150 cells.

M_LIST=(16 32 64 128 256)
ETA_LIST=(1e-4 1e-3 1e-2)
D_LIST=(5 10 15 20 25 30 35 40 45 50)
N_TRAIN_LIST=(500)
N_TEST_LIST=(5000)

# Number of independent replications (seeds 0..N_SEEDS-1) per cell.
# Used by run_exp{1,2}.sh for the seed loop and by run_exp{1,2}_vis.sh
# to pass --n_seeds to the error-bar aggregation in global mode.
N_SEEDS="${N_SEEDS:-10}"

# Tail-window size for the global aggregation plots
# (run_exp{1,2}_vis.sh / run_exp{1,2}_vis.py --mode global): number
# of trailing eval points averaged into each (m, eta) curve's y-value
# on the `loss vs d` / `w1 vs d` figures. Bigger = smoother but lags
# late dynamics; smaller = noisier but follows the tail more closely.
# Honors externally-set TAIL, so
#     TAIL=10 sbatch scripts/run_exp2_vis.sh
# still wins over this default.
TAIL="${TAIL:-50}"

N_M=${#M_LIST[@]}
N_ETA=${#ETA_LIST[@]}
N_D=${#D_LIST[@]}
N_NTR=${#N_TRAIN_LIST[@]}
N_NTE=${#N_TEST_LIST[@]}
GRID_SIZE=$(( N_M * N_ETA * N_D * N_NTR * N_NTE ))

# A "setting" is a (m, eta, n_train, n_test) tuple -- the cross-product
# of the grid axes excluding `d`. Used by the d-axis aggregation plots
# (run_exp_d_vis.py / run_exp_d_vis.sh), which fix a setting and sweep
# d on the x-axis.
N_SETTINGS=$(( N_M * N_ETA * N_NTR * N_NTE ))

# Resolve a grid index into (M, ETA, D, N_TRAIN, N_TEST, RUN_NAME).
# Caller passes the index as $1; result is exported via shell vars.
resolve_cell() {
  local idx=$1
  local nte_i=$(( idx % N_NTE )); idx=$(( idx / N_NTE ))
  local ntr_i=$(( idx % N_NTR )); idx=$(( idx / N_NTR ))
  local d_i=$((   idx % N_D   )); idx=$(( idx / N_D   ))
  local e_i=$((   idx % N_ETA )); idx=$(( idx / N_ETA ))
  local m_i=$idx
  M=${M_LIST[$m_i]}
  ETA=${ETA_LIST[$e_i]}
  D=${D_LIST[$d_i]}
  N_TRAIN=${N_TRAIN_LIST[$ntr_i]}
  N_TEST=${N_TEST_LIST[$nte_i]}
  RUN_NAME="m${M}_eta${ETA}_d${D}_ntr${N_TRAIN}_nte${N_TEST}"
}

# Indices to process this invocation. Honors (in order):
#
#   1. IDX env var          -- single-cell mode for interactive testing,
#                              e.g.  IDX=42 bash scripts/run_exp1.sh
#
#   2. SLURM_ARRAY_TASK_ID  -- sharded sweep: each array task k of N
#                              processes the round-robin slice
#                                  cells {k, k+N, k+2N, ...} & < GRID_SIZE
#                              where N == SLURM_ARRAY_TASK_COUNT.
#                              Default scripts header is `--array=0-9`,
#                              so N = 10 shards covering all GRID_SIZE
#                              cells exactly once. Stride sharding spreads
#                              heavy and light cells across shards rather
#                              than packing all heavy cells into one shard.
#                              Submission with a single SLURM_ARRAY_TASK
#                              (e.g. `sbatch --array=42 ...`) is treated
#                              as "run cell 42 only" instead.
#                              Note: requires --array=0-(N-1) contiguous;
#                              non-contiguous arrays such as --array=2,5,7
#                              are not supported (would alias / leave gaps).
#
#   3. otherwise            -- iterate the entire grid sequentially. Only
#                              practical for tiny grids; submitting via
#                              sbatch is the intended path for the full
#                              sweep.
cell_indices() {
  if [[ -n "${IDX:-}" ]]; then
    echo "$IDX"
  elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    local n=${SLURM_ARRAY_TASK_COUNT:-1}
    local k=$SLURM_ARRAY_TASK_ID
    if (( n <= 1 )); then
      # Single-task submission (e.g. sbatch --array=42): TASK_ID is one cell.
      echo "$k"
    else
      # Sharded sweep: stride n starting at offset k.
      seq "$k" "$n" $(( GRID_SIZE - 1 ))
    fi
  else
    seq 0 $(( GRID_SIZE - 1 ))
  fi
}

# ---------------------------------------------------------------------------
# Setting-level (d-collapsed) iteration. Mirrors cell_indices/resolve_cell
# but over the (m, eta, n_train, n_test) axes only -- d is the x-axis of
# the aggregation plots, so it is NOT part of the setting key.
#
# Index mapping (m outermost, n_test innermost):
#   sidx = m_i * (N_ETA * N_NTR * N_NTE)
#        + e_i * (N_NTR * N_NTE)
#        + ntr_i * N_NTE
#        + nte_i
#
# Default: 5 * 3 * 1 * 1 = 15 settings. Stride sharding mirrors
# cell_indices: shard k of N processes settings {k, k+N, k+2N, ...}.
# ---------------------------------------------------------------------------

# Resolve a setting index into (M, ETA, N_TRAIN, N_TEST, SETTING_NAME).
# SETTING_NAME = "m{M}_eta{ETA}_ntr{N_TRAIN}_nte{N_TEST}" (no d).
resolve_setting() {
  local sidx=$1
  local nte_i=$(( sidx % N_NTE )); sidx=$(( sidx / N_NTE ))
  local ntr_i=$(( sidx % N_NTR )); sidx=$(( sidx / N_NTR ))
  local e_i=$((   sidx % N_ETA )); sidx=$(( sidx / N_ETA ))
  local m_i=$sidx
  M=${M_LIST[$m_i]}
  ETA=${ETA_LIST[$e_i]}
  N_TRAIN=${N_TRAIN_LIST[$ntr_i]}
  N_TEST=${N_TEST_LIST[$nte_i]}
  SETTING_NAME="m${M}_eta${ETA}_ntr${N_TRAIN}_nte${N_TEST}"
}

# Setting indices for this invocation. Same dispatch convention as
# cell_indices: SIDX env var > SLURM_ARRAY_TASK_ID + COUNT > full sweep.
# Note: we use SIDX (not IDX) so a job submitting both cell- and
# setting-level steps can target one without disturbing the other.
setting_indices() {
  if [[ -n "${SIDX:-}" ]]; then
    echo "$SIDX"
  elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    local n=${SLURM_ARRAY_TASK_COUNT:-1}
    local k=$SLURM_ARRAY_TASK_ID
    if (( n <= 1 )); then
      echo "$k"
    else
      seq "$k" "$n" $(( N_SETTINGS - 1 ))
    fi
  else
    seq 0 $(( N_SETTINGS - 1 ))
  fi
}
