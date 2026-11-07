#!/usr/bin/env bash
# NTK-experiment grid definition: (d, n, m, eta_mult, seed).
# Sourced by run_ntk.sh and run_ntk_vis.sh so the grid lives in one place.
#
# Index mapping (lexicographic, d outermost / seed innermost):
#   idx = d_i   * (N_N * N_M * N_EM * N_SEEDS_NTK)
#       + n_i   * (N_M * N_EM * N_SEEDS_NTK)
#       + m_i   * (N_EM * N_SEEDS_NTK)
#       + em_i  * N_SEEDS_NTK
#       + seed
#
# Default: 3 * 2 * 7 * 3 * 5 = 630 cells.

D_LIST_NTK=(5 10 20)
N_LIST_NTK=(100 500)
# Widths span the OVERPARAMETERISED range the paper studies. The
# lower end is 256: Theorem 3.2 concerns m >> n, and at n = 500 a
# width of 128 is not overparameterised at all, so it probes a
# regime the theorem makes no statement about.
M_LIST_NTK=(256 512 1024 2048 4096 8192 16384)
ETA_MULT_LIST=(0.25 0.5 1.0)
N_SEEDS_NTK="${N_SEEDS_NTK:-5}"

# Iterations per cell (spectral eta => contraction visible within this).
T_NTK="${T_NTK:-1000}"

N_D=${#D_LIST_NTK[@]}
N_N=${#N_LIST_NTK[@]}
N_M=${#M_LIST_NTK[@]}
N_EM=${#ETA_MULT_LIST[@]}
NTK_GRID_SIZE=$(( N_D * N_N * N_M * N_EM * N_SEEDS_NTK ))

# Resolve a grid index into (D, N, M, EM, SEED).
resolve_ntk_cell() {
  local idx=$1
  SEED=$(( idx % N_SEEDS_NTK )); idx=$(( idx / N_SEEDS_NTK ))
  local em_i=$(( idx % N_EM ));  idx=$(( idx / N_EM ))
  local m_i=$((  idx % N_M ));   idx=$(( idx / N_M ))
  local n_i=$((  idx % N_N ));   idx=$(( idx / N_N ))
  local d_i=$idx
  D=${D_LIST_NTK[$d_i]}
  N=${N_LIST_NTK[$n_i]}
  M=${M_LIST_NTK[$m_i]}
  EM=${ETA_MULT_LIST[$em_i]}
}

# Cell indices for this invocation (same dispatch as _grid.sh):
#   1. IDX env var                       — single-cell mode
#   2. SLURM_ARRAY_TASK_ID + COUNT       — stride-N sharding
#   3. otherwise                         — full grid sequentially
ntk_cell_indices() {
  if [[ -n "${IDX:-}" ]]; then
    echo "$IDX"
  elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    local n=${SLURM_ARRAY_TASK_COUNT:-1}
    local k=$SLURM_ARRAY_TASK_ID
    if (( n <= 1 )); then
      echo "$k"
    else
      seq "$k" "$n" $(( NTK_GRID_SIZE - 1 ))
    fi
  else
    seq 0 $(( NTK_GRID_SIZE - 1 ))
  fi
}
