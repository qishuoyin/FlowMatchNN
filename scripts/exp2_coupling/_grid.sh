#!/usr/bin/env bash
# E4 grid definition (revision plan): (coupling, net, d, seed).
# Sourced by run_coupling.sh and run_coupling_vis.sh.
#
# The grid is a clean 2 x 2 (+ one extra coupling) at fixed (n, m):
#   coupling in {conditional, permuted, ot}   how (x_0, x_1) are paired
#   net      in {conditional, unconditional}  f(x_tau,tau,x_0) vs f(x_tau,tau)
# All 6 arms of one (d, seed) share ONE data draw, so the factorial
# isolates the coupling effect from the architecture effect. The
# paper's setting is (conditional, conditional); standard practice is
# (permuted, unconditional).
#
# Index mapping (lexicographic, coupling outermost / seed innermost):
#   idx = coupling_i * (NUM_NETS * NUM_D_COUP * N_SEEDS_COUP)
#       + net_i      * (NUM_D_COUP * N_SEEDS_COUP)
#       + d_i        * N_SEEDS_COUP
#       + seed
#
# Default: 3 * 2 * 3 * 5 = 90 cells.

COUPLING_LIST=(conditional permuted ot)
NET_LIST=(conditional unconditional)
D_LIST_COUP=(3 5 10)
N_SEEDS_COUP="${N_SEEDS_COUP:-5}"

N_COUP="${N_COUP:-2000}"
M_COUP="${M_COUP:-1024}"
N_TEST_COUP="${N_TEST_COUP:-4000}"
# Iteration CAP only: each cell trains to convergence (stop when the
# relative training-loss improvement over the last 10 recorded evals
# drops below 1%), so most cells stop well before this. The step size
# is NOT set here -- it is calibrated per arm from that arm's own NTK
# spectrum inside run_coupling.py, because re-pairing changes the Gram
# matrix and a shared eta would silently change the effective step.
T_COUP="${T_COUP:-50000}"

NUM_COUPLINGS=${#COUPLING_LIST[@]}
NUM_NETS=${#NET_LIST[@]}
NUM_D_COUP=${#D_LIST_COUP[@]}
COUP_GRID_SIZE=$(( NUM_COUPLINGS * NUM_NETS * NUM_D_COUP * N_SEEDS_COUP ))

# Resolve idx -> (COUPLING, NET, D, SEED, COUP_RUN_NAME).
resolve_coup_cell() {
  local idx=$1
  SEED=$(( idx % N_SEEDS_COUP ));   idx=$(( idx / N_SEEDS_COUP ))
  local d_i=$(( idx % NUM_D_COUP )); idx=$(( idx / NUM_D_COUP ))
  local net_i=$(( idx % NUM_NETS )); idx=$(( idx / NUM_NETS ))
  local coupling_i=$idx
  COUPLING=${COUPLING_LIST[$coupling_i]}
  NET=${NET_LIST[$net_i]}
  D=${D_LIST_COUP[$d_i]}
  COUP_RUN_NAME="coup_${COUPLING}_${NET}_d${D}_n${N_COUP}_m${M_COUP}_seed${SEED}"
}

coup_cell_indices() {
  if [[ -n "${IDX:-}" ]]; then
    echo "$IDX"
  elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    local n=${SLURM_ARRAY_TASK_COUNT:-1}
    local k=$SLURM_ARRAY_TASK_ID
    if (( n <= 1 )); then
      echo "$k"
    else
      seq "$k" "$n" $(( COUP_GRID_SIZE - 1 ))
    fi
  else
    seq 0 $(( COUP_GRID_SIZE - 1 ))
  fi
}
