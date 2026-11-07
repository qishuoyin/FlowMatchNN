#!/usr/bin/env bash
# PCA-sweep grid definition: (dataset, d_pca, n, seed) at a
# fixed width m and a fixed iteration cap T. Sourced by
# run_pca_sweep.sh and run_pca_sweep_vis.sh.
#
# The (d_pca, n) grid is RAGGED on purpose: d_pca = 2, 4 already
# saturate at n = 2000 (the representation, not the sample size, is the
# bottleneck there), so the expensive n = 8000 cells are spent only on
# d_pca >= 8 where the n-dependence is the point. The pairs are
# therefore enumerated explicitly in DN_LIST_PCA rather than taken as a
# cross product; DPCA_LIST_PCA / N_LIST_PCA are the marginal axes and
# exist only so the vis driver can name them (it skips the absent
# combinations with a warning).
#
# Index mapping (dataset outermost / seed innermost):
#   idx = ds_i * (N_DN * N_SEEDS_PCA) + dn_i * N_SEEDS_PCA + seed
#
# Default: 2 * 16 * 3 = 96 cells (~6.8 GPU-h; the dominant per-cell cost
# is decoding the parquet split and fitting PCA, ~15 s, not the flow).
# Needs the dataset parquets
# (scripts/exp3_real_data/download_{mnist,fashion_mnist}.sh).

DATASET_LIST_PCA=(mnist fashion_mnist)

# "d_pca:n" pairs, d_pca outer / n inner.
DN_LIST_PCA=(
  2:500   2:2000
  4:500   4:2000
  8:500   8:2000   8:8000
  16:500  16:2000  16:8000
  32:500  32:2000  32:8000
  64:500  64:2000  64:8000
)

# Marginal axes (vis only).
DPCA_LIST_PCA=(2 4 8 16 32 64)
N_LIST_PCA=(500 2000 8000)

N_SEEDS_PCA="${N_SEEDS_PCA:-3}"
M_PCA="${M_PCA:-1024}"
# Iteration CAP only: run_pca_sweep.py stops on the loss-plateau rule
# and records the achieved improvement ratio as `converged_ratio`.
T_PCA="${T_PCA:-2000}"

N_DS=${#DATASET_LIST_PCA[@]}
N_DN=${#DN_LIST_PCA[@]}
PCA_GRID_SIZE=$(( N_DS * N_DN * N_SEEDS_PCA ))

resolve_pca_cell() {
  local idx=$1
  SEED=$(( idx % N_SEEDS_PCA )); idx=$(( idx / N_SEEDS_PCA ))
  local dn_i=$(( idx % N_DN ));  idx=$(( idx / N_DN ))
  local ds_i=$idx
  DATASET=${DATASET_LIST_PCA[$ds_i]}
  local dn=${DN_LIST_PCA[$dn_i]}
  DPCA=${dn%%:*}
  N_PCA=${dn##*:}
  PCA_RUN_NAME="pca_${DATASET}_dpca${DPCA}_n${N_PCA}_m${M_PCA}_seed${SEED}"
}

pca_cell_indices() {
  if [[ -n "${IDX:-}" ]]; then
    echo "$IDX"
  elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    local n=${SLURM_ARRAY_TASK_COUNT:-1}
    local k=$SLURM_ARRAY_TASK_ID
    if (( n <= 1 )); then
      echo "$k"
    else
      seq "$k" "$n" $(( PCA_GRID_SIZE - 1 ))
    fi
  else
    seq 0 $(( PCA_GRID_SIZE - 1 ))
  fi
}
