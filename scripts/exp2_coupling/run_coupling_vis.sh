#!/usr/bin/env bash
#SBATCH --job-name=fmnn-coup-vis
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=ailab
#SBATCH --account=jqfan
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qy1448@princeton.edu

# E4 figure (Fig E4) + coupling_table.csv. CPU-only — can be run
# locally:
#   bash scripts/exp2_coupling/run_coupling_vis.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

source scripts/_env.sh
mkdir -p logs

source scripts/exp2_coupling/_grid.sh
FIGS_DIR="${FIGS_DIR:-figs/exp2_coupling}"

python exp/exp2_coupling/run_coupling_vis.py \
  --couplings "${COUPLING_LIST[@]}" \
  --nets      "${NET_LIST[@]}" \
  --d_list    "${D_LIST_COUP[@]}" \
  --n         "$N_COUP" \
  --m         "$M_COUP" \
  --n_seeds   "$N_SEEDS_COUP" \
  --figs_dir  "$FIGS_DIR" \
  "$@"
