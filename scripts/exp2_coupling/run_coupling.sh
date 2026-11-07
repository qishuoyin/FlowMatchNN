#!/usr/bin/env bash
#SBATCH --job-name=fmnn-coup
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --account=jqfan
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qy1448@princeton.edu

# Coupling comparison over the (coupling, net, d,
# seed) grid in _grid.sh -- 90 cells, 6 shards of 15.
#
# Each cell draws its (x_1, x_0, tau) once from the seed, applies one
# pairing, calibrates eta from that arm's own NTK spectrum, trains to
# convergence, and writes final/exact W1 plus the do-nothing and
# finite-sample-floor baselines. Simulation only -- no data download
# prerequisite.
#
# Submit options:
#   sbatch scripts/exp2_coupling/run_coupling.sh        # full grid
#   IDX=0 bash scripts/exp2_coupling/run_coupling.sh    # single cell

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

source scripts/_env.sh
mkdir -p logs results

source scripts/exp2_coupling/_grid.sh

for IDXC in $(coup_cell_indices); do
  resolve_coup_cell "$IDXC"
  echo "[run_coupling] cell idx=$IDXC coupling=$COUPLING net=$NET d=$D seed=$SEED"
  python exp/exp2_coupling/run_coupling.py \
    --coupling "$COUPLING" --net "$NET" \
    --d "$D" --n "$N_COUP" --m "$M_COUP" \
    --n_test "$N_TEST_COUP" --T "$T_COUP" --seed "$SEED" \
    "$@"
done
