#!/usr/bin/env bash
#SBATCH --job-name=fmnn-ntk-vis
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

# NTK figures (rate overlay, width error, spectrum) + summary table.
# CPU-only — can also just be run locally:
#   bash scripts/exp1_optimization/run_ntk_vis.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

source scripts/_env.sh
mkdir -p logs

source scripts/exp1_optimization/_ntk_grid.sh
FIGS_DIR="${FIGS_DIR:-figs/exp1_optimization}"

python exp/exp1_optimization/run_ntk_vis.py \
  --d_list        "${D_LIST_NTK[@]}" \
  --n_list        "${N_LIST_NTK[@]}" \
  --m_list        "${M_LIST_NTK[@]}" \
  --eta_mult_list "${ETA_MULT_LIST[@]}" \
  --n_seeds       "$N_SEEDS_NTK" \
  --figs_dir      "$FIGS_DIR" \
  "$@"
