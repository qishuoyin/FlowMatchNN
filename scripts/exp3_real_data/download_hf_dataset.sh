#!/usr/bin/env bash
#SBATCH --job-name=fmnn-download
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=jqfan
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qy1448@princeton.edu

# Generic HuggingFace dataset downloader.  For known datasets prefer the
# dedicated per-dataset scripts which pre-fill the correct parameters:
#
#   scripts/exp3_real_data/download_mnist.sh      MNIST  (ylecun/mnist,  image col: image)
#
# Use this script for any other HuggingFace dataset:
#   DATASET=owner/name sbatch scripts/exp3_real_data/download_hf_dataset.sh
#   sbatch scripts/exp3_real_data/download_hf_dataset.sh --dataset owner/name
#   bash   scripts/exp3_real_data/download_hf_dataset.sh --dataset owner/name
#
# If the dataset stores images under a non-standard column name, add:
#   --normalize_col <old_col_name>
# to rename it to 'image' in the saved parquet.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

source scripts/_env.sh

mkdir -p logs data/application

# Redirect HuggingFace cache to scratch to avoid filling the home-dir quota.
# Override at submit time: HF_HOME=/path sbatch scripts/exp3_real_data/download_hf_dataset.sh
HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export HF_HOME
echo "[download_hf_dataset] HF_HOME=$HF_HOME"

# ---------------------------------------------------------------------------
# Editable args. Override at submit time:
#     DATASET=owner/name sbatch scripts/exp3_real_data/download_hf_dataset.sh
#     sbatch scripts/exp3_real_data/download_hf_dataset.sh --dataset owner/name  # also works
# ---------------------------------------------------------------------------
DATASET="${DATASET:-}"
SAVE_FORMAT="${SAVE_FORMAT:-parquet}"
NORMALIZE_COL="${NORMALIZE_COL:-}"

if [[ -z "$DATASET" ]]; then
  echo "[download_hf_dataset] ERROR: set DATASET=owner/name or pass --dataset." >&2
  echo "  Known datasets have dedicated scripts:" >&2
  echo "    bash scripts/exp3_real_data/download_mnist.sh" >&2
  exit 1
fi

NORMALIZE_FLAG=()
[[ -n "$NORMALIZE_COL" ]] && NORMALIZE_FLAG=(--normalize_col "$NORMALIZE_COL")

echo "[download_hf_dataset] dataset=$DATASET save_format=$SAVE_FORMAT"
python exp/exp3_real_data/download_hf_dataset.py \
  --dataset     "$DATASET" \
  --save_format "$SAVE_FORMAT" \
  "${NORMALIZE_FLAG[@]}" \
  "$@"
