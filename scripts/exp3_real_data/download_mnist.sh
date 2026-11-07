#!/usr/bin/env bash
#SBATCH --job-name=fmnn-download-mnist
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=jqfan
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qy1448@princeton.edu

# Download MNIST from HuggingFace to data/application/mnist/.
#
# Dataset:  ylecun/mnist
# Splits:   train (60 000 images), test (10 000 images)
# Image:    28×28 greyscale, stored in column 'image' as PNG bytes
# Output:   data/application/mnist/{train,test}.parquet
#
# No column normalisation needed: HuggingFace MNIST already uses 'image'.
#
# Usage:
#   sbatch scripts/exp3_real_data/download_mnist.sh
#   bash   scripts/exp3_real_data/download_mnist.sh    (local, no GPU needed)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
source scripts/_env.sh
mkdir -p logs data/application

HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export HF_HOME
echo "[download_mnist] HF_HOME=$HF_HOME"

python exp/exp3_real_data/download_hf_dataset.py \
  --dataset     ylecun/mnist \
  --output_name mnist \
  --save_format parquet \
  "$@"
