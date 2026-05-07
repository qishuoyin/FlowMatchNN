#!/usr/bin/env bash
#SBATCH --job-name=fmnn-download-fashion-mnist
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

# Download Fashion-MNIST to data/application/fashion_mnist/.
#
# Source:  https://github.com/zalandoresearch/fashion-mnist (Zalando S3)
#          Downloads the 4 IDX binary files directly — no HuggingFace needed.
# Splits:  train (60 000 images), test (10 000 images)
# Image:   28×28 greyscale, stored as PNG bytes
# Output:  data/application/fashion_mnist/{train,test}.parquet
#
# Schema matches MNIST / CIFAR-10 parquets (image: struct<bytes, path>).
#
# NOTE: Run from a login node (outbound HTTP required).
#       Compute nodes on this cluster have no internet access.
#
# Usage:
#   sbatch scripts/download_fashion_mnist.sh
#   bash   scripts/download_fashion_mnist.sh    (local, no GPU needed)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source scripts/_env.sh
mkdir -p logs data/application

python exp/download_fashion_mnist_idx.py "$@"
