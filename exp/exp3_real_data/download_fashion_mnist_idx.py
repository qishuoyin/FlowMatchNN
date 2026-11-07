"""
Download Fashion-MNIST from the official Zalando S3 bucket (IDX binary format),
convert to PNG-encoded parquet, and save to data/application/fashion_mnist/.

Fallback for HPC environments where compute nodes lack internet access.
Run from a login node or any machine with outbound HTTP access.

Dataset source: https://github.com/zalandoresearch/fashion-mnist
  train-images: 60 000 × 28×28 greyscale
  test-images:  10 000 × 28×28 greyscale
  10 classes (T-shirt, Trouser, Pullover, Dress, Coat,
               Sandal, Shirt, Sneaker, Bag, Ankle boot)

Output schema matches the HuggingFace parquet used for MNIST / CIFAR-10:
  image  struct<bytes: binary, path: string>   PNG-encoded 28×28 greyscale
  label  int64

Usage:
    python exp/download_fashion_mnist_idx.py
    python exp/download_fashion_mnist_idx.py --out_dir /custom/path
"""

import argparse
import gzip
import io
import struct
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

_BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"

_SPLITS = {
    "train": {
        "images": f"{_BASE_URL}/train-images-idx3-ubyte.gz",
        "labels": f"{_BASE_URL}/train-labels-idx1-ubyte.gz",
    },
    "test": {
        "images": f"{_BASE_URL}/t10k-images-idx3-ubyte.gz",
        "labels": f"{_BASE_URL}/t10k-labels-idx1-ubyte.gz",
    },
}

_IMAGE_SCHEMA = pa.struct([
    pa.field("bytes", pa.binary()),
    pa.field("path",  pa.string()),
])

_TABLE_SCHEMA = pa.schema([
    pa.field("image", _IMAGE_SCHEMA),
    pa.field("label", pa.int64()),
])


def _download(url: str) -> bytes:
    print(f"  Downloading {url} …", flush=True)
    with urllib.request.urlopen(url, timeout=180) as resp:
        data = resp.read()
    print(f"    received {len(data):,} bytes", flush=True)
    return data


def _parse_idx_images(gz_data: bytes) -> np.ndarray:
    """IDX3 (gzipped) → uint8 ndarray (N, H, W)."""
    with gzip.open(io.BytesIO(gz_data)) as f:
        raw = f.read()
    magic, n, h, w = struct.unpack_from(">IIII", raw, 0)
    assert magic == 0x00000803, f"Unexpected IDX3 magic: {magic:#010x}"
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, h, w)


def _parse_idx_labels(gz_data: bytes) -> np.ndarray:
    """IDX1 (gzipped) → int64 ndarray (N,)."""
    with gzip.open(io.BytesIO(gz_data)) as f:
        raw = f.read()
    magic, n = struct.unpack_from(">II", raw, 0)
    assert magic == 0x00000801, f"Unexpected IDX1 magic: {magic:#010x}"
    return np.frombuffer(raw, dtype=np.uint8, offset=8).astype(np.int64)


def _to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _save_split(images: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    print(f"  Encoding {len(images)} images to PNG …", flush=True)
    png_bytes = [_to_png_bytes(img) for img in images]

    image_col = pa.StructArray.from_arrays(
        [
            pa.array(png_bytes, type=pa.binary()),
            pa.array([""] * len(png_bytes), type=pa.string()),
        ],
        names=["bytes", "path"],
    )
    table = pa.table(
        {"image": image_col, "label": pa.array(labels, type=pa.int64())},
        schema=_TABLE_SCHEMA,
    )
    pq.write_table(table, str(out_path))
    print(f"  Saved → {out_path}  ({len(images)} rows)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: data/application/fashion_mnist/ "
             "relative to the repo root).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else (
        repo_root / "data" / "application" / "fashion_mnist"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, urls in _SPLITS.items():
        print(f"\n[download_fashion_mnist_idx] split={split}")
        images = _parse_idx_images(_download(urls["images"]))
        labels = _parse_idx_labels(_download(urls["labels"]))
        print(f"  parsed: images={images.shape}  labels={labels.shape}")
        _save_split(images, labels, out_dir / f"{split}.parquet")

    print("\n[download_fashion_mnist_idx] Done.")


if __name__ == "__main__":
    main()
