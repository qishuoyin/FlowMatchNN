"""
Parquet image loading shared by the revision experiments (the coupling comparison
and the PCA sweep).

Mirrors the loader embedded in exp/exp3_real_data/run_real_train.py —
duplicated here (rather than imported across exp/ folders) so package
code never depends on experiment entry points. Pixel values are scaled
to [0, 1].
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import numpy as np


def _decode_row(row) -> np.ndarray:
    """
    Decode one value from the image column.

    Handles:
      - dict {'bytes': <bytes>}       — HuggingFace PNG/JPEG
      - numpy array (H, W) or (H,W,C)
      - raw bytes
    """
    if isinstance(row, dict) and "bytes" in row:
        from PIL import Image
        return np.array(Image.open(io.BytesIO(row["bytes"])).convert("L"),
                        dtype=np.float32)
    if isinstance(row, np.ndarray):
        return row.astype(np.float32)
    if isinstance(row, (bytes, bytearray)):
        from PIL import Image
        return np.array(Image.open(io.BytesIO(row)).convert("L"),
                        dtype=np.float32)
    raise TypeError(f"Unrecognised image format: {type(row)}")


def load_split(
    parquet_path: Path | str, image_col: str = "image"
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load a parquet split → (X_flat, (img_h, img_w)).

    X_flat has shape (N, H*W) with pixel values in [0, 1].
    img_h, img_w are auto-detected from the first row.
    """
    import pandas as pd
    df = pd.read_parquet(str(parquet_path))
    first = _decode_row(df[image_col].iloc[0])
    if first.ndim == 1:
        raise ValueError(
            f"Image in {parquet_path} is already 1-D; "
            "cannot auto-detect (img_h, img_w). Check the dataset format."
        )
    if first.ndim == 3:
        first = first.mean(axis=-1)
    img_h, img_w = first.shape

    imgs = []
    for row in df[image_col]:
        arr = _decode_row(row)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        imgs.append(arr.ravel())

    return (np.stack(imgs, axis=0) / 255.0).astype(np.float32), (img_h, img_w)
