"""
Thin numpy PCA: fit on training images, transform/inverse-transform.

Used by the real-data experiments to compress images from d_original
(e.g. 784 for MNIST 28×28) down to d_pca = d_original // 100 dimensions
before passing them through the flow-matching pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAModel:
    components: np.ndarray   # (k, d_original)  — principal directions
    mean: np.ndarray         # (d_original,)     — training-set mean
    explained_var: np.ndarray  # (k,)            — variance per component


def fit(X: np.ndarray, n_components: int) -> PCAModel:
    """
    Fit PCA on X (n_samples, d_original).

    Centres X, runs a full SVD, and retains the top n_components
    principal directions. Normalisation is per sklearn convention:
    explained variance = S^2 / (n - 1).

    Args:
        X: float array of shape (n, d_original).
        n_components: number of components to keep.

    Returns:
        PCAModel with .components, .mean, .explained_var.
    """
    mean = X.mean(axis=0)
    X_c = X - mean
    _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    return PCAModel(
        components=Vt[:n_components].astype(np.float32),
        mean=mean.astype(np.float32),
        explained_var=(S[:n_components] ** 2 / max(len(X) - 1, 1)).astype(np.float32),
    )


def transform(X: np.ndarray, pca: PCAModel) -> np.ndarray:
    """
    Project X (n, d_original) onto PCA components.

    Returns:
        Z of shape (n, k).
    """
    return (X - pca.mean) @ pca.components.T


def inverse_transform(Z: np.ndarray, pca: PCAModel) -> np.ndarray:
    """
    Reconstruct pixel values from PCA codes Z (n, k).

    Returns:
        X_rec of shape (n, d_original).  Values are in the original
        pixel scale (not clamped — caller must clip if desired).
    """
    return Z @ pca.components + pca.mean
