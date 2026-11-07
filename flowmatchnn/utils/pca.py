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


# ---------------------------------------------------------------------------
# Pixel-space evaluation (PCA-dimension sweep)
# ---------------------------------------------------------------------------
#
# The legacy real-data pipeline evaluates W_1 in PCA space, which is not
# comparable across different d_pca. These helpers map samples back to
# pixel space and evaluate the sliced W_1 there, plus the PCA
# "reconstruction floor" — the sliced W_1 between PCA-reconstructed test
# images and the raw test images, i.e. the error PCA compression alone
# already commits before any flow error.


def reconstruct_clipped(Z: np.ndarray, pca: PCAModel) -> np.ndarray:
    """Inverse-transform Z (n, k) and clip to the [0, 1] pixel range."""
    return np.clip(inverse_transform(Z, pca), 0.0, 1.0)


def pixel_sliced_w1(
    samples_pca,
    pca: PCAModel,
    x_test_pixel,
    directions,
) -> float:
    """
    Sliced W_1 in PIXEL space between generated samples (given in PCA
    coordinates) and raw test images.

    Args:
        samples_pca: (n, k) torch tensor or numpy array of PCA codes.
        pca: fitted PCAModel with k components.
        x_test_pixel: (N, d_flat) torch tensor of raw test images
            in [0, 1].
        directions: (K, d_flat) torch tensor of unit directions (drawn
            once by the caller so values are comparable across d_pca).

    Returns:
        float sliced-W_1 value.
    """
    import torch
    from . import metrics as _metrics

    z = (
        samples_pca.detach().cpu().numpy()
        if hasattr(samples_pca, "detach") else np.asarray(samples_pca)
    )
    rec = reconstruct_clipped(z, pca)
    rec_t = torch.from_numpy(rec).to(
        device=x_test_pixel.device, dtype=x_test_pixel.dtype
    )
    return float(_metrics.sliced_wasserstein_1(
        rec_t, x_test_pixel, directions=directions,
    ))


def pca_reconstruction_w1(
    pca: PCAModel,
    x_test_pixel,
    directions,
) -> float:
    """
    Sliced W_1 in pixel space between PCA-reconstructed test images
    and the raw test images -- the error PCA compression alone commits
    before any flow error.

    NOTE ON INTERPRETATION. This is a *reference level*, not a lower
    bound: sliced W_1 along random pixel directions is not monotone
    under adding variance, so a model whose samples carry extra
    variance in the discarded subspace can score below it. Always
    report it alongside `holdout_floor_w1`, which IS the honest
    finite-sample floor.
    """
    import torch
    from . import metrics as _metrics

    x_np = x_test_pixel.detach().cpu().numpy()
    rec = reconstruct_clipped(transform(x_np, pca), pca)
    rec_t = torch.from_numpy(rec).to(
        device=x_test_pixel.device, dtype=x_test_pixel.dtype
    )
    return float(_metrics.sliced_wasserstein_1(
        rec_t, x_test_pixel, directions=directions,
    ))


def holdout_floor_w1(n, x_test_pixel, directions, *, generator=None):
    """
    Honest finite-sample floor: sliced W_1 in pixel space between n
    REAL held-out images and the evaluation reference.

    No generative model evaluated on n samples against this reference
    can do better than this in expectation, so it is the correct
    "perfect model" level to draw on a pixel-space W_1 figure.

    Args:
        n: number of real images to draw (match the model's n_gen).
        x_test_pixel: (N, d_flat) torch tensor of raw held-out images.
        directions: (K, d_flat) unit directions, shared across cells.
        generator: torch.Generator for the subsample.
    """
    import torch
    from . import metrics as _metrics

    idx = torch.randperm(
        x_test_pixel.shape[0], generator=generator,
        device=x_test_pixel.device,
    )[:n]
    return float(_metrics.sliced_wasserstein_1(
        x_test_pixel[idx], x_test_pixel, directions=directions,
    ))


def standardize_codes(codes_train, *codes_other):
    """
    Rescale PCA codes to unit per-component variance (PCA-dimension sweep).

    The conditional coupling P(z_0 | z_1) = N(z_1, I) adds noise of
    variance 1 to EVERY retained coordinate. Raw PCA codes have
    component variances that decay steeply with the component index, so
    on raw codes the source's noise-to-signal ratio grows with d_pca
    and a sweep over d_pca confounds dimension with corruption level.
    Standardising first makes the coupling's SNR identical across
    coordinates and across d_pca, so the sweep is a sweep in dimension.

    Returns (scale, codes_train_std, *codes_other_std); undo with
    `z * scale`.
    """
    scale = codes_train.std(axis=0)
    scale[scale <= 0] = 1.0
    out = [scale, codes_train / scale]
    out += [c / scale for c in codes_other]
    return tuple(out)
