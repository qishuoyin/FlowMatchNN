"""
Sliced Wasserstein-1 distance between two empirical distributions.

The manuscript (Theorem 3.4) bounds W_1(P_1, hat P_1), so we want to
evaluate convergence in the same metric. Computing the multivariate
W_1 in R^d exactly costs O(n^3) (linear program), which is impractical
inside a training loop swept over d up to 100 and n_test up to 10000.
The standard practical surrogate -- and the one the optimal-transport
literature uses for high-dimensional W_1 estimation -- is the
*sliced* W_1: average the 1-D Wasserstein-1 distance along K random
unit-vector projections of R^d,

    SW_1(P, Q) = (1 / K) sum_{k=1}^K W_1(u_k . P, u_k . Q),
    u_k iid uniform on the unit sphere S^{d-1}.

SW_1 is a metric, equals 0 iff P == Q, and is a lower bound on the
true multivariate W_1; it agrees with the per-axis estimator only in
the special case where the random directions are restricted to the
standard basis. Each 1-D W_1 has a closed form via the CDF
formulation (no LP needed),

    W_1(p, q) = integral_R | F_p(t) - F_q(t) | dt,

evaluated as a piecewise-constant integrand on the sorted union of
the two sample sets so that arbitrary unequal sample sizes are
handled exactly (here n_train model samples vs n_test reference
samples).

Used by Exp 2 to track how close the model distribution is to P_1
at each evaluated training step.
"""

from __future__ import annotations

from typing import Optional

import torch


def random_unit_directions(
    n_proj: int,
    d: int,
    *,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample `n_proj` uniformly random unit vectors on the sphere S^{d-1}.

    Standard trick: a Gaussian draw normalised by its L2 norm is
    uniform on the sphere by rotation invariance of N(0, I_d).

    Args:
        n_proj: number of directions K to draw.
        d: ambient dimension.
        generator: optional torch.Generator for reproducible draws.
            Pre-drawing the directions once and reusing them across
            evaluations removes projection-noise jitter from the
            W_1-vs-t trajectory; the training loop does exactly that.
        device: torch device (defaults to CPU when omitted).
        dtype: tensor dtype.

    Returns:
        Tensor of shape (n_proj, d), each row of unit L2 norm.
    """
    g = torch.randn(
        n_proj, d, generator=generator, device=device, dtype=dtype,
    )
    return g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def sliced_wasserstein_1(
    samples_p: torch.Tensor,
    samples_q: torch.Tensor,
    *,
    n_proj: int = 128,
    directions: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sliced Wasserstein-1 distance SW_1(P, Q) between two sample sets.

    Each direction's 1-D W_1 is computed via the CDF formulation,
    summed over the sorted union of the two projected sample sets.
    Supports unequal sample sizes (the typical case here:
    `samples_p` has n_train rows, `samples_q` has n_test rows).

    Args:
        samples_p: (N_p, d) sample matrix (e.g. model samples).
        samples_q: (N_q, d) sample matrix (e.g. reference P_1 samples).
        n_proj: number of random directions K (only used when
            `directions` is None).
        directions: optional (K, d) precomputed unit-vector directions.
            When provided, `n_proj` and `generator` are ignored. The
            training loop pre-draws once so the trace is comparable
            across t (no projection-noise jitter).
        generator: optional torch.Generator for fresh direction draws.

    Returns:
        Scalar non-negative tensor. Same device / dtype as `samples_p`.
    """
    if samples_p.ndim != 2 or samples_q.ndim != 2:
        raise ValueError("Expected 2-D sample arrays")
    if samples_p.shape[1] != samples_q.shape[1]:
        raise ValueError(
            f"Coord dim mismatch: {samples_p.shape[1]} vs {samples_q.shape[1]}"
        )

    device = samples_p.device
    dtype = samples_p.dtype
    d = samples_p.shape[1]

    if directions is None:
        directions = random_unit_directions(
            n_proj, d,
            generator=generator, device=device, dtype=dtype,
        )
    elif directions.shape[1] != d:
        raise ValueError(
            f"directions has dim {directions.shape[1]}, expected {d}"
        )

    # Project samples onto each direction. Shapes: (N_p, K) and (N_q, K).
    p_proj = samples_p @ directions.T
    q_proj = samples_q @ directions.T
    return _sliced_w1_from_projections(p_proj, q_proj)


def _sliced_w1_from_projections(
    p_proj: torch.Tensor, q_proj: torch.Tensor,
) -> torch.Tensor:
    """
    Closed-form mean over K projections of the 1-D W_1 distance.

    `p_proj` and `q_proj` are projected sample matrices of shapes
    (N_p, K) and (N_q, K) -- each column is one projection. We
    sort each column independently, then evaluate the CDF integral
    on the sorted union along that column.
    """
    n_p, n_proj = p_proj.shape
    n_q = q_proj.shape[0]
    dtype = p_proj.dtype

    # Per-column sort. Resulting shapes: (N_p, K), (N_q, K).
    p_sorted, _ = torch.sort(p_proj, dim=0)
    q_sorted, _ = torch.sort(q_proj, dim=0)

    # Sorted union per projection: stack and sort, shape (N_p+N_q, K).
    all_vals = torch.sort(
        torch.cat([p_sorted, q_sorted], dim=0), dim=0
    ).values

    # `torch.searchsorted` searches along the *last* dim, so we move
    # the projection axis to the front. Shapes become:
    #   p_sorted_K, q_sorted_K : (K, N_p), (K, N_q)
    #   all_vals_K             : (K, N_p + N_q)
    p_sorted_K = p_sorted.transpose(0, 1).contiguous()
    q_sorted_K = q_sorted.transpose(0, 1).contiguous()
    all_vals_K = all_vals.transpose(0, 1).contiguous()

    # CDF F_p, F_q evaluated at the left edges of every interval. Using
    # right=True so a tied value is counted as already accumulated, which
    # matches scipy.stats.wasserstein_distance. `.contiguous()` because
    # torch.searchsorted complains about strided value tensors.
    edges_left = all_vals_K[:, :-1].contiguous()            # (K, M-1)
    deltas = all_vals_K[:, 1:] - edges_left                 # (K, M-1)
    cdf_p = torch.searchsorted(p_sorted_K, edges_left, right=True).to(dtype) / n_p
    cdf_q = torch.searchsorted(q_sorted_K, edges_left, right=True).to(dtype) / n_q

    # Per-projection W_1, then average over projections.
    w1_per_proj = (deltas * (cdf_p - cdf_q).abs()).sum(dim=-1)  # (K,)
    return w1_per_proj.mean()


# ---------------------------------------------------------------------------
# Exact multivariate W_1 (sampling-error evaluation)
# ---------------------------------------------------------------------------


def wasserstein1_exact(
    samples_p: torch.Tensor,
    samples_q: torch.Tensor,
    *,
    max_points: int = 5000,
) -> float:
    """
    Exact multivariate Wasserstein-1 distance between two sample sets,
    solved as a discrete optimal-transport LP via POT's network-simplex
    (`ot.emd2`) with Euclidean ground cost.

    Complements the sliced estimator above: used in low dimension
    (d <= 5) to cross-check the sliced estimator and to
    report the exact metric the theory bounds.

    Args:
        samples_p: (N_p, d) sample matrix.
        samples_q: (N_q, d) sample matrix. Sizes may differ (uniform
            weights on each side).
        max_points: guard on the LP size — the cost matrix is
            N_p x N_q, so both sides must stay <= max_points.

    Returns:
        Non-negative float.
    """
    import numpy as np
    import ot

    n_p, n_q = samples_p.shape[0], samples_q.shape[0]
    if max(n_p, n_q) > max_points:
        raise ValueError(
            f"wasserstein1_exact: {max(n_p, n_q)} points > max_points = "
            f"{max_points}; subsample or use sliced_wasserstein_1."
        )
    xp = samples_p.detach().cpu().numpy().astype(np.float64)
    xq = samples_q.detach().cpu().numpy().astype(np.float64)
    M = ot.dist(xp, xq, metric="euclidean")
    a = np.full(n_p, 1.0 / n_p)
    b = np.full(n_q, 1.0 / n_q)
    return float(ot.emd2(a, b, M, numItermax=1_000_000))
