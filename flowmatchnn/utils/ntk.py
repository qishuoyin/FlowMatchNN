"""
Closed-form and empirical NTK Gram matrices for the two-layer ReLU
velocity network (Theorem 3.2).

The paper's infinite-width Gram matrix factorizes as

    H^(inf) = H_tilde  (x)  I_d          (Kronecker product)

with H_tilde an (n, n) matrix whose entries are expectations over
w ~ N(0, kappa^2 I) of arc-cosine-kernel type. Writing
z_i = [x_tau_i; tau_i; x_0_i] in R^{2d+1} and theta_ij for the angle
between z_i and z_j:

    B_ij = E[(w.z_i)(w.z_j) 1{w.z_i>0, w.z_j>0}]
         = kappa^2 ||z_i|| ||z_j|| (sin th + (pi - th) cos th) / (2 pi)
    C_ij = (z_i . z_j) P(w.z_i>0, w.z_j>0)
         = (z_i . z_j) (pi - th) / (2 pi)
    H_tilde = B + C.

B is the infinite-width kernel of the trainable output weights a_r
(relu(w.z_i) relu(w.z_j) = (w.z_i)(w.z_j) 1{both > 0}) and C is the
kernel of the trainable first-layer weights w_r (E over a_r ~ N(0, I_d)
gives the delta_ab factor absorbed into the Kronecker I_d).

All linear algebra here is float64 — lambda_min(H_tilde) can be small
and the (1 - eta * lambda)^t powers amplify rounding.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Augmented inputs
# ---------------------------------------------------------------------------


def augment(
    x_tau: torch.Tensor, tau: torch.Tensor, x_0: torch.Tensor
) -> torch.Tensor:
    """
    Stack the network inputs into z_i = [x_tau_i; tau_i; x_0_i].

    Shapes: x_tau (n, d), tau (n,) or (n, 1), x_0 (n, d).
    Returns (n, 2d+1) in float64.
    """
    if tau.ndim == 1:
        tau = tau.unsqueeze(-1)
    return torch.cat([x_tau, tau, x_0], dim=-1).to(torch.float64)


# ---------------------------------------------------------------------------
# Closed-form H_tilde
# ---------------------------------------------------------------------------


def gram_htilde(
    x_tau: torch.Tensor,
    tau: torch.Tensor,
    x_0: torch.Tensor,
    *,
    kappa: float,
) -> torch.Tensor:
    """
    Closed-form (n, n) Gram matrix H_tilde = B + C in float64.

    The angle theta_ij depends only on directions, so kappa enters only
    through the kappa^2 prefactor of B.
    """
    z = augment(x_tau, tau, x_0)
    norms = z.norm(dim=-1).clamp_min(1e-30)          # (n,)
    G = z @ z.T                                       # (n, n)
    norms_outer = norms[:, None] * norms[None, :]
    cos = (G / norms_outer).clamp(-1.0, 1.0)
    theta = torch.arccos(cos)
    pi = math.pi

    B = kappa ** 2 * norms_outer * (
        torch.sin(theta) + (pi - theta) * cos
    ) / (2.0 * pi)
    C = G * (pi - theta) / (2.0 * pi)
    H = B + C
    return 0.5 * (H + H.T)  # exact symmetry for eigh


def monte_carlo_htilde(
    x_tau: torch.Tensor,
    tau: torch.Tensor,
    x_0: torch.Tensor,
    *,
    kappa: float,
    n_samples: int,
    generator: torch.Generator | None = None,
    chunk: int = 100_000,
) -> torch.Tensor:
    """
    Monte-Carlo estimate of H_tilde over w ~ N(0, kappa^2 I).

    Test-only reference implementation for validating gram_htilde;
    O(n_samples * n^2) memory-chunked over the w draws.
    """
    z = augment(x_tau, tau, x_0)
    n, p = z.shape
    G = z @ z.T
    acc = torch.zeros((n, n), dtype=torch.float64, device=z.device)
    done = 0
    while done < n_samples:
        k = min(chunk, n_samples - done)
        w = kappa * torch.randn(
            k, p, generator=generator, device=z.device, dtype=torch.float64
        )
        proj = z @ w.T                                # (n, k)
        ind = (proj > 0).to(torch.float64)
        pb = proj * ind
        # B-part: (w.z_i)(w.z_j) 1{both>0};  C-part: (z_i.z_j) 1{both>0}
        acc += pb @ pb.T + G * (ind @ ind.T)
        done += k
    return acc / n_samples


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------


def eig_htilde(H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eigendecomposition H = U diag(evals) U^T (ascending evals), float64.

    Returns (evals (n,), U (n, n)).
    """
    evals, U = torch.linalg.eigh(H.to(torch.float64))
    return evals, U


def lambda_min_htilde(H: torch.Tensor) -> float:
    """Smallest eigenvalue of H_tilde (== lambda_min of H^(inf))."""
    return float(torch.linalg.eigvalsh(H.to(torch.float64))[0])


def kernel_residual_norms(
    evals: torch.Tensor,
    U: torch.Tensor,
    V_mat: torch.Tensor,
    eta: float,
    T: int,
) -> torch.Tensor:
    """
    ||(I - eta * H^(inf))^t V||_2 for t = 0..T, via one eigendecomposition.

    Because H^(inf) = H_tilde (x) I_d, powers act column-wise on the
    (n, d) matrix V_mat: with coef = U^T V_mat,

        ||(I - eta H)^t V||^2 = sum_k (1 - eta * evals_k)^{2t} ||coef_k||^2.

    Returns a float64 tensor of shape (T+1,).
    """
    coef_sq = (U.T.to(torch.float64) @ V_mat.to(torch.float64)) ** 2
    coef_sq = coef_sq.sum(dim=1)                       # (n,)
    factors = (1.0 - eta * evals.to(torch.float64))    # (n,)
    ts = torch.arange(T + 1, dtype=torch.float64, device=evals.device)
    # (T+1, n): factors^ (2t); guard |factor| -> log-space is unnecessary
    # for the T ~ O(1e3) used here.
    pow_ = factors[None, :] ** (2.0 * ts[:, None])
    return torch.sqrt((pow_ * coef_sq[None, :]).sum(dim=1))


def kernel_residual_stepper(
    evals: torch.Tensor,
    U: torch.Tensor,
    V_mat: torch.Tensor,
    eta: float,
):
    """
    Incremental evaluator of (I - eta H^(inf))^t V for use inside the
    training loop (one O(n^2 d) matmul per step instead of matrix powers).

    Returns a closure step() that, at its k-th call, returns the (n, d)
    float64 matrix (I - eta H_tilde)^k V_mat; the 0-th call returns V_mat.
    """
    U64 = U.to(torch.float64)
    coef = (U64.T @ V_mat.to(torch.float64)).clone()   # (n, d)
    factors = (1.0 - eta * evals.to(torch.float64))[:, None]
    state = {"first": True}

    def step() -> torch.Tensor:
        if state["first"]:
            state["first"] = False
        else:
            coef.mul_(factors)
        return U64 @ coef

    return step


# ---------------------------------------------------------------------------
# Finite-width empirical NTK at initialization
# ---------------------------------------------------------------------------


def empirical_ntk_gram(
    model,
    x_tau: torch.Tensor,
    tau: torch.Tensor,
    x_0: torch.Tensor,
    *,
    chunk_m: int = 4096,
    max_nd: int = 4096,
) -> torch.Tensor:
    """
    Exact finite-m NTK Gram matrix of the TwoLayerReLU model, (nd, nd).

    With z_i the augmented inputs, S = 1{Z W^T > 0}, R = relu(Z W^T):

        K[(i,a),(j,b)] = (1/m) [ (z_i.z_j) sum_r S_ir S_jr A_ra A_rb
                                 + delta_ab (R R^T)_ij ]

    (first term = trainable-W kernel, second = trainable-A kernel).
    Flattened index order is (i, a) -> i * d + a. Accumulated in chunks
    over neurons; float64.

    Memory guard: raises if n * d > max_nd — run this only on the small
    (n = 100) cells of the optimization grid.
    """
    z = augment(x_tau, tau, x_0)
    n = z.shape[0]
    W = model.W.detach().to(torch.float64)
    A = model.A.detach().to(torch.float64)
    m, d = A.shape
    nd = n * d
    if nd > max_nd:
        raise ValueError(
            f"empirical_ntk_gram: n*d = {nd} > max_nd = {max_nd}; restrict "
            "the empirical NTK check to small cells."
        )

    G = z @ z.T                                        # (n, n)
    KW = torch.zeros((nd, nd), dtype=torch.float64, device=z.device)
    RRT = torch.zeros((n, n), dtype=torch.float64, device=z.device)
    for lo in range(0, m, chunk_m):
        hi = min(lo + chunk_m, m)
        pre = z @ W[lo:hi].T                           # (n, mc)
        S = (pre > 0).to(torch.float64)
        RRT += torch.relu(pre) @ torch.relu(pre).T
        # P[i, a, r] = S[i, r] * A[r, a]  -> (n*d, mc)
        P = (S.unsqueeze(1) * A[lo:hi].T.unsqueeze(0)).reshape(nd, hi - lo)
        KW += P @ P.T
    KW /= m
    RRT /= m

    G_exp = G.repeat_interleave(d, dim=0).repeat_interleave(d, dim=1)
    KA = torch.kron(RRT, torch.eye(d, dtype=torch.float64, device=z.device))
    K = KW * G_exp + KA
    return 0.5 * (K + K.T)


def lambda_min_empirical(
    model,
    x_tau: torch.Tensor,
    tau: torch.Tensor,
    x_0: torch.Tensor,
    **kwargs,
) -> float:
    """lambda_min of the finite-m empirical NTK Gram matrix."""
    K = empirical_ntk_gram(model, x_tau, tau, x_0, **kwargs)
    return float(torch.linalg.eigvalsh(K)[0])


# ---------------------------------------------------------------------------
# Step-size calibration from the kernel spectrum
# ---------------------------------------------------------------------------


def lambda_max_htilde(H: torch.Tensor) -> float:
    """Largest eigenvalue of H_tilde."""
    return float(torch.linalg.eigvalsh(H.to(torch.float64))[-1])


def eta_spectral(
    lambda_max: float,
    *,
    n: int,
    safety: float = 0.5,
    mean_loss: bool = True,
) -> float:
    """
    Step size calibrated to the kernel spectrum.

    Theorem 3.2 states the condition eta < 2 lambda_min / (n^2 d^2),
    which comes from bounding ||H^(inf)||_2 by the worst-case surrogate
    n*d*max||A||^2 in the proof of the T1 term. Recombining the T1 and
    T2 terms into the single spectral step

        ||(I - eta H) Delta||_2 <= max_lambda |1 - eta lambda| ||Delta||_2

    yields the SAME contraction factor (1 - eta lambda_min / 2) for any
    eta <= 1 / lambda_max(H^(inf)), which is the sharp form of the same
    condition. We therefore calibrate eta = safety / lambda_max.

    `mean_loss=True` converts to the units of the mean-squared
    `utils.losses.empirical_loss`, whose gradient w.r.t. the network
    outputs is (2/n) * Delta rather than Delta: the kernel step is
    2*eta/n, so the admissible eta scales by n/2.
    """
    eta = safety / lambda_max
    return eta * (n / 2.0) if mean_loss else eta


def lambda_max_subsampled(
    x_tau, tau, x_0, *, kappa: float, n_sub: int = 1000,
    generator=None,
) -> float:
    """
    Estimate lambda_max(H_tilde) for a large batch from a subsample.

    lambda_max of the Gram matrix grows linearly in n (verified: 9071
    at n=500 vs 36101 at n=2000, ratio 3.98 for a 4x sample increase),
    so estimating on n_sub rows and rescaling by n/n_sub agrees with
    the full computation to <1% while avoiding an O(n^3) eigensolve and
    an O(n^2) allocation. Use for n >~ 4000.
    """
    n = x_tau.shape[0]
    if n <= n_sub:
        return lambda_max_htilde(gram_htilde(x_tau, tau, x_0, kappa=kappa))
    idx = torch.randperm(n, generator=generator, device=x_tau.device)[:n_sub]
    H = gram_htilde(x_tau[idx], tau[idx], x_0[idx], kappa=kappa)
    return lambda_max_htilde(H) * (n / n_sub)
