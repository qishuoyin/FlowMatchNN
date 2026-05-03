"""
Schedules sigma/mu, target/source samplers, and the inner loop that
turns paired (x_0, x_1) into the triple (tau, x_tau, x_0).

Schedule
    The interpolation x_tau = sigma(tau) * x_0 + mu(tau) * x_1 must
    satisfy x_{tau=0} = x_0 and x_{tau=1} = x_1, i.e. boundary
    conditions sigma(0)=mu(1)=1 and sigma(1)=mu(0)=0. The unique
    LINEAR choice satisfying both is

        sigma(tau) = 1 - tau          mu(tau) = tau
        sigma'(tau) = -1              mu'(tau) = +1.

    To swap the labels x_0 <-> x_1 (data and noise), exchange the
    two functions.

Target P_1
    Two-component Gaussian mixture in R^d with centers at
    +- center_norm * e_1 (first basis vector), identity covariance,
    equal mixture weights.

Source coupling P(x_0 | x_1) = N(x_1, I_d)
    The source x_0 is *not* independent of the target; instead
    x_0 = x_1 + noise. The induced marginal P_0 of x_0 is therefore
    the convolution P_1 * N(0, I_d). At sampling time we need a
    fresh source sample drawn from this true marginal -- see
    sample_p0_marginal.
"""

from __future__ import annotations

from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def sigma(tau: torch.Tensor) -> torch.Tensor:
    """
    Interpolation weight on the source x_0:  sigma(tau) = 1 - tau.
    """
    return 1.0 - tau


def mu(tau: torch.Tensor) -> torch.Tensor:
    """
    Interpolation weight on the target x_1:  mu(tau) = tau.
    """
    return tau


def sigma_prime(tau: torch.Tensor) -> torch.Tensor:
    """
    Derivative d/dtau sigma(tau) = -1.
    """
    return -torch.ones_like(tau)


def mu_prime(tau: torch.Tensor) -> torch.Tensor:
    """
    Derivative d/dtau mu(tau) = +1.
    """
    return torch.ones_like(tau)


def interpolate(
    x_0: torch.Tensor, x_1: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    """
    Linear interpolant x_tau = sigma(tau) * x_0 + mu(tau) * x_1.

    Shapes:
        x_0:  (n, d)
        x_1:  (n, d)
        tau:  (n,)

    Returns:
        x_tau of shape (n, d).
    """
    s = sigma(tau).unsqueeze(-1)
    m = mu(tau).unsqueeze(-1)
    return s * x_0 + m * x_1


# ---------------------------------------------------------------------------
# Target P_1
# ---------------------------------------------------------------------------


def sample_p1_gmm(
    n: int,
    d: int,
    *,
    center_norm: float = 2.0,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Draw n samples from the target distribution P_1.

    P_1 is a two-component Gaussian mixture with centers at
    +- center_norm on the first axis, identity covariance and equal
    mixture weights. We sample by flipping a coin to pick the center
    then adding standard-normal noise.

    Returns:
        Tensor of shape (n, d).
    """
    signs = torch.randint(
        0, 2, (n,), generator=generator, device=device, dtype=torch.int64
    ) * 2 - 1
    centers = torch.zeros((n, d), device=device, dtype=dtype)
    centers[:, 0] = signs.to(dtype) * center_norm
    eps = torch.randn((n, d), generator=generator, device=device, dtype=dtype)
    return centers + eps


# ---------------------------------------------------------------------------
# Source P(x_0 | x_1)
# ---------------------------------------------------------------------------


def sample_p_x0_given_x1(
    x_1: torch.Tensor,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Draw x_0 from the conditional source P(x_0 | x_1) = N(x_1, I_d).

    Shapes:
        x_1: (n, d)

    Returns:
        x_0 of shape (n, d), elementwise x_1 + standard normal noise.
    """
    eps = torch.randn(
        x_1.shape, generator=generator, device=x_1.device, dtype=dtype
    )
    return x_1 + eps


def sample_p0_marginal(
    n: int,
    d: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    center_norm: float = 2.0,
) -> torch.Tensor:
    """
    Draw n samples from the source marginal P_0 = P_1 * N(0, I_d).

    Concretely: draw a fresh x_1' from P_1 and then x_0 ~ N(x_1', I_d).
    Used at sampling time, when we need source samples that are
    consistent with the same coupling that produced the training set.
    """
    x1_prime = sample_p1_gmm(
        n,
        d,
        center_norm=center_norm,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return sample_p_x0_given_x1(x1_prime, generator=generator, dtype=dtype)


# ---------------------------------------------------------------------------
# Inner loop of the dataset construction
# ---------------------------------------------------------------------------


def sample_taus(
    n: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Draw n iid interpolation times from Uniform([0, 1]).

    Returns:
        Tensor of shape (n,).
    """
    return torch.rand(n, generator=generator, device=device, dtype=dtype)


def make_dataset(
    x_1: torch.Tensor,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorised dataset constructor: from x_1 produce (tau, x_tau, x_0).

    For each row of x_1 (shape (n, d)) we draw an iid time tau and
    source sample x_0, then form the interpolation x_tau:

        tau_i   ~ Uniform([0, 1])
        x_{0,i} ~ P(x_0 | x_{1,i})
        x_{tau,i} = sigma(tau_i) * x_{0,i} + mu(tau_i) * x_{1,i}.

    Returns:
        Tuple (tau, x_tau, x_0), each of length n.
    """
    n = x_1.shape[0]
    tau = sample_taus(
        n, generator=generator, device=x_1.device, dtype=dtype
    )
    x_0 = sample_p_x0_given_x1(x_1, generator=generator, dtype=dtype)
    x_tau = interpolate(x_0, x_1, tau)
    return tau, x_tau, x_0
