"""
Closed-form regression target the network is trained to predict.

The interpolation x_tau = sigma(tau) * x_0 + mu(tau) * x_1 has
time-derivative

    d x_tau / d tau = sigma'(tau) * x_0 + mu'(tau) * x_1.        (path)

This is the velocity that pushes a particle from x_0 (at tau=0) to
x_1 (at tau=1) along the straight line connecting them. Two equivalent
forms are useful:

    1. *On the path*: when (x_0, x_1) come paired from the dataset we
       can use the (path) formula directly. This is the form the
       training loss uses.

    2. *General*: at sampling time we evaluate the velocity at
       arbitrary x_tau (not necessarily the training-time point on the
       path). Eliminating x_1 via x_1 = (x_tau - sigma(tau) x_0) / mu(tau)
       gives

           v(x_tau, tau, x_0) = sigma'(tau) * x_0
                              + (mu'(tau) / mu(tau)) * (x_tau - sigma(tau) * x_0).

       This form blows up at tau=0 because mu(0)=0, so we never use it
       for training; it is exposed only for tests / completeness.
"""

from __future__ import annotations

import torch

from . import data


def velocity_on_path(
    x_0: torch.Tensor, x_1: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    """
    Velocity evaluated on the (x_0, x_1) interpolation path.

    Implements v = sigma'(tau) * x_0 + mu'(tau) * x_1, the form that is
    finite for every tau and only requires the paired x_1 from the
    training dataset.

    Shapes:
        x_0: (n, d)
        x_1: (n, d)
        tau: (n,)

    Returns:
        v of shape (n, d).
    """
    sp = data.sigma_prime(tau).unsqueeze(-1)
    mp = data.mu_prime(tau).unsqueeze(-1)
    return sp * x_0 + mp * x_1


def velocity_general(
    x_tau: torch.Tensor, x_0: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    """
    Velocity evaluated at an arbitrary x_tau (need not lie on the path).

        v = sigma'(tau) * x_0
          + (mu'(tau) / mu(tau)) * (x_tau - sigma(tau) * x_0).

    Caller must ensure mu(tau) > 0 (i.e. tau > 0) since this form
    divides by mu(tau). Not used in training or sampling; kept for
    tests and as documentation of the equivalent general form.
    """
    s = data.sigma(tau).unsqueeze(-1)
    sp = data.sigma_prime(tau).unsqueeze(-1)
    m = data.mu(tau).unsqueeze(-1)
    mp = data.mu_prime(tau).unsqueeze(-1)
    return sp * x_0 + (mp / m) * (x_tau - s * x_0)


def posterior_mean_x1(
    x_0: torch.Tensor, *, center_norm: float = 2.0
) -> torch.Tensor:
    """
    E[x_1 | x_0] under the paper's simulation model (sampling-error evaluation).

    Prior:      x_1 ~ 0.5 N(+c e_1, I) + 0.5 N(-c e_1, I),  c = center_norm
    Coupling:   x_0 | x_1 ~ N(x_1, I)

    Per mixture component k the posterior is Gaussian,
    x_1 | x_0, k ~ N((x_0 + mu_k) / 2, I/2), and the component weights
    are w_k ∝ N(x_0; mu_k, 2I) ∝ exp(-||x_0 - mu_k||^2 / 4), so

        E[x_1 | x_0] = sum_k w_k (x_0 + mu_k) / 2.

    Needed by the oracle sampler: at tau = 0 the population velocity is
    sigma'(0) x_0 + mu'(0) E[x_1 | x_0] (for tau > 0 the pair
    (x_0, x_tau) pins x_1 down and velocity_general applies).

    Shapes: x_0 (n, d) -> (n, d).
    """
    n, d = x_0.shape
    mu_plus = torch.zeros(d, device=x_0.device, dtype=x_0.dtype)
    mu_plus[0] = center_norm
    # log-weights: -||x0 -/+ c e1||^2 / 4; softmax over the two components
    lw_p = -((x_0 - mu_plus) ** 2).sum(dim=1) / 4.0
    lw_m = -((x_0 + mu_plus) ** 2).sum(dim=1) / 4.0
    w_p = torch.sigmoid(lw_p - lw_m).unsqueeze(-1)   # (n, 1)
    mean_p = (x_0 + mu_plus) / 2.0
    mean_m = (x_0 - mu_plus) / 2.0
    return w_p * mean_p + (1.0 - w_p) * mean_m
