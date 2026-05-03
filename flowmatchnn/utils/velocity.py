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
