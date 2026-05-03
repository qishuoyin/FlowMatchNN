"""
Mean-squared error against the closed-form velocity target.

This is the objective minimised during training:

    L(theta) = (1/n) sum_i || f_theta(x_{tau,i}, tau_i, x_{0,i})
                              - v(x_{tau,i}, tau_i, x_{0,i}) ||^2.

Because the dataset carries the paired x_1, we compute the target via
velocity_on_path(x_0, x_1, tau), which is finite at every tau
(including tau=0 where the general formula would divide by zero).
"""

from __future__ import annotations

import torch

from . import velocity


def empirical_loss(
    f_theta_pred: torch.Tensor,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """
    Compute L(theta) on a batch of n training pairs.

    Shapes:
        f_theta_pred: (n, d)   model output on (x_tau_i, tau_i, x_{0,i})
        x_0:          (n, d)
        x_1:          (n, d)
        tau:          (n,)

    Returns:
        Scalar tensor (the value of L on the batch).
    """
    target = velocity.velocity_on_path(x_0, x_1, tau)
    diff = f_theta_pred - target
    return diff.pow(2).sum(dim=-1).mean()
