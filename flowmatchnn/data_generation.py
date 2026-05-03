"""
Build the regression dataset that trains the velocity field.

For each i in 1..n we
    - sample a time           tau_i  ~ Uniform([0, 1]),
    - sample a source point   x_{0,i} ~ P(x_0 | x_{1,i}),
    - form the interpolation  x_{tau,i} = sigma(tau_i) * x_{0,i}
                                          + mu(tau_i) * x_{1,i}.

The model will later be trained to predict the closed-form velocity
sigma'(tau_i) * x_{0,i} + mu'(tau_i) * x_{1,i} at each
(x_{tau,i}, tau_i, x_{0,i}). That target depends on the paired x_1,
which is *not* an input to the network, so this module returns x_1
alongside the inputs and the loss assembles the target itself.
"""

from __future__ import annotations

from typing import Dict

import torch

from .utils import data as data_utils


def generate_dataset(
    n: int,
    d: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    target_center_norm: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """
    Draw the full training set in one call.

    Pipeline:
        1. Sample n iid x_1 from the target P_1 (a 2-component GMM).
        2. For each x_{1,i}, draw tau_i and x_{0,i} ~ P(x_0 | x_{1,i})
           and form x_{tau,i} via the linear interpolation.

    Args:
        n: training set size.
        d: ambient dimension.
        seed: master RNG seed.
        device: torch device (must be CUDA in this project).
        dtype: tensor dtype, default float32.
        target_center_norm: GMM center norm c so the two centers are
            +- c * e_1 (first standard basis vector).

    Returns:
        Dict with keys
            "tau"   shape (n,)     interpolation times
            "x_tau" shape (n, d)   interpolated points
            "x_0"   shape (n, d)   source draws
            "x_1"   shape (n, d)   target draws (kept around so the
                                    velocity-field target can be
                                    computed inside the loss).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    x_1 = data_utils.sample_p1_gmm(
        n,
        d,
        center_norm=target_center_norm,
        generator=g,
        device=device,
        dtype=dtype,
    )
    tau, x_tau, x_0 = data_utils.make_dataset(x_1, generator=g, dtype=dtype)
    return {"tau": tau, "x_tau": x_tau, "x_0": x_0, "x_1": x_1}
