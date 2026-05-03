"""
Two-layer ReLU velocity-field network.

    f_theta(x_tau, tau, x_0) = (1/sqrt(m)) * sum_{r=1..m}
                                 a_r * ReLU(w_r^T [x_tau; tau; x_0])

Shape recap:
    input  dim = 2d + 1   -- concat of x_tau (R^d), tau (R), x_0 (R^d)
    hidden dim = m        -- chosen large ("overparameterised", m >> n)
    output dim = d        -- predicted velocity in R^d

Both layers are trainable: w_r are the first-layer rows, a_r are the
d-dim output rows. The 1/sqrt(m) at the output is the standard NTK
normalisation that keeps the magnitude of the output O(1) as m grows.

Initialisation:
    w_r ~ N(0, kappa^2 * I_{2d+1})   -- kappa controls init scale
    a_r ~ N(0, I_d)                  -- standard normal
"""

from __future__ import annotations

import math

import torch
from torch import nn


class TwoLayerReLU(nn.Module):
    """
    Vectorised implementation of f_theta.

    Parameters are stored as two matrices (rather than a list of m
    per-neuron vectors) so the forward pass is one matmul per layer:

        W  shape (m, 2d + 1)   rows are w_r
        A  shape (m, d)        rows are a_r
    """

    def __init__(
        self,
        d: int,
        m: int,
        *,
        kappa: float = 1.0,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Construct the network with random initialisation.

        Args:
            d: ambient dimension of x_0 / x_1.
            m: hidden width (large for the overparameterised regime).
            kappa: scale of the W initialisation; w_r ~ N(0, kappa^2 I).
            generator: optional torch.Generator for reproducible init.
            device: tensor device.
            dtype: tensor dtype.
        """
        super().__init__()
        self.d = d
        self.m = m
        self.kappa = kappa

        if generator is None:
            W = torch.randn(m, 2 * d + 1, device=device, dtype=dtype) * kappa
            A = torch.randn(m, d, device=device, dtype=dtype)
        else:
            W = (
                torch.randn(
                    m, 2 * d + 1,
                    generator=generator, device=device, dtype=dtype,
                )
                * kappa
            )
            A = torch.randn(
                m, d, generator=generator, device=device, dtype=dtype
            )
        self.W = nn.Parameter(W)
        self.A = nn.Parameter(A)

    def forward(
        self,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        x_0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate f_theta on a batch of (x_tau, tau, x_0) triples.

        Shapes:
            x_tau: (B, d)
            tau:   (B,) or (B, 1)
            x_0:   (B, d)

        Returns:
            (B, d) tensor of predicted velocities.
        """
        if tau.ndim == 1:
            tau = tau.unsqueeze(-1)
        z = torch.cat([x_tau, tau, x_0], dim=-1)
        h = torch.relu(z @ self.W.T)
        return (h @ self.A) / math.sqrt(self.m)
