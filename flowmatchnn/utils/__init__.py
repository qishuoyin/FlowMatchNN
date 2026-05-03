"""
Building blocks shared across the package.

    data     -- schedules sigma/mu, source / target samplers, and the
                inner loop that turns (x_1) into (tau, x_tau, x_0).
    velocity -- the closed-form regression target the network is
                trained to predict.
    losses   -- mean-squared error against that target.
    metrics  -- sliced Wasserstein-1 distance used by Exp 2.
"""

from . import data, losses, metrics, velocity

__all__ = ["data", "losses", "metrics", "velocity"]
