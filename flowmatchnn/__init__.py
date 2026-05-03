"""
FlowMatchNN: flow matching with two-layer ReLU networks.

What is flow matching
---------------------
Flow matching learns a time-dependent velocity field v(x, tau) that
transports samples from a simple source distribution P_0 (here, a
Gaussian-mixture-convolved-with-noise) to a target data distribution
P_1. The recipe:

  1. Pick an interpolation x_tau = sigma(tau) * x_0 + mu(tau) * x_1
     between a source draw x_0 ~ P_0 and a target draw x_1 ~ P_1,
     parametrised by tau in [0, 1] with sigma(0)=mu(1)=1 and
     sigma(1)=mu(0)=0.
  2. The velocity along this path is closed-form:
        d x_tau / d tau = sigma'(tau) x_0 + mu'(tau) x_1.
     For any sampled (tau, x_0, x_1) we can write down the regression
     target exactly.
  3. Fit a neural network f_theta(x_tau, tau, x_0) to that target by
     least squares.
  4. To generate a fresh sample, draw x_hat_0 ~ P_0 and integrate
     the ODE  dx/dtau = f_theta(x, tau, x_hat_0)  from tau=0 to 1.

Submodules
----------
data_generation       -- build the regression dataset (steps 1-2).
learning_and_sampling -- train f_theta and sample from it (steps 3-4).
model                 -- the two-layer ReLU network f_theta itself.
utils                 -- schedules, samplers, the velocity target,
                         the MSE loss, and the sliced W_1 metric used
                         for eval (matches Theorem 3.4 in the paper).
"""

from . import data_generation, learning_and_sampling, model, utils

__all__ = ["data_generation", "learning_and_sampling", "model", "utils"]
