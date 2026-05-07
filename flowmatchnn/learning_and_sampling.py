"""
Train the velocity field f_theta and sample from it.

Two halves:

Training
    Full-batch gradient descent on the MSE between the network output
    and the closed-form velocity target,

        L(theta) = (1/n) sum_i || f_theta(x_{tau,i}, tau_i, x_{0,i})
                                  - sigma'(tau_i) * x_{0,i}
                                  - mu'(tau_i)    * x_{1,i} ||^2.

    No torch.optim wrapper: we update W and A by hand to keep the
    optimiser exactly equal to vanilla GD with fixed step size eta.

Sampling
    1. Draw a fresh source sample x_hat_0 from the marginal P_0.
    2. Numerically solve the ODE
           dx/dtau = f_theta(x, tau, x_hat_0)
       from tau=0 to tau=1 with forward Euler (T_euler steps).
    3. If the result lies outside the ball of radius 2*sqrt(d)*B_1,
       project it onto that sphere. This truncation is required for
       the theoretical analysis; in practice it almost never fires
       once training has converged.

Evaluation interface
    The training loop does not log every step. Every `eval_every`
    iterations it records (a) the current training loss and (b) the
    sliced Wasserstein-1 distance between n_train freshly drawn model
    samples and a fixed reference set of n_test P_1 samples (W_1 is
    the metric the manuscript bounds in Theorem 3.4). These two
    scalar series are exactly what Exp 1 and Exp 2 plot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .model import TwoLayerReLU
from .utils import data as data_utils
from .utils import losses as loss_utils
from .utils import metrics as metric_utils


@dataclass
class TrainConfig:
    """
    All hyperparameters for one training+evaluation run.

    Mirrors the CLI flags exposed by exp/_common.py::parse_args() so
    the experiment scripts can build it directly from the argparse
    namespace without manual unpacking.
    """

    n_train: int
    d: int
    m: int
    kappa: float
    T: int
    eta: float
    B1: float
    T_euler: int
    n_test: int
    eval_every: int
    seed: int
    target_center_norm: float = 2.0
    n_proj: int = 128  # # random directions for the sliced-W_1 estimator


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_algorithm2(
    model: TwoLayerReLU,
    *,
    n_eval: int,
    d: int,
    B1: float,
    T_euler: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    target_center_norm: float,
) -> torch.Tensor:
    """
    Generate n_eval fresh samples from the trained flow.

    Steps:
        1. Draw n_eval source points x_hat_0 from the marginal P_0
           (which here is P_1 convolved with N(0, I_d) thanks to the
           coupling P(x_0|x_1) = N(x_1, I_d)).
        2. Run forward Euler on dx/dtau = f_theta(x, tau, x_hat_0)
           from tau=0 to tau=1 with step dt = 1/T_euler.
        3. Truncate the endpoint into the ball of radius 2*sqrt(d)*B_1
           (project onto the sphere if the norm exceeds the cap).

    Returns:
        x_hat_1 of shape (n_eval, d).
    """
    x_hat_0 = data_utils.sample_p0_marginal(
        n_eval,
        d,
        generator=generator,
        device=device,
        dtype=dtype,
        center_norm=target_center_norm,
    )

    x = x_hat_0.clone()
    dt = 1.0 / T_euler
    for k in range(T_euler):
        tau_scalar = k * dt
        tau_batch = torch.full(
            (n_eval,), tau_scalar, device=device, dtype=dtype
        )
        v = model(x, tau_batch, x_hat_0)
        x = x + dt * v

    R = 2.0 * math.sqrt(d) * B1
    norms = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    inside = (norms <= R).to(dtype)
    projected = x * (R / norms)
    x_hat_1 = inside * x + (1.0 - inside) * projected
    return x_hat_1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_and_trace(
    cfg: TrainConfig,
    *,
    dataset: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, List[float]]:
    """
    Run vanilla GD on the velocity-matching loss and record evals.

    The training loop is a literal `theta <- theta - eta * grad L`
    update on the two parameter matrices (W and A) using the full
    batch of n_train pairs. Every `eval_every` iterations we
    additionally record the loss and the sliced Wasserstein-1 distance
    between fresh model samples and a fixed reference set drawn from
    P_1; those traces are what the plots consume.

    Args:
        cfg: TrainConfig holding all hyperparameters.
        dataset: dict with keys "tau", "x_tau", "x_0", "x_1" as
            produced by `data_generation.generate_dataset`. Already
            on `device`.
        device: CUDA device.
        dtype: tensor dtype, default float32.

    Returns:
        {
            "t":    [eval_iter_0, eval_iter_1, ...],
            "loss": [L(theta) at each evaluated iter],
            "w1":   [SW_1(samples@theta, P_1) at each evaluated iter],
        }
    """
    tau = dataset["tau"].to(device=device, dtype=dtype)
    x_tau = dataset["x_tau"].to(device=device, dtype=dtype)
    x_0 = dataset["x_0"].to(device=device, dtype=dtype)
    x_1 = dataset["x_1"].to(device=device, dtype=dtype)

    g_init = torch.Generator(device=device).manual_seed(
        cfg.seed * 1_000_003 + 1
    )
    g_sample = torch.Generator(device=device).manual_seed(
        cfg.seed * 1_000_003 + 2
    )
    g_test = torch.Generator(device=device).manual_seed(
        cfg.seed * 1_000_003 + 3
    )
    g_proj = torch.Generator(device=device).manual_seed(
        cfg.seed * 1_000_003 + 4
    )

    model = TwoLayerReLU(
        d=cfg.d,
        m=cfg.m,
        kappa=cfg.kappa,
        generator=g_init,
        device=device,
        dtype=dtype,
    )

    # Reference P_1 samples used by every W_1 evaluation. Drawn once so
    # that W_1 trends across t are not contaminated by reference noise.
    x_1_test = data_utils.sample_p1_gmm(
        cfg.n_test,
        cfg.d,
        center_norm=cfg.target_center_norm,
        generator=g_test,
        device=device,
        dtype=dtype,
    )

    # Random unit-vector directions for the sliced W_1 estimator. Drawn
    # once and reused at every evaluation for the same reason: a fixed
    # set of slices makes the trace comparable across t (no extra noise
    # from re-sampling directions every eval).
    proj_directions = metric_utils.random_unit_directions(
        cfg.n_proj,
        cfg.d,
        generator=g_proj,
        device=device,
        dtype=dtype,
    )

    ts: List[int] = []
    losses: List[float] = []
    w1s: List[float] = []

    def evaluate(t: int) -> None:
        """
        Append (t, loss, W_1) to the trace lists.

        Closure captures the training tensors and the model so it can
        be called from inside the GD loop without arg threading.
        """
        model.eval()
        with torch.no_grad():
            pred = model(x_tau, tau, x_0)
            loss_val = loss_utils.empirical_loss(pred, x_0, x_1, tau).item()
        x_hat_1 = sample_algorithm2(
            model,
            n_eval=cfg.n_train,
            d=cfg.d,
            B1=cfg.B1,
            T_euler=cfg.T_euler,
            generator=g_sample,
            device=device,
            dtype=dtype,
            target_center_norm=cfg.target_center_norm,
        )
        w1_val = metric_utils.sliced_wasserstein_1(
            x_hat_1, x_1_test, directions=proj_directions,
        ).item()
        ts.append(t)
        losses.append(loss_val)
        w1s.append(w1_val)
        model.train()

    evaluate(0)
    for t in range(1, cfg.T + 1):
        model.train()
        if model.W.grad is not None:
            model.W.grad = None
        if model.A.grad is not None:
            model.A.grad = None
        pred = model(x_tau, tau, x_0)
        loss = loss_utils.empirical_loss(pred, x_0, x_1, tau)
        loss.backward()
        with torch.no_grad():
            model.W.add_(model.W.grad, alpha=-cfg.eta)
            model.A.add_(model.A.grad, alpha=-cfg.eta)
        if t % cfg.eval_every == 0:
            evaluate(t)

    return {"t": ts, "loss": losses, "w1": w1s}


# ---------------------------------------------------------------------------
# Real-data variants (empirical P_1 reference instead of GMM)
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_algorithm2_real(
    model: TwoLayerReLU,
    *,
    n_eval: int,
    d: int,
    B1: float,
    T_euler: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    x_1_pool: torch.Tensor,
) -> torch.Tensor:
    """
    Like sample_algorithm2 but draws x_hat_0 from the empirical P_0.

    The empirical P_0 mirrors the training coupling: we pick a random
    x_1' from x_1_pool (the real-data reference set) and set
    x_hat_0 = x_1' + N(0, I_d).  This is consistent with the coupling
    P(x_0 | x_1) = N(x_1, I_d) used to build the training dataset.

    Args:
        x_1_pool: tensor of shape (N_pool, d) — real samples from P_1
            (e.g. PCA-embedded test images).  Sampled with replacement.
    """
    n_pool = x_1_pool.shape[0]
    idx = torch.randint(n_pool, (n_eval,), generator=generator, device=device)
    x_1_drawn = x_1_pool[idx].to(dtype=dtype)
    noise = torch.randn(x_1_drawn.shape, generator=generator,
                        device=device, dtype=dtype)
    x_hat_0 = x_1_drawn + noise

    x = x_hat_0.clone()
    dt = 1.0 / T_euler
    for k in range(T_euler):
        tau_scalar = k * dt
        tau_batch = torch.full(
            (n_eval,), tau_scalar, device=device, dtype=dtype
        )
        v = model(x, tau_batch, x_hat_0)
        x = x + dt * v

    R = 2.0 * math.sqrt(d) * B1
    norms = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    inside = (norms <= R).to(dtype)
    projected = x * (R / norms)
    return inside * x + (1.0 - inside) * projected


def train_and_trace_real(
    cfg: TrainConfig,
    *,
    dataset: Dict[str, torch.Tensor],
    x_1_ref: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, List[float]]:
    """
    Train on real-data PCA embeddings and record loss + W_1 traces.

    Identical to train_and_trace except:
      - The W_1 reference set is the pre-computed x_1_ref (PCA-embedded
        test images) rather than fresh GMM draws.
      - Source samples at eval time are drawn from the empirical P_0
        (random x_1 from x_1_ref + Gaussian noise) rather than the GMM.

    Args:
        cfg: TrainConfig — d must equal the PCA dimension.
        dataset: dict with keys "tau", "x_tau", "x_0", "x_1" (all in
            PCA space, shape (n_train, d_pca)), as produced by
            exp/run_real_exp.py.
        x_1_ref: tensor of shape (n_ref, d_pca) — held-out test
            images in PCA space, used as the W_1 reference.
        device: CUDA device.
        dtype: tensor dtype, default float32.

    Returns:
        {"t": [...], "loss": [...], "w1": [...]} — same format as
        train_and_trace.
    """
    tau = dataset["tau"].to(device=device, dtype=dtype)
    x_tau = dataset["x_tau"].to(device=device, dtype=dtype)
    x_0 = dataset["x_0"].to(device=device, dtype=dtype)
    x_1 = dataset["x_1"].to(device=device, dtype=dtype)
    x_1_ref = x_1_ref.to(device=device, dtype=dtype)

    g_init = torch.Generator(device=device).manual_seed(
        cfg.seed * 1_000_003 + 1
    )
    g_sample = torch.Generator(device=device).manual_seed(
        cfg.seed * 1_000_003 + 2
    )
    g_proj = torch.Generator(device=device).manual_seed(
        cfg.seed * 1_000_003 + 4
    )

    model = TwoLayerReLU(
        d=cfg.d,
        m=cfg.m,
        kappa=cfg.kappa,
        generator=g_init,
        device=device,
        dtype=dtype,
    )

    proj_directions = metric_utils.random_unit_directions(
        cfg.n_proj,
        cfg.d,
        generator=g_proj,
        device=device,
        dtype=dtype,
    )

    ts: List[int] = []
    losses: List[float] = []
    w1s: List[float] = []

    def evaluate(t: int) -> None:
        model.eval()
        with torch.no_grad():
            pred = model(x_tau, tau, x_0)
            loss_val = loss_utils.empirical_loss(pred, x_0, x_1, tau).item()
        x_hat_1 = sample_algorithm2_real(
            model,
            n_eval=cfg.n_train,
            d=cfg.d,
            B1=cfg.B1,
            T_euler=cfg.T_euler,
            generator=g_sample,
            device=device,
            dtype=dtype,
            x_1_pool=x_1_ref,
        )
        w1_val = metric_utils.sliced_wasserstein_1(
            x_hat_1, x_1_ref, directions=proj_directions,
        ).item()
        ts.append(t)
        losses.append(loss_val)
        w1s.append(w1_val)
        model.train()

    evaluate(0)
    for t in range(1, cfg.T + 1):
        model.train()
        if model.W.grad is not None:
            model.W.grad = None
        if model.A.grad is not None:
            model.A.grad = None
        pred = model(x_tau, tau, x_0)
        loss = loss_utils.empirical_loss(pred, x_0, x_1, tau)
        loss.backward()
        with torch.no_grad():
            model.W.add_(model.W.grad, alpha=-cfg.eta)
            model.A.add_(model.A.grad, alpha=-cfg.eta)
        if t % cfg.eval_every == 0:
            evaluate(t)

    return {"t": ts, "loss": losses, "w1": w1s, "model": model}
