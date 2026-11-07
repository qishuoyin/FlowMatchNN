"""
Validation of flowmatchnn.utils.ntk (revision plan, testing protocol).

Runs on CPU in a few seconds:
    pytest test/test_ntk.py -q
"""

import math

import pytest
import torch

from flowmatchnn.model import TwoLayerReLU
from flowmatchnn.utils import ntk


N, D, KAPPA = 8, 3, 0.7


@pytest.fixture(scope="module")
def tiny_inputs():
    g = torch.Generator().manual_seed(0)
    x_tau = torch.randn(N, D, generator=g)
    tau = torch.rand(N, generator=g)
    x_0 = torch.randn(N, D, generator=g)
    return x_tau, tau, x_0


def test_htilde_symmetric_psd(tiny_inputs):
    H = ntk.gram_htilde(*tiny_inputs, kappa=KAPPA)
    assert H.shape == (N, N)
    assert torch.allclose(H, H.T)
    evals = torch.linalg.eigvalsh(H)
    assert evals.min().item() > -1e-10  # PSD up to rounding


def test_htilde_matches_monte_carlo(tiny_inputs):
    """Closed form vs 4e5-sample MC estimate, entrywise."""
    H = ntk.gram_htilde(*tiny_inputs, kappa=KAPPA)
    g = torch.Generator().manual_seed(1)
    H_mc = ntk.monte_carlo_htilde(
        *tiny_inputs, kappa=KAPPA, n_samples=400_000, generator=g
    )
    scale = H.abs().max().item()
    assert (H - H_mc).abs().max().item() < 0.01 * scale


def test_diagonal_closed_form(tiny_inputs):
    """theta_ii = 0 so H_ii = kappa^2 ||z_i||^2 / 2 + ||z_i||^2 / 2."""
    x_tau, tau, x_0 = tiny_inputs
    z = ntk.augment(x_tau, tau, x_0)
    H = ntk.gram_htilde(x_tau, tau, x_0, kappa=KAPPA)
    expected = (KAPPA ** 2 + 1.0) * (z ** 2).sum(dim=1) / 2.0
    # arccos at cos ~ 1 loses ~1e-8 absolute precision on the diagonal
    assert torch.allclose(torch.diag(H), expected, rtol=1e-6, atol=1e-6)


def test_kernel_residual_norms_vs_matrix_powers(tiny_inputs):
    H = ntk.gram_htilde(*tiny_inputs, kappa=KAPPA)
    evals, U = ntk.eig_htilde(H)
    g = torch.Generator().manual_seed(2)
    V = torch.randn(N, D, generator=g, dtype=torch.float64)
    eta = 0.5 / evals.max().item()
    T = 12

    norms = ntk.kernel_residual_norms(evals, U, V, eta, T)
    M = torch.eye(N, dtype=torch.float64)
    step_mat = torch.eye(N, dtype=torch.float64) - eta * H
    for t in range(T + 1):
        direct = torch.linalg.matrix_norm(M @ V, ord="fro")
        assert math.isclose(norms[t].item(), direct.item(), rel_tol=1e-9)
        M = step_mat @ M


def test_kernel_residual_stepper_matches_norms(tiny_inputs):
    H = ntk.gram_htilde(*tiny_inputs, kappa=KAPPA)
    evals, U = ntk.eig_htilde(H)
    g = torch.Generator().manual_seed(3)
    V = torch.randn(N, D, generator=g, dtype=torch.float64)
    eta = 0.3 / evals.max().item()
    T = 10
    norms = ntk.kernel_residual_norms(evals, U, V, eta, T)
    step = ntk.kernel_residual_stepper(evals, U, V, eta)
    for t in range(T + 1):
        vec = step()
        assert math.isclose(
            vec.norm().item(), norms[t].item(), rel_tol=1e-9
        )


def test_empirical_ntk_converges_to_htilde(tiny_inputs):
    """
    Finite-m NTK at init -> H_tilde (x) I_d as m grows (law of large
    numbers over the m random neurons; 1/sqrt(m) fluctuations).
    """
    x_tau, tau, x_0 = tiny_inputs
    H = ntk.gram_htilde(x_tau, tau, x_0, kappa=KAPPA)
    target = torch.kron(H, torch.eye(D, dtype=torch.float64))

    g = torch.Generator().manual_seed(4)
    model = TwoLayerReLU(d=D, m=200_000, kappa=KAPPA, generator=g)
    K = ntk.empirical_ntk_gram(model, x_tau, tau, x_0, chunk_m=50_000)

    scale = target.abs().max().item()
    rel_err = (K - target).abs().max().item() / scale
    assert rel_err < 0.05, f"finite-m NTK deviates {rel_err:.3f} from H"


def test_empirical_ntk_memory_guard(tiny_inputs):
    x_tau, tau, x_0 = tiny_inputs
    g = torch.Generator().manual_seed(5)
    model = TwoLayerReLU(d=D, m=32, kappa=KAPPA, generator=g)
    with pytest.raises(ValueError, match="max_nd"):
        ntk.empirical_ntk_gram(model, x_tau, tau, x_0, max_nd=4)
