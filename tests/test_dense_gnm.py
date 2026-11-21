"""src/dense_gnm.py against the GNM toolbox (toolbox convention) and a loop transcription
of BCT's generative_model.m (bct convention)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import dense_gnm as dg  # noqa: E402

warnings.filterwarnings("ignore", module="gnm")


def _state(n=20, p=0.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.triu(torch.rand(n, n, generator=g) < p, 1)
    A = A | A.T
    A[0] = A[:, 0] = False  # an isolated node: zero denominators
    pts = torch.rand(n, 3, generator=g) * 100
    D = torch.cdist(pts, pts)
    return A, D


def _weights(A, D, eta, gamma, conv):
    n = A.shape[0]
    iu = torch.triu_indices(n, n, 1)
    Af = A.float()[None]
    C = (Af @ Af)[:, iu[0], iu[1]]
    lw = dg.log_weights(Af[:, iu[0], iu[1]], Af.sum(-1), C, iu,
                        D.clamp_min(1e-12).log()[iu[0], iu[1]],
                        torch.tensor([[eta]]), torch.tensor([[gamma]]), conv)[0]
    W = torch.zeros(n, n, dtype=torch.float64)
    W[iu[0], iu[1]] = torch.exp(lw.double() - lw.max())
    return W / W.sum()


@pytest.mark.parametrize("eta,gamma", [(-2.0, 0.3), (-1.0, 2.5), (-3.0, -0.5)])
def test_toolbox_convention_matches_toolbox(eta, gamma):
    from gnm import BinaryGenerativeParameters, GenerativeNetworkModel
    from gnm.generative_rules import MatchingIndex

    A, D = _state()
    n = A.shape[0]
    p = BinaryGenerativeParameters(
        eta=eta, gamma=gamma, lambdah=0.0, distance_relationship_type="powerlaw",
        preferential_relationship_type="powerlaw", heterochronicity_relationship_type="powerlaw",
        generative_rule=MatchingIndex(divisor="mean"), num_iterations=1)
    m = GenerativeNetworkModel(binary_parameters=p, num_simulations=1, num_nodes=n,
                               distance_matrix=D, seed_adjacency_matrix=A.float(),
                               device=torch.device("cpu"))
    captured = {}
    orig = torch.multinomial
    torch.multinomial = lambda w, num_samples: (captured.setdefault("w", w.clone()), orig(w, num_samples))[1]
    try:
        m.binary_update()
    finally:
        torch.multinomial = orig
    P = captured["w"].view(n, n).double()
    P = torch.triu(P + P.T, 1)  # toolbox samples over the full matrix, symmetric
    P = P / P.sum()
    torch.testing.assert_close(_weights(A, D, eta, gamma, "toolbox"), P, rtol=2e-5, atol=1e-12)


def _bct_loop(A, D, eta, gamma, eps=1e-5):
    """Transcription of BCT generative_model.m fcn_matching: K(u, v) =
    2 * sum(c1(use) & c2(use)) / ncon + epsilon with use = ~(i | j) columns, ncon =
    sum(c1(use)) + sum(c2(use)); epsilon if ncon == 0. Ff = D.^eta .* K.^gam .* ~A."""
    A = A.numpy().astype(bool)
    D = D.numpy().astype(np.float64)
    n = A.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                continue
            use = np.ones(n, bool)
            use[[i, j]] = False
            c1, c2 = A[i, use], A[j, use]
            ncon = c1.sum() + c2.sum()
            K = 2 * (c1 & c2).sum() / ncon + eps if ncon else eps
            W[i, j] = D[i, j] ** eta * K ** gamma
    return torch.tensor(W / W.sum())


@pytest.mark.parametrize("eta,gamma", [(-2.0, 0.3), (-1.0, 2.5), (-3.0, -0.5)])
def test_bct_convention_matches_bct_transcription(eta, gamma):
    A, D = _state(seed=1)
    torch.testing.assert_close(_weights(A, D, eta, gamma, "bct"), _bct_loop(A, D, eta, gamma),
                               rtol=2e-5, atol=1e-14)


@pytest.mark.parametrize("conv", dg.CONVENTIONS)
def test_simulate_keeps_seed_and_edge_count(conv):
    A, D = _state(n=16, p=0.1, seed=2)
    T = int(torch.triu(A, 1).sum()) + 25
    eta, gamma = torch.tensor([-2.0, -1.0, 0.0]), torch.tensor([0.5, 2.0, -1.0])
    out = dg.simulate(D, eta, gamma, T, conv, seed=A, chunk=2,
                      generator=torch.Generator().manual_seed(0))
    assert torch.equal(out, out.transpose(1, 2))
    assert not out.diagonal(dim1=1, dim2=2).any()
    assert (torch.triu(out, 1).sum((1, 2)) == T).all()
    assert (out & A).sum((1, 2)).eq(A.sum()).all()  # seed edges kept
