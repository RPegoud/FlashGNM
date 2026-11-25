"""Correctness of the flash kernel (gnm.fast.FlashGNM) against the toolbox's own maths.

The reference for every test is the toolbox: wiring weights come from `MatchingIndex` and
the same operations as `GenerativeNetworkModel.binary_update`, and the slow test compares
whole networks with `GenerativeNetworkModel` itself.

    pytest tests/test_flash_kernel.py -v                 # on a CUDA GPU (~1 min)
    GNM_RUN_SLOW=1 pytest tests/test_flash_kernel.py -v  # + distribution vs the toolbox
    TRITON_INTERPRET=1 pytest tests/test_flash_kernel.py -v   # CPU interpreter (slow)

What each test establishes:

- test_initial_row_sums: the kernel's starting state (R = row sums of P) equals the
  toolbox's weights, for empty and seeded networks, including eta = 0, zero distances
  and the prob_offset.
- test_teacher_forced_state: after any sequence of added edges, A, C = A^T A, the degrees
  and R equal a from-scratch recomputation (the O(n) incremental updates are exact).
- test_sampling_follows_weights: with given uniforms, each sampled edge is the one that
  inverse-CDF sampling picks from the toolbox's weights for the current network.
- test_sampling_distribution: the sampled edges follow P / sum(P) (chi-square), with the
  kernel's own random numbers.
- test_edges_valid_until_complete: edges are new, off-diagonal, and the network can be
  grown to completion; asking for more edges than are missing raises.
- test_clamping: extreme parameters give inf weights; they are clamped as in the toolbox
  and counted.
- test_long_run_drift: the maintained row sums stay equal to fresh sums over long runs.
- test_distribution_vs_toolbox (slow): networks grown by the kernel and by
  GenerativeNetworkModel have the same distribution of degree, clustering and edge length.
"""

import os

import numpy as np
import pytest
import torch

from gnm import BinaryGenerativeParameters, GenerativeNetworkModel
from gnm.defaults import get_binary_network, get_distance_matrix
from gnm.fast import FlashGNM, flash_available
from gnm.generative_rules import MatchingIndex

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INTERPRETED = DEVICE.type == "cpu"  # Triton's CPU interpreter: keep cases small

pytestmark = pytest.mark.skipif(
    not flash_available(DEVICE),
    reason="needs Triton and a CUDA GPU (or TRITON_INTERPRET=1 set before importing triton)",
)
slow = pytest.mark.skipif(
    not os.environ.get("GNM_RUN_SLOW"),
    reason="slow statistical test: set GNM_RUN_SLOW=1",
)

PARAMS = [(-2.0, 0.3), (-1.0, -0.5), (0.0, 0.0), (1.5, 1.0)]


# ---------------------------------------------------------------- reference (toolbox)
def toolbox_weights(A, D, eta, gamma, offset=1e-6):
    """Unnormalised wiring weights, [B, n, n], float64 from the toolbox's float32 maths.

    Mirrors GenerativeNetworkModel.binary_update (powerlaw terms, heterochronicity 1):
    affinity = MatchingIndex(A); affinity[affinity == 0] += offset; P = D^eta * K^gamma
    + offset; zero on edges and the diagonal; nan -> 0, inf -> 1e30.
    """
    A = A.to(torch.float32)
    K = MatchingIndex(divisor="mean")(A)
    K[K == 0] += offset
    eta = torch.as_tensor(eta, dtype=torch.float32, device=A.device).view(-1, 1, 1)
    gamma = torch.as_tensor(gamma, dtype=torch.float32, device=A.device).view(-1, 1, 1)
    P = torch.pow(D, eta) * K.pow(gamma) + offset
    P = P * (1 - A)
    idx = torch.arange(A.shape[-1], device=A.device)
    P[..., idx, idx] = 0.0
    P = torch.nan_to_num(P, nan=0.0, posinf=1e30, neginf=0.0)
    return P.to(torch.float64)


def distances(n, seed=0):
    """Euclidean distances between random points in the unit cube, zero diagonal."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, generator=g)
    return torch.cdist(x, x).to(DEVICE)


def random_graph(B, n, p, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(B, n, n, generator=g) < p).triu(1)
    return (A | A.transpose(-1, -2)).to(torch.float32).to(DEVICE)


def random_new_edges(A, T, seed=0):
    """T edges per simulation not in A, as [B, T, 2], in random order."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for a in A.cpu().bool():
        n = a.shape[0]
        i, j = torch.triu_indices(n, n, 1)
        free = ~a[i, j]
        pick = torch.randperm(int(free.sum()), generator=g)[:T]
        e = torch.stack([i[free][pick], j[free][pick]], -1)
        flip = torch.rand(T, generator=g) < 0.5  # both orientations
        e[flip] = e[flip].flip(-1)
        out.append(e)
    return torch.stack(out).to(DEVICE)


def params(B):
    eta = torch.tensor([PARAMS[b % len(PARAMS)][0] for b in range(B)], device=DEVICE)
    gamma = torch.tensor([PARAMS[b % len(PARAMS)][1] for b in range(B)], device=DEVICE)
    return eta, gamma


def assert_state(sim, A, D, eta, gamma, offset=1e-6, rtol=1e-5):
    """Kernel state equals a from-scratch recomputation from the adjacency A."""
    st = sim.state()
    assert torch.equal(st["A"], A.bool()), "adjacency"
    C = A @ A
    assert torch.equal(st["C"].to(torch.float32), C), "C = A^T A"
    assert torch.equal(st["deg"].to(torch.float32), A.sum(-1)), "degrees"
    R = toolbox_weights(A, D, eta, gamma, offset).sum(-1)
    torch.testing.assert_close(st["r"], R, rtol=rtol, atol=1e-30)


# ---------------------------------------------------------------- tests
@pytest.mark.parametrize("n,B", [(12, 4), (33, 8)] + ([] if INTERPRETED else [(200, 16)]))
@pytest.mark.parametrize("seeded", [False, True])
def test_initial_row_sums(n, B, seeded):
    D = distances(n)
    eta, gamma = params(B)
    seed = random_graph(B, n, 0.2) if seeded else None
    sim = FlashGNM(D, eta, gamma, seed_adjacency_matrix=seed)
    A = seed if seeded else torch.zeros(B, n, n, device=DEVICE)
    assert_state(sim, A, D, eta, gamma)


def test_initial_row_sums_zero_distances_and_offset():
    """Zero off-diagonal distances (0 ** eta: inf, 0 or 1) and a non-default offset."""
    n, B = 12, 4
    D = distances(n)
    D[0, 1] = D[1, 0] = 0.0
    eta, gamma = torch.tensor([-1.0, 0.0, 1.0, 0.0], device=DEVICE), torch.zeros(B, device=DEVICE)
    sim = FlashGNM(D, eta, gamma, prob_offset=1e-3)
    assert_state(sim, torch.zeros(B, n, n, device=DEVICE), D, eta, gamma, offset=1e-3)


@pytest.mark.parametrize("mode", ["step", "run"])
@pytest.mark.parametrize("seeded", [False, True])
def test_teacher_forced_state(mode, seeded):
    n, B, T = (20, 4, 30) if INTERPRETED else (64, 16, 400)
    D = distances(n)
    eta, gamma = params(B)
    A = random_graph(B, n, 0.1) if seeded else torch.zeros(B, n, n, device=DEVICE)
    sim = FlashGNM(D, eta, gamma, seed_adjacency_matrix=A if seeded else None)
    edges = random_new_edges(A, T)
    b = torch.arange(B, device=DEVICE)
    chunks = [edges[:, t:t + 1] for t in range(T)] if mode == "step" else [edges[:, : T // 2], edges[:, T // 2:]]
    for k, chunk in enumerate(chunks):
        out = sim.run(chunk.shape[1], edges=chunk)
        assert torch.equal(out.long(), chunk.long())
        for t in range(chunk.shape[1]):
            A[b, chunk[:, t, 0], chunk[:, t, 1]] = 1
            A[b, chunk[:, t, 1], chunk[:, t, 0]] = 1
        if mode == "run" or k % 10 == 9 or k == len(chunks) - 1:
            assert_state(sim, A, D, eta, gamma)


def _inverse_cdf(w, u):
    """Index chosen by inverse-CDF sampling from weights w with uniform u, and the
    distance from u to the nearest boundary (relative), to skip float near-ties."""
    c = torch.cumsum(w, -1)
    x = u * c[-1]
    k = int(torch.searchsorted(c, x, right=True))
    lo = c[k - 1] if k > 0 else torch.zeros((), dtype=c.dtype, device=c.device)
    margin = float(torch.minimum(x - lo, c[k] - x) / c[-1])
    return k, margin


def test_sampling_follows_weights():
    """With given uniforms (u_row, u_col), the kernel picks row i by R and column j by
    row i of P; compare with inverse-CDF sampling on the toolbox's weights, step by step."""
    n, B, T = (16, 4, 40) if INTERPRETED else (90, 32, 600)
    D = distances(n)
    eta, gamma = params(B)
    sim = FlashGNM(D, eta, gamma)
    g = torch.Generator().manual_seed(1)
    A = torch.zeros(B, n, n, device=DEVICE)
    checked = 0
    for t in range(T):
        u = torch.rand(B, 2, generator=g)
        W = toolbox_weights(A, D, eta, gamma)
        out = sim.step(u=u)
        for s in range(B):
            i, m_i = _inverse_cdf(W[s].sum(-1), float(u[s, 0]))
            if m_i < 1e-6:  # float near-tie between two rows: either answer is right
                continue
            assert int(out[s, 0]) == i, f"step {t}, sim {s}: row {int(out[s, 0])} != {i}"
            j, m_j = _inverse_cdf(W[s, i], float(u[s, 1]))
            if m_j >= 1e-6:
                assert int(out[s, 1]) == j, f"step {t}, sim {s}: column {int(out[s, 1])} != {j}"
                checked += 1
        b = torch.arange(B, device=DEVICE)
        A[b, out[:, 0].long(), out[:, 1].long()] = 1
        A[b, out[:, 1].long(), out[:, 0].long()] = 1
    assert checked > 0.95 * B * T
    assert_state(sim, A, D, eta, gamma)


# gamma < 0 favours pairs without shared neighbours, with weight prob_offset ** gamma:
# keep |gamma| small so the distribution is spread over many pairs (gamma = -1 puts it all
# on one pair, and the test would say nothing)
@pytest.mark.parametrize("eta,gamma", [(-2.0, 0.3), (1.0, -0.2)])
def test_sampling_distribution(eta, gamma):
    """B copies of one seeded network take one step each with the kernel's own RNG; the
    counts of each unordered edge must follow (P + P^T) / sum(P) (chi-square)."""
    from scipy.stats import chisquare

    n, B = 10, (1024 if INTERPRETED else 65536)
    D = distances(n)
    seed = random_graph(1, n, 0.3, seed=3)[0]
    torch.manual_seed(0)
    sim = FlashGNM(D, torch.full((B,), eta, device=DEVICE), torch.full((B,), gamma, device=DEVICE),
                   seed_adjacency_matrix=seed)
    e = sim.step().long().cpu()
    lo, hi = torch.minimum(e[:, 0], e[:, 1]), torch.maximum(e[:, 0], e[:, 1])
    counts = torch.bincount(lo * n + hi, minlength=n * n).view(n, n).to(torch.float64)

    W = toolbox_weights(seed[None], D, eta, gamma)[0].cpu()
    p = torch.triu(W + W.T, 1)
    p = p / p.sum()
    keep = p > 0
    assert counts[~keep].sum() == 0, "sampled an existing edge or the diagonal"
    expected = p[keep] * B
    order = torch.argsort(expected)
    exp_sorted, obs_sorted = expected[order], counts[keep][order]
    # merge the rarest cells until each expected count is >= 5
    cut = int(torch.searchsorted(torch.cumsum(exp_sorted, 0), torch.tensor(5.0, dtype=torch.float64)))
    exp_m = torch.cat([exp_sorted[: cut + 1].sum().view(1), exp_sorted[cut + 1:]])
    obs_m = torch.cat([obs_sorted[: cut + 1].sum().view(1), obs_sorted[cut + 1:]])
    assert len(exp_m) >= 10, "test design: the distribution is too concentrated to test"
    _, pval = chisquare(obs_m.numpy(), exp_m.numpy())
    assert pval > 1e-4, f"chi-square p = {pval:.2e}"


@pytest.mark.parametrize("rng", ["uniforms", "internal"])
def test_edges_valid_until_complete(rng):
    n, B = 12, 4
    D = distances(n)
    eta, gamma = params(B)
    sim = FlashGNM(D, eta, gamma)
    T = n * (n - 1) // 2
    u = torch.rand(B, T, 2) if rng == "uniforms" else None
    e = sim.run(T, u=u).long().cpu()
    assert (e[..., 0] != e[..., 1]).all(), "self-loop"
    assert ((e >= 0) & (e < n)).all(), "index out of range"
    key = torch.minimum(e[..., 0], e[..., 1]) * n + torch.maximum(e[..., 0], e[..., 1])
    for s in range(B):
        assert key[s].unique().numel() == T, "an edge was added twice"
    full = torch.ones(n, n, dtype=torch.bool, device=DEVICE).fill_diagonal_(False)
    assert (sim.state()["A"] == full).all()
    with pytest.raises(ValueError):
        sim.run(1)


def test_clamping():
    """gamma = -20 with K = 1e-6 overflows float32: weights are clamped to 1e30 and the
    clamps counted, as in the toolbox; the toolbox model clamps too."""
    n, B, T = 12, 2, 10
    D = distances(n)
    eta = torch.tensor([0.0, -1.0], device=DEVICE)
    gamma = torch.tensor([-20.0, 0.3], device=DEVICE)
    sim = FlashGNM(D, eta, gamma)
    assert torch.isfinite(sim.state()["r"]).all()
    assert sim.clamp_count.tolist()[0] > 0 and sim.clamp_count.tolist()[1] == 0
    e = sim.run(T).long()
    assert ((e >= 0) & (e < n)).all()
    A = sim.state()["A"].to(torch.float32)
    assert_state(sim, A, D, eta, gamma)

    model = GenerativeNetworkModel(
        binary_parameters=BinaryGenerativeParameters(
            eta=0.0, gamma=-20.0, lambdah=0.0, distance_relationship_type="powerlaw",
            preferential_relationship_type="powerlaw", heterochronicity_relationship_type="powerlaw",
            generative_rule=MatchingIndex(), num_iterations=1),
        num_simulations=1, distance_matrix=D.cpu(), device=torch.device("cpu"))
    with pytest.warns(UserWarning):
        model.run_model()
    assert model._clamp_count > 0


def test_long_run_drift():
    """R is updated by differences for thousands of steps; it must stay equal to fresh
    row sums (float64 accumulation of float32 rows)."""
    D = get_distance_matrix().to(DEVICE)  # n = 90
    n = D.shape[0]
    B, T = (4, 100) if INTERPRETED else (64, 3000)
    eta, gamma = params(B)
    sim = FlashGNM(D, eta, gamma)
    sim.run(T)
    A = sim.state()["A"].to(torch.float32)
    assert (A.sum((-1, -2)) == 2 * T).all()
    assert_state(sim, A, D, eta, gamma, rtol=1e-5)


@slow
@pytest.mark.skipif(INTERPRETED, reason="too slow in the interpreter")
@pytest.mark.parametrize("eta,gamma", [(-2.0, 0.3), (-1.0, 0.6)])
def test_distribution_vs_toolbox(eta, gamma):
    """Grow S networks with GenerativeNetworkModel and S with the kernel (same
    parameters, the toolbox's distance matrix and the edge count of its binary network);
    the per-network statistics must have the same distributions (two-sample KS)."""
    from scipy.stats import ks_2samp

    S = 300
    D = get_distance_matrix().to(DEVICE)
    T = int(get_binary_network()[0].sum().item()) // 2  # 400 edges on 90 nodes
    bp = BinaryGenerativeParameters(
        eta=eta, gamma=gamma, lambdah=0.0, distance_relationship_type="powerlaw",
        preferential_relationship_type="powerlaw", heterochronicity_relationship_type="powerlaw",
        generative_rule=MatchingIndex(), num_iterations=T)
    torch.manual_seed(0)
    model = GenerativeNetworkModel(binary_parameters=bp, num_simulations=S,
                                   distance_matrix=D, device=DEVICE)
    model.run_model()
    A_ref = model.adjacency_matrix
    sim = FlashGNM(D, torch.full((S,), eta, device=DEVICE), torch.full((S,), gamma, device=DEVICE))
    sim.run(T)
    A_new = sim.adjacency_matrix()

    def summaries(A):
        deg = A.sum(-1)
        tri = torch.diagonal(A @ A @ A, dim1=-2, dim2=-1)
        clust = torch.where(deg > 1, tri / (deg * (deg - 1)).clamp(min=1), torch.zeros_like(deg))
        length = (A * D).sum((-1, -2)) / A.sum((-1, -2))
        return {
            "max degree": deg.max(-1).values,
            "degree std": deg.std(-1),
            "mean clustering": clust.mean(-1),
            "mean edge length": length,
            "isolated nodes": (deg == 0).sum(-1).to(torch.float32),
        }

    ref, new = summaries(A_ref), summaries(A_new)
    for name in ref:
        p = ks_2samp(ref[name].cpu().numpy(), new[name].cpu().numpy()).pvalue
        assert p > 1e-3, f"{name}: KS p = {p:.2e} (toolbox vs flash)"
