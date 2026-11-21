"""Evaluation pipeline (src/evaluation.py) against the GNM toolbox, networkx and scipy.

    pytest tests/test_evaluation.py -v                 # CPU (GPU cases skip)
    modal run tests/modal_tests.py --args "tests/test_evaluation.py -v -rs"

Networks under test are grown by the toolbox itself on a Griffa target (Lausanne 68, the
first subject's edge count), at three (eta, gamma), plus random and degenerate graphs.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch
from scipy.stats import ks_2samp as scipy_ks

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import evaluation as ev  # noqa: E402

warnings.filterwarnings("ignore", module="gnm")
gnm = pytest.importorskip("gnm")
from gnm import BinaryGenerativeParameters, GenerativeNetworkModel  # noqa: E402
from gnm.evaluation import (  # noqa: E402
    BetweennessKS,
    ClusteringKS,
    DegreeKS,
    EdgeLengthKS,
    MaxCriteria,
)
from gnm.generative_rules import MatchingIndex  # noqa: E402
from gnm.utils import (  # noqa: E402
    binary_betweenness_centrality,
    binary_clustering_coefficients,
    ks_statistic,
)

DATA = ROOT / "data" / "griffa" / "scale033.npz"
PARAMS = [(-2.0, 0.3), (-1.0, 0.0), (-3.0, 0.6)]  # (eta, gamma)
SIMS_PER_PARAM = 8
CUDA = torch.cuda.is_available()
GPU = pytest.mark.skipif(not CUDA, reason="needs CUDA")


# ------------------------------------------------------------------ fixtures
def _target(z, s):
    n = int(z["n"])
    m = z["sub"] == s
    A = np.zeros((n, n), bool)
    A[z["i"][m], z["j"][m]] = True
    A |= A.T
    np.fill_diagonal(A, False)
    return torch.from_numpy(A)


@pytest.fixture(scope="session")
def griffa():
    if not DATA.exists():  # tests/modal_tests.py doesn't upload data/
        pytest.skip(f"needs {DATA.relative_to(ROOT)}")
    z = np.load(DATA)
    D = torch.tensor(z["dist_mm"], dtype=torch.float32)
    targets = torch.stack([_target(z, s) for s in range(3)])  # [3, n, n] bool
    return D, targets


@pytest.fixture(scope="session")
def toolbox_nets(griffa):
    """Toolbox-grown networks [24, 68, 68] bool, plus the order edges were added in."""
    D, targets = griffa
    T = int(targets[0].sum().item()) // 2
    torch.manual_seed(0)
    nets, edges = [], []
    for eta, gamma in PARAMS:
        p = BinaryGenerativeParameters(
            eta=eta, gamma=gamma, lambdah=0.0,
            distance_relationship_type="powerlaw",
            preferential_relationship_type="powerlaw",
            heterochronicity_relationship_type="powerlaw",
            generative_rule=MatchingIndex(divisor="mean"),
            num_iterations=T,
        )
        model = GenerativeNetworkModel(
            binary_parameters=p, num_simulations=SIMS_PER_PARAM, num_nodes=D.shape[0],
            distance_matrix=D, device=torch.device("cpu"),
        )
        model.run_model()
        nets.append(model.adjacency_matrix.bool())
    A = torch.cat(nets)
    # an edge list in a random order, standing in for the kernel's output
    for a in A:
        e = torch.triu(a, 1).nonzero()
        e = e[torch.randperm(len(e))]
        flip = torch.rand(len(e)) < 0.5  # either orientation is valid
        e[flip] = e[flip].flip(-1)
        edges.append(e)
    return A, torch.stack(edges).int()


def random_graphs(B, n, p, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.rand(B, n, n, generator=g) < p
    A = torch.triu(A, 1)
    return A | A.transpose(-1, -2)


def toolbox_ks(A_syn, A_real, D):
    """Toolbox criteria, each [B, S], and their max."""
    s, r = A_syn.float(), A_real.float()
    crit = [DegreeKS(), ClusteringKS(), BetweennessKS(), EdgeLengthKS(D)]
    parts = torch.stack([c(s, r).cpu().double() for c in crit], -1)
    energy = MaxCriteria(crit)(s, r).cpu().double()
    return parts, energy


# ------------------------------------------------------------------ KS
@pytest.mark.parametrize("seed", range(4))
def test_ks_matches_scipy_with_ties_and_padding(seed):
    g = torch.Generator().manual_seed(seed)
    B, S, m, k = 7, 3, 40, 55
    x = torch.randint(0, 12, (B, m), generator=g).double()  # many ties
    y = torch.randint(0, 12, (S, k), generator=g).double() + 0.5 * (seed % 2)
    nx_ = torch.randint(5, m + 1, (B,), generator=g)
    ny_ = torch.randint(5, k + 1, (S,), generator=g)
    xs = ev.Samples.masked(x, torch.arange(m) < nx_[:, None])
    ys = ev.Samples.masked(y, torch.arange(k) < ny_[:, None])
    got = ev.ks_2samp(xs, ys, mem_bytes=4096)  # tiny budget: exercises chunking
    for b in range(B):
        for s in range(S):
            ref = scipy_ks(x[b, : nx_[b]].numpy(), y[s, : ny_[s]].numpy()).statistic
            assert got[b, s].item() == pytest.approx(ref, abs=1e-12)


def test_ks_matches_toolbox():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(9, 68, generator=g), torch.randn(2, 68, generator=g)
    x[:, :10] = 0.0  # ties across samples
    ref = ks_statistic(x, y).double()
    got = ev.ks_2samp(ev.Samples.full(x), ev.Samples.full(y))
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=0)


# ------------------------------------------------------------------ statistics
def test_clustering_matches_toolbox_and_networkx(toolbox_nets):
    A, _ = toolbox_nets
    A = torch.cat([A, random_graphs(4, 68, 0.1), torch.zeros(1, 68, 68, dtype=torch.bool)])
    ref = binary_clustering_coefficients(A.float()).double()
    torch.testing.assert_close(ev.clustering(A), ref, atol=1e-6, rtol=0)
    # from the kernel's state: int16 C and int16 degrees
    Af = A.float()
    C = (Af @ Af).to(torch.int16)
    got = ev.clustering(A, C=C, deg=A.sum(-1).to(torch.int16))
    torch.testing.assert_close(got, ev.clustering(A), atol=0, rtol=0)
    for a, c in zip(A[:5], got[:5]):
        cn = nx.clustering(nx.from_numpy_array(a.numpy().astype(int)))
        np.testing.assert_allclose(c.numpy(), [cn[i] for i in range(68)], atol=1e-12)


def _nx_betweenness(a):
    g = nx.from_numpy_array(a.numpy().astype(int))
    bc = nx.betweenness_centrality(g, normalized=True)
    return np.array([bc[i] for i in range(a.shape[0])])


def test_betweenness_matches_networkx(toolbox_nets):
    A, _ = toolbox_nets
    cases = [A[::3], random_graphs(6, 50, 0.04, seed=3), random_graphs(3, 30, 0.5)]
    path = torch.zeros(1, 12, 12, dtype=torch.bool)
    i = torch.arange(11)
    path[0, i, i + 1] = path[0, i + 1, i] = True
    complete = ~torch.eye(9, dtype=torch.bool)[None]
    star = torch.zeros(1, 8, 8, dtype=torch.bool)
    star[0, 0, 1:] = star[0, 1:, 0] = True
    empty = torch.zeros(1, 10, 10, dtype=torch.bool)
    cases += [path, complete, star, empty]
    for G in cases:
        got = ev.betweenness(G, mem_bytes=G.shape[1] ** 2 * 200)  # several chunks
        ref = np.stack([_nx_betweenness(a) for a in G])
        np.testing.assert_allclose(got.numpy(), ref, rtol=1e-10, atol=1e-12)


def test_betweenness_float32_close():
    G = random_graphs(4, 60, 0.08, seed=5)
    torch.testing.assert_close(
        ev.betweenness(G, dtype=torch.float32), ev.betweenness(G), rtol=1e-5, atol=1e-7)


def test_betweenness_is_quarter_of_toolbox(toolbox_nets):
    """The known quirk (SPEC §5): toolbox betweenness = 4 x the standard normalised one."""
    A, _ = toolbox_nets
    G = torch.cat([A[:6], random_graphs(3, 68, 0.03, seed=7)])  # incl. disconnected
    tb = binary_betweenness_centrality(G.float()).double()
    torch.testing.assert_close(
        ev.TOOLBOX_BETWEENNESS_SCALE * ev.betweenness(G), tb, rtol=2e-6, atol=1e-7)


def test_betweenness_matches_igraph():
    ig = pytest.importorskip("igraph")
    G = random_graphs(4, 80, 0.06, seed=11)
    got = ev.betweenness(G).numpy()
    n = G.shape[1]
    for a, row in zip(G, got):
        e = torch.triu(a, 1).nonzero().tolist()
        bc = np.array(ig.Graph(n=n, edges=e).betweenness(directed=False))
        np.testing.assert_allclose(row, bc / ((n - 1) * (n - 2) / 2), rtol=1e-10, atol=1e-12)


def test_edge_lengths_from_edge_list_and_adjacency(griffa, toolbox_nets):
    D, _ = griffa
    A, edges = toolbox_nets
    from_A = ev.edge_lengths(D, A=A)
    from_E = ev.edge_lengths(D, edges=edges)
    torch.testing.assert_close(from_E.values, from_A.values, atol=0, rtol=0)
    assert torch.equal(from_E.counts, from_A.counts)
    # per-network edge counts (SPEC §4.0 item 3): only the first n_edges count
    n_edges = torch.tensor([edges.shape[1] - 7 * i for i in range(len(edges))])
    part = ev.edge_lengths(D, edges=edges, n_edges=n_edges)
    for b in range(len(edges)):
        ref = np.sort(D[edges[b, : n_edges[b], 0].long(), edges[b, : n_edges[b], 1].long()].double().numpy())
        np.testing.assert_array_equal(part.values[b, : n_edges[b]].numpy(), ref)
        assert torch.isinf(part.values[b, n_edges[b]:]).all()


# ------------------------------------------------------------------ against the toolbox's criteria
def _check_against_toolbox(res, A, targets, D):
    parts, energy = toolbox_ks(A, targets, D)
    n = A.shape[1]
    # degree and edge length: identical up to fp32 CDF rounding in the toolbox
    for k in ("degree", "edge_length"):
        i = ev.STATS.index(k)
        torch.testing.assert_close(res.ks[..., i], parts[..., i], atol=1e-6, rtol=0)
    # clustering and betweenness are float statistics: a tie broken by rounding in one
    # implementation (fp32 in the toolbox) moves one ECDF step, i.e. 1/n.
    for k in ("clustering", "betweenness"):
        i = ev.STATS.index(k)
        diff = (res.ks[..., i] - parts[..., i]).abs()
        assert diff.max() <= 1 / n + 1e-6, k
        assert (diff <= 1e-6).float().mean() >= 0.9, k
    diff = (res.energy - energy).abs()
    assert diff.max() <= 1 / n + 1e-6
    assert (diff <= 1e-6).float().mean() >= 0.9
    return parts, energy


def test_evaluate_matches_toolbox_from_adjacency(griffa, toolbox_nets):
    D, targets = griffa
    A, _ = toolbox_nets
    res = ev.evaluate(A, D, ev.Targets.from_adjacency(targets, D))
    assert res.exact.all()
    _check_against_toolbox(res, A, targets, D)


def test_evaluate_matches_toolbox_from_kernel_state(griffa, toolbox_nets):
    """The fast path, as fed by the kernel: int16 C and degrees, int32 edge list."""
    D, targets = griffa
    A, edges = toolbox_nets
    Af = A.float()
    res = ev.evaluate(A, D, ev.Targets.from_adjacency(targets, D),
                      C=(Af @ Af).to(torch.int16), deg=A.sum(-1).to(torch.int16), edges=edges)
    ref = ev.evaluate(A, D, ev.Targets.from_adjacency(targets, D))
    torch.testing.assert_close(res.ks, ref.ks, atol=0, rtol=0)
    _check_against_toolbox(res, A, targets, D)


def test_target_vs_itself_is_zero(griffa):
    D, targets = griffa
    res = ev.evaluate(targets, D, ev.Targets.from_adjacency(targets, D))
    assert torch.all(res.energy.diagonal() == 0)
    assert torch.all(res.energy + torch.eye(3, dtype=torch.float64) > 0)


# ------------------------------------------------------------------ pruning
@pytest.mark.parametrize("margin", [0.0, 0.05])
@pytest.mark.parametrize("chunk", [1, 5])
def test_pruning_keeps_argmin_and_near_optimal_set(griffa, toolbox_nets, margin, chunk):
    D, targets = griffa
    A, edges = toolbox_nets
    A = torch.cat([A, random_graphs(8, 68, 0.25, seed=2)])  # a few hopeless networks
    T = ev.Targets.from_adjacency(targets, D)
    full = ev.evaluate(A, D, T)
    pr = ev.evaluate(A, D, T, prune=True, margin=margin, chunk=chunk)
    best = full.energy.amin(0)  # [S]
    # exact where computed, bitwise
    torch.testing.assert_close(pr.ks[pr.exact], full.ks[pr.exact], atol=0, rtol=0)
    # every network within margin of the optimum of some target was scored
    near = (full.energy <= best + margin).any(1)
    assert pr.exact[near].all()
    torch.testing.assert_close(pr.energy.amin(0), best, atol=0, rtol=0)
    assert torch.equal(pr.energy[near], full.energy[near])
    # pruned: NaN betweenness, energy is a valid lower bound that can't reach best + margin
    assert torch.isnan(pr.ks[~pr.exact][..., ev.BETW]).all()
    assert (pr.energy <= full.energy).all()
    assert (pr.energy[~pr.exact] > best + margin).all()
    if margin == 0.0:
        assert (~pr.exact).sum() > 0, "nothing pruned on hopeless networks"


def test_pruning_handles_all_pruned_but_first():
    D = torch.rand(20, 20).double()
    D = D + D.T
    tgt = random_graphs(1, 20, 0.2, seed=1)
    A = torch.cat([tgt, random_graphs(5, 20, 0.9, seed=4)])
    res = ev.evaluate(A, D, ev.Targets.from_adjacency(tgt, D), prune=True, chunk=1)
    assert res.energy[0, 0] == 0
    assert res.exact.tolist() == [True] + [False] * 5


# ------------------------------------------------------------------ GPU
@GPU
def test_gpu_matches_cpu(griffa, toolbox_nets):
    D, targets = griffa
    A, edges = toolbox_nets
    T = ev.Targets.from_adjacency(targets, D)
    cpu = ev.evaluate(A, D, T, edges=edges)
    gpu = ev.evaluate(A.cuda(), D.cuda(), T, edges=edges.cuda())
    torch.testing.assert_close(gpu.ks.cpu(), cpu.ks, atol=1e-12, rtol=0)
    bc_cpu = ev.betweenness(A)
    torch.testing.assert_close(ev.betweenness(A.cuda()).cpu(), bc_cpu, rtol=1e-12, atol=1e-14)


@GPU
def test_gpu_large_n_betweenness_against_networkx():
    G = random_graphs(3, 500, 0.01, seed=9).cuda()  # sparse: deep BFS, disconnected
    got = ev.betweenness(G).cpu().numpy()
    ref = np.stack([_nx_betweenness(a) for a in G.cpu()])
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)
