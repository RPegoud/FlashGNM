"""Assess LeanDense and RowSampled against the toolbox.

    python assess.py [exact] [dist] [timing]

exact:  P from LeanDense and RowSampled vs the P the toolbox hands to torch.multinomial, same state
dist:   300 toolbox runs vs 300 RowSampled runs on AAL (n=90, 400 edges), KS on per-network stats
timing: ms/step for toolbox, LeanDense and RowSampled, CPU, 2 threads
"""

import sys
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from bench_gnm import make_dist, time_steps, toolbox_step_time
from lean_dense import LeanDense
from row_sampled import RowSampled, time_rowsampled
from scipy.stats import ks_2samp

from gnm import BinaryGenerativeParameters, GenerativeNetworkModel
from gnm.generative_rules import MatchingIndex

HERE = Path(__file__).resolve().parent  # AAL_DISTANCES.pt copied from the toolbox (MIT)
AAL = torch.load(HERE / "AAL_DISTANCES.pt").float()
ETA, GAMMA, EDGES = -2.0, 0.3, 400


def toolbox_model(dist, B, device, eta=ETA, gamma=GAMMA):
    p = BinaryGenerativeParameters(
        eta=eta,
        gamma=gamma,
        lambdah=0.0,
        distance_relationship_type="powerlaw",
        preferential_relationship_type="powerlaw",
        heterochronicity_relationship_type="powerlaw",
        generative_rule=MatchingIndex(divisor="mean"),
        num_iterations=1,
    )
    return GenerativeNetworkModel(
        binary_parameters=p,
        num_simulations=B,
        num_nodes=dist.shape[0],
        distance_matrix=dist,
        device=device,
    )


def ours_dist(dist):
    d = dist.clone()
    d.fill_diagonal_(1.0)
    return d


def normalise(P):
    return P / P.sum((-2, -1), keepdim=True)


# ------------------------------------------------------------------ exactness
def exact(device, steps=300, B=4):
    torch.manual_seed(1)
    aal = AAL.to(device)
    m = toolbox_model(aal, B, device=device)
    for _ in range(steps):
        m.binary_update()
    captured = {}
    orig = torch.multinomial
    torch.multinomial = lambda p, num_samples: (
        captured.setdefault("p", p.clone()),
        orig(p, num_samples),
    )[1]
    A_before = m.adjacency_matrix.clone()
    m.binary_update()
    torch.multinomial = orig
    P_tb = captured["p"].view(B, *aal.shape)

    eta, gamma = torch.full((B,), ETA), torch.full((B,), GAMMA)
    ld = LeanDense(aal, eta, gamma)
    ld.A = A_before.clone()
    rs = RowSampled(ours_dist(aal), eta, gamma)
    rs.A = A_before.clone()
    rs.C = rs.A.transpose(-1, -2) @ rs.A
    rs.deg = rs.A.sum(-1)
    for name, P in [("LeanDense", ld.probs()), ("RowSampled", rs.full_P())]:
        err = ((normalise(P) - P_tb).abs().max() / P_tb.max()).item()
        print(f"exact: max rel err of normalised P, {name} vs toolbox: {err:.2e}")


# ------------------------------------------------------------------ distribution
def stats(A, dist):
    out = {
        "max_deg": [],
        "mean_clust": [],
        "mean_betw": [],
        "mean_len": [],
        "deg_var": [],
    }
    D = dist.numpy()
    for a in A.numpy():
        g = nx.from_numpy_array(a)
        deg = a.sum(1)
        out["max_deg"].append(deg.max())
        out["deg_var"].append(deg.var())
        out["mean_clust"].append(np.mean(list(nx.clustering(g).values())))  # type: ignore
        out["mean_betw"].append(np.mean(list(nx.betweenness_centrality(g).values())))
        out["mean_len"].append(D[np.triu(a, 1) > 0].mean())
    return {k: np.array(v) for k, v in out.items()}


def dist_test(device, R=300):
    torch.manual_seed(2)
    aal = AAL.to(device)
    m = toolbox_model(aal, R, device=device)
    t_tb = time_steps(m.binary_update, EDGES, device, warmup=0) * EDGES
    rs = RowSampled(ours_dist(aal), torch.full((R,), ETA), torch.full((R,), GAMMA))
    t_rs = time_steps(rs.step, EDGES, device, warmup=0) * EDGES
    print(
        f"dist: {R} sims x {EDGES} edges: toolbox {t_tb:.1f}s, RowSampled {t_rs:.1f}s"
    )
    s_tb, s_rs = stats(m.adjacency_matrix.cpu(), AAL), stats(rs.A.cpu(), AAL)
    for k in s_tb:
        r = ks_2samp(s_tb[k], s_rs[k])
        print(
            f"   {k:<11} toolbox {s_tb[k].mean():.4f}  ours {s_rs[k].mean():.4f}  KS={r.statistic:.3f} p={r.pvalue:.3f}"  # type: ignore
        )


# ------------------------------------------------------------------ timing
def time_model(m, steps):
    """ms per step of a model exposing `.step()` and an `.A` on its device."""
    return time_steps(m.step, steps, m.A.device) * 1e3


def timing(device: torch.device):
    print("timing, ms/step   n     B   toolbox  lean-dense   O(n)   tb/O(n)  lean/O(n)")
    for n, B, s_tb, s in [
        (68, 1, 200, 400),
        (68, 64, 100, 200),
        (200, 16, 30, 100),
        (1000, 1, 10, 100),
        (1000, 4, 5, 50),
        (2000, 1, 3, 50),
    ]:
        eta, gamma = torch.empty(B).uniform_(-3, 0), torch.empty(B).uniform_(0.1, 0.5)
        tb = toolbox_step_time(n, B, s_tb, device=device) * 1e3
        ld = time_model(LeanDense(make_dist(n).to(device), eta, gamma), max(5, s_tb))
        rs = time_rowsampled(n, B, s, device=device) * 1e3
        print(
            f"                {n:>5} {B:>5} {tb:>9.2f} {ld:>10.2f} {rs:>7.3f} {tb / rs:>8.0f}x {ld / rs:>8.0f}x",
            flush=True,
        )


if __name__ == "__main__":
    torch.set_num_threads(2)
    cpu = torch.device("cpu")
    tasks = {
        "exact": partial(exact, device=cpu),
        "dist": partial(dist_test, device=cpu),
        "timing": partial(timing, device=cpu),
    }
    for name in sys.argv[1:] or tasks:
        tasks[name]()
