"""src/recovery.py metrics on hand-built networks; src/analysis_individual.py helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import analysis_individual as ai  # noqa: E402
from src import recovery as rc  # noqa: E402


def _sym(n, pairs):
    A = torch.zeros(n, n, dtype=torch.bool)
    for i, j in pairs:
        A[i, j] = A[j, i] = True
    return A


def _line(n, step):
    """Nodes on a line, `step` mm apart."""
    x = torch.arange(n, dtype=torch.float64) * step
    return (x[:, None] - x[None]).abs().float()


def test_perfect_and_disjoint_recovery():
    n = 6
    D = _line(n, 15.0)  # neighbours 15 mm, up to 75 mm
    tgt = _sym(n, [(0, 1), (2, 3), (0, 5)])  # 15, 15, 75 mm
    other = _sym(n, [(1, 2), (3, 4), (1, 5)])  # disjoint, same lengths 15, 15, 60
    m = rc.metrics(torch.stack([tgt, other]), tgt, D)
    assert m["T"] == 3
    assert m["recovery"] == pytest.approx(0.5)  # 1 and 0
    # bin 0 (0-20 mm): target has 2 edges, recalled by sim 0 only
    assert m["target_edges_per_bin"][0] == 2
    assert m["recall_per_bin"][0] == pytest.approx(0.5)
    # 75 mm edge in bin 3 (60-80): recalled by sim 0 only
    assert m["target_edges_per_bin"][3] == 1 and m["recall_per_bin"][3] == pytest.approx(0.5)
    # chance in bin 0 = sim edges there / pairs there (5 neighbour pairs), mean over sims
    assert m["pairs_per_bin"][0] == 5
    assert m["chance_per_bin"][0] == pytest.approx(2 / 5)


def test_long_range_top25_uses_probability():
    n = 8
    D = _line(n, 20.0)  # pairs > 60 mm: distance >= 80 (|i-j| >= 4): 4+3+2+1 = 10 pairs
    tgt = _sym(n, [(0, 4), (0, 1)])
    prob = torch.zeros(n, n)
    prob[0, 4] = prob[4, 0] = 1.0  # most probable long pair is a real edge
    prob[1, 7] = prob[7, 1] = 0.9
    m = rc.metrics(tgt[None], tgt, D, prob=prob)
    # k = round(0.25 * 10) = 2 -> {(0,4), (1,7)}: one exists
    assert m["long_top25_hit"] == pytest.approx(0.5)
    assert m["long_base_rate"] == pytest.approx(1 / 10)
    assert m["degree_spearman"] == pytest.approx(1.0)


def test_group_consensus_leaves_subject_out():
    n = 5
    a = _sym(n, [(0, 1), (1, 2)])
    b = _sym(n, [(0, 1), (3, 4)])
    c = _sym(n, [(0, 1), (3, 4)])
    freq, A = rc.group_consensus(torch.stack([a, b, c]), 0)
    assert freq[3, 4] == 1.0 and freq[1, 2] == 0.0  # subject 0 excluded
    assert torch.equal(A, _sym(n, [(0, 1), (3, 4)]))  # top T = 2 pairs


def test_argmin_and_near_set():
    E = np.full((3, 4, 5), 0.5)
    E[1, 2] = 0.1
    E[1, 2, 0] = 0.12  # nonzero SD
    g, e = ai.argmin2(E.mean(2))
    assert (g, e) == (1, 2)
    near = ai.near_set(E)
    assert near[1, 2] and near.sum() == 1


def test_cochran_q_homogeneous_vs_heterogeneous():
    rng = np.random.default_rng(0)
    v = np.full(200, 0.01)
    Q0, p0 = ai.cochran_q(rng.normal(0, 0.1, 200), v)
    Q1, p1 = ai.cochran_q(rng.normal(0, 0.3, 200), v)
    assert p0 > 0.01 and p1 < 1e-6


def test_voronoi_search_concentrates_on_minimum():
    """Synthetic 3-D landscape with a noisy minimum: later levels sample near it, all
    points stay in the box, and fixed parameters stay fixed."""
    from src import voronoi_search as vs
    rng = np.random.default_rng(0)
    target = np.array([-1.0, 2.0, 0.5])

    def f(p):
        return 0.2 + np.abs(p - target).sum(1) / 10 + rng.normal(0, 0.01, len(p))

    lo, hi = np.array([-2, -10, 0.0]), np.array([0, 10, 10.0])
    P, E, lvl = vs.search(f, lo, hi, ndraw=300, nlvl=5, seed=1)
    assert P.shape == (1500, 3) and np.all(P >= lo) and np.all(P <= hi)
    d0 = np.abs(P[lvl == 0] - target).sum(1).mean()
    d4 = np.abs(P[lvl == 4] - target).sum(1).mean()
    assert d4 < d0 / 2
    # 2-D with a fixed third parameter
    P2, _, _ = vs.search(f, lo, np.array([0, 10, 0.0]), ndraw=200, nlvl=3, seed=2)
    assert np.all(P2[:, 2] == 0.0)
