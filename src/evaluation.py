"""Batched evaluation of simulated networks against targets (SPEC §5).

Energy = max of four two-sample KS statistics: degree, clustering, betweenness and edge
length, as in the GNM toolbox (`gnm.evaluation.MaxCriteria` over `DegreeKS`,
`ClusteringKS`, `BetweennessKS`, `EdgeLengthKS`).

Pipeline, for B simulated networks and S targets on the same n nodes and distances D:

1. **Cheap statistics from simulation state** (`network_stats`): degree is the degree
   vector; clustering is (A ∘ C) summed over a row, over d(d − 1) (C = AᵀA, so this is
   (A³)ᵢᵢ); edge length is D gathered at the edge list. O(n²) per network, no matmul when
   C comes from the kernel.
2. **Batched KS** (`ks_2samp`): sort each sample once, evaluate both ECDFs at every point
   of both samples by binary search. Memory is O(B · S · (m + k)); no B × n × distinct-values
   table.
3. **Betweenness** (`betweenness`): Brandes with all sources at once. Each BFS level is one
   batched matmul over (network, source, vertex); the backward pass is the same. fp64 by
   default (sigma counts stay exact up to 2^53; no TF32).
4. **Pruning** (`evaluate(..., prune=True)`): energy ≥ max of the three cheap KS. Networks
   whose lower bound is above the best exact energy found so far plus `margin` never get
   their betweenness computed (ties are kept: KS values are multiples of 1/n). Betweenness is computed in chunks, in order of increasing
   lower bound, so the best energy drops fast and later chunks are pruned.

Conventions (checked in tests/test_evaluation.py):
- betweenness is the standard normalised value (networkx `normalized=True`: divided by
  (n − 1)(n − 2) over ordered pairs). The toolbox's `binary_betweenness_centrality` is
  exactly 4× this (its Floyd-Warshall path counting double-counts at k = s and k = t), so
  KS values agree; use `TOOLBOX_BETWEENNESS_SCALE` to compare raw values.
- ECDFs use ≤ (right-continuous), like scipy's `ks_2samp` and the toolbox.
- clustering is 0 for nodes of degree < 2, like the toolbox.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

STATS = ("degree", "clustering", "betweenness", "edge_length")
CHEAP = ("degree", "clustering", "edge_length")  # indices 0, 1, 3 of STATS
BETW = STATS.index("betweenness")
TOOLBOX_BETWEENNESS_SCALE = 4.0  # gnm.utils.binary_betweenness_centrality / ours

_DEFAULT_MEM = 2 * 1024**3  # bytes of temporaries per chunk


def _chunks(total: int, size: int):
    size = max(1, int(size))
    for lo in range(0, total, size):
        yield slice(lo, min(total, lo + size))


# ------------------------------------------------------------------ samples
@dataclass
class Samples:
    """Per-network samples, sorted ascending along the last axis.

    values [N, m] float64; entries at positions >= counts[b] are +inf padding.
    counts [N] int64.
    """

    values: torch.Tensor
    counts: torch.Tensor

    @staticmethod
    def full(x: torch.Tensor) -> "Samples":
        """Every entry of x [N, m] is a sample."""
        x = x.to(torch.float64)
        counts = torch.full((x.shape[0],), x.shape[1], dtype=torch.int64, device=x.device)
        return Samples(torch.sort(x, dim=-1).values, counts)

    @staticmethod
    def masked(x: torch.Tensor, valid: torch.Tensor) -> "Samples":
        """Entries of x [N, m] where valid [N, m]; the rest become +inf padding."""
        x = torch.where(valid, x.to(torch.float64), math.inf)
        counts = valid.sum(-1).to(torch.int64)
        return Samples(torch.sort(x, dim=-1).values, counts)

    def __getitem__(self, idx) -> "Samples":
        return Samples(self.values[idx], self.counts[idx])

    def to(self, device) -> "Samples":
        return Samples(self.values.to(device), self.counts.to(device))


# ------------------------------------------------------------------ KS
def ks_2samp(x: Samples, y: Samples, mem_bytes: int = _DEFAULT_MEM) -> torch.Tensor:
    """Two-sample KS statistic between every network in x and every target in y.

    x: B samples (sorted, +inf padded), y: S samples. Returns [B, S] float64.
    sup_z |F_x(z) − F_y(z)| is attained at a sample point, so both ECDFs are evaluated at
    every point of both samples, with searchsorted(right=True) giving #{≤ z}.
    """
    B, m = x.values.shape
    S, k = y.values.shape
    out = torch.empty(B, S, dtype=torch.float64, device=x.values.device)
    if B == 0:
        return out
    ny = y.counts.to(torch.float64)  # [S]
    y_valid = torch.arange(k, device=y.values.device) < y.counts[:, None]  # [S, k]
    Fy_y = torch.searchsorted(y.values, y.values, right=True) / ny[:, None]  # [S, k]
    # ~4 tensors of [b, S, m + k] 8-byte entries per chunk
    per_net = 4 * 8 * S * (m + k)
    for sl in _chunks(B, mem_bytes // per_net):
        xv, nx = x.values[sl], x.counts[sl].to(torch.float64)
        b = xv.shape[0]
        x_valid = torch.arange(m, device=xv.device) < x.counts[sl, None]  # [b, m]
        # points of x
        Fx_x = torch.searchsorted(xv, xv, right=True) / nx[:, None]  # [b, m]
        q = xv.reshape(1, -1).expand(S, -1).contiguous()
        Fy_x = torch.searchsorted(y.values, q, right=True).view(S, b, m) / ny[:, None, None]
        d1 = (Fx_x.unsqueeze(0) - Fy_x).abs().masked_fill_(~x_valid.unsqueeze(0), 0)
        d1 = d1.amax(-1).T  # [b, S]
        # points of y
        q = y.values.reshape(1, -1).expand(b, -1).contiguous()
        Fx_y = torch.searchsorted(xv, q, right=True).view(b, S, k) / nx[:, None, None]
        d2 = (Fx_y - Fy_y.unsqueeze(0)).abs().masked_fill_(~y_valid.unsqueeze(0), 0)
        out[sl] = torch.maximum(d1, d2.amax(-1))
    return out


# ------------------------------------------------------------------ statistics
def clustering(A: torch.Tensor, C: torch.Tensor | None = None,
               deg: torch.Tensor | None = None,
               mem_bytes: int = _DEFAULT_MEM) -> torch.Tensor:
    """Clustering coefficients [B, n] float64 of binary networks A [B, n, n].

    C = AᵀA (shared-neighbour counts, e.g. the kernel's int16 C) is computed if not given.
    triangles·2 = (A³)ᵢᵢ = Σⱼ Aᵢⱼ Cᵢⱼ; clustering = (A³)ᵢᵢ / (d(d − 1)), 0 when d < 2.
    """
    B, n, _ = A.shape
    tri2 = torch.empty(B, n, dtype=torch.float64, device=A.device)
    per_net = 3 * 4 * n * n
    for sl in _chunks(B, mem_bytes // per_net):
        a = A[sl].to(torch.float32)  # counts < 2^24: exact
        c = (a @ a) if C is None else C[sl].to(torch.float32)
        tri2[sl] = (a * c).sum(-1, dtype=torch.float64)
    d = (A.sum(-1) if deg is None else deg).to(torch.float64)
    pairs = d * (d - 1)
    return torch.where(pairs > 0, tri2 / pairs.clamp_min(1), 0.0)


def edge_lengths(D: torch.Tensor, A: torch.Tensor | None = None,
                 edges: torch.Tensor | None = None,
                 n_edges: torch.Tensor | None = None) -> Samples:
    """Edge-length samples, from an edge list (fast path) or from adjacency.

    edges [B, T, 2] int (the kernel's output; order is irrelevant), optionally with
    n_edges [B] valid edges per network (the rest is ignored). Otherwise A [B, n, n]:
    lengths of the upper-triangle edges.
    """
    D = D.to(torch.float64)
    if edges is not None:
        edges = edges.long()
        lengths = D[edges[..., 0], edges[..., 1]]  # [B, T]
        if n_edges is None:
            return Samples.full(lengths)
        valid = torch.arange(lengths.shape[1], device=lengths.device) < n_edges.to(lengths.device)[:, None]
        return Samples.masked(lengths, valid)
    assert A is not None
    B, n, _ = A.shape
    iu = torch.triu_indices(n, n, 1, device=A.device)
    vals = D[iu[0], iu[1]]  # [n(n-1)/2]
    counts = torch.triu(A.bool(), 1).sum((1, 2)).to(torch.int64)
    m = int(counts.max().item()) if B else 0
    out = torch.full((B, m), math.inf, dtype=torch.float64, device=A.device)
    for sl in _chunks(B, _DEFAULT_MEM // (16 * vals.numel())):
        valid = A[sl, iu[0], iu[1]].bool()
        s = Samples.masked(vals.expand(valid.shape[0], -1), valid)
        out[sl] = s.values[:, :m]
    return Samples(out, counts)


@torch.no_grad()
def betweenness(A: torch.Tensor, dtype: torch.dtype = torch.float64,
                mem_bytes: int = _DEFAULT_MEM) -> torch.Tensor:
    """Normalised betweenness [B, n] (networkx convention) of binary undirected A [B, n, n].

    Brandes with all n sources at once, in linear-algebra form. Forward: BFS levels, with
    the path counts of the next level = (counts on the frontier) @ A, masked to unvisited
    vertices. Backward, from the deepest level: δ[s, v] += σ[s, v] · Σ_w A[v, w] (1 + δ[s, w])
    / σ[s, w] over w one level below v. Cost per chunk is set by its deepest network.
    """
    B, n, _ = A.shape
    out = torch.zeros(B, n, dtype=torch.float64, device=A.device)
    if n <= 2 or B == 0:
        return out
    esize = torch.finfo(dtype).bits // 8
    per_net = (6 * esize + 4) * n * n  # a, sigma, frontier, delta, matmul tmp, coef; depth, visited
    eye = torch.eye(n, dtype=torch.bool, device=A.device)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for sl in _chunks(B, mem_bytes // per_net):
            a = A[sl].to(dtype)
            b = a.shape[0]
            visited = eye.expand(b, n, n).clone()
            depth = torch.where(visited, 0, -1).to(torch.int16)
            sigma = visited.to(dtype)
            frontier = sigma.clone()
            d = 0
            while True:
                nxt = torch.bmm(frontier, a).masked_fill_(visited, 0)
                new = nxt > 0
                if not bool(new.any()):
                    break
                d += 1
                depth.masked_fill_(new, d)
                visited |= new
                sigma += nxt
                frontier = nxt
            del frontier, visited
            delta = torch.zeros_like(sigma)
            inv_sigma = torch.where(sigma > 0, 1.0 / sigma, 0.0)
            for lvl in range(d, 1, -1):  # sources (level 0) get no dependency
                coef = torch.where(depth == lvl, (1 + delta) * inv_sigma, 0.0)
                delta += torch.where(depth == lvl - 1, sigma * torch.bmm(coef, a), 0.0)
            out[sl] = delta.sum(1, dtype=torch.float64)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return out / ((n - 1) * (n - 2))


# ------------------------------------------------------------------ targets
@dataclass
class Targets:
    """Sorted samples of each statistic for S target networks."""

    stats: dict[str, Samples]
    bc_dtype: torch.dtype = torch.float64

    @property
    def S(self) -> int:
        return self.stats["degree"].values.shape[0]

    @staticmethod
    def from_adjacency(A: torch.Tensor, D: torch.Tensor, mem_bytes: int = _DEFAULT_MEM,
                       bc_dtype: torch.dtype = torch.float64) -> "Targets":
        """A [S, n, n] (or [n, n]) binary symmetric targets, zero diagonal."""
        if A.dim() == 2:
            A = A[None]
        A = A.bool()
        s = network_stats(A, D, mem_bytes=mem_bytes)
        s["betweenness"] = Samples.full(betweenness(A, dtype=bc_dtype, mem_bytes=mem_bytes))
        return Targets(s, bc_dtype)

    def to(self, device) -> "Targets":
        return Targets({k: v.to(device) for k, v in self.stats.items()}, self.bc_dtype)


def network_stats(A: torch.Tensor, D: torch.Tensor, C: torch.Tensor | None = None,
                  deg: torch.Tensor | None = None, edges: torch.Tensor | None = None,
                  n_edges: torch.Tensor | None = None,
                  mem_bytes: int = _DEFAULT_MEM) -> dict[str, Samples]:
    """The three cheap statistics as sorted samples. C, deg and edges come from the
    simulation state when available (no matmul); otherwise they are rebuilt from A."""
    deg = A.sum(-1) if deg is None else deg
    return {
        "degree": Samples.full(deg),
        "clustering": Samples.full(clustering(A, C, deg, mem_bytes=mem_bytes)),
        "edge_length": edge_lengths(D, A=A, edges=edges, n_edges=n_edges),
    }


# ------------------------------------------------------------------ energy
@dataclass
class Evaluation:
    """ks [B, S, 4] in STATS order; energy [B, S] = max over the four.

    exact [B] marks networks whose betweenness was computed. For the others (pruned),
    ks[..., BETW] is NaN and energy is the lower bound max(cheap KS), which is > the best
    exact energy for every target + margin.
    """

    ks: torch.Tensor
    energy: torch.Tensor
    exact: torch.Tensor

    @property
    def lower_bound(self) -> torch.Tensor:
        return self.ks[..., list(map(STATS.index, CHEAP))].amax(-1)


@torch.no_grad()
def evaluate(A: torch.Tensor, D: torch.Tensor, targets: Targets,
             C: torch.Tensor | None = None, deg: torch.Tensor | None = None,
             edges: torch.Tensor | None = None, n_edges: torch.Tensor | None = None,
             prune: bool = False, margin: float = 0.0, chunk: int | None = None,
             mem_bytes: int = _DEFAULT_MEM, bc_dtype: torch.dtype = torch.float64) -> Evaluation:
    """Energy of B simulated networks A [B, n, n] (bool) against S targets.

    C [B, n, n], deg [B, n], edges [B, T, 2] (and n_edges [B]) are the kernel's outputs;
    pass them to skip rebuilding from A.

    prune: skip betweenness for a network once max(its cheap KS) > best exact energy +
        margin for every target. margin = 0 scores every network tied at the optimum of
        some target; margin > 0 scores every network whose energy is within margin of it (e.g. a
        few noise SDs, for the near-optimal set). Pruned networks get NaN betweenness KS
        and a lower-bound energy: don't average pruned energies into landscape means.
    chunk: networks per betweenness batch when pruning (smaller prunes at a finer grain,
        larger keeps the GPU busier). Default: B / 8, capped by mem_bytes.
    bc_dtype: betweenness precision. fp32 is 10-30x faster on cards with slow fp64 (L4,
        A10, L40S, T4); targets are always scored in the same dtype as the networks.
    """
    B, n, _ = A.shape
    S = targets.S
    dev = A.device
    t = targets.to(dev)
    if bc_dtype != torch.float64 and t.bc_dtype != bc_dtype:
        raise ValueError("score targets with the same bc_dtype (Targets.from_adjacency)")
    stats = network_stats(A, D, C=C, deg=deg, edges=edges, n_edges=n_edges, mem_bytes=mem_bytes)
    ks = torch.full((B, S, len(STATS)), math.nan, dtype=torch.float64, device=dev)
    for name in CHEAP:
        ks[..., STATS.index(name)] = ks_2samp(stats[name], t.stats[name], mem_bytes)
    lb = ks[..., [STATS.index(c) for c in CHEAP]].amax(-1)  # [B, S]
    exact = torch.zeros(B, dtype=torch.bool, device=dev)

    def score(idx):
        bc = betweenness(A[idx], dtype=bc_dtype, mem_bytes=mem_bytes)
        ks[idx, :, BETW] = ks_2samp(Samples.full(bc), t.stats["betweenness"], mem_bytes)
        exact[idx] = True

    if not prune:
        score(torch.arange(B, device=dev))
    else:
        if chunk is None:  # ~8 rounds: early rounds set the bar, later ones get pruned
            chunk = max(1, min(mem_bytes // (52 * n * n), math.ceil(B / 8)))
        best = torch.full((S,), math.inf, dtype=torch.float64, device=dev)
        while True:
            # slack <= 0: the lower bound still allows energy <= best + margin on a target
            open_ = (lb <= best + margin) & ~exact[:, None]  # [B, S]
            n_open = int(open_.any(1).sum().item())
            if n_open == 0:
                break
            # most promising first; before a target has a best, order by the bound alone
            # (lb - inf would tie every network at -inf)
            ref = torch.where(torch.isinf(best), 0.0, best + margin)
            slack = torch.where(open_, lb - ref, math.inf).amin(1)  # [B]
            idx = torch.topk(slack, min(chunk, n_open), largest=False).indices
            score(idx)
            best = torch.minimum(best, ks[idx].amax(-1).amin(0))
    energy = torch.where(exact[:, None], ks.amax(-1), lb)
    return Evaluation(ks=ks, energy=energy, exact=exact)
