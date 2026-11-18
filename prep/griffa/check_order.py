"""Check that Griffa's region order matches the Lausanne atlas (cortex, id order, R first).

Uses only distance vs connection probability, so it is independent of the atlas labels.

    python prep/griffa/check_order.py [griffa_dir] [atlas_dir]

Tests (cons = fraction of the 70 subjects with an edge):
1. Global: Spearman(dist, cons) and edge-weighted mean distance, atlas order vs L-first, 10%
   adjacent swaps (weak by construction: neighbouring parcels are close anyway) and random
   within-hemisphere permutations. Within-hemisphere edge fraction for both split points.
2. Local: hits = how many of each node's 5 nearest centroids it connects to (cons >= 0.5),
   with 10 simulated mislabels as a positive control.
3. Within-gyrus assignment: score[k, m] = sum_j C[k, j] exp(-D[m, j] / 10 mm), Hungarian
   assignment within each (hemisphere, gyrus). A label permutation would move many nodes.

Dropped test: argmax_m Spearman(C[k], D[m]) failed even at 68 nodes (where tests 1-2 are
clean), because a distance row's rank profile mostly rewards generic/central nodes.

Results 2026-11-02: see Claude.md "Griffa data".
"""
import re
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from common import DEFAULT_ATLAS, DEFAULT_GRIFFA, SCALES, euclid, load_cortex_centroids, load_sparse

gdir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GRIFFA
adir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ATLAS
rng = np.random.default_rng(0)

for s in SCALES:
    z, n, S, cons = load_sparse(gdir, s)
    rows, xyz = load_cortex_centroids(adir, s)
    assert len(rows) == n
    D = euclid(xyz)
    hemi = np.array([r["hemisphere"] for r in rows])
    nR = int((hemi == "R").sum())
    iu = np.triu_indices(n, 1)
    w = cons[iu]
    I = np.arange(n)

    # 1. global
    def score(order):
        d = D[np.ix_(order, order)][iu]
        return spearmanr(d, w)[0], (d * w).sum() / w.sum()

    def within(split):
        h = I < split
        return (w * (h[:, None] == h[None])[iu]).sum() / w.sum()

    def jitter():
        o = I.copy()
        for k in rng.choice(np.r_[np.arange(nR - 1), np.arange(nR, n - 1)], n // 10, replace=False):
            o[k], o[k + 1] = o[k + 1], o[k]
        return o

    rho, md = score(I)
    rho_sw, _ = score(np.r_[np.arange(n - nR, n), np.arange(n - nR)])
    perms = [score(np.r_[rng.permutation(nR), nR + rng.permutation(n - nR)]) for _ in range(20)]
    jit = [score(jitter())[0] for _ in range(20)]
    print(f"scale{s}: n={n} nR={nR}")
    print(f"  within-hemi edge frac: split@nR={within(nR):.3f} split@n-nR={within(n - nR):.3f}")
    print(f"  spearman(dist, cons): atlas={rho:.3f} L-first={rho_sw:.3f} 10%adjswap={np.mean(jit):.3f} "
          f"within-hemi perm={np.mean([p[0] for p in perms]):.3f}+-{np.std([p[0] for p in perms]):.3f}")
    print(f"  edge-weighted mean dist {md:.1f} mm vs perm {np.mean([p[1] for p in perms]):.1f} mm")

    # 2. local nearest-neighbour hits
    C = cons + cons.T
    Dn = D + np.diag(np.full(n, np.inf))
    nn5 = np.argsort(Dn, 1)[:, :5]

    def hits(Cm):
        return (Cm[I[:, None], nn5] >= 0.5).sum(1)

    h = hits(C)
    idx = rng.choice(np.where(hemi == "R")[0], 10, replace=False)
    p = I.copy()
    p[idx] = rng.permutation(idx)
    moved = idx[p[idx] != idx]
    hp = hits(C[np.ix_(p, p)])
    print(f"  hits/5 mean={h.mean():.2f}, zero-hit nodes={(h == 0).sum()}, <=1 hit={(h <= 1).sum()} | "
          f"simulated mislabels: hits mean {hp[moved].mean():.2f}, zero-hit {(hp[moved] == 0).sum()}/{len(moved)}")
    if s == "500":
        zh = np.where(h == 0)[0]
        print("  zero-hit consensus degrees:", sorted((C[zh] >= 0.5).sum(1).tolist()))

    # 3. within-gyrus optimal assignment
    if s in ("250", "500"):
        gy = np.array([r["hemisphere"] + re.sub(r"_\d+$", "", r["label"]) for r in rows])
        K = np.exp(-D / 10.0)
        np.fill_diagonal(K, 0)
        tot_id = tot_opt = 0.0
        n_self = 0
        worst = []
        for g in np.unique(gy):
            G = np.where(gy == g)[0]
            Cg = C[G].copy()
            Cg[np.arange(len(G)), G] = 0
            Sc = Cg @ K[:, G]
            r_, c_ = linear_sum_assignment(-Sc)
            idv, opt = np.trace(Sc), Sc[r_, c_].sum()
            tot_id += idv
            tot_opt += opt
            n_self += int((r_ == c_).sum())
            worst.append((opt / idv - 1, g, len(G)))
        print(f"  within-gyrus assignment keeps {n_self}/{n} in place; score gain {tot_opt / tot_id - 1:.3%}; "
              "largest gains: " + ", ".join(f"{g} {x:.1%}" for x, g, _ in sorted(worst, reverse=True)[:4]))
