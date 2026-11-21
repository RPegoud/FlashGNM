"""How often does a row's max of K over non-edges decrease, per step? (SPEC §4.1 worst case)
Multiplicative-rule trajectories at the Griffa optima (toolbox K: mean-degree denominator,
zero -> 1e-6). Counts rows k != i, j whose argmax column is i or j and whose max drops."""

Result (2026-11-21, Griffa subject 0, seed 0): extra row recomputes per step, mean / p99 /
max: n = 68: 1.13 / 4 / 6; n = 219: 1.39 / 5 / 9; n = 1000: 1.63 / 6 / 13. Ties (max
unchanged) 6-10% of them.
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from src.landscape import load_griffa  # noqa: E402

rng = np.random.default_rng(0)
EPS = 1e-6


def K_rows(C, A, deg, rows):
    den = (deg[rows, None] + deg[None, :] - 2 * A[rows]) / 2.0
    den[den == 0] = 1
    K = C[rows] / den
    K[K == 0] = EPS
    return K


for scale, eta, gamma in [("033", -1.8, 2.3), ("125", -1.5, 1.85), ("500", -1.32, 1.2)]:
    D, target, _ = load_griffa(scale)
    D = D.numpy().astype(np.float64)
    n = D.shape[0]
    np.fill_diagonal(D, 1)
    T = int(target(0).sum()) // 2
    A = np.zeros((n, n), np.int64)
    C = np.zeros((n, n), np.int64)
    deg = np.zeros(n, np.int64)
    De = D ** eta
    K = K_rows(C, A, deg, np.arange(n))
    valid = (A == 0) & ~np.eye(n, dtype=bool)
    Km = np.where(valid, K, -np.inf)
    amax, mmax = Km.argmax(1), Km.max(1)
    P = np.where(valid, De * K ** gamma + EPS, 0)
    rec, ties = [], []
    t0 = time.time()
    for t in range(T):
        c = np.cumsum(P.ravel())
        e = np.searchsorted(c, rng.random() * c[-1], side="right")
        i, j = divmod(int(e), n)
        Ai, Aj = A[i].copy(), A[j].copy()
        A[i, j] = A[j, i] = 1
        deg[i] += 1
        deg[j] += 1
        C[i] += Aj
        C[:, i] += Aj
        C[j] += Ai
        C[:, j] += Ai
        C[i, i], C[j, j] = deg[i], deg[j]
        rows = np.array([i, j])
        Kr = K_rows(C, A, deg, rows)  # rows i, j (= columns i, j by symmetry)
        K[rows] = Kr
        K[:, i], K[:, j] = Kr[0], Kr[1]
        valid[i, j] = valid[j, i] = False
        Pr = np.where(valid[rows], De[rows] * Kr ** gamma + EPS, 0)
        P[rows] = Pr
        P[:, i], P[:, j] = Pr[0], Pr[1]
        others = np.ones(n, bool)
        others[[i, j]] = False
        newv = np.where(amax == i, K[:, i], np.where(amax == j, K[:, j], np.inf))
        r = np.flatnonzero(others & (newv < mmax))
        if len(r):
            true_max = np.where(valid[r], K[r], -np.inf).max(1)
            ties.append(int((true_max >= mmax[r]).sum()))
        else:
            ties.append(0)
        rec.append(len(r))
        for col in (i, j):
            up = others & (K[:, col] > mmax) & valid[:, col]
            mmax[up] = K[up, col]
            amax[up] = col
        for x in list(r) + [i, j]:
            row = np.where(valid[x], K[x], -np.inf)
            amax[x], mmax[x] = row.argmax(), row.max()
    rec, ties = np.array(rec), np.array(ties)
    print(f"scale {scale} n={n} T={T}: rows recomputed/step mean {rec.mean():.2f} "
          f"({rec.mean() / n:.2%} of n), p99 {np.quantile(rec, .99):.0f}, max {rec.max()}; "
          f"max unchanged (tie) in {ties.sum() / max(rec.sum(), 1):.0%} of them; "
          f"last 10% of steps mean {rec[-T // 10:].mean():.2f}; {time.time() - t0:.0f}s",
          flush=True)
