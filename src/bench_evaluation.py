"""Cost of each evaluation stage, and what pruning saves (SPEC §5, §10).

    python src/bench_evaluation.py --scale 500 --B 1024        # Griffa n = 1000

Networks are spatial proxies, not GNM runs: T edges sampled without replacement with
weight D^eta (Gumbel top-k), eta spread over [eta_lo, eta_hi], T = the target's edge count.
They exercise realistic sizes, densities and BFS depths; swap in kernel output for the real
landscape. Times are medians over repeats after a warm-up, synchronised.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import evaluation as ev  # noqa: E402


def load(scale: str, subjects: int, device):
    z = np.load(ROOT / "data" / "griffa" / f"scale{scale}.npz")
    n = int(z["n"])
    D = torch.tensor(z["dist_mm"], dtype=torch.float32, device=device)
    tg = torch.zeros(subjects, n, n, dtype=torch.bool)
    for s in range(subjects):
        m = z["sub"] == s
        tg[s, z["i"][m], z["j"][m]] = True
    tg = (tg | tg.transpose(1, 2)) & ~torch.eye(n, dtype=torch.bool)
    return D, tg.to(device)


def proxies(D, T, B, eta_lo, eta_hi, seed=0):
    n = D.shape[0]
    g = torch.Generator(device=D.device).manual_seed(seed)
    iu = torch.triu_indices(n, n, 1, device=D.device)
    logd = D[iu[0], iu[1]].clamp_min(1e-3).log()
    eta = torch.linspace(eta_lo, eta_hi, B, device=D.device)
    A = torch.zeros(B, n, n, dtype=torch.bool, device=D.device)
    edges = torch.empty(B, T, 2, dtype=torch.int32, device=D.device)
    for sl in ev._chunks(B, max(1, 2**27 // logd.numel())):
        u = torch.rand(sl.stop - sl.start, logd.numel(), generator=g, device=D.device)
        keys = eta[sl, None] * logd - torch.log(-torch.log(u.clamp_min(1e-12)))
        top = keys.topk(T, dim=-1).indices
        i, j = iu[0][top], iu[1][top]
        b = torch.arange(sl.start, sl.stop, device=D.device)[:, None]
        A[b, i, j] = True
        A[b, j, i] = True
        edges[sl] = torch.stack([i, j], -1).int()
    return A, edges


def timed(fn, reps):
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    fn()
    out, ts = None, []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="033")
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--subjects", type=int, default=1)
    ap.add_argument("--eta", type=float, nargs=2, default=(-4.0, 0.0))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, nargs="*", default=[0], help="0: default")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, tg = load(args.scale, args.subjects, dev)
    n = D.shape[0]
    T = int(tg[0].sum().item()) // 2
    A, edges = proxies(D, T, args.B, *args.eta)
    Af = A.float()
    C = (Af @ Af).to(torch.int16) if n <= 500 else None  # stands in for the kernel's C
    deg = A.sum(-1).to(torch.int16)
    del Af
    targets = ev.Targets.from_adjacency(tg, D)
    print(f"device {torch.cuda.get_device_name() if dev.type == 'cuda' else 'cpu'}, "
          f"torch {torch.__version__}, n={n}, B={args.B}, T={T}, S={args.subjects}, "
          f"eta in {tuple(args.eta)}")

    rows = []
    t, st = timed(lambda: ev.network_stats(A, D, C=C, deg=deg, edges=edges), args.reps)
    rows.append(("cheap stats (deg, clust, len)", t))
    t, _ = timed(lambda: [ev.ks_2samp(st[k], targets.stats[k]) for k in ev.CHEAP], args.reps)
    rows.append(("3 cheap KS", t))
    t, bc = timed(lambda: ev.betweenness(A), args.reps)
    rows.append(("betweenness (fp64)", t))
    t, _ = timed(lambda: ev.betweenness(A, dtype=torch.float32), args.reps)
    rows.append(("betweenness (fp32, no TF32)", t))
    t, _ = timed(lambda: ev.ks_2samp(ev.Samples.full(bc), targets.stats["betweenness"]), args.reps)
    rows.append(("betweenness KS", t))
    t, full = timed(lambda: ev.evaluate(A, D, targets, C=C, deg=deg, edges=edges), args.reps)
    rows.append(("evaluate, no pruning", t))
    for ch in args.chunk:
        t, pr = timed(lambda: ev.evaluate(A, D, targets, C=C, deg=deg, edges=edges, prune=True,
                                          margin=args.margin, chunk=ch or None), args.reps)
        assert torch.equal(pr.energy.amin(0), full.energy.amin(0))
        rows.append((f"pruned, chunk {ch or 'default'}: {int(pr.exact.sum())}/{args.B} scored", t))
    for name, t in rows:
        print(f"  {name:<44} {t * 1e3:10.1f} ms   {args.B / t:10.0f} nets/s")
    print(f"  pruning margin {args.margin}; best energy {full.energy.amin(0).tolist()} "
          "(identical with and without pruning)")
    if torch.cuda.is_available():
        print(f"  peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
