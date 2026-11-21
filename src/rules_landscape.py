"""Energy landscapes for the Oldham et al. rule forms (SPEC H9), per subject.

Like src/landscape.py, with src/kernel_additive.TritonRuleGNM and a third grid axis, alpha
(additive form only). KS arrays are [alpha, gamma, eta, repeat, 4] (n_alpha = 1 for the
product forms). Conventions: Oldham, Fornito & Ball 2025 (see src/kernel_additive.py).

    RULES["add_exp"]   additive, exponential cost:   eta in [-2, 0], gamma, alpha
    RULES["mult_exp"]  multiplicative, exponential:   eta in [-2, 0], gamma
    RULES["mult_pow"]  multiplicative, power law:     eta in [-10, 0], gamma
Default ranges are theirs (run_fitGNM.m): gamma in [-10, 10], alpha in [0, 10].

Full-run validation on the first `n_validate` simulations of every batch: A rebuilt from
the edge list, degrees and C exact, row sums against an exact fp64 recompute (relative to
the total), maintained maxima against brute force.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from src.landscape import encode_ks, load_griffa

RULES = {
    "add_exp": {"form": "add", "cost": "exp"},
    "mult_exp": {"form": "mult", "cost": "exp"},
    "mult_pow": {"form": "mult", "cost": "power"},
}


@dataclass
class Grid3:
    eta_lo: float = -2.0
    eta_hi: float = 0.0
    n_eta: int = 21
    gamma_lo: float = -10.0
    gamma_hi: float = 10.0
    n_gamma: int = 21
    alpha_lo: float = 0.0
    alpha_hi: float = 10.0
    n_alpha: int = 21
    repeats: int = 10

    @property
    def etas(self):
        return np.linspace(self.eta_lo, self.eta_hi, self.n_eta)

    @property
    def gammas(self):
        return np.linspace(self.gamma_lo, self.gamma_hi, self.n_gamma)

    @property
    def alphas(self):
        return np.linspace(self.alpha_lo, self.alpha_hi, self.n_alpha)

    @property
    def shape(self):
        return (self.n_alpha, self.n_gamma, self.n_eta, self.repeats)

    def params(self):
        """eta, gamma, alpha per simulation, flat in [alpha, gamma, eta, repeat] order."""
        Al, G, E = np.meshgrid(self.alphas, self.gammas, self.etas, indexing="ij")
        r = self.repeats
        f = lambda x: torch.tensor(np.repeat(x.ravel(), r), dtype=torch.float32)  # noqa: E731
        return f(E), f(G), f(Al)


def default_batch(n: int) -> int:
    return 8192 if n <= 114 else 4096 if n <= 219 else 2048 if n <= 448 else 1024


@torch.no_grad()
def validate(sim, edges: torch.Tensor, k: int) -> dict:
    k = min(k, sim.B)
    n = sim.N
    ok, r_err = True, 0.0
    x = sim.exact(0, k)
    for b in range(k):
        e = edges[b].long()
        A = torch.zeros(n, n, dtype=torch.bool, device=e.device)
        A[e[:, 0], e[:, 1]] = True
        A[e[:, 1], e[:, 0]] = True
        Ab = sim.A[b]
        Af = Ab.double()
        ok &= torch.equal(A, Ab) and int(Ab.sum()) == 2 * e.shape[0]
        ok &= not bool(Ab.diagonal().any())
        ok &= torch.equal(sim.deg[b].long(), Ab.sum(-1))
        ok &= torch.equal(sim.C[b].double(), Af @ Af)
        for key in ("Ra", "Rb"):
            g, ex = getattr(sim, key)[b], x[key][b]
            tot = float(ex.sum())
            if tot > 0:
                r_err = max(r_err, float((g - ex).abs().max()) / tot)
        my = x["MaxY"][b]
        fin = torch.isfinite(my)
        ok &= bool(((sim.MaxY[b][fin] - my[fin]).abs() <= 1e-9 * my[fin].abs().clamp_min(1)).all())
    return {"ok": bool(ok) and r_err < 1e-6, "R_max_rel_err": r_err, "checked": k}


@torch.no_grad()
def evaluate_params(A_t: torch.Tensor, D: torch.Tensor, rule: str, eta, gamma, alpha,
                    device, batch: int, seed: int, n_validate: int = 2,
                    mem_bytes: int = 8 * 1024**3, log=lambda *a: None):
    """KS [N, 4] float64 of one simulation per parameter triple against target A_t."""
    from src import evaluation as ev
    from src.kernel_additive import TritonRuleGNM

    T = int(A_t.sum()) // 2
    D = D.to(device)
    targets = ev.Targets.from_adjacency(A_t.to(device), D)
    N = len(eta)
    ks = torch.empty(N, 4, dtype=torch.float64)
    t_gen = t_eval = 0.0
    checks, stats = [], torch.zeros(2, dtype=torch.int64)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    for bi, lo in enumerate(range(0, N, batch)):
        hi = min(N, lo + batch)
        torch.manual_seed(seed * 1_000_003 + bi)
        sync()
        t0 = time.perf_counter()
        sim = TritonRuleGNM(D, eta[lo:hi].to(device), gamma[lo:hi].to(device),
                            alpha[lo:hi].to(device), **RULES[rule])
        edges = sim.run(T)
        sync()
        t1 = time.perf_counter()
        checks.append(validate(sim, edges, n_validate))
        stats += sim.stats.sum(0).cpu().long()
        sync()
        t2 = time.perf_counter()
        res = ev.evaluate(sim.A, D, targets, C=sim.C, deg=sim.deg, edges=edges,
                          mem_bytes=mem_bytes)
        ks[lo:hi] = res.ks[:, 0].cpu()
        sync()
        t_gen += t1 - t0
        t_eval += time.perf_counter() - t2
        del sim, edges, res
        log(f"  {hi}/{N} sims, gen {t_gen:.1f}s, eval {t_eval:.1f}s, valid {checks[-1]['ok']}")
    info = {
        "t_gen_s": t_gen, "t_eval_s": t_eval,
        "validation": {"ok": all(c["ok"] for c in checks),
                       "R_max_rel_err": max(c["R_max_rel_err"] for c in checks),
                       "checked": sum(c["checked"] for c in checks)},
        "row_max_recomputes_per_step": float(stats[0]) / (N * T),
        "resyncs_per_sim": float(stats[1]) / N,
    }
    return ks, info


@torch.no_grad()
def run_subject(scale: str, subject: int, rule: str, grid: Grid3, device: torch.device,
                batch: int | None = None, seed: int = 0, log=lambda *a: None) -> dict:
    if RULES[rule]["form"] == "mult":
        assert grid.n_alpha == 1, "product forms have no alpha"
    D, target_fn, _ = load_griffa(scale)
    n = D.shape[0]
    A_t = target_fn(subject)
    T = int(A_t.sum()) // 2
    eta, gamma, alpha = grid.params()
    t0 = time.perf_counter()
    ks, info = evaluate_params(A_t, D, rule, eta, gamma, alpha, device,
                               batch or default_batch(n), seed * 1_000_003 + subject * 1009,
                               log=log)
    den = np.array([n, n, n, T], dtype=np.float64)
    return {
        "ks_num": encode_ks(ks, den).reshape(*grid.shape, 4),
        "ks_den": den,
        "etas": grid.etas, "gammas": grid.gammas, "alphas": grid.alphas,
        "meta": json.dumps({
            "scale": scale, "subject": subject, "n": n, "T": T, "rule": rule,
            **RULES[rule], "grid": asdict(grid), "seed": seed,
            "t_total_s": time.perf_counter() - t0, **info,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        }),
    }


def load(path) -> dict:
    """Saved rule landscape -> KS [alpha, gamma, eta, repeat, 4], energy, axes, meta."""
    z = np.load(path)
    ks = z["ks_num"].astype(np.float64) / z["ks_den"]
    return {"ks": ks, "energy": ks.max(-1), "etas": z["etas"], "gammas": z["gammas"],
            "alphas": z["alphas"], "meta": json.loads(str(z["meta"]))}
