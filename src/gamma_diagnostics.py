"""Why is the fitted gamma ~2 at n = 68 when Akarca et al. 2021 report 0.21-0.50?

Factorial over the three protocol differences, on Griffa Lausanne 68 (src/dense_gnm.py as
the simulator, src/evaluation.py for the energy, no pruning):

    convention  toolbox (ours: K=0 -> 1e-6, P + 1e-6) | bct (Akarca: K + 1e-5, no offset)
    seed        empty | consensus (edges present in all 70 subjects, as Akarca's 25 edges
                common to all their 270 subjects)
    density     native (~27%) | 10% (each subject's strongest 10% of pairs by weight;
                Akarca thresholded at 27 streamlines for a 10% mean density)

Plus a pipeline check: the dense toolbox/empty/native cell against the kernel landscapes of
the main run, and slices through the GNM toolbox itself.

Output files use the landscape format (src/landscape.py: load, landscape_plots).
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict
from functools import lru_cache

import numpy as np
import torch

from src import dense_gnm as dg
from src import evaluation as ev
from src.landscape import GRIFFA, Grid, encode_ks

SCALE = "033"
CONFIGS = [f"{c}-{s}-{d}" for c, s, d in
           itertools.product(dg.CONVENTIONS, ("empty", "seed"), ("native", "d10"))]


@lru_cache(maxsize=None)
def griffa_targets(density: str) -> tuple[torch.Tensor, torch.Tensor]:
    """All 70 targets [S, n, n] bool and distances, at native density or the strongest 10%."""
    z = np.load(GRIFFA / f"scale{SCALE}.npz")
    n = int(z["n"])
    S = len(z["subjects"])
    k = round(0.1 * n * (n - 1) / 2) if density == "d10" else None
    A = torch.zeros(S, n, n, dtype=torch.bool)
    for s in range(S):
        m = z["sub"] == s
        i, j, w = z["i"][m], z["j"][m], z["w"][m]
        if k is not None:
            top = np.argsort(-w, kind="stable")[:k]
            i, j = i[top], j[top]
        A[s, i, j] = True
    A = (A | A.transpose(1, 2)) & ~torch.eye(n, dtype=torch.bool)
    return A, torch.tensor(z["dist_mm"], dtype=torch.float32)


def consensus_seed(density: str) -> torch.Tensor:
    A, _ = griffa_targets(density)
    return A.all(0)


@torch.no_grad()
def run_config(config: str, subject: int, grid: Grid, device: torch.device, seed: int = 0) -> dict:
    conv, seed_kind, density = config.split("-")
    A_all, D = griffa_targets(density)
    A_t = A_all[subject]
    T = int(A_t.sum()) // 2
    seed_A = consensus_seed(density) if seed_kind == "seed" else None
    n_seed = 0 if seed_A is None else int(seed_A.sum()) // 2
    D = D.to(device)
    eta, gamma = grid.params()
    gen = torch.Generator(device=device).manual_seed(seed * 1_000_003 + subject * 1009
                                                     + CONFIGS.index(config))
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    t0 = time.perf_counter()
    A = dg.simulate(D, eta, gamma, T, conv, seed=None if seed_A is None else seed_A.to(device),
                    generator=gen)
    sync()
    t1 = time.perf_counter()
    # structural checks on every network: symmetric, no self-loops, T edges, seed kept
    ok = bool(torch.equal(A, A.transpose(1, 2)) and not A.diagonal(dim1=1, dim2=2).any()
              and (torch.triu(A, 1).sum((1, 2)) == T).all()
              and (seed_A is None or (A & seed_A.to(device)).sum((1, 2)).eq(2 * n_seed).all()))
    targets = ev.Targets.from_adjacency(A_t.to(device), D)
    res = ev.evaluate(A, D, targets, mem_bytes=8 * 1024**3)
    sync()
    t2 = time.perf_counter()
    den = np.array([D.shape[0]] * 3 + [T], dtype=np.float64)
    return {
        "ks_num": encode_ks(res.ks[:, 0], den).reshape(grid.n_gamma, grid.n_eta, grid.repeats, 4),
        "ks_den": den,
        "etas": grid.etas,
        "gammas": grid.gammas,
        "meta": json.dumps({
            "scale": SCALE, "subject": subject, "n": int(D.shape[0]), "T": T,
            "config": config, "convention": conv, "seed": seed_kind, "density": density,
            "n_seed_edges": n_seed, "grid": asdict(grid), "rng_seed": seed,
            "stats": list(ev.STATS), "t_gen_s": t1 - t0, "t_eval_s": t2 - t1,
            "t_total_s": t2 - t0,
            "validation": {"ok": ok, "R_max_rel_err": 0.0, "checked": int(A.shape[0]),
                           "kind": "structural (dense reference has no R)"},
            "simulator": "src/dense_gnm.py",
        }),
    }


@torch.no_grad()
def toolbox_slice(subject: int, eta: float, gammas: list[float], repeats: int,
                  seeded: bool, device: torch.device) -> dict:
    """Energies [len(gammas), repeats] from the GNM toolbox itself (native density, toolbox
    convention), empty or consensus-seeded, scored with src/evaluation.py."""
    from gnm import BinaryGenerativeParameters, GenerativeNetworkModel
    from gnm.generative_rules import MatchingIndex

    A_all, D = griffa_targets("native")
    A_t = A_all[subject]
    T = int(A_t.sum()) // 2
    seed_A = consensus_seed("native") if seeded else None
    n_seed = 0 if seed_A is None else int(seed_A.sum()) // 2
    D = D.to(device)
    targets = ev.Targets.from_adjacency(A_t.to(device), D)
    torch.manual_seed(subject)
    E, t0 = [], time.perf_counter()
    for g in gammas:
        p = BinaryGenerativeParameters(
            eta=eta, gamma=float(g), lambdah=0.0, distance_relationship_type="powerlaw",
            preferential_relationship_type="powerlaw",
            heterochronicity_relationship_type="powerlaw",
            generative_rule=MatchingIndex(divisor="mean"), num_iterations=T - n_seed)
        m = GenerativeNetworkModel(
            binary_parameters=p, num_simulations=repeats, num_nodes=D.shape[0],
            distance_matrix=D,
            seed_adjacency_matrix=None if seed_A is None else seed_A.float().to(device),
            device=device)
        m.run_model()
        A = m.adjacency_matrix.bool()
        assert (torch.triu(A, 1).sum((1, 2)) == T).all()
        E.append(ev.evaluate(A, D, targets).energy[:, 0].cpu().numpy())
    return {"subject": subject, "eta": eta, "gammas": list(gammas), "seeded": seeded,
            "energy": np.stack(E).tolist(), "t_s": time.perf_counter() - t0}
