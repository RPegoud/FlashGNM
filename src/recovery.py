"""Connection recovery by distance at the fitted optimum (SPEC H7, after Oldham et al. 2025).

For one subject at one scale, simulate `n_sims` networks at the subject's optimum (eta,
gamma) and at its distance-only optimum (gamma = 0, best eta on that row of the
landscape), each with T = the subject's edge count, and score them against the target:

- overall recovery: |sim ∩ target| / T (precision = recall, since both have T edges);
- per distance bin (`BINS_MM`, Euclidean centroid distance): recall = share of the
  target's edges in the bin that the simulation has, against chance = the simulation's
  density in that bin (the recall of edges placed at random within the bin, keeping the
  model's count there). Also the simulated and target edge counts per bin;
- Oldham's test: over pairs with D > 60 mm, take the 25% with the highest edge probability
  (frequency across the simulations) and report the share that exist in the target,
  against the base rate over all pairs > 60 mm;
- Spearman correlation of nodal degree (mean over simulations) with the target's.

The same metrics for the leave-one-out group consensus (edge frequency over the other
subjects, top T pairs as a network) give a ceiling for what is predictable from other
brains without any model.
"""

from __future__ import annotations

import numpy as np
import torch

BINS_MM = np.array([0, 20, 40, 60, 80, 100, 120, np.inf])
LONG_MM = 60.0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats
    return float(stats.spearmanr(x, y)[0])


@torch.no_grad()
def metrics(A_sim: torch.Tensor, A_t: torch.Tensor, D: torch.Tensor,
            prob: torch.Tensor | None = None) -> dict:
    """A_sim [B, n, n] bool, A_t [n, n] bool, D [n, n]. prob [n, n] edge probabilities
    for the top-25% test (default: frequency over A_sim)."""
    n = A_t.shape[0]
    iu = torch.triu_indices(n, n, 1, device=A_t.device)
    d = D[iu[0], iu[1]].double().cpu().numpy()
    t = A_t[iu[0], iu[1]].cpu().numpy()
    s = A_sim[:, iu[0], iu[1]].cpu().numpy()  # [B, pairs]
    T = int(t.sum())
    b = np.digitize(d, BINS_MM) - 1  # bin per pair
    nb = len(BINS_MM) - 1
    pairs_b = np.bincount(b, minlength=nb)
    t_b = np.bincount(b, weights=t, minlength=nb)
    hit = s & t[None]
    s_b = np.stack([np.bincount(b, weights=x, minlength=nb) for x in s])  # [B, nb]
    o_b = np.stack([np.bincount(b, weights=x, minlength=nb) for x in hit])
    with np.errstate(invalid="ignore", divide="ignore"):
        recall_b = o_b / t_b
        chance_b = s_b / pairs_b
    p = s.mean(0) if prob is None else prob[iu[0], iu[1]].double().cpu().numpy()
    long = d > LONG_MM
    pl, tl = p[long], t[long]
    k = max(1, int(round(0.25 * long.sum())))
    top = np.argsort(-pl, kind="stable")[:k]
    deg_sim = A_sim.sum(-1).double().mean(0).cpu().numpy()
    deg_t = A_t.sum(-1).double().cpu().numpy()
    return {
        "T": T,
        "recovery": float((hit.sum(1) / max(T, 1)).mean()),
        "recovery_sd": float((hit.sum(1) / max(T, 1)).std()),
        "bins_mm": BINS_MM.tolist(),
        "pairs_per_bin": pairs_b.tolist(),
        "target_edges_per_bin": t_b.tolist(),
        "sim_edges_per_bin": s_b.mean(0).tolist(),
        "recall_per_bin": np.nanmean(recall_b, 0).tolist(),
        "chance_per_bin": np.nanmean(chance_b, 0).tolist(),
        "long_top25_hit": float(tl[top].mean()),
        "long_base_rate": float(tl.mean()),
        "degree_spearman": _spearman(deg_sim, deg_t),
    }


def group_consensus(A_all: torch.Tensor, s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Leave-one-out edge frequency over the other subjects, and the network of its top T
    pairs (T = subject s's edge count; ties broken by index)."""
    S, n, _ = A_all.shape
    keep = torch.arange(S, device=A_all.device) != s
    freq = A_all[keep].double().mean(0)
    T = int(A_all[s].sum()) // 2
    iu = torch.triu_indices(n, n, 1, device=A_all.device)
    f = freq[iu[0], iu[1]]
    top = torch.topk(f, T, sorted=False).indices
    A = torch.zeros(n, n, dtype=torch.bool, device=A_all.device)
    A[iu[0][top], iu[1][top]] = True
    A |= A.T.clone()
    return freq, A


@torch.no_grad()
def simulate(D: torch.Tensor, eta: float, gamma: float, T: int, n_sims: int,
             seed: int) -> torch.Tensor:
    """n_sims networks [B, n, n] bool from the kernel at one (eta, gamma)."""
    from src.kernel import TritonGNM
    D1 = D.clone()
    D1.fill_diagonal_(1.0)
    torch.manual_seed(seed)
    e = torch.full((n_sims,), eta, dtype=torch.float32, device=D.device)
    g = torch.full((n_sims,), gamma, dtype=torch.float32, device=D.device)
    sim = TritonGNM(D1, e, g)
    sim.run(T)
    return sim.A.clone()


@torch.no_grad()
def run_subject(scale: str, subject: int, eta: float, gamma: float, eta_dist: float,
                device: torch.device, n_sims: int = 100, seed: int = 0) -> dict:
    from src.landscape import load_griffa
    D, target, S = load_griffa(scale)
    D = D.to(device)
    A_all = torch.stack([target(s) for s in range(S)]).to(device)
    A_t = A_all[subject]
    T = int(A_t.sum()) // 2
    out = {"scale": scale, "subject": subject, "n": int(D.shape[0]), "T": T,
           "params": {"homophily": [eta, gamma], "distance_only": [eta_dist, 0.0]}}
    for name, (e, g) in out["params"].items():
        A = simulate(D, e, g, T, n_sims, seed * 1_000_003 + subject * 1009 + (name == "distance_only"))
        assert int(A[0].sum()) == 2 * T
        out[name] = metrics(A, A_t, D)
        del A
    freq, A_g = group_consensus(A_all, subject)
    out["group_consensus"] = metrics(A_g[None], A_t, D, prob=freq)
    return out


@torch.no_grad()
def run_rule_subject(scale: str, subject: int, optima: dict, device: torch.device,
                     n_sims: int = 100, seed: int = 0) -> dict:
    """Recovery metrics at each rule's optimum (SPEC H9). optima: {rule: (eta, gamma,
    alpha)} with rules from src/rules_landscape.RULES."""
    from src.kernel_additive import TritonRuleGNM
    from src.landscape import load_griffa
    from src.rules_landscape import RULES
    D, target, _ = load_griffa(scale)
    D = D.to(device)
    A_t = target(subject).to(device)
    T = int(A_t.sum()) // 2
    from src import evaluation as ev
    out = {"scale": scale, "subject": subject, "n": int(D.shape[0]), "T": T, "optima": optima}
    targets = ev.Targets.from_adjacency(A_t, D)
    for r, (e, g, a) in optima.items():
        torch.manual_seed(seed * 1_000_003 + subject * 1009 + sorted(RULES).index(r))
        full = lambda x: torch.full((n_sims,), x, dtype=torch.float32, device=device)  # noqa: E731
        sim = TritonRuleGNM(D, full(e), full(g), full(a), **RULES[r])
        edges = sim.run(T)
        assert int(sim.A[0].sum()) == 2 * T
        out[r] = metrics(sim.A, A_t, D)
        # held-out energy: fresh networks at the optimum (the landscape minimum is biased
        # low by the search over many noisy cells)
        ks = ev.evaluate(sim.A, D, targets, C=sim.C, deg=sim.deg, edges=edges).ks[:, 0]
        out[r]["heldout_energy"] = float(ks.amax(-1).mean())
        out[r]["heldout_energy_sd"] = float(ks.amax(-1).std())
        out[r]["heldout_ks"] = ks.mean(0).tolist()
        del sim, edges
    return out
