"""Per-subject energy landscapes: generate with the kernel, validate, evaluate (SPEC §8).

For one target (a Griffa subject at one scale), simulate every cell of an (eta, gamma) grid
`repeats` times with T = the subject's edge count, and keep all four KS components of
every run (no pruning). Arrays are indexed [gamma, eta, repeat, stat]
(`meshgrid(indexing="xy")` order, eta on the x axis).

    python src/landscape.py --scale 033 --subject 0            # one subject, local GPU
    modal run src/modal_landscape.py --scales 033 500          # many subjects, Modal

Distances: Euclidean between template centroids (`dist_mm`), shared across subjects.
The model is the kernel's: toolbox conventions (matching index, mean-degree divisor,
offset 1e-6). Every batch is checked on its first `n_validate` simulations (SPEC §6
full-run validation): A rebuilt from the edge list, degrees and C exact, row sums R
against a full recompute.

Storage: KS between samples of equal size m is k/m exactly, and both sides have n nodes
(degree, clustering, betweenness) or T edges (edge length). So KS is stored exactly as
uint16 numerators `ks_num` with denominators `ks_den` [4].
"""

from __future__ import annotations

import argparse
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GRIFFA = ROOT / "data" / "griffa"
SCALES = {"033": 68, "060": 114, "125": 219, "250": 448, "500": 1000}
# Synthetic-subject null (SPEC H5): (eta, gamma) per scale = median optimum over the 70
# Griffa subjects (results/landscape_griffa_full/summary.md).
NULL_PARAMS = {"033": (-1.8, 2.3), "060": (-1.7, 2.0), "125": (-1.5, 1.85),
               "250": (-1.4, 1.5), "500": (-1.3, 1.1)}


@dataclass
class Grid:
    eta_lo: float = -4.0
    eta_hi: float = 1.0
    n_eta: int = 51
    gamma_lo: float = -1.0
    gamma_hi: float = 4.0
    n_gamma: int = 51
    repeats: int = 20

    @property
    def etas(self) -> np.ndarray:
        return np.linspace(self.eta_lo, self.eta_hi, self.n_eta)

    @property
    def gammas(self) -> np.ndarray:
        return np.linspace(self.gamma_lo, self.gamma_hi, self.n_gamma)

    @property
    def size(self) -> int:
        return self.n_eta * self.n_gamma * self.repeats

    def params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """eta, gamma per simulation, flat in [gamma, eta, repeat] order."""
        E, G = np.meshgrid(self.etas, self.gammas, indexing="xy")  # [gamma, eta]
        eta = np.repeat(E.ravel(), self.repeats)
        gamma = np.repeat(G.ravel(), self.repeats)
        return torch.tensor(eta, dtype=torch.float32), torch.tensor(gamma, dtype=torch.float32)


def default_batch(n: int) -> int:
    """Simulations per kernel launch: sims/s flat in B beyond these (H100, results/
    26-11-21_15-42-39_sim_v2_bigB), memory ~6 GB for the kernel plus evaluation."""
    return 12500 if n <= 114 else 5000 if n <= 219 else 2500 if n <= 448 else 1250


def load_griffa(scale: str):
    """Distances [n, n] float32 (zero diagonal) and a function subject -> target A bool."""
    z = np.load(GRIFFA / f"scale{scale}.npz")
    n = int(z["n"])
    D = torch.tensor(z["dist_mm"], dtype=torch.float32)
    i, j, sub = z["i"], z["j"], z["sub"]

    def target(s: int) -> torch.Tensor:
        m = sub == s
        A = torch.zeros(n, n, dtype=torch.bool)
        A[i[m], j[m]] = True
        A |= A.T.clone()
        A.fill_diagonal_(False)
        return A

    return D, target, int(len(z["subjects"]))


@torch.no_grad()
def validate(sim, edges: torch.Tensor, k: int) -> dict:
    """SPEC §6 full-run checks on the first k simulations of a batch."""
    k = min(k, sim.B)
    n = sim.N
    ok, r_err = True, 0.0
    for b in range(k):
        e = edges[b].long()
        A = torch.zeros(n, n, dtype=torch.bool, device=e.device)
        A[e[:, 0], e[:, 1]] = True
        A[e[:, 1], e[:, 0]] = True
        Ab = sim.A[b]
        Af = Ab.float()
        ok &= torch.equal(A, Ab)  # rebuilt from the edge list; implies no duplicates
        ok &= int(Ab.sum()) == 2 * e.shape[0] and not bool(Ab.diagonal().any())
        ok &= torch.equal(sim.deg[b].long(), Ab.sum(-1))
        ok &= torch.equal(sim.C[b].float(), Af @ Af)
        R_ref = sim._block_P(b, b + 1).sum(-1, dtype=torch.float64)[0]
        r_err = max(r_err, float(((sim.R[b] - R_ref).abs() / R_ref.abs().clamp_min(1e-300)).max()))
    return {"ok": bool(ok), "R_max_rel_err": r_err, "checked": k}


def encode_ks(ks: torch.Tensor, den: np.ndarray) -> np.ndarray:
    """ks [..., 4] float64 -> exact uint16 numerators (see module doc)."""
    num = ks.cpu().double().numpy() * den
    r = np.rint(num)
    assert np.abs(num - r).max() < 1e-6, "KS is not a multiple of 1/m"
    return r.astype(np.uint16)


@torch.no_grad()
def null_target(scale: str, subject: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    """Synthetic subject: one model network at NULL_PARAMS[scale] with T = the real
    subject's edge count, so the null keeps the cohort's spread of T. [n, n] bool on CPU."""
    from src.kernel import TritonGNM

    D, target_fn, _ = load_griffa(scale)
    T = int(target_fn(subject).sum()) // 2
    D1 = D.to(device)
    D1.fill_diagonal_(1.0)
    eta, gamma = NULL_PARAMS[scale]
    torch.manual_seed(seed * 1_000_003 + subject * 1009 + 7_777_777)  # disjoint from fits
    sim = TritonGNM(D1, torch.tensor([eta], device=device), torch.tensor([gamma], device=device))
    sim.run(T)
    A = sim.A[0].cpu()
    assert int(A.sum()) == 2 * T and torch.equal(A, A.T)
    return A


@torch.no_grad()
def run_subject(scale: str, subject: int, grid: Grid, device: torch.device,
                batch: int | None = None, seed: int = 0, n_validate: int = 4,
                mem_bytes: int = 8 * 1024**3, log=print,
                target: torch.Tensor | None = None) -> dict:
    """target: [n, n] bool network to fit instead of the subject's own (e.g. null_target);
    T is its edge count."""
    from src import evaluation as ev
    from src.kernel import TritonGNM

    D, target_fn, _ = load_griffa(scale)
    n = D.shape[0]
    A_t = target_fn(subject) if target is None else target.cpu().bool()
    T = int(A_t.sum()) // 2
    D = D.to(device)
    D1 = D.clone()
    D1.fill_diagonal_(1.0)  # the kernel's contract: masked diagonal, avoids 0 ** eta
    targets = ev.Targets.from_adjacency(A_t.to(device), D)
    eta, gamma = grid.params()
    N = len(eta)
    batch = batch or default_batch(n)
    ks = torch.empty(N, 4, dtype=torch.float64)
    t_gen = t_eval = 0.0
    checks = []
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    t_start = time.perf_counter()
    for bi, lo in enumerate(range(0, N, batch)):
        hi = min(N, lo + batch)
        torch.manual_seed(seed * 1_000_003 + subject * 1009 + bi)  # kernel seed from torch RNG
        sync()
        t0 = time.perf_counter()
        sim = TritonGNM(D1, eta[lo:hi].to(device), gamma[lo:hi].to(device))
        edges = sim.run(T)
        sync()
        t1 = time.perf_counter()
        checks.append(validate(sim, edges, n_validate))
        sync()
        t2 = time.perf_counter()
        res = ev.evaluate(sim.A, D, targets, C=sim.C, deg=sim.deg, edges=edges,
                          mem_bytes=mem_bytes)
        ks[lo:hi] = res.ks[:, 0].cpu()
        sync()
        t3 = time.perf_counter()
        t_gen += t1 - t0
        t_eval += t3 - t2
        del sim, edges, res
        log(f"  scale {scale} sub {subject:02d}: {hi}/{N} sims, gen {t_gen:.1f}s, "
            f"eval {t_eval:.1f}s, valid {checks[-1]['ok']}")
    den = np.array([n, n, n, T], dtype=np.float64)
    shape = (grid.n_gamma, grid.n_eta, grid.repeats, 4)
    return {
        "ks_num": encode_ks(ks, den).reshape(shape),
        "ks_den": den,
        "etas": grid.etas,
        "gammas": grid.gammas,
        "meta": json.dumps({
            "scale": scale, "subject": subject, "n": n, "T": T, "grid": asdict(grid),
            "target": "griffa" if target is None else "custom",
            "batch": batch, "seed": seed, "stats": list(ev.STATS),
            "t_gen_s": t_gen, "t_eval_s": t_eval,
            "t_total_s": time.perf_counter() - t_start,
            "validation": {
                "ok": all(c["ok"] for c in checks),
                "R_max_rel_err": max(c["R_max_rel_err"] for c in checks),
                "checked": sum(c["checked"] for c in checks),
            },
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
        }),
    }


def to_bytes(result: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **result)
    return buf.getvalue()


def load(path) -> dict:
    """Saved landscape -> dict with float KS [gamma, eta, repeat, 4] and energy."""
    z = np.load(path)
    ks = z["ks_num"].astype(np.float64) / z["ks_den"]
    return {"ks": ks, "energy": ks.max(-1), "etas": z["etas"], "gammas": z["gammas"],
            "meta": json.loads(str(z["meta"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="033", choices=list(SCALES))
    ap.add_argument("--subject", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    for f, v in asdict(Grid()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()
    grid = Grid(**{f: getattr(args, f) for f in asdict(Grid())})
    res = run_subject(args.scale, args.subject, grid, torch.device("cuda"))
    out = args.out or ROOT / "results" / f"landscape_scale{args.scale}_sub{args.subject:02d}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(to_bytes(res))
    print(json.loads(res["meta"]))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
