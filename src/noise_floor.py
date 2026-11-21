"""Empirical noise floor (SPEC H6): max(KS) between real subjects' connectomes.

    python src/noise_floor.py [--scales 033 500] [--device cpu] [--landscapes results/landscape_griffa_full]

For each scale, every ordered pair of distinct subjects (s, t): the four KS statistics
(degree, clustering, betweenness, edge length) of s against t, with the same evaluation
code as the landscapes (src/evaluation.py) and the same distances (template centroids).
KS is symmetric, so the matrix is too, up to floating point.

Compared with the model: for each subject, the share of other subjects t whose network is
further from s (max KS) than the model's best mean energy for s (from the landscape
summary). A model energy at or below the empirical range can't be improved on max(KS).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import evaluation as ev  # noqa: E402
from src.landscape import SCALES, load_griffa  # noqa: E402


@torch.no_grad()
def pairwise(scale: str, device: torch.device, mem_bytes: int = 2 * 1024**3) -> np.ndarray:
    """[S, S, 4] KS of subject s (row) against subject t (column); diagonal NaN."""
    D, target, S = load_griffa(scale)
    A = torch.stack([target(s) for s in range(S)]).to(device)
    D = D.to(device)
    tg = ev.Targets.from_adjacency(A, D, mem_bytes=mem_bytes)
    res = ev.evaluate(A, D, tg, mem_bytes=mem_bytes)
    ks = res.ks.cpu().numpy()
    idx = np.arange(S)
    ks[idx, idx] = np.nan
    return ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", default=list(SCALES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--landscapes", type=Path, default=ROOT / "results" / "landscape_griffa_full")
    ap.add_argument("--mem-gb", type=float, default=2.0, help="evaluation temporaries")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "noise_floor")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = {}
    with open(args.landscapes / "summary.csv") as f:
        for r in csv.DictReader(f):
            model[(r["scale"], int(r["subject"]))] = float(r["E_opt"])
    path = args.out / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for s in args.scales:
        ks = pairwise(s, torch.device(args.device), int(args.mem_gb * 1024**3))
        np.save(args.out / f"ks_scale{s}.npy", ks)
        E = ks.max(-1)
        S = E.shape[0]
        off = ~np.eye(S, dtype=bool)
        e = E[off]
        comp = {name: float(np.median(ks[..., i][off])) for i, name in enumerate(ev.STATS)}
        binding = np.bincount(ks[off].argmax(-1), minlength=4) / off.sum()
        m = np.array([model[(s, i)] for i in range(S)])
        # per subject: share of other subjects further away than the model's best energy
        worse = np.array([np.mean(E[i, off[i]] > m[i]) for i in range(S)])
        # model energy relative to subject's median empirical distance
        rel = m / np.array([np.median(E[i, off[i]]) for i in range(S)])
        summary[s] = {
            "n": SCALES[s], "subjects": S,
            "emp_energy_median": float(np.median(e)),
            "emp_energy_q10_q90": [float(np.quantile(e, .1)), float(np.quantile(e, .9))],
            "emp_ks_median": comp,
            "emp_binding_share": dict(zip(ev.STATS, map(float, binding))),
            "model_energy_median": float(np.median(m)),
            "model_over_emp_median": float(np.median(rel)),
            "share_emp_pairs_worse_than_model_median": float(np.median(worse)),
            "subjects_model_below_emp_median": int((rel < 1).sum()),
        }
        print(s, json.dumps(summary[s]), flush=True)
        path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
