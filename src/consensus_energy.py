"""Energy of the leave-one-out group-consensus network (SPEC H7 follow-up).

    python src/consensus_energy.py [--scales 033 500] [--mem-gb 0.4]

For each subject, the consensus network (the subject's T most frequent edges among the
other 69 subjects, src/recovery.group_consensus) scored with the same max-KS energy as the
models (src/evaluation.py, template distances). Compared with:
- the classic rule's best mean energy (results/landscape_griffa_full/summary.csv);
- held-out energies of Oldham's rule forms (results/rules/recovery, n = 68 and 219);
- the empirical noise floor (results/noise_floor).
Recovery of the consensus network is in results/recovery (group_consensus).

Question: is energy (distributions) aligned with wiring (which edges)? The consensus
recovers ~10x more edges than the model; if its energy is no better, the two axes differ.
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
from src.recovery import group_consensus  # noqa: E402


@torch.no_grad()
def consensus_ks(scale: str, mem_bytes: int) -> np.ndarray:
    """[S, 4] KS of each subject's consensus network against the subject."""
    D, target, S = load_griffa(scale)
    A_all = torch.stack([target(s) for s in range(S)])
    G = torch.stack([group_consensus(A_all, s)[1] for s in range(S)])
    out = np.empty((S, 4))
    for s in range(S):
        tg = ev.Targets.from_adjacency(A_all[s], D, mem_bytes=mem_bytes)
        out[s] = ev.evaluate(G[s:s + 1], D, tg, mem_bytes=mem_bytes).ks[0, 0].numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", default=list(SCALES))
    ap.add_argument("--mem-gb", type=float, default=0.4)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "consensus_energy")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    classic = {}
    with open(ROOT / "results/landscape_griffa_full/summary.csv") as f:
        for r in csv.DictReader(f):
            classic[(r["scale"], int(r["subject"]))] = float(r["E_opt"])
    floor = json.loads((ROOT / "results/noise_floor/summary.json").read_text())
    rules = {}
    for s in ("033", "125"):
        p = ROOT / f"results/rules/recovery/scale{s}.jsonl"
        if p.exists():
            for r in map(json.loads, open(p)):
                rules[(s, r["subject"])] = {k: r[k]["heldout_energy"]
                                            for k in ("add_exp", "mult_exp", "mult_pow")}
    path = args.out / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for s in args.scales:
        ks = consensus_ks(s, int(args.mem_gb * 1024**3))
        np.save(args.out / f"ks_scale{s}.npy", ks)
        E = ks.max(-1)
        S = len(E)
        m = np.array([classic[(s, i)] for i in range(S)])
        row = {
            "n": SCALES[s],
            "consensus_energy_median": float(np.median(E)),
            "consensus_ks_median": dict(zip(ev.STATS, np.median(ks, 0).tolist())),
            "consensus_binding": {k: int((ks.argmax(-1) == i).sum()) for i, k in enumerate(ev.STATS)},
            "classic_model_energy_median": float(np.median(m)),
            "consensus_better_than_classic": int((E < m).sum()),
            "empirical_floor_median": floor[s]["emp_energy_median"],
        }
        if (s, 0) in rules:
            for k in ("add_exp", "mult_exp", "mult_pow"):
                h = np.array([rules[(s, i)][k] for i in range(S)])
                row[f"{k}_heldout_median"] = float(np.median(h))
                row[f"consensus_better_than_{k}"] = int((E < h).sum())
        summary[s] = row
        print(s, json.dumps(row), flush=True)
        path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
