"""Oldham et al.'s Voronoi search (one simulation per point) against replicated grids (H3, H9).

    python src/compare_voronoi.py optima        # -> results/rules/optima_vor.json
    modal run src/modal_rules.py::recovery --optima results/rules/optima_vor.json \
        --out results/rules/recovery_vor
    python src/compare_voronoi.py compare       # -> results/rules/voronoi_vs_grid.md

Per subject and rule:
- the search's optimum: the point with the lowest single-run maxKS (as in their code);
- its reported energy (that single run) against the held-out energy of fresh networks at
  the same point (winner's curse), and against the held-out energy at the replicated
  zoom grid's optimum;
- where it lands relative to the zoom grid's optimum, per parameter.
Across subjects: spread of the optima and the eta-gamma correlation, search vs grid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RULES = ["add_exp", "mult_exp", "mult_pow"]
SCALES = ["033", "125"]
R = ROOT / "results" / "rules"


def vor_optima():
    out = {}
    for rule in RULES:
        for s in SCALES:
            for f in sorted((R / f"vor_{rule}" / f"scale{s}").glob("sub*.json")):
                v = json.loads(f.read_text())
                E = np.max(np.array(v["ks"]), -1)
                k = int(np.argmin(E))
                out.setdefault(s, {}).setdefault(str(v["subject"]), {})[rule] = v["params"][k]
                out[s][str(v["subject"])].setdefault("_reported", {})[rule] = float(E[k])
    return out


def jl(path):
    return {(r["scale"], r["subject"]): r for r in map(json.loads, open(path))}


def compare():
    vo = json.loads((R / "optima_vor.json").read_text())
    rows = ["| rule | n | search: reported energy | search: held-out at its optimum | grid: held-out at its optimum | search worse than grid (subjects) | Wilcoxon p | median abs. diff from grid optimum (eta, gamma, alpha) | between-subject SD of gamma, search / grid | corr(eta, gamma), search / grid |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for s in SCALES:
        hv = jl(R / "recovery_vor" / f"scale{s}.jsonl")
        hg = jl(R / "recovery" / f"scale{s}.jsonl")
        n = {"033": 68, "125": 219}[s]
        for rule in RULES:
            subs = sorted(int(k) for k in vo[s])
            rep = np.array([vo[s][str(i)]["_reported"][rule] for i in subs])
            ev = np.array([hv[(s, i)][rule]["heldout_energy"] for i in subs])
            eg = np.array([hg[(s, i)][rule]["heldout_energy"] for i in subs])
            pv = np.array([vo[s][str(i)][rule] for i in subs])  # eta, gamma, alpha
            pg = np.array([hg[(s, i)]["optima"][rule] for i in subs])
            d = np.median(np.abs(pv - pg), 0)
            dl = ", ".join(f"{x:.2f}" for x in (d if rule == "add_exp" else d[:2]))
            corr_v = stats.pearsonr(pv[:, 0], pv[:, 1])[0]
            corr_g = stats.pearsonr(pg[:, 0], pg[:, 1])[0] if pg[:, 1].std() > 0 else float("nan")
            rows.append(
                f"| {rule} | {n} | {np.median(rep):.3f} | {np.median(ev):.3f} | {np.median(eg):.3f} | "
                f"{(ev > eg).sum()}/70 | {stats.wilcoxon(ev - eg).pvalue:.1e} | {dl} | "
                f"{pv[:, 1].std(ddof=1):.2f} / {pg[:, 1].std(ddof=1):.2f} | {corr_v:+.2f} / {corr_g:+.2f} |")
    text = "\n".join(rows) + "\n"
    (R / "voronoi_vs_grid.md").write_text(text)
    print(text)


if __name__ == "__main__":
    if sys.argv[1] == "optima":
        (R / "optima_vor.json").write_text(json.dumps(vor_optima(), indent=1))
        print("wrote", R / "optima_vor.json")
    else:
        compare()
