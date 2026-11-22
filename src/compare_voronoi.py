"""Oldham et al.'s Voronoi search against replicated grids (H3, H9), with replicated
variants of the search.

    python src/compare_voronoi.py optima VARIANT     # -> results/rules/optima_<VARIANT>.json
    modal run src/modal_rules.py::recovery --optima results/rules/optima_<VARIANT>.json \
        --out results/rules/recovery_<VARIANT>
    python src/compare_voronoi.py compare            # -> results/rules/voronoi_vs_grid.md

Variants (run directories results/rules/<prefix>_<rule>):
    vor   their procedure: 5 levels x 2000 points, 1 simulation per point   (10,000 sims)
    vorA  same budget, replicated: 5 x 400 points x 5 simulations          (10,000 sims)
    vorB  their points, replicated: 5 x 2000 points x 20 simulations        (200,000 sims)
The search's optimum is the point with the lowest (mean) energy. Its "reported" energy is
that value; the held-out energy is the mean over 100 fresh networks at the same point.
The replicated zoom grid's optimum, scored the same way, is the reference.
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
VARIANTS = {"vor": "single run (their procedure)", "vorA": "same budget, 5 repeats",
            "vorB": "20 repeats per point"}


def optima(prefix: str) -> dict:
    out = {}
    for rule in RULES:
        for s in SCALES:
            for f in sorted((R / f"{prefix}_{rule}" / f"scale{s}").glob("sub*.json")):
                v = json.loads(f.read_text())
                E = np.array(v["energy"]) if "energy" in v else np.max(np.array(v["ks"]), -1)
                k = int(np.argmin(E))
                d = out.setdefault(s, {}).setdefault(str(v["subject"]), {})
                d[rule] = v["params"][k]
                d.setdefault("_reported", {})[rule] = float(E[k])
    return out


def jl(path):
    return {(r["scale"], r["subject"]): r for r in map(json.loads, open(path))}


def compare():
    rows = ["| search | rule | n | reported | held-out at its optimum | optimism | grid held-out | worse than grid (subjects) | p |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for prefix, label in VARIANTS.items():
        op = R / ("optima_vor.json" if prefix == "vor" else f"optima_{prefix}.json")
        rec = R / ("recovery_vor" if prefix == "vor" else f"recovery_{prefix}")
        if not op.exists() or not rec.exists():
            continue
        vo = json.loads(op.read_text())
        for s in SCALES:
            if not (rec / f"scale{s}.jsonl").exists():
                continue
            hv = jl(rec / f"scale{s}.jsonl")
            hg = jl(R / "recovery" / f"scale{s}.jsonl")
            n = {"033": 68, "125": 219}[s]
            for rule in RULES:
                subs = [i for i in range(70) if (s, i) in hv and rule in hv[(s, i)]]
                if not subs:
                    continue
                rep = np.array([vo[s][str(i)]["_reported"][rule] for i in subs])
                ev = np.array([hv[(s, i)][rule]["heldout_energy"] for i in subs])
                eg = np.array([hg[(s, i)][rule]["heldout_energy"] for i in subs])
                p = stats.wilcoxon(ev - eg).pvalue if len(subs) > 5 else float("nan")
                rows.append(
                    f"| {label} | {rule} | {n} | {np.median(rep):.3f} | {np.median(ev):.3f} | "
                    f"{np.median(ev - rep):+.3f} | {np.median(eg):.3f} | "
                    f"{(ev > eg).sum()}/{len(subs)} | {p:.1e} |")
    text = "\n".join(rows) + "\n"
    (R / "voronoi_vs_grid.md").write_text(text)
    print(text)


if __name__ == "__main__":
    if sys.argv[1] == "optima":
        prefix = sys.argv[2] if len(sys.argv) > 2 else "vor"
        path = R / ("optima_vor.json" if prefix == "vor" else f"optima_{prefix}.json")
        path.write_text(json.dumps(optima(prefix), indent=1))
        print("wrote", path)
    else:
        compare()
