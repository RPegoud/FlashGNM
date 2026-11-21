"""Summaries of the rule-form landscapes (SPEC H9).

    python src/analysis_rules.py results/rules/add_exp_wide results/rules/mult_exp_wide ...

Per run and scale, per subject: optimum (argmin of the mean energy over repeats), best
mean energy and its SD, near-optimal set (cells within 2 SE of the best, SE of the
difference) with its size and extent per parameter, whether the optimum is interior (not
on a grid face), and which KS component binds at the optimum. Writes <run>/summary.csv
and <run>/summary.json (medians per scale), and results/rules/optima.json ({scale:
{subject: {rule: [eta, gamma, alpha]}}}) for src/modal_rules.py::recovery.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rules_landscape import load  # noqa: E402

STATS = ("degree", "clustering", "betweenness", "edge_length")


def analyse(L: dict) -> dict:
    E = L["energy"]  # [alpha, gamma, eta, rep]
    R = E.shape[-1]
    mean, sd = E.mean(-1), E.std(-1, ddof=1)
    se = sd / np.sqrt(R)
    idx = np.unravel_index(np.argmin(mean), mean.shape)
    near = mean - mean[idx] <= 2 * np.sqrt(se**2 + se[idx] ** 2)
    axes = {"alpha": L["alphas"], "gamma": L["gammas"], "eta": L["etas"]}
    out = {"E_opt": float(mean[idx]), "E_opt_sd": float(sd[idx]), "near_cells": int(near.sum()),
           "cells": int(mean.size)}
    interior = True
    for k, (name, ax) in enumerate(axes.items()):
        out[f"{name}_opt"] = float(ax[idx[k]])
        nz = np.flatnonzero(near.any(axis=tuple(j for j in range(3) if j != k)))
        out[f"near_{name}_lo"], out[f"near_{name}_hi"] = float(ax[nz.min()]), float(ax[nz.max()])
        if len(ax) > 1 and idx[k] in (0, len(ax) - 1):
            interior = False
    out["interior"] = interior
    ks = L["ks"][idx].mean(0)  # [4]
    out.update({f"ks_opt_{s}": float(v) for s, v in zip(STATS, ks)})
    out["binding"] = STATS[int(np.argmax(ks))]
    return out


def main(runs: list[Path]):
    optima_path = ROOT / "results" / "rules" / "optima.json"
    optima = json.loads(optima_path.read_text()) if optima_path.exists() else {}
    for run in runs:
        rows = []
        for f in sorted(run.glob("scale*/sub*.npz")):
            L = load(f)
            m = L["meta"]
            a = analyse(L)
            row = {"scale": m["scale"], "subject": m["subject"], "n": m["n"], "T": m["T"],
                   "rule": m["rule"], **a, "valid": m["validation"]["ok"],
                   "rec_per_step": m["row_max_recomputes_per_step"],
                   "resyncs_per_sim": m["resyncs_per_sim"],
                   "t_gen_s": m["t_gen_s"], "t_eval_s": m["t_eval_s"]}
            rows.append(row)
            optima.setdefault(m["scale"], {}).setdefault(str(m["subject"]), {})[m["rule"]] = [
                a["eta_opt"], a["gamma_opt"], a["alpha_opt"]]
        if not rows:
            continue
        with open(run / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        summ = {}
        for s in sorted({r["scale"] for r in rows}):
            rs = [r for r in rows if r["scale"] == s]
            med = lambda k: float(np.median([r[k] for r in rs]))  # noqa: E731
            q = lambda k: [float(np.quantile([r[k] for r in rs], x)) for x in (.25, .75)]  # noqa: E731
            summ[s] = {
                "subjects": len(rs), "rule": rs[0]["rule"],
                "E_opt": med("E_opt"), "E_opt_sd": med("E_opt_sd"),
                "eta_opt": med("eta_opt"), "eta_iqr": q("eta_opt"),
                "gamma_opt": med("gamma_opt"), "gamma_iqr": q("gamma_opt"),
                "alpha_opt": med("alpha_opt"), "alpha_iqr": q("alpha_opt"),
                "near_cells": med("near_cells"), "cells": rs[0]["cells"],
                "near_share": med("near_cells") / rs[0]["cells"],
                "near_eta_width": float(np.median([r["near_eta_hi"] - r["near_eta_lo"] for r in rs])),
                "near_gamma_width": float(np.median([r["near_gamma_hi"] - r["near_gamma_lo"] for r in rs])),
                "near_alpha_width": float(np.median([r["near_alpha_hi"] - r["near_alpha_lo"] for r in rs])),
                "interior": sum(r["interior"] for r in rs),
                "binding": {k: sum(r["binding"] == k for r in rs) for k in STATS},
                "all_valid": all(r["valid"] for r in rs),
                "rec_per_step": med("rec_per_step"), "resyncs_per_sim": med("resyncs_per_sim"),
                "t_gen_s": med("t_gen_s"), "t_eval_s": med("t_eval_s"),
            }
            print(run.name, s, json.dumps(summ[s]))
        (run / "summary.json").write_text(json.dumps(summ, indent=2))
    optima_path.write_text(json.dumps(optima, indent=1))


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
