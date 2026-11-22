"""Data for the 5-minute talk clips (A-E) -> src/data/talk.json, src/data/connectome.json.

    python3 scripts/talk_export.py

Reads only files committed in this repo (results/, data/griffa/). Nothing typed by hand.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "src" / "data"


def landscape_cost() -> dict:
    """One subject's full landscape: toolbox (extrapolated) vs ours (measured), n = 1000."""
    run = ROOT / "results/landscape_griffa_full"
    grid = json.loads((run / "meta.json").read_text())["grid"]
    sims = grid["n_eta"] * grid["n_gamma"] * grid["repeats"]
    rows = [r for r in csv.DictReader(open(run / "summary.csv")) if r["scale"] == "500"]
    t_ours = np.array([float(r["t_total_s"]) for r in rows])
    tb = [json.loads(l) for l in open(ROOT / "results/26-11-21_14-40-12_sim_baseline/results.jsonl")]
    tb = next(r for r in tb if r["impl"] == "toolbox" and r["n"] == 1000 and r["B"] == 1024)
    t_tool = sims / tb["sims_per_s"]
    return {
        "source": ["results/landscape_griffa_full/{meta.json,summary.csv}",
                   "results/26-11-21_14-40-12_sim_baseline/results.jsonl (toolbox, n=1000, B=1024)"],
        "n": 1000, "grid": [grid["n_eta"], grid["n_gamma"]], "repeats": grid["repeats"],
        "sims_per_subject": sims, "subjects": len(rows),
        "toolbox_sims_per_s": tb["sims_per_s"], "toolbox_extrapolated": bool(tb.get("extrapolated")),
        "toolbox_s_per_subject": t_tool,
        "ours_s_per_subject_median": float(np.median(t_ours)),  # simulation + scoring, measured
        "ours_s_cohort_total": float(t_ours.sum()),
        "toolbox_s_cohort_total": t_tool * len(rows),
    }


def connectome(scale: str = "500", subject: int = 0) -> dict:
    d = np.load(ROOT / f"data/griffa/scale{scale}.npz", allow_pickle=True)
    xyz = d["xyz"].astype(float)
    xyz -= xyz.mean(0)
    m = d["sub"] == subject
    E = np.stack([d["i"][m], d["j"][m]], 1).astype(int)
    rng = np.random.default_rng(0)
    E = E[rng.permutation(len(E))]  # reveal order
    return {
        "source": f"data/griffa/scale{scale}.npz (Griffa et al., Lausanne {int(d['n'])} regions), subject {subject}",
        "n": int(d["n"]), "subject": subject,
        "xyz": np.round(xyz, 1).tolist(),  # MNI mm, centred: x right-left, y posterior-anterior, z inferior-superior
        "hemi": d["hemi"].astype(int).tolist(),  # order: right then left (validation.json)
        "edges": E.tolist(),
        "n_edges": int(len(E)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lc = landscape_cost()
    (OUT / "talk.json").write_text(json.dumps({"landscape": lc}, indent=1))
    print(f"landscape: {lc['sims_per_subject']:,} sims/subject; toolbox "
          f"{lc['toolbox_s_per_subject'] / 86400:.2f} days vs ours {lc['ours_s_per_subject_median']:.1f} s; "
          f"cohort {lc['toolbox_s_cohort_total'] / 86400 / 365:.2f} yr vs {lc['ours_s_cohort_total'] / 3600:.2f} h")
    c = connectome()
    (OUT / "connectome.json").write_text(json.dumps(c))
    print(f"connectome: n={c['n']}, edges={c['n_edges']}")


if __name__ == "__main__":
    main()
