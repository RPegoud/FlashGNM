"""Export the numbers plotted in the science clips (S1-S3) to src/data/science_*.json.

The science results live on branch worktree-analysis-h3-h7. Extract them first, e.g.

    git archive worktree-analysis-h3-h7 results src/landscape.py | tar -x -C /tmp/analysis
    # S2 also needs results/rules/vor_add_exp/scale125/sub42.json (untracked; Modal volume
    # vor_add_exp, or the analysis worktree) copied into the same tree.
    python3 scripts/science_export.py /tmp/analysis

Every number in the clips comes from these files; nothing is typed by hand.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "src" / "data"


def null_params(root: Path) -> dict:
    src = (root / "src" / "landscape.py").read_text()
    m = re.search(r"NULL_PARAMS\s*=\s*(\{.*?\})", src, re.S)
    return ast.literal_eval(m.group(1))


def ellipse(be: np.ndarray, bg: np.ndarray) -> dict:
    """1-SD covariance ellipse of a subject's bootstrap optima (grid units: eta, gamma)."""
    cov = np.cov(np.vstack([be, bg]))
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0, None)
    major = vecs[:, 1]
    return {
        "sd_major": float(np.sqrt(vals[1])),
        "sd_minor": float(np.sqrt(vals[0])),
        # angle of the major axis in (eta, gamma) data space, radians
        "angle": float(np.arctan2(major[1], major[0])),
    }


def s1(root: Path) -> dict:
    """Synthetic twins: real optima vs the synthetic-subject null (H5, B1)."""
    real = np.load(root / "results/analysis_individual/per_subject.npz")
    null = np.load(root / "results/analysis_null/per_subject.npz")
    NP = null_params(root)
    out = {"source": ["results/analysis_individual/per_subject.npz",
                      "results/analysis_null/per_subject.npz",
                      "src/landscape.py NULL_PARAMS", "results/analysis_null/compare.md"]}
    for s, n in (("033", 68), ("500", 1000)):
        re_, rg = real[f"{s}_eta"], real[f"{s}_gamma"]
        ne, ng = null[f"{s}_eta"], null[f"{s}_gamma"]
        F = rg.var(ddof=1) / ng.var(ddof=1)
        out[s] = {
            "n": n,
            "true": list(NP[s]),
            "real": {"eta": re_.round(4).tolist(), "gamma": rg.round(4).tolist()},
            "null": {"eta": ne.round(4).tolist(), "gamma": ng.round(4).tolist()},
            "real_ellipses": [ellipse(real[f"{s}_boot_eta"][k], real[f"{s}_boot_gamma"][k])
                              for k in range(len(re_))],
            "corr_real": float(np.corrcoef(re_, rg)[0, 1]),
            "corr_null": float(np.corrcoef(ne, ng)[0, 1]),
            "sd_gamma_real": float(rg.std(ddof=1)),
            "sd_gamma_null": float(ng.std(ddof=1)),
            "sd_eta_real": float(re_.std(ddof=1)),
            "sd_eta_null": float(ne.std(ddof=1)),
            "F_p_gamma_real_gt_null": float(stats.f.sf(F, len(rg) - 1, len(ng) - 1)),
            "n_subjects": int(len(re_)),
            "grid_step": 0.1,
        }
    return out


def s2(root: Path, subject: int = 42, scale: str = "125", rule: str = "add_exp") -> dict:
    """Luckiest of 10,000: Oldham-style single-run search vs replicated grid (A4).
    Same statistics as src/compare_voronoi.py on the analysis branch."""
    R = root / "results/rules"
    v = json.loads((R / f"vor_{rule}/scale{scale}/sub{subject}.json").read_text())
    P = np.array(v["params"])
    E = np.max(np.array(v["ks"]), -1)
    L = np.array(v["level"])
    k = int(np.argmin(E))
    jl = lambda f: {(r["scale"], r["subject"]): r for r in map(json.loads, open(f))}
    hv = jl(R / f"recovery_vor/scale{scale}.jsonl")
    hg = jl(R / f"recovery/scale{scale}.jsonl")
    vo = json.loads((R / "optima_vor.json").read_text())[scale]
    subs = sorted(int(x) for x in vo)
    rep = np.array([vo[str(i)]["_reported"][rule] for i in subs])
    ev = np.array([hv[(scale, i)][rule]["heldout_energy"] for i in subs])
    eg = np.array([hg[(scale, i)][rule]["heldout_energy"] for i in subs])
    a, b = hv[(scale, subject)][rule], hg[(scale, subject)][rule]
    assert abs(vo[str(subject)]["_reported"][rule] - E[k]) < 1e-12
    return {
        "source": [f"results/rules/vor_{rule}/scale{scale}/sub{subject}.json",
                   f"results/rules/recovery_vor/scale{scale}.jsonl",
                   f"results/rules/recovery/scale{scale}.jsonl",
                   "results/rules/optima_vor.json", "results/rules/voronoi_vs_grid.md"],
        "subject": subject, "n": int(v["n"]), "rule": rule, "T": int(v["T"]),
        "n_levels": int(L.max() + 1), "per_level": int((L == 0).sum()),
        # eta, gamma, level (alpha not shown in the (eta, gamma) projection)
        "points": [[round(float(P[i, 0]), 4), round(float(P[i, 1]), 3), int(L[i])] for i in range(len(P))],
        "best": {"eta": float(P[k, 0]), "gamma": float(P[k, 1]), "alpha": float(P[k, 2]),
                 "reported": float(E[k])},
        "heldout_search": {"mean": a["heldout_energy"], "sd": a["heldout_energy_sd"]},
        "heldout_grid": {"mean": b["heldout_energy"], "sd": b["heldout_energy_sd"],
                         "optimum": hg[(scale, subject)]["optima"][rule]},
        "n_fresh": 100,  # modal_rules.py recovery(n_sims=100)
        "cohort": {"worse": int((ev > eg).sum()), "n": len(subs),
                   "wilcoxon_p": float(stats.wilcoxon(ev - eg).pvalue),
                   "median_optimism": float(np.median(ev - rep)),
                   "median_gap": float(np.median(ev - eg)),
                   "subject_optimism": float(ev[subs.index(subject)] - rep[subs.index(subject)]),
                   "subject_gap": float(ev[subs.index(subject)] - eg[subs.index(subject)])},
        "box": {"eta": [-2.0, 0.0], "gamma": [-10.0, 10.0]},
    }


def s3(root: Path) -> dict:
    """Recall = chance: per distance bin, recall of the target's edges vs the model's own
    density in that bin (A3). Mean over subjects of the per-subject records."""
    def panel(path, model, label, n):
        R = [json.loads(l) for l in open(path)]
        rc = np.nanmean([r[model]["recall_per_bin"] for r in R], 0)
        ch = np.nanmean([r[model]["chance_per_bin"] for r in R], 0)
        return {"label": label, "n": n, "model": model, "subjects": len(R),
                "recall": rc.tolist(), "chance": ch.tolist(),
                "long_top25_hit": float(np.mean([r[model]["long_top25_hit"] for r in R])),
                "long_base_rate": float(np.mean([r[model]["long_base_rate"] for r in R]))}
    rec = root / "results/recovery"
    rul = root / "results/rules/recovery"
    first = json.loads(open(rec / "scale500.jsonl").readline())
    bins = [b if np.isfinite(b) else None for b in first["homophily"]["bins_mm"]]  # inf -> null
    return {
        "source": ["results/recovery/scale{033,125,500}.jsonl",
                   "results/rules/recovery/scale125.jsonl"],
        "bins_mm": bins,
        "classic": [panel(rec / f"scale{s}.jsonl", "homophily", "classic rule", n)
                    for s, n in (("033", 68), ("125", 219), ("500", 1000))],
        "rules": [panel(rul / "scale125.jsonl", m, lab, 219) for m, lab in
                  (("add_exp", "additive rule"), ("mult_exp", "multiplicative, exponential"),
                   ("mult_pow", "multiplicative, power law"))],
        "consensus": [panel(rec / f"scale{s}.jsonl", "group_consensus", "other brains (group consensus)", n)
                      for s, n in (("033", 68), ("125", 219), ("500", 1000))],
    }


def main(root: Path):
    OUT.mkdir(parents=True, exist_ok=True)
    d = s1(root)
    (OUT / "science_s1.json").write_text(json.dumps(d, indent=1))
    for s in ("033", "500"):
        x = d[s]
        print(f"S1 {s}: corr real {x['corr_real']:+.3f} null {x['corr_null']:+.3f}, "
              f"sd_gamma {x['sd_gamma_real']:.3f} / {x['sd_gamma_null']:.3f}, "
              f"F p {x['F_p_gamma_real_gt_null']:.1e}")
    d2 = s2(root)
    (OUT / "science_s2.json").write_text(json.dumps(d2))
    print(f"S2 sub {d2['subject']}: reported {d2['best']['reported']:.3f}, fresh "
          f"{d2['heldout_search']['mean']:.3f}+-{d2['heldout_search']['sd']:.3f}, grid "
          f"{d2['heldout_grid']['mean']:.3f}; cohort {d2['cohort']}")
    d3 = s3(root)
    (OUT / "science_s3.json").write_text(json.dumps(d3, indent=1))
    c, g = d3["classic"][2], d3["consensus"][2]
    print(f"S3 bins {d3['bins_mm']}; n=1000 long-range top-25% hit: model {c['long_top25_hit']:.4f} "
          f"base {c['long_base_rate']:.4f} consensus {g['long_top25_hit']:.4f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
