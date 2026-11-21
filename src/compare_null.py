"""Real subjects vs synthetic-subject null (SPEC H5).

    python src/analysis_individual.py results/landscape_griffa_null --out results/analysis_null
    python src/compare_null.py

The null's "subjects" are model networks generated at one (eta, gamma) per scale
(src/landscape.py NULL_PARAMS) with each real subject's T. Any spread of their optima, and
any eta-gamma correlation across them, comes from estimation noise and single-network
target variability alone. Compare against the real cohort: if the real between-subject SD
and correlation are no larger than the null's, the data don't support individual
parameters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.landscape import NULL_PARAMS  # noqa: E402


def main(real=ROOT / "results/analysis_individual", null=ROOT / "results/analysis_null",
         out=ROOT / "results/analysis_null/compare.md"):
    R = json.loads((real / "summary.json").read_text())["scales"]
    N = json.loads((null / "summary.json").read_text())["scales"]
    zr = np.load(real / "per_subject.npz")
    zn = np.load(null / "per_subject.npz")
    rows = ["| scale | true (eta, gamma) | null median optimum | null bias (eta, gamma) | "
            "eta between SD real / null | gamma between SD real / null | gamma ratio | "
            "corr(eta, gamma) real / null | F-test p (gamma var real > null) |",
            "|---|---|---|---|---|---|---:|---|---:|"]
    from scipy import stats
    for s in R:
        if s not in N:
            continue
        te, tg = NULL_PARAMS[s]
        ne, ng = zn[f"{s}_eta"], zn[f"{s}_gamma"]
        re_, rg = zr[f"{s}_eta"], zr[f"{s}_gamma"]
        F = rg.var(ddof=1) / ng.var(ddof=1)
        p = stats.f.sf(F, len(rg) - 1, len(ng) - 1)
        rows.append(
            f"| {s} | ({te}, {tg}) | ({np.median(ne):.2f}, {np.median(ng):.2f}) | "
            f"({np.mean(ne) - te:+.2f}, {np.mean(ng) - tg:+.2f}) | "
            f"{re_.std(ddof=1):.3f} / {ne.std(ddof=1):.3f} | "
            f"{rg.std(ddof=1):.2f} / {ng.std(ddof=1):.2f} | {np.sqrt(F):.2f} | "
            f"{R[s]['corr_replicated']:+.2f} / {N[s]['corr_replicated']:+.2f} | {p:.1e} |")
    text = "\n".join(rows) + "\n"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
