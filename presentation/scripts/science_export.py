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


def main(root: Path):
    OUT.mkdir(parents=True, exist_ok=True)
    d = s1(root)
    (OUT / "science_s1.json").write_text(json.dumps(d, indent=1))
    for s in ("033", "500"):
        x = d[s]
        print(f"S1 {s}: corr real {x['corr_real']:+.3f} null {x['corr_null']:+.3f}, "
              f"sd_gamma {x['sd_gamma_real']:.3f} / {x['sd_gamma_null']:.3f}, "
              f"F p {x['F_p_gamma_real_gt_null']:.1e}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
