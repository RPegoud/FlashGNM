"""Fraction of edges longer than 90 mm: all pairs, individual subjects, and consensus thresholds.

    python prep/griffa/long_range.py [griffa_dir] [atlas_dir]

Shows that uniform consensus thresholding removes long edges at high resolution, which matters
for target construction (see Claude.md "Griffa data").
"""
import sys

import numpy as np

from common import DEFAULT_ATLAS, DEFAULT_GRIFFA, SCALES, load_sparse

gdir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GRIFFA
adir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ATLAS
print("scale  frac>90mm: all-pairs | individual (mean) | consensus>=0.3 / 0.5 / 0.75")
for s in SCALES:
    z, n, S, cons = load_sparse(gdir, s)
    D = np.load(f"{adir}/griffa_order_scale{s}_cortex_dist_mm.npy")
    d = D[z["i"], z["j"]]
    ind = np.mean([(d[z["subject"] == k] > 90).mean() for k in range(S)])
    iu = np.triu_indices(n, 1)
    print(f"{s}   {(D[iu] > 90).mean():.3f} | {ind:.3f} | "
          + " / ".join(f"{(D[cons >= t] > 90).mean():.3f}" for t in (0.3, 0.5, 0.75)))
