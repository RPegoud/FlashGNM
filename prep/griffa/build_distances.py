"""Euclidean centroid distance matrices in Griffa order, plus edge-count summary per scale.

    python prep/griffa/build_distances.py [griffa_dir] [atlas_dir]

Writes <atlas_dir>/griffa_order_scale{s}_cortex_dist_mm.npy (float64, symmetric, zero diagonal,
as the toolbox's checks require). Griffa order = atlas id order with subcortex removed (R first);
verified by check_order.py.
"""
import sys

import numpy as np

from common import DEFAULT_ATLAS, DEFAULT_GRIFFA, SCALES, euclid, load_cortex_centroids, load_sparse

gdir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GRIFFA
adir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ATLAS
print("scale    n  edges/subj mean sd min max  density | consensus edges at frac>=0.3 0.5 0.75 1.0")
for s in SCALES:
    z, n, S, cons = load_sparse(gdir, s)
    rows, xyz = load_cortex_centroids(adir, s)
    assert len(rows) == n
    D = euclid(xyz)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    assert np.array_equal(D, D.T) and (np.diag(D) == 0).all()
    np.save(f"{adir}/griffa_order_scale{s}_cortex_dist_mm.npy", D)
    e = np.bincount(z["subject"], minlength=S)
    npairs = n * (n - 1) // 2
    print(f"{s}  {n:5d}  {e.mean():7.0f} {e.std():5.0f} {e.min():6d} {e.max():6d}  {e.mean() / npairs:.4f} | "
          + " ".join(f"{(cons >= t).sum():6d}" for t in (0.3, 0.5, 0.75, 1.0)))
