"""Package and validate Griffa et al. cortical SC targets (Lausanne scales 033-500, n = 68-1000).

Inputs (produced before the event by prep/griffa/: convert_modal.py, centroids.py, build_distances.py):
  GRIFFA/griffa_sc_scale{s}_sparse.npz   strict upper triangle per subject (Zenodo 2872624, CC BY 4.0)
  ATLAS/lausanne_scale{s}_centroids_mni.csv, ATLAS/MNI152NLin2009aSym/atl-Cammoun2012_..._info.csv  (netneurotools)

    python -I prep/griffa_package.py --griffa DIR --atlas DIR --out data/griffa

Outputs per scale, data/griffa/scale{s}.npz, same fields as data/mica:
  n, subjects (0..69), i, j (int16, i < j), sub (uint8), w (float32 fibre density > 0), dist_mm (n x n float32,
  Euclidean between template centroids), hemi (0 = L, 1 = R; Griffa order is R first), names, xyz, yeo7, von_economo.
Validation goes to data/griffa/validation.json.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCALES = ("033", "060", "125", "250", "500")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--griffa", required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    info = {}
    for r in csv.DictReader(open(next(Path(a.atlas).glob("MNI152NLin2009aSym/*info.csv")))):
        info[(r["scale"], r["id"])] = r
    report = {"source": "Griffa, Aleman-Gomez & Hagmann, Zenodo 2872624 (CC BY 4.0); Lausanne (Cammoun 2012) "
                        "MNI152NLin2009aSym centroids from netneurotools", "scales": {}}
    for s in SCALES:
        z = np.load(f"{a.griffa}/griffa_sc_scale{s}_sparse.npz")
        n, S = int(z["n"]), int(z["n_subjects"])
        rows = [r for r in csv.DictReader(open(f"{a.atlas}/lausanne_scale{s}_centroids_mni.csv"))
                if r["structure"] == "cortex"]
        assert len(rows) == n
        xyz = np.array([[float(r[k]) for k in "xyz"] for r in rows])
        D = np.sqrt(((xyz[:, None] - xyz[None]) ** 2).sum(-1)).astype(np.float32)
        D_saved = np.load(f"{a.atlas}/griffa_order_scale{s}_cortex_dist_mm.npy")
        meta = [info[(f"scale{s}", r["id"])] for r in rows]
        hemi = np.array([0 if r["hemisphere"] == "L" else 1 for r in rows], np.uint8)
        i, j, sub, w = z["i"], z["j"], z["subject"], z["w"]
        iu = np.triu_indices(n, 1)
        cons = np.zeros((n, n)); np.add.at(cons, (i, j), 1.0 / S)
        A = np.zeros((S, n, n), bool); A[sub, i, j] = True
        edges = A.sum((1, 2))
        r_true = spearmanr(D[iu], cons[iu])[0]
        null = []
        for _ in range(a.perms):
            q = np.concatenate([rng.permutation(np.where(hemi == h)[0]) for h in (1, 0)])
            Dq = D[np.ix_(q, q)]
            null.append(spearmanr(Dq[iu], cons[iu])[0])
        C = cons + cons.T
        near5 = np.argsort(D + np.eye(n) * 1e9, 1)[:, :5]
        hits = np.array([(C[k, near5[k]] >= 0.5).sum() for k in range(n)])
        checks = {
            "source_symmetry_error": float(z["sym_err"]), "source_diag_max_dropped": float(z["diag_max"]),
            "strict_upper": bool(np.all(i < j)), "positive_weights": bool(np.all(w > 0)),
            "no_duplicates": bool(len(np.unique(np.stack([sub, i, j]), axis=1).T) == len(i)),
            "dist_matches_saved_matrix": bool(np.allclose(D, D_saved, atol=1e-3)),
            "dist_symmetric_zero_diag": bool(np.allclose(D, D.T) and np.all(np.diag(D) == 0)),
            "hemisphere_order_R_then_L": bool(np.all(np.diff(hemi.astype(int)) <= 0) and hemi[0] == 1),
        }
        assert all(v for k, v in checks.items() if isinstance(v, bool)), checks
        np.savez_compressed(
            out / f"scale{s}.npz", n=n, subjects=np.arange(S), i=i.astype(np.int16), j=j.astype(np.int16),
            sub=sub.astype(np.uint8), w=w.astype(np.float32), dist_mm=D, hemi=hemi,
            names=np.array([r["label"] for r in rows]), xyz=xyz.astype(np.float32),
            yeo7=np.array([m["yeo_7"] for m in meta]), von_economo=np.array([m["von_economo"] for m in meta]),
        )
        report["scales"][s] = {
            "n": n, "subjects": S, "checks": checks,
            "edges_mean_sd_min_max": [float(edges.mean()), float(edges.std()), int(edges.min()), int(edges.max())],
            "density_mean": float(edges.mean() / len(iu[0])),
            "order_check": {
                "spearman_dist_vs_consensus": round(float(r_true), 4),
                "null_within_hemi_perm_mean_sd": [round(float(np.mean(null)), 4), round(float(np.std(null)), 4)],
                "nodes_with_a_5_nearest_neighbour_at_consensus_0.5": round(float((hits > 0).mean()), 4),
            },
        }
        print(s, json.dumps(report["scales"][s]), flush=True)
    json.dump(report, open(out / "validation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
