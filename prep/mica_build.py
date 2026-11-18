"""Build and validate MICA-MICs cortical SC targets (Schaefer 100/200/400/1000).

Inputs (not in the repo):
  - micapipe_MICs_v1.1.zip   OSF j532r (https://osf.io/download/x7qr2/), CC0. Royer et al. 2022, Sci Data.
  - Schaefer centroid tables, CBIG (MIT): Schaefer2018_{p}Parcels_{7,17}Networks_order_FSLMNI152_1mm.Centroid_RAS.csv

    python -I prep/mica_build.py --zip PATH --centroids DIR --out data/mica

Matrix layout in the zip (checked at every scale): (p + 16) x (p + 16), upper triangle only, nonzero diagonal.
Rows 0-13 subcortical, row 14 and row 15 + p/2 medial wall (all zero), then left cortex 15 .. 14 + p/2 and
right cortex 16 + p/2 .. 15 + p.

Outputs per scale, data/mica/schaefer{p}.npz:
  n, subjects, i, j (int16, i < j, cortex indices), sub (uint8), w (float32, SIFT2 weight > 0), dist_mm (n x n float32, Euclidean between
  CBIG MNI centroids, symmetric, zero diagonal), hemi (0 = L, 1 = R), names.
and data/mica/schaefer{p}_length.npz: length (float32, mean streamline length in mm, aligned with i, j, sub).
Validation goes to data/mica/validation.json.
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCALES = (100, 200, 400, 1000)


def load_sym(z, f):
    M = np.loadtxt(z.open(f), delimiter=",")
    assert np.count_nonzero(np.tril(M, -1)) == 0, f"{f}: lower triangle not empty"
    return M + np.triu(M, 1).T


def centroids(path):
    rows = [l.strip().split(",") for l in open(path).read().strip().splitlines()[1:]]
    names = [r[1] for r in rows]
    xyz = np.array([[float(v) for v in r[2:5]] for r in rows])
    return names, xyz


def euclid(xyz):
    return np.sqrt(((xyz[:, None] - xyz[None]) ** 2).sum(-1)).astype(np.float32)


def perm_within_hemi(n, rng):
    h = n // 2
    return np.r_[rng.permutation(h), h + rng.permutation(h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--centroids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(a.zip)
    subjects = sorted({x.split("/")[1] for x in z.namelist() if x.startswith("micapipe/sub-")})
    rng = np.random.default_rng(0)
    report = {"source": "MICA-MICs micapipe_MICs_v1.1.zip (OSF j532r, CC0)", "subjects": len(subjects), "scales": {}}

    for p in SCALES:
        h = p // 2
        ctx = np.r_[15:15 + h, 16 + h:16 + p]
        iu = np.triu_indices(p, 1)
        I, J, S, Wv, Lv = [], [], [], [], []
        Lmats, Wmats = [], []
        checks = {"medial_wall_zero": True, "diag_dropped": True, "length_missing_where_w": 0, "length_where_no_w": 0}
        for k, s in enumerate(subjects):
            base = f"micapipe/{s}/ses-01/dwi/{s}_ses-01_space-dwinative_atlas-schaefer{p}_desc-"
            Wf = load_sym(z, base + "sc.txt")
            Lf = load_sym(z, base + "edgeLength.txt")
            assert Wf.shape == (p + 16, p + 16)
            for mw in (14, 15 + h):
                checks["medial_wall_zero"] &= bool(Wf[mw].sum() == 0)
            W, L = Wf[np.ix_(ctx, ctx)], Lf[np.ix_(ctx, ctx)]
            w, l = W[iu], L[iu]
            e = w > 0
            checks["length_missing_where_w"] += int(((l <= 0) & e).sum())
            checks["length_where_no_w"] += int(((l > 0) & ~e).sum())
            I.append(iu[0][e]); J.append(iu[1][e]); S.append(np.full(e.sum(), k)); Wv.append(w[e]); Lv.append(l[e])
            Wmats.append(w); Lmats.append(l)
        Wm, Lm = np.array(Wmats), np.array(Lmats)

        # Region order: centroid distance should track each subject's tract lengths and (inversely) weights,
        # far better than under within-hemisphere permutations. Compare CBIG 7- and 17-network orders.
        mean_len = np.where((Lm > 0).sum(0) > 0, Lm.sum(0) / np.maximum((Lm > 0).sum(0), 1), np.nan)
        has = ~np.isnan(mean_len)
        mean_logw = np.log1p(Wm).mean(0)
        order = {}
        for net in (7, 17):
            names, xyz = centroids(Path(a.centroids) / f"s{p}_{net}net.csv")
            assert len(names) == p
            hemi_ok = all(n.split("_")[1] == ("LH" if i < h else "RH") for i, n in enumerate(names))
            D = euclid(xyz)
            r_len = spearmanr(D[iu][has], mean_len[has])[0]
            r_w = spearmanr(D[iu], mean_logw)[0]
            null_len = []
            for _ in range(a.perms):
                q = perm_within_hemi(p, rng)
                Dq = D[np.ix_(q, q)]
                null_len.append(spearmanr(Dq[iu][has], mean_len[has])[0])
            # top-weight neighbour among the 5 nearest centroids
            Wmean = np.zeros((p, p)); Wmean[iu] = Wm.mean(0); Wmean += Wmean.T
            Dn = D + np.eye(p) * 1e9
            near5 = np.argsort(Dn, 1)[:, :5]
            top = Wmean.argmax(1)
            nn_hit = float(np.mean([top[i] in near5[i] for i in range(p)]))
            order[f"{net}net"] = {
                "hemisphere_labels_match_rows": hemi_ok,
                "spearman_dist_vs_mean_tract_length": round(float(r_len), 4),
                "null_within_hemi_perm_mean_sd": [round(float(np.mean(null_len)), 4), round(float(np.std(null_len)), 4)],
                "spearman_dist_vs_mean_log_weight": round(float(r_w), 4),
                "strongest_edge_in_5_nearest": round(nn_hit, 4),
                "_D": D, "_names": names,
            }
        best = max(order, key=lambda k: order[k]["spearman_dist_vs_mean_tract_length"])
        D, names = order[best]["_D"], order[best]["_names"]
        for v in order.values():
            v.pop("_D"); v.pop("_names")
        assert np.allclose(D, D.T) and np.all(np.diag(D) == 0)
        dens = (Wm > 0).mean(1)
        np.savez_compressed(
            out / f"schaefer{p}.npz",
            n=p, subjects=np.array(subjects), i=np.concatenate(I).astype(np.int16), j=np.concatenate(J).astype(np.int16),
            sub=np.concatenate(S).astype(np.uint8), w=np.concatenate(Wv).astype(np.float32),
            dist_mm=D, hemi=(np.arange(p) >= h).astype(np.uint8),
            names=np.array(names), centroid_order=best,
        )
        # Separate file keeps every file under GitHub's 100 MB limit; entries align with i, j, sub above.
        np.savez_compressed(out / f"schaefer{p}_length.npz", length=np.concatenate(Lv).astype(np.float32))
        report["scales"][str(p)] = {
            "checks": checks,
            "density_mean_sd_min_max": [round(float(x), 4) for x in (dens.mean(), dens.std(), dens.min(), dens.max())],
            "subjects_reaching_density": {f"{k}%": int((dens >= k / 100).sum()) for k in (10, 20, 30, 50)},
            "order_check": order,
            "centroid_order_used": best,
        }
        print(p, json.dumps(report["scales"][str(p)]), flush=True)
    json.dump(report, open(out / "validation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
