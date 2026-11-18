"""Centroids (MNI mm) of each Lausanne (Cammoun 2012) region, per scale, in atlas id order.

Atlas: netneurotools `atl-cammoun2012` MNI152NLin2009aSym package (OSF udpv8, file
67326efaee12e85a3662f7c6, md5 7cc281d6916baf78bb53b12806d28454), extracted to <atlas_dir>.
Needs nibabel.

    python prep/griffa/centroids.py /workspace/data/lausanne_mni/MNI152NLin2009aSym /workspace/data/lausanne_mni

Writes lausanne_scale{res}_centroids_mni.csv (id, label, hemisphere, structure, x, y, z, n_voxels).
Atlas order at scale500: R cortex 1-501, R subcortex 502-508, L cortex 509-1007, L subcortex + brainstem 1008-1015.
"""
import csv
import sys

import nibabel as nib
import numpy as np

atlas_dir, out_dir = sys.argv[1], sys.argv[2]
prefix = f"{atlas_dir}/atl-Cammoun2012_space-MNI152NLin2009aSym"
info = list(csv.DictReader(open(f"{prefix}_info.csv")))
for res in ["033", "060", "125", "250", "500"]:
    img = nib.load(f"{prefix}_res-{res}_deterministic.nii.gz")
    lab = np.asarray(img.dataobj).astype(np.int64)
    rows = [r for r in info if r["scale"] == f"scale{res}"]
    ids = np.array([int(r["id"]) for r in rows])
    vox = np.argwhere(lab > 0)
    l = lab[lab > 0]
    counts = np.bincount(l, minlength=ids.max() + 1)
    sums = np.stack([np.bincount(l, weights=vox[:, k], minlength=ids.max() + 1) for k in range(3)], 1)
    assert set(np.unique(l)) == set(ids), (res, set(np.unique(l)) ^ set(ids))
    c_mm = nib.affines.apply_affine(img.affine, sums[ids] / counts[ids, None])
    with open(f"{out_dir}/lausanne_scale{res}_centroids_mni.csv", "w") as f:
        f.write("id,label,hemisphere,structure,x,y,z,n_voxels\n")
        for r, c, n in zip(rows, c_mm, counts[ids]):
            f.write(f"{r['id']},{r['label']},{r['hemisphere']},{r['structure']},{c[0]:.3f},{c[1]:.3f},{c[2]:.3f},{n}\n")
    hemi = np.array([r["hemisphere"] for r in rows])
    print(res, len(ids), "mean x R/L", c_mm[hemi == "R", 0].mean().round(1), c_mm[hemi == "L", 0].mean().round(1))
