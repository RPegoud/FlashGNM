"""Shared loaders for the Griffa / Lausanne data prepared by convert_modal.py and centroids.py."""
import csv

import numpy as np

SCALES = ["033", "060", "125", "250", "500"]
DEFAULT_GRIFFA = "/workspace/data/griffa"
DEFAULT_ATLAS = "/workspace/data/lausanne_mni"


def load_sparse(gdir: str, scale: str):
    """Returns (z, n, S, cons) where cons[i, j] (upper triangle) is the fraction of subjects with edge i-j."""
    z = np.load(f"{gdir}/griffa_sc_scale{scale}_sparse.npz")
    n, S = int(z["n"]), int(z["n_subjects"])
    cons = np.zeros((n, n))
    np.add.at(cons, (z["i"], z["j"]), 1.0)
    return z, n, S, cons / S


def load_cortex_centroids(adir: str, scale: str):
    """Atlas rows with structure == cortex, in atlas id order (= Griffa order, R hemisphere first)."""
    rows = [r for r in csv.DictReader(open(f"{adir}/lausanne_scale{scale}_centroids_mni.csv"))
            if r["structure"] == "cortex"]
    xyz = np.array([[float(r[k]) for k in "xyz"] for r in rows])
    return rows, xyz


def euclid(xyz):
    return np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
