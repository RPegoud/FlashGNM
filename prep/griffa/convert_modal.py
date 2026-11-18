"""Download Griffa et al. (Zenodo 2872624, CC BY 4.0) and convert SC to compact sparse npz per scale.

Runs on a Modal CPU container because scipy.io.loadmat of the 700 MB file needs ~1.5 GB+
(it decompresses SC and FC at all five scales as float64), which OOMs the devcontainer.

    modal run bench/griffa/convert_modal.py --out-dir /workspace/data/griffa

Output per scale: griffa_sc_scale{033,060,125,250,500}_sparse.npz with the strict upper
triangle only (so self-loops are dropped): i, j (int16), subject (int8), w (float32), plus
n, n_subjects and sanity stats (sym_err, diag_max, min_w, max_w).
"""
import io

import modal

app = modal.App("griffa-convert")
image = modal.Image.debian_slim().pip_install("numpy", "scipy")
URL = "https://zenodo.org/api/records/2872624/files/Individual_Connectomes.mat/content"
MD5 = "87ef63debc0cbe44afc1e6cf9ab9717e"
SCALES = ["033", "060", "125", "250", "500"]


@app.function(image=image, memory=8192, cpu=2, timeout=900)
def convert() -> dict:
    import hashlib
    import urllib.request

    import numpy as np
    import scipy.io as sio

    path = "/tmp/ic.mat"
    urllib.request.urlretrieve(URL, path)
    h = hashlib.md5(open(path, "rb").read()).hexdigest()
    assert h == MD5, h
    # connMatrices.SC / .FC: (5,) object arrays, each n x n x 70 float64, cortex only.
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["connMatrices"]
    out = {}
    for s, SC in zip(SCALES, m.SC):
        n, _, S = SC.shape
        iu, ju = np.triu_indices(n, 1)
        sym = float(max(np.abs(SC[:, :, k] - SC[:, :, k].T).max() for k in range(S)))
        diag = float(np.abs(SC[np.arange(n), np.arange(n), :]).max())
        U = SC[iu, ju, :]  # (pairs, S)
        p, subj = np.nonzero(U)
        buf = io.BytesIO()
        np.savez_compressed(
            buf, n=n, n_subjects=S, i=iu[p].astype(np.int16), j=ju[p].astype(np.int16),
            subject=subj.astype(np.int8), w=U[p, subj].astype(np.float32),
            sym_err=sym, diag_max=diag, min_w=float(U[U > 0].min()), max_w=float(U.max()),
        )
        out[s] = buf.getvalue()
        print(s, n, S, "nnz/subj", len(p) / S, "sym", sym, "diag", diag)
    return out


@app.local_entrypoint()
def main(out_dir: str):
    for s, b in convert.remote().items():
        with open(f"{out_dir}/griffa_sc_scale{s}_sparse.npz", "wb") as f:
            f.write(b)
        print("wrote", s, len(b))
