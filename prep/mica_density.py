"""MICA-MICs SC: cortical density, >90 mm tract share and interhemispheric share, raw and after per-subject
weight thresholding (SPEC §7.1). Usage: python -I prep/mica_density.py (reads /workspace/data/mica/micapipe_MICs_v1.1.zip)."""
import zipfile, numpy as np, sys
z = zipfile.ZipFile('/workspace/data/mica/micapipe_MICs_v1.1.zip')
subs = sorted({x.split('/')[1] for x in z.namelist() if x.startswith('micapipe/sub-')})


def load(f):
    M = np.loadtxt(z.open(f), delimiter=',')
    return M + np.triu(M, 1).T


for p in [100, 200, 400, 1000]:
    h = p // 2
    ctx = np.r_[15:15 + h, 16 + h:16 + p]
    hemi = np.r_[np.zeros(h), np.ones(h)]
    inter = hemi[:, None] != hemi[None, :]
    iu = np.triu_indices(p, 1)
    dens, long, ih, Ws = [], [], [], []
    thr = {10: [], 20: [], 30: [], 50: []}
    thr_ih = {k: [] for k in thr}
    for s in subs:
        base = f'micapipe/{s}/ses-01/dwi/{s}_ses-01_space-dwinative_atlas-schaefer{p}_desc-'
        try:
            W = load(base + 'sc.txt')[np.ix_(ctx, ctx)]
            L = load(base + 'edgeLength.txt')[np.ix_(ctx, ctx)]
        except KeyError:
            continue
        w, l = W[iu], L[iu]
        e = w > 0
        dens.append(e.mean()); long.append((l[e] > 90).mean()); ih.append(inter[iu][e].mean()); Ws.append(e)
        order = np.argsort(-w)
        for k in thr:
            keep = order[:int(k / 100 * len(w))]
            thr[k].append((l[keep] > 90).mean()); thr_ih[k].append(inter[iu][keep].mean())
    cons = np.array(Ws).mean(0)
    f = lambda a: f'{np.mean(a):.3f}±{np.std(a):.3f}'
    print(f'Schaefer{p} ({len(dens)} subj): density {f(dens)} [min {min(dens):.3f}, max {max(dens):.3f}]; '
          f'>90mm tract share {f(long)}; inter-hemi {f(ih)}; pairs in >=50% subj {(cons >= .5).mean():.3f}')
    for k in thr:
        print(f'   top {k}% by weight: >90mm {f(thr[k])}, inter-hemi {f(thr_ih[k])}')
    sys.stdout.flush()
