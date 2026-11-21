import json
import sys

import numpy as np

sys.path.insert(0, ".")
from src.landscape import load
from src.landscape_plots import analyse

RUN = "results/gamma_diag_v1"
MAIN = "results/landscape_griffa_full/scale033"
SUBS = range(20)

# ---------------- 1. pipeline check: dense toolbox-empty-native vs kernel
print("== pipeline check: dense reference (toolbox, empty, native) vs Triton kernel, 20 subjects")
zs, opt_k, opt_d, Ek, Ed = [], [], [], [], []
for s in SUBS:
    Lk, Ld = load(f"{MAIN}/sub{s:02d}.npz"), load(f"{RUN}/toolbox-empty-native/sub{s:02d}.npz")
    ek, ed = Lk["energy"], Ld["energy"]  # [g, e, r]
    se = np.sqrt(ek.var(2, ddof=1) / ek.shape[2] + ed.var(2, ddof=1) / ed.shape[2])
    d = ek.mean(2) - ed.mean(2)
    ok = se > 0
    zs.append(d[ok] / se[ok])
    ak, ad = analyse(Lk), analyse(Ld)
    opt_k.append((ak["eta_opt"], ak["gamma_opt"])); opt_d.append((ad["eta_opt"], ad["gamma_opt"]))
    Ek.append(ak["E_opt"]); Ed.append(ad["E_opt"])
z = np.concatenate(zs)
print(f"  per-cell z of mean-energy difference over {z.size} cells: mean {z.mean():+.3f}, "
      f"SD {z.std():.3f}, |z|>3: {np.mean(np.abs(z) > 3):.4f} (N(0,1): 0.0027)")
opt_k, opt_d = np.array(opt_k), np.array(opt_d)
print(f"  median optimum kernel eta {np.median(opt_k[:,0]):.2f} gamma {np.median(opt_k[:,1]):.2f} "
      f"E {np.median(Ek):.3f} | dense eta {np.median(opt_d[:,0]):.2f} gamma {np.median(opt_d[:,1]):.2f} "
      f"E {np.median(Ed):.3f}")

print("== toolbox slices at eta = -1.8 (mean energy, 20 runs; kernel 20 runs; dense 10 runs)")
for sl in json.load(open(f"{RUN}/toolbox_slices.json")):
    s, seeded = sl["subject"], sl["seeded"]
    cfg = "toolbox-seed-native" if seeded else "toolbox-empty-native"
    Ld = load(f"{RUN}/{cfg}/sub{s:02d}.npz")
    Lk = None if seeded else load(f"{MAIN}/sub{s:02d}.npz")
    ie = int(np.argmin(np.abs(Ld["etas"] + 1.8)))
    rows = []
    for g, e in zip(sl["gammas"], sl["energy"]):
        ig = int(np.argmin(np.abs(Ld["gammas"] - g)))
        e = np.array(e)
        tb = f"{e.mean():.3f}±{e.std(ddof=1)/np.sqrt(len(e)):.3f}"
        dn = Ld["energy"][ig, ie]
        dn = f"{dn.mean():.3f}±{dn.std(ddof=1)/np.sqrt(len(dn)):.3f}"
        kn = "" if Lk is None else "{:.3f}±{:.3f}".format(Lk["energy"][ig, ie].mean(),
                                                          Lk["energy"][ig, ie].std(ddof=1) / np.sqrt(20))
        rows.append(f"    g={Ld['gammas'][ig]:.1f}: toolbox {tb}  dense {dn}  kernel {kn}")
    print(f"  subject {s}, {'consensus seed' if seeded else 'empty start'}:")
    print("\n".join(rows))

# ---------------- 2. factorial
print("== factorial (20 subjects each): medians [IQR]")
print("  config                 eta_opt              gamma_opt            E_opt  interior  near-opt gamma span")
cfgs = ["toolbox-empty-native", "toolbox-seed-native", "bct-empty-native", "bct-seed-native",
        "toolbox-empty-d10", "toolbox-seed-d10", "bct-empty-d10", "bct-seed-d10"]
res = {}
for c in cfgs:
    A = [analyse(load(f"{RUN}/{c}/sub{s:02d}.npz")) for s in SUBS]
    eta = np.array([a["eta_opt"] for a in A]); gam = np.array([a["gamma_opt"] for a in A])
    E = np.array([a["E_opt"] for a in A])
    inter = sum(min(a["edge_min"].values()) > a["E_opt"] for a in A)
    span = np.median([a["near_gamma"][1] - a["near_gamma"][0] for a in A])
    q = lambda x: "{:5.2f} [{:5.2f},{:5.2f}]".format(*np.percentile(x, [50, 25, 75]))
    print(f"  {c:22s} {q(eta)}  {q(gam)}  {np.median(E):.3f}   {inter:2d}/20    {span:.2f}")
    res[c] = (eta, gam, E)
print("  corr(eta, gamma) across subjects:", {c: round(float(np.corrcoef(*res[c][:2])[0, 1]), 2) for c in cfgs})
