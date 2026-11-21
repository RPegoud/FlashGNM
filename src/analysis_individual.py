"""Individual differences vs estimation noise (SPEC H3, H5), from replicated landscapes.

    python src/analysis_individual.py results/griffa_raw [--out results/analysis_individual]

Input: a landscape run directory with scale<s>/sub<ss>.npz (src/landscape.py format,
[gamma, eta, repeat, 4] KS). No new simulations.

Per subject:
- point estimate: argmin of the mean energy over repeats;
- bootstrap (resample repeats with replacement, `--boot` times): the argmin's spread
  (within-subject SD of eta and gamma, and the eta-gamma correlation of the bootstrap
  cloud, i.e. the orientation of the noise along the valley);
- Akarca-style estimates: the argmin of each single-repeat landscape (one simulation per
  cell), on the full grid and on coarse subgrids (every 2nd and 5th cell).

Per scale:
- between-subject SD of the point estimates against the mean within-subject SD;
  share of the observed between-subject variance that noise explains, within / between;
- Cochran's Q heterogeneity test: Q = sum_s (x_s - xbar_w)^2 / v_s with inverse-variance
  weights, ~ chi2(S - 1) if all subjects share one parameter. v_s is the bootstrap
  variance plus step^2 / 12 (grid quantisation), so a subject whose argmin never moves
  doesn't get infinite weight;
- cross-subject corr(eta, gamma) from replicated estimates and from single-run estimates
  (mean and range over the repeats), next to the mean within-subject bootstrap corr;
- coarse/single-run accuracy (H3): distance of the single-run argmin from the replicated
  optimum, and the share of single-run argmins inside the replicated near-optimal set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.landscape import SCALES, load  # noqa: E402


def near_set(E: np.ndarray) -> np.ndarray:
    """Cells within 2 SE (of the difference) of the best mean, as in landscape_plots."""
    R = E.shape[2]
    mean, se = E.mean(2), E.std(2, ddof=1) / np.sqrt(R)
    ig, ie = np.unravel_index(np.argmin(mean), mean.shape)
    return mean - mean[ig, ie] <= 2 * np.sqrt(se**2 + se[ig, ie] ** 2)


def argmin2(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Argmin over the last two axes of [..., gamma, eta] -> (gamma idx, eta idx)."""
    flat = M.reshape(*M.shape[:-2], -1).argmin(-1)
    return np.unravel_index(flat, M.shape[-2:])


def subject(path: Path, n_boot: int, rng: np.random.Generator) -> dict:
    L = load(path)
    E, etas, gammas = L["energy"], L["etas"], L["gammas"]  # E [g, e, r]
    G, Ne, R = E.shape
    ig, ie = argmin2(E.mean(2))
    near = near_set(E)
    # bootstrap the argmin over repeats
    idx = rng.integers(0, R, size=(n_boot, R))
    means = np.stack([E[:, :, i].mean(2) for i in idx])  # [boot, g, e]
    bg, be = argmin2(means)
    b_eta, b_gamma = etas[be], gammas[bg]
    # single-run argmins on the full grid and on coarse subgrids (every k-th cell)
    single = {}
    for k in (1, 2, 5):
        sub = E[::k, ::k]  # [g', e', r]
        sg, se_ = argmin2(np.moveaxis(sub, 2, 0))  # per repeat
        eg, ee = sg * k, se_ * k  # back to fine-grid indices
        single[k] = {
            "eta": etas[ee], "gamma": gammas[eg],
            "in_near": near[eg, ee],
        }
    return {
        "eta": float(etas[ie]), "gamma": float(gammas[ig]),
        "boot_eta_sd": float(b_eta.std(ddof=1)), "boot_gamma_sd": float(b_gamma.std(ddof=1)),
        "boot_corr": (float(np.corrcoef(b_eta, b_gamma)[0, 1])
                      if b_eta.std() > 0 and b_gamma.std() > 0 else float("nan")),
        "boot_eta": b_eta, "boot_gamma": b_gamma,
        "single": single,
        "d_eta": float(etas[1] - etas[0]), "d_gamma": float(gammas[1] - gammas[0]),
        "near_cells": int(near.sum()),
    }


def cochran_q(x: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    w = 1 / v
    xbar = (w * x).sum() / w.sum()
    Q = float((w * (x - xbar) ** 2).sum())
    return Q, float(stats.chi2.sf(Q, len(x) - 1))


def scale_summary(subs: list[dict]) -> dict:
    out = {"subjects": len(subs)}
    d = {"eta": subs[0]["d_eta"], "gamma": subs[0]["d_gamma"]}
    for p in ("eta", "gamma"):
        x = np.array([s[p] for s in subs])
        w_sd = np.array([s[f"boot_{p}_sd"] for s in subs])
        v = w_sd**2 + d[p] ** 2 / 12
        between = x.std(ddof=1)
        within = float(np.sqrt(np.mean(v)))
        Q, pval = cochran_q(x, v)
        # single-run (one simulation per cell) estimates: within-subject SD over repeats
        sr = np.stack([s["single"][1][p] for s in subs])  # [S, R]
        out[p] = {
            "median": float(np.median(x)), "between_sd": float(between),
            "within_sd": within, "noise_share": float(min(1.0, within**2 / between**2))
            if between > 0 else float("nan"),
            "Q": Q, "Q_df": len(x) - 1, "Q_p": pval,
            "single_run_within_sd": float(sr.std(1, ddof=1).mean()),
            "single_run_between_sd": float(sr.std(0, ddof=1).mean()),
        }
    e = np.array([s["eta"] for s in subs])
    g = np.array([s["gamma"] for s in subs])
    out["corr_replicated"] = float(stats.pearsonr(e, g)[0])
    out["corr_replicated_p"] = float(stats.pearsonr(e, g)[1])
    R = len(subs[0]["single"][1]["eta"])
    sr_corr = []
    for r in range(R):
        se_ = np.array([s["single"][1]["eta"][r] for s in subs])
        sg = np.array([s["single"][1]["gamma"][r] for s in subs])
        sr_corr.append(stats.pearsonr(se_, sg)[0])
    out["corr_single_run"] = {"mean": float(np.mean(sr_corr)), "min": float(np.min(sr_corr)),
                              "max": float(np.max(sr_corr))}
    bc = np.array([s["boot_corr"] for s in subs])
    out["within_boot_corr_median"] = float(np.nanmedian(bc))
    out["within_boot_corr_defined"] = int(np.isfinite(bc).sum())
    # H3: where single-run argmins land, full grid vs coarse subgrids
    out["single_run"] = {}
    for k in (1, 2, 5):
        de = np.concatenate([np.abs(s["single"][k]["eta"] - s["eta"]) for s in subs])
        dg = np.concatenate([np.abs(s["single"][k]["gamma"] - s["gamma"]) for s in subs])
        inn = np.concatenate([s["single"][k]["in_near"] for s in subs])
        out["single_run"][f"step_x{k}"] = {
            "median_abs_d_eta": float(np.median(de)), "p90_abs_d_eta": float(np.quantile(de, .9)),
            "median_abs_d_gamma": float(np.median(dg)),
            "p90_abs_d_gamma": float(np.quantile(dg, .9)),
            "in_near_share": float(inn.mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "analysis_individual")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    summary, per_subject = {}, {}
    for s in SCALES:
        files = sorted((args.run / f"scale{s}").glob("sub*.npz"))
        if not files:
            continue
        subs = [subject(f, args.boot, rng) for f in files]
        summary[s] = scale_summary(subs)
        per_subject[s] = {
            "eta": [x["eta"] for x in subs], "gamma": [x["gamma"] for x in subs],
            "boot_eta": np.stack([x["boot_eta"] for x in subs]),
            "boot_gamma": np.stack([x["boot_gamma"] for x in subs]),
            "single_eta": np.stack([x["single"][1]["eta"] for x in subs]),
            "single_gamma": np.stack([x["single"][1]["gamma"] for x in subs]),
        }
        print(s, json.dumps(summary[s]["eta"]), json.dumps(summary[s]["gamma"]), flush=True)
    (args.out / "summary.json").write_text(json.dumps(
        {"run": str(args.run), "boot": args.boot, "seed": args.seed, "scales": summary},
        indent=2))
    np.savez_compressed(args.out / "per_subject.npz", **{
        f"{s}_{k}": np.asarray(v) for s, d in per_subject.items() for k, v in d.items()})


if __name__ == "__main__":
    main()
