"""Plots and summaries for a landscape run (SPEC §8).

    python src/landscape_plots.py results/landscape_<run>

Per subject, plots/scale<s>/sub<ss>.pdf:
    page 1: mean of each KS component over repeats;
    page 2: SD of each component;
    page 3: mean energy (max KS), optimum (×) and the near-optimal set (outlined: cells
            whose mean is within 2 SE of the best cell's, SE of the difference).
Per run: summary.csv (one row per subject: optimum, best energy and its SD, near-optimal
set size and extent, minimum energy on each grid edge, timings, validation), summary.md
(per-scale medians), and summary.pdf (optima of all subjects per scale; mean landscape
over subjects per scale).

Arrays are [gamma, eta]; eta on the x axis. Sequential single-hue maps: blue for means
and energy (light = low = better), orange for SDs.
"""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.landscape import SCALES, load  # noqa: E402

STAT_LABELS = ["degree KS", "clustering KS", "betweenness KS", "edge length KS"]
BLUE = LinearSegmentedColormap.from_list("blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
ORANGE = LinearSegmentedColormap.from_list("orange", [
    "#fde5d9", "#f8b897", "#f28b5c", "#eb6834", "#c4501f", "#943b15", "#62270d"])
INK, MUTED = "#1f1f1f", "#6b6b6b"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlesize": 9, "axes.titlecolor": INK,
})


def analyse(L: dict) -> dict:
    """Optimum, near-optimal set and edge minima of one landscape."""
    E = L["energy"]  # [gamma, eta, rep]
    R = E.shape[2]
    mean, sd = E.mean(2), E.std(2, ddof=1)
    se = sd / np.sqrt(R)
    ig, ie = np.unravel_index(np.argmin(mean), mean.shape)
    near = mean - mean[ig, ie] <= 2 * np.sqrt(se**2 + se[ig, ie] ** 2)
    etas, gammas = L["etas"], L["gammas"]
    return {
        "mean": mean, "sd": sd, "near": near, "ig": ig, "ie": ie,
        "eta_opt": float(etas[ie]), "gamma_opt": float(gammas[ig]),
        "E_opt": float(mean[ig, ie]), "E_opt_sd": float(sd[ig, ie]),
        "near_cells": int(near.sum()),
        "near_eta": (float(etas[near.any(0)].min()), float(etas[near.any(0)].max())),
        "near_gamma": (float(gammas[near.any(1)].min()), float(gammas[near.any(1)].max())),
        "edge_min": {"eta_lo": float(mean[:, 0].min()), "eta_hi": float(mean[:, -1].min()),
                     "gamma_lo": float(mean[0].min()), "gamma_hi": float(mean[-1].min())},
        "comp_opt": L["ks"].mean(2)[ig, ie].tolist(),
    }


def _heat(ax, Z, L, cmap, title, cbar_label, vmax=None):
    etas, gammas = L["etas"], L["gammas"]
    de, dg = etas[1] - etas[0], gammas[1] - gammas[0]
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap, interpolation="nearest",
                   vmax=vmax,
                   extent=[etas[0] - de / 2, etas[-1] + de / 2, gammas[0] - dg / 2, gammas[-1] + dg / 2])
    ax.set_xlabel("η (distance)")
    ax.set_ylabel("γ (matching index)")
    ax.set_title(title, loc="left")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                            extend="max" if vmax is not None else "neither")
    cb.set_label(cbar_label, color=MUTED)
    cb.outline.set_edgecolor(MUTED)
    return im


def _mark_opt(ax, L, a, near=True):
    e, g = L["etas"], L["gammas"]
    if near:
        ax.contour(e, g, a["near"].astype(float), levels=[0.5], colors="white", linewidths=2.2)
        ax.contour(e, g, a["near"].astype(float), levels=[0.5], colors=INK, linewidths=1.0)
    ax.plot(a["eta_opt"], a["gamma_opt"], marker="X", ms=9, color=INK, mec="white", mew=1.2)


def plot_subject(path: Path, out: Path) -> dict:
    L = load(path)
    m = L["meta"]
    a = analyse(L)
    head = (f"Griffa scale {m['scale']} (n = {m['n']}), subject {m['subject']:02d}, "
            f"T = {m['T']} edges, {m['grid']['repeats']} repeats per cell")
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for page, (Z, cmap, what) in enumerate([
                (L["ks"].mean(2), BLUE, "mean"), (L["ks"].std(2, ddof=1), ORANGE, "SD")]):
            fig, axs = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
            for k, ax in enumerate(axs.flat):
                _heat(ax, Z[..., k], L, cmap, f"{what} {STAT_LABELS[k]}", what)
                if page == 0:
                    _mark_opt(ax, L, a, near=False)
            fig.suptitle(f"{head}: {what} of each KS component", x=0.01, ha="left", color=INK)
            pdf.savefig(fig)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 6.5), layout="constrained")
        _heat(ax, a["mean"], L, BLUE, "mean energy (max of the four KS); colour clipped at the "
              "60th percentile", "mean energy", vmax=np.percentile(a["mean"], 60))
        _mark_opt(ax, L, a)
        em = a["edge_min"]
        ax.text(0.01, -0.13, (
            f"optimum η = {a['eta_opt']:.2f}, γ = {a['gamma_opt']:.2f}: energy {a['E_opt']:.3f} "
            f"(SD {a['E_opt_sd']:.3f}); outlined: {a['near_cells']} cells within 2 SE.\n"
            f"Minimum on the grid edges: η low {em['eta_lo']:.3f}, η high {em['eta_hi']:.3f}, "
            f"γ low {em['gamma_lo']:.3f}, γ high {em['gamma_hi']:.3f}."),
            transform=ax.transAxes, va="top", color=INK)
        fig.suptitle(head, x=0.01, ha="left", color=INK)
        pdf.savefig(fig)
        plt.close(fig)
    v = m["validation"]
    return {
        "scale": m["scale"], "n": m["n"], "subject": m["subject"], "T": m["T"],
        "eta_opt": a["eta_opt"], "gamma_opt": a["gamma_opt"], "E_opt": a["E_opt"],
        "E_opt_sd": a["E_opt_sd"], "near_cells": a["near_cells"],
        "near_eta_lo": a["near_eta"][0], "near_eta_hi": a["near_eta"][1],
        "near_gamma_lo": a["near_gamma"][0], "near_gamma_hi": a["near_gamma"][1],
        **{f"edge_{k}": v_ for k, v_ in a["edge_min"].items()},
        **{f"ks_opt_{s}": c for s, c in zip(("deg", "clu", "bet", "len"), a["comp_opt"])},
        "interior": min(a["edge_min"].values()) > a["E_opt"],
        "t_gen_s": m["t_gen_s"], "t_eval_s": m["t_eval_s"], "t_total_s": m["t_total_s"],
        "valid": v["ok"], "R_max_rel_err": v["R_max_rel_err"],
    }


def _plot_one(args):
    return plot_subject(*args)


def summarise(run: Path, workers: int = 8):
    files = sorted(run.glob("scale*/sub*.npz"))
    jobs = [(f, run / "plots" / f.parent.name / f.with_suffix(".pdf").name) for f in files]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_plot_one, jobs, chunksize=4))
    rows.sort(key=lambda r: (SCALES[r["scale"]], r["subject"]))
    with open(run / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    lines = [f"# Landscapes: {run.name}", "", f"Grid: {meta.get('grid')}, seed {meta.get('seed')}, "
             f"git {meta.get('git')}", "",
             "Medians over subjects (IQR for the optimum). Near-optimal: cells within 2 SE of the "
             "best mean. Interior: every grid-edge cell is worse than the optimum.", "",
             "| scale | n | subjects | eta opt | gamma opt | energy | SD at opt | near-opt cells | "
             "interior | all valid | gen s | eval s |",
             "|---|---:|---:|---|---|---:|---:|---:|---:|---|---:|---:|"]
    by_scale = {}
    for r in rows:
        by_scale.setdefault(r["scale"], []).append(r)
    for s, rs in by_scale.items():
        col = lambda k: np.array([r[k] for r in rs], dtype=float)  # noqa: E731
        q = lambda k: "{:.2f} [{:.2f}, {:.2f}]".format(*np.percentile(col(k), [50, 25, 75]))  # noqa: E731
        lines.append(
            f"| {s} | {rs[0]['n']} | {len(rs)} | {q('eta_opt')} | {q('gamma_opt')} | "
            f"{np.median(col('E_opt')):.3f} | {np.median(col('E_opt_sd')):.3f} | "
            f"{np.median(col('near_cells')):.0f} | {int(col('interior').sum())}/{len(rs)} | "
            f"{all(r['valid'] for r in rs)} | {np.median(col('t_gen_s')):.1f} | "
            f"{np.median(col('t_eval_s')):.1f} |")
    (run / "summary.md").write_text("\n".join(lines) + "\n")

    # cross-subject figure: optima scatter and mean landscape per scale
    scales = list(by_scale)
    with PdfPages(run / "summary.pdf") as pdf:
        fig, axs = plt.subplots(1, len(scales), figsize=(3.6 * len(scales), 3.6),
                                layout="constrained", squeeze=False, sharex=True, sharey=True)
        for ax, s in zip(axs[0], scales):
            rs = by_scale[s]
            L = load(run / f"scale{s}" / "sub00.npz") if (run / f"scale{s}" / "sub00.npz").exists() \
                else load(next((run / f"scale{s}").glob("sub*.npz")))
            ax.set_xlim(L["etas"][0], L["etas"][-1])
            ax.set_ylim(L["gammas"][0], L["gammas"][-1])
            jit = np.random.default_rng(0).uniform(-0.03, 0.03, (len(rs), 2))
            ax.scatter([r["eta_opt"] for r in rs] + jit[:, 0], [r["gamma_opt"] for r in rs] + jit[:, 1],
                       s=16, color="#2a78d6", edgecolor="white", linewidth=0.6)
            ax.set_title(f"scale {s} (n = {rs[0]['n']}), {len(rs)} subjects", loc="left")
            ax.set_xlabel("η at optimum")
            ax.grid(color="#e6e6e6", linewidth=0.5)
        axs[0, 0].set_ylabel("γ at optimum")
        fig.suptitle("Per-subject optimum (argmin of the mean energy), small jitter", x=0.01,
                     ha="left", color=INK)
        pdf.savefig(fig)
        plt.close(fig)
        fig, axs = plt.subplots(1, len(scales), figsize=(3.9 * len(scales), 3.6),
                                layout="constrained", squeeze=False)
        for ax, s in zip(axs[0], scales):
            Ls = [load(f) for f in sorted((run / f"scale{s}").glob("sub*.npz"))]
            M = np.mean([L["energy"].mean(2) for L in Ls], 0)
            _heat(ax, M, Ls[0], BLUE, f"scale {s}: mean over {len(Ls)} subjects", "mean energy",
                  vmax=np.percentile(M, 60))
            ig, ie = np.unravel_index(M.argmin(), M.shape)
            ax.plot(Ls[0]["etas"][ie], Ls[0]["gammas"][ig], marker="X", ms=8, color=INK, mec="white")
        fig.suptitle("Mean energy landscape across subjects", x=0.01, ha="left", color=INK)
        pdf.savefig(fig)
        plt.close(fig)
    print((run / "summary.md").read_text())


if __name__ == "__main__":
    summarise(Path(sys.argv[1]))
