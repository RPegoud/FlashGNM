"""Export the simulation benchmark numbers plotted in clip 6 to src/data/bench.json.

Reads the committed results directories (H100, bench/bench_sim.py) and keeps only the rows
the clip shows. Every number in the clip comes from this file; nothing is typed by hand.

    python3 scripts/bench_export.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # repo root (presentation/ lives inside it)
RES = ROOT / "results"
OUT = HERE.parent / "src" / "data" / "bench.json"

RUNS = {
    # baselines: toolbox, leandense, rowsampled (their memory is valid, see NOTE.md)
    "baseline": "26-11-21_14-40-12_sim_baseline",
    # prototype and v2 measured head to head in one run
    "v2": "26-11-21_15-10-58_sim_v2",
    # v2 with batched init (commit 4cc1b27), B up to 16384
    "bigB": "26-11-21_15-42-39_sim_v2_bigB",
}

# (label, run, impl name in results.jsonl)
IMPLS = [
    ("toolbox", "Toolbox (reference)", "baseline", "toolbox"),
    ("dense", "Dense baseline", "baseline", "leandense"),
    ("rowsampled", "Row-sampled PyTorch", "baseline", "rowsampled"),
    ("proto", "Triton prototype", "v2", "triton_proto"),
    ("v2", "Kernel v2", "v2", "src.kernel:TritonGNM"),
    ("v2bi", "v2 + batched init", "bigB", "src.kernel:TritonGNM"),
]


def load(run: str) -> list[dict]:
    p = RES / RUNS[run] / "results.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def find(rows, impl, n, B):
    for r in rows:
        if r["impl"] == impl and r["n"] == n and r["B"] == B:
            return r
    raise KeyError((impl, n, B))


def cell(r: dict) -> dict:
    """The fields the clip uses; status 'oom' rows keep only their status."""
    if r["status"] != "ok":
        return {"status": r["status"]}
    return {
        "status": "ok",
        "sims_per_s": r["sims_per_s"],
        "setup_s": r["setup_s"],
        "run_s": r["run_s"],
        "peak_gb": r.get("peak_mem_gb"),
        "extrapolated": bool(r.get("extrapolated")),
    }


def main():
    rows = {k: load(k) for k in RUNS}
    meta = json.loads((RES / RUNS["bigB"] / "meta.json").read_text())
    base_meta = json.loads((RES / RUNS["baseline"] / "meta.json").read_text())
    NS = [68, 219, 1000, 2000]
    T = {n: find(rows["baseline"], "toolbox", n, 1)["T"] for n in NS}
    dist = {n: find(rows["baseline"], "toolbox", n, 1)["dist"] for n in NS}

    # Beat B: every implementation at n = 1000, B = 1024.
    nB, BB = 1000, 1024
    impls = []
    for key, label, run, impl in IMPLS:
        c = cell(find(rows[run], impl, nB, BB))
        impls.append({"key": key, "label": label, "impl": impl, "run": RUNS[run], **c})
    tb = impls[0]["sims_per_s"]
    for x in impls:
        x["speedup_vs_toolbox"] = x["sims_per_s"] / tb

    # Beat C: scaling with n at B = 1024 (toolbox, dense baseline, v2 + batched init).
    scaling = []
    for key, label, run, impl in IMPLS:
        if key not in ("toolbox", "dense", "v2bi"):
            continue
        pts = []
        for n in NS:
            pts.append({"n": n, **cell(find(rows[run], impl, n, 1024))})
        scaling.append({"key": key, "label": label, "points": pts})
    tbs = {p["n"]: p for p in scaling[0]["points"]}
    v2s = {p["n"]: p for p in scaling[-1]["points"]}
    speedup_n = []
    for n in NS:
        if tbs[n]["status"] == "ok":
            speedup_n.append({"n": n, "speedup": v2s[n]["sims_per_s"] / tbs[n]["sims_per_s"]})
        else:
            speedup_n.append({"n": n, "speedup": None, "toolbox": tbs[n]["status"]})

    # Beat D: v2 + batched init against B.
    Bs = meta["args"]["B"]
    bigB = []
    for n in NS:
        bigB.append({"n": n, "points": [{"B": B, **cell(find(rows["bigB"], "src.kernel:TritonGNM", n, B))} for B in Bs]})

    # Beat E: peak memory at n = 2000, B = 1024. Dense state model: bytes/pair x n^2 x B.
    nM, BM = 2000, 1024
    memory = []
    for key, label, run, impl in IMPLS:
        c = cell(find(rows[run], impl, nM, BM))
        memory.append({"key": key, "label": label, **c})
    state_gb = {"proto_9B": 9 * nM * nM * BM / 1e9, "v2_3B": 3 * nM * nM * BM / 1e9}

    # Headline: v2 + batched init vs toolbox and vs the fair dense baseline at n=1000, B=1024.
    v2bi = impls[-1]
    head = {
        "n": nB, "B": BB, "T": T[nB],
        "v2bi": v2bi["sims_per_s"], "toolbox": tb, "dense": impls[1]["sims_per_s"],
        "vs_toolbox": v2bi["sims_per_s"] / tb,
        "vs_dense": v2bi["sims_per_s"] / impls[1]["sims_per_s"],
    }
    best = max((p for s in bigB for p in s["points"] if p["status"] == "ok"), key=lambda p: p["sims_per_s"])
    best_n = next(s["n"] for s in bigB if best in s["points"])

    out = {
        "gpu": meta["gpu"],
        "gpu_mem_gb": meta["gpu_mem_gb"],
        "probe_steps": base_meta["args"]["probe_steps"],
        "repeats": base_meta["args"]["repeats"],
        "ns": NS, "T": {str(n): T[n] for n in NS}, "dist": {str(n): dist[n] for n in NS},
        "B_grid": base_meta["args"]["B"], "B_big": Bs,
        "impls_n": nB, "impls_B": BB, "impls": impls,
        "scaling": scaling, "speedup_n": speedup_n,
        "bigB": bigB, "best": {"n": best_n, **best},
        "mem_n": nM, "mem_B": BM, "memory": memory, "state_gb": state_gb,
        "headline": head,
        "sources": RUNS,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    print(json.dumps(head, indent=1))
    print("best", best_n, best["B"], best["sims_per_s"])
    print("speedup_n", speedup_n)
    print("state_gb", state_gb)


if __name__ == "__main__":
    main()
