"""Connection recovery by distance at each subject's optimum (SPEC H7), on Modal.

    modal run src/modal_recovery.py                       # all scales, all 70 subjects
    modal run src/modal_recovery.py --scales "033" --subjects "0 1"

Parameters per subject come from a landscape run: the optimum from <landscapes>/summary.csv
and the distance-only optimum (gamma = 0, best mean energy on that row) from the
per-subject .npz in <raw> (fetch with `modal volume get flashgnm-landscapes griffa_full`).
Writes results/recovery/scale<s>.jsonl (one line per subject, src/recovery.py metrics) and
summary.json (medians per scale).
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if modal.is_local() else Path("/workspace")
REMOTE = PurePosixPath("/workspace")
TRITON_CACHE = "/cache/triton"


def _git_sha() -> str:
    if not modal.is_local():
        return ""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "src"], cwd=ROOT, text=True)
        return sha + ("-dirty" if dirty.strip() else "")
    except Exception:
        return ""


image = (
    modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT)
    .entrypoint([])
    .env({"TRITON_CACHE_DIR": TRITON_CACHE, "PYTHONPATH": str(REMOTE), "GIT_SHA": _git_sha()})
    .add_local_dir(
        ROOT,
        remote_path=str(REMOTE),
        ignore=[".git", ".git/**", ".claude", ".claude/**", ".venv", ".venv/**", ".env",
                "papers", "papers/**", "data/mica", "data/mica/**", "results", "results/**",
                "**/__pycache__", "**/__pycache__/**"],
    )
)
app = modal.App("FlashGNM_recovery", image=image)
triton_vol = modal.Volume.from_name("triton_kernels", create_if_missing=True)


@app.function(gpu="H100!", volumes={TRITON_CACHE: triton_vol}, timeout=3600,
              max_containers=24, retries=1)
def recovery_remote(scale: str, subject: int, eta: float, gamma: float, eta_dist: float,
                    n_sims: int, seed: int) -> dict:
    import os

    import torch

    from src import recovery

    out = recovery.run_subject(scale, subject, eta, gamma, eta_dist, torch.device("cuda"),
                               n_sims=n_sims, seed=seed)
    out["git"] = os.environ.get("GIT_SHA", "")
    try:
        triton_vol.commit()
    except Exception:
        pass
    return out


def _params(landscapes: Path, raw: Path, scales: list[str]) -> dict:
    sys.path.insert(0, str(ROOT))
    import numpy as np

    from src.landscape import load

    opt = {}
    with open(landscapes / "summary.csv") as f:
        for r in csv.DictReader(f):
            opt[(r["scale"], int(r["subject"]))] = (float(r["eta_opt"]), float(r["gamma_opt"]))
    out = {}
    for (s, sub), (e, g) in opt.items():
        if s not in scales:
            continue
        L = load(raw / f"scale{s}" / f"sub{sub:02d}.npz")
        i0 = int(np.argmin(np.abs(L["gammas"])))
        assert abs(L["gammas"][i0]) < 1e-9, "grid has no gamma = 0 row"
        row = L["energy"][i0].mean(-1)  # [eta]
        out[(s, sub)] = (e, g, float(L["etas"][int(np.argmin(row))]), float(row.min()))
    return out


@app.local_entrypoint()
def main(scales: str = "033 060 125 250 500", subjects: str = "all", n_sims: int = 100,
         seed: int = 0, landscapes: str = "results/landscape_griffa_full",
         raw: str = "results/griffa_raw", out: str = "results/recovery"):
    import numpy as np

    sc = scales.split()
    params = _params(ROOT / landscapes, ROOT / raw, sc)
    keep = None if subjects == "all" else set(map(int, subjects.split()))
    jobs = [(s, sub, e, g, ed, n_sims, seed) for (s, sub), (e, g, ed, _) in sorted(params.items())
            if keep is None or sub in keep]
    od = ROOT / out
    od.mkdir(parents=True, exist_ok=True)
    (od / "meta.json").write_text(json.dumps({"n_sims": n_sims, "seed": seed, "git": _git_sha(),
                                              "landscapes": landscapes, "gpu": "H100",
                                              "started": time.ctime()}, indent=2))
    print(f"{len(jobs)} recovery jobs", flush=True)
    rows: dict[str, list] = {}
    t0 = time.time()
    for i, res in enumerate(recovery_remote.starmap(jobs, order_outputs=False,
                                                    return_exceptions=True)):
        if isinstance(res, Exception):
            print(f"FAILED: {repr(res)[:300]}", flush=True)
            continue
        res["distance_only_energy"] = params[(res["scale"], res["subject"])][3]
        rows.setdefault(res["scale"], []).append(res)
        print(f"[{i + 1}/{len(jobs)} {time.time() - t0:5.0f}s] scale {res['scale']} "
              f"sub {res['subject']:02d}: recovery {res['homophily']['recovery']:.3f} "
              f"(distance only {res['distance_only']['recovery']:.3f}, "
              f"group {res['group_consensus']['recovery']:.3f})", flush=True)
    summary = {}
    for s, rs in sorted(rows.items()):
        rs.sort(key=lambda r: r["subject"])
        with open(od / f"scale{s}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        summary[s] = {"subjects": len(rs)}
        for m in ("homophily", "distance_only", "group_consensus"):
            med = lambda k: float(np.nanmedian([r[m][k] for r in rs]))  # noqa: E731
            summary[s][m] = {
                "recovery": med("recovery"), "long_top25_hit": med("long_top25_hit"),
                "long_base_rate": med("long_base_rate"), "degree_spearman": med("degree_spearman"),
                "recall_per_bin": np.nanmedian([r[m]["recall_per_bin"] for r in rs], 0).tolist(),
                "chance_per_bin": np.nanmedian([r[m]["chance_per_bin"] for r in rs], 0).tolist(),
            }
        summary[s]["bins_mm"] = rs[0]["homophily"]["bins_mm"][:-1] + ["inf"]
    (od / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({s: {m: round(v[m]["recovery"], 3) for m in
                          ("homophily", "distance_only", "group_consensus")}
                      for s, v in summary.items()}), flush=True)
