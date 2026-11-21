"""Energy landscapes for many Griffa subjects and scales, one subject per GPU call (SPEC §8).

    modal run src/modal_landscape.py                                   # all scales, all 70 subjects
    modal run src/modal_landscape.py --scales "033 500" --subjects "0 1 2" --run smoke
    modal run src/modal_landscape.py --run <name>                      # resume: skips done subjects

Each call simulates the full grid for one subject (src/landscape.py), writes the result to
the Modal volume `flashgnm-landscapes` under <run>/scale<s>/sub<ss>.npz (so nothing is lost
if the local side dies; fetch with `modal volume get flashgnm-landscapes <run> results/`), and
returns it; the local side writes results/landscape_<run>/scale<s>/sub<ss>.npz and prints
progress. Progress is reported locally (remote progress bars garble).
"""

import json
import subprocess
import time
from pathlib import Path, PurePosixPath

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if modal.is_local() else Path("/workspace")
REMOTE = PurePosixPath("/workspace")
TRITON_CACHE = "/cache/triton"
VOL_DIR = "/landscapes"


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
app = modal.App("FlashGNM_landscapes", image=image)
triton_vol = modal.Volume.from_name("triton_kernels", create_if_missing=True)
out_vol = modal.Volume.from_name("flashgnm-landscapes", create_if_missing=True)


@app.function(gpu="A100-40GB", volumes={TRITON_CACHE: triton_vol, VOL_DIR: out_vol},
              timeout=2 * 3600, max_containers=24, retries=1)
def subject_remote(run: str, scale: str, subject: int, grid: dict, seed: int,
                   bc_dtype: str = "float64") -> bytes:
    import os

    import torch

    from src import landscape as ls

    res = ls.run_subject(scale, subject, ls.Grid(**grid), torch.device("cuda"), seed=seed,
                         log=lambda *a: None, bc_dtype=bc_dtype)
    meta = json.loads(res["meta"])
    meta["git"] = os.environ.get("GIT_SHA", "")
    res["meta"] = json.dumps(meta)
    data = ls.to_bytes(res)
    path = Path(VOL_DIR) / run / f"scale{scale}" / f"sub{subject:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    out_vol.commit()
    try:
        triton_vol.commit()
    except Exception:
        pass
    return data


# Cheapest card per network size, by cost per subject (price x runtime) with fp64
# betweenness, measured 2026-11-21 (results/gpu_probe/summary.md). Small n: jobs take
# seconds, so start-up dominates and a cheap card wins; n >= 219: evaluation is fp64-bound
# and the H100 is cheaper per job than the A100 (fast fp64). Override with --gpu <name>.
GPU_BY_N = [(114, "T4"), (10**9, "H100")]


def gpu_for(n: int) -> str:
    return next(g for lim, g in GPU_BY_N if n <= lim)


@app.local_entrypoint()
def main(scales: str = "033 060 125 250 500", subjects: str = "all", run: str = "",
         seed: int = 0, eta_lo: float = -4.0, eta_hi: float = 1.0, n_eta: int = 51,
         gamma_lo: float = -1.0, gamma_hi: float = 4.0, n_gamma: int = 51, repeats: int = 20,
         gpu: str = "auto", bc_dtype: str = "float64"):
    import io
    import sys
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    sys.path.insert(0, str(ROOT))
    from src.landscape import SCALES

    run = run or time.strftime("%y-%m-%d_%H-%M-%S")
    grid = dict(eta_lo=eta_lo, eta_hi=eta_hi, n_eta=n_eta, gamma_lo=gamma_lo,
                gamma_hi=gamma_hi, n_gamma=n_gamma, repeats=repeats)
    out = ROOT / "results" / f"landscape_{run}"
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        assert old["grid"] == grid and old["seed"] == seed, "resume with the same grid and seed"
    else:
        meta_path.write_text(json.dumps({"run": run, "grid": grid, "seed": seed, "gpu": gpu,
                                         "gpu_by_n": GPU_BY_N, "bc_dtype": bc_dtype, "git": _git_sha(),
                                         "started": time.ctime()}, indent=2))
    groups: dict[str, list] = {}
    for s in scales.split():
        assert s in SCALES, s
        subs = range(70) if subjects == "all" else map(int, subjects.split())
        card = gpu_for(SCALES[s]) if gpu == "auto" else gpu
        for sub in subs:
            if not (out / f"scale{s}" / f"sub{sub:02d}.npz").exists():
                groups.setdefault(card, []).append((run, s, sub, grid, seed, bc_dtype))
    n_jobs = sum(map(len, groups.values()))
    print(f"run {run}: {n_jobs} subject-scale jobs -> {out}; "
          + ", ".join(f"{g}: {len(j)}" for g, j in groups.items()), flush=True)
    t0, lock, state = time.time(), threading.Lock(), {"done": 0, "failed": []}

    def run_group(card, jobs):
        jobs.sort(key=lambda j: -SCALES[j[1]])  # biggest first
        fn = subject_remote.with_options(gpu=card)
        for res in fn.starmap(jobs, order_outputs=False, return_exceptions=True):
            with lock:
                if isinstance(res, Exception):  # which job: whatever is missing at the end
                    state["failed"].append(repr(res)[:300])
                    print(f"FAILED ({card}): {repr(res)[:300]}", flush=True)
                    continue
                m = json.loads(str(np.load(io.BytesIO(res))["meta"]))
                s, sub = m["scale"], m["subject"]
                p = out / f"scale{s}" / f"sub{sub:02d}.npz"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(res)
                state["done"] += 1
                print(f"[{state['done']}/{n_jobs} {time.time() - t0:7.0f}s] {m['device']}: "
                      f"scale {s} sub {sub:02d}: T={m['T']} gen {m['t_gen_s']:.1f}s "
                      f"eval {m['t_eval_s']:.1f}s valid={m['validation']['ok']} "
                      f"R_err={m['validation']['R_max_rel_err']:.1e}", flush=True)

    with ThreadPoolExecutor(max(1, len(groups))) as ex:
        list(ex.map(lambda kv: run_group(*kv), groups.items()))
    print(f"done: {state['done']} ok, {len(state['failed'])} failed, "
          f"{time.time() - t0:.0f}s wall", flush=True)
    missing = [(s, sub) for jobs in groups.values() for _, s, sub, *_ in jobs
               if not (out / f"scale{s}" / f"sub{sub:02d}.npz").exists()]
    if state["failed"] or missing:
        (out / "failed.json").write_text(json.dumps(
            {"errors": state["failed"], "missing": missing}, indent=2))
        print(f"missing {missing}; rerun with --run {run} to retry", flush=True)
