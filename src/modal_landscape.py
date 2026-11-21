"""Energy landscapes for many Griffa subjects and scales, one subject per GPU call (SPEC §8).

    modal run src/modal_landscape.py                                   # all scales, all 70 subjects
    modal run src/modal_landscape.py --scales "033 500" --subjects "0 1 2" --run smoke
    modal run src/modal_landscape.py --run <name>                      # resume: skips done subjects
    modal run src/modal_landscape.py --null --run griffa_null          # synthetic-subject null (H5)

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


@app.function(gpu="H100!", volumes={TRITON_CACHE: triton_vol, VOL_DIR: out_vol},
              timeout=2 * 3600, max_containers=24, retries=1)
def subject_remote(run: str, scale: str, subject: int, grid: dict, seed: int,
                   null: bool = False) -> bytes:
    import os

    import numpy as np
    import torch

    from src import landscape as ls

    dev = torch.device("cuda")
    target = ls.null_target(scale, subject, dev, seed) if null else None
    res = ls.run_subject(scale, subject, ls.Grid(**grid), dev, seed=seed,
                         log=lambda *a: None, target=target)
    meta = json.loads(res["meta"])
    if null:
        meta["target"] = {"null": True, "params": ls.NULL_PARAMS[scale]}
        res["target_A"] = np.packbits(target.numpy())
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


@app.local_entrypoint()
def main(scales: str = "033 060 125 250 500", subjects: str = "all", run: str = "",
         seed: int = 0, eta_lo: float = -4.0, eta_hi: float = 1.0, n_eta: int = 51,
         gamma_lo: float = -1.0, gamma_hi: float = 4.0, n_gamma: int = 51, repeats: int = 20,
         null: bool = False):
    import io
    import sys

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
        meta_path.write_text(json.dumps({"run": run, "grid": grid, "seed": seed, "null": null,
                                         "gpu": "H100",
                                         "git": _git_sha(), "started": time.ctime()}, indent=2))
    jobs = []
    for s in scales.split():
        assert s in SCALES, s
        n_sub = 70
        subs = range(n_sub) if subjects == "all" else map(int, subjects.split())
        for sub in subs:
            if not (out / f"scale{s}" / f"sub{sub:02d}.npz").exists():
                jobs.append((run, s, sub, grid, seed, null))
    # biggest first, so the long n = 1000 calls don't trail at the end
    jobs.sort(key=lambda j: -SCALES[j[1]])
    print(f"run {run}: {len(jobs)} subject-scale jobs to do -> {out}", flush=True)
    t0, done, failed = time.time(), 0, []
    for res in subject_remote.starmap(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):  # which job: whatever is missing at the end
            failed.append(repr(res)[:300])
            print(f"FAILED: {repr(res)[:300]}", flush=True)
            continue
        m = json.loads(str(np.load(io.BytesIO(res))["meta"]))
        s, sub = m["scale"], m["subject"]
        p = out / f"scale{s}" / f"sub{sub:02d}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(res)
        done += 1
        print(f"[{done}/{len(jobs)} {time.time() - t0:7.0f}s] scale {s} sub {sub:02d}: "
              f"T={m['T']} gen {m['t_gen_s']:.1f}s eval {m['t_eval_s']:.1f}s "
              f"valid={m['validation']['ok']} R_err={m['validation']['R_max_rel_err']:.1e}",
              flush=True)
    print(f"done: {done} ok, {len(failed)} failed, {time.time() - t0:.0f}s wall", flush=True)
    missing = [(s, sub) for _, s, sub, *_ in jobs
               if not (out / f"scale{s}" / f"sub{sub:02d}.npz").exists()]
    if failed or missing:
        (out / "failed.json").write_text(json.dumps({"errors": failed, "missing": missing}, indent=2))
        print(f"missing {missing}; rerun with --run {run} to retry", flush=True)
