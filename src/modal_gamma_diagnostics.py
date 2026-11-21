"""Run src/gamma_diagnostics.py on Modal GPUs: 8 configurations x N subjects, plus toolbox
slices. Results in results/gamma_diag_<run>/<config>/sub<ss>.npz and toolbox_slices.json.

    modal run src/modal_gamma_diagnostics.py --subjects 20 --repeats 10
"""

import json
import time
from pathlib import Path, PurePosixPath

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if modal.is_local() else Path("/workspace")
REMOTE = PurePosixPath("/workspace")

image = (
    modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT)
    .entrypoint([])
    .env({"PYTHONPATH": str(REMOTE)})
    .add_local_dir(
        ROOT, remote_path=str(REMOTE),
        ignore=[".git", ".git/**", ".claude", ".claude/**", ".venv", ".venv/**", ".env",
                "papers", "papers/**", "data/mica", "data/mica/**", "results", "results/**",
                "**/__pycache__", "**/__pycache__/**"],
    )
)
app = modal.App("FlashGNM_gamma_diag", image=image)


@app.function(gpu="H100!", timeout=3600, max_containers=24, retries=1)
def config_remote(config: str, subject: int, grid: dict) -> bytes:
    import torch

    from src import gamma_diagnostics as gd
    from src.landscape import Grid, to_bytes

    return to_bytes(gd.run_config(config, subject, Grid(**grid), torch.device("cuda")))


@app.function(gpu="H100!", timeout=3600, max_containers=8, retries=1)
def slice_remote(subject: int, eta: float, gammas: list, repeats: int, seeded: bool) -> dict:
    import torch

    from src import gamma_diagnostics as gd

    return gd.toolbox_slice(subject, eta, gammas, repeats, seeded, torch.device("cuda"))


@app.local_entrypoint()
def main(subjects: int = 20, repeats: int = 10, run: str = ""):
    import io

    import numpy as np

    run = run or time.strftime("%y-%m-%d_%H-%M-%S")
    out = ROOT / "results" / f"gamma_diag_{run}"
    out.mkdir(parents=True, exist_ok=True)
    grid = dict(eta_lo=-4.0, eta_hi=1.0, n_eta=51, gamma_lo=-1.0, gamma_hi=4.0, n_gamma=51,
                repeats=repeats)
    import sys
    sys.path.insert(0, str(ROOT))
    from src.gamma_diagnostics import CONFIGS

    jobs = [(c, s, grid) for c in CONFIGS for s in range(subjects)
            if not (out / c / f"sub{s:02d}.npz").exists()]
    slices = [(s, -1.8, [0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 20, seeded)
              for s in (0, 1) for seeded in (False, True)]
    print(f"run {run}: {len(jobs)} landscape jobs, {len(slices)} toolbox slices -> {out}", flush=True)
    t0 = time.time()
    sl_calls = [slice_remote.spawn(*a) for a in slices]
    done = 0
    for res in config_remote.starmap(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"FAILED: {repr(res)[:300]}", flush=True)
            continue
        m = json.loads(str(np.load(io.BytesIO(res))["meta"]))
        p = out / m["config"] / f"sub{m['subject']:02d}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(res)
        done += 1
        if done % 20 == 0 or done == len(jobs):
            print(f"[{done}/{len(jobs)} {time.time() - t0:5.0f}s] last {m['config']} sub "
                  f"{m['subject']:02d}: gen {m['t_gen_s']:.1f}s valid={m['validation']['ok']}",
                  flush=True)
    sl = [c.get() for c in sl_calls]
    (out / "toolbox_slices.json").write_text(json.dumps(sl, indent=1))
    (out / "meta.json").write_text(json.dumps({"run": run, "grid": grid, "subjects": subjects},
                                              indent=1))
    print(f"done {done}/{len(jobs)}, slices {len(sl)}, {time.time() - t0:.0f}s", flush=True)
