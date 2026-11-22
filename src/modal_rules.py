"""Rule-form experiments (SPEC H9) on Modal: replicated grids, Oldham's Voronoi search,
recovery at each rule's optimum.

    modal run src/modal_rules.py::grids --rule add_exp --scales "033 125" --run add_exp_wide
    modal run src/modal_rules.py::grids --rule mult_exp --n-alpha 1 --n-eta 41 --n-gamma 41 \
        --repeats 20 --run mult_exp_wide
    modal run src/modal_rules.py::voronoi --rule add_exp --scales "033" --run vor_add_exp
    modal run src/modal_rules.py::recovery --scales "033 125" --optima results/rules/optima.json

Grids: one subject per GPU call; results on the volume `flashgnm-landscapes` under
<run>/scale<s>/sub<ss>.npz and locally under results/rules/<run>/. Voronoi: one call per
(scale, chunk of subjects); per subject all 10,000 points with their four KS values.
"""

import io
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
app = modal.App("FlashGNM_rules", image=image)
triton_vol = modal.Volume.from_name("triton_kernels", create_if_missing=True)
out_vol = modal.Volume.from_name("flashgnm-landscapes", create_if_missing=True)
GPU = dict(gpu="H100!", volumes={TRITON_CACHE: triton_vol, VOL_DIR: out_vol},
           max_containers=20, retries=1)


def _commit_triton():
    try:
        triton_vol.commit()
    except Exception:
        pass


@app.function(timeout=2 * 3600, **GPU)
def grid_remote(run: str, scale: str, subject: int, rule: str, grid: dict, seed: int) -> bytes:
    import os

    import numpy as np  # noqa: F401
    import torch

    from src import rules_landscape as rl
    from src.landscape import to_bytes

    res = rl.run_subject(scale, subject, rule, rl.Grid3(**grid), torch.device("cuda"), seed=seed)
    meta = json.loads(res["meta"])
    meta["git"] = os.environ.get("GIT_SHA", "")
    res["meta"] = json.dumps(meta)
    data = to_bytes(res)
    path = Path(VOL_DIR) / run / f"scale{scale}" / f"sub{subject:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    out_vol.commit()
    _commit_triton()
    return data


@app.function(timeout=4 * 3600, **GPU)
def voronoi_remote(scale: str, subjects: list, rule: str, seed: int, ndraw: int,
                   nlvl: int, repeats: int = 1) -> list:
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch

    from src import rules_landscape as rl
    from src import voronoi_search as vs
    from src.landscape import load_griffa

    dev = torch.device("cuda")
    D, target_fn, _ = load_griffa(scale)
    n = D.shape[0]
    lock = threading.Lock()  # GPU work one subject at a time; Voronoi geometry overlaps
    if rule == "add_exp":
        lo, hi = [-2.0, -10.0, 0.0], [0.0, 10.0, 10.0]
    elif rule == "mult_exp":
        lo, hi = [-2.0, -10.0, 1.0], [0.0, 10.0, 1.0]
    else:
        lo, hi = [-10.0, -10.0, 1.0], [0.0, 10.0, 1.0]

    def one(sub):
        A_t = target_fn(sub)
        store = []

        def evaluate(p):
            # each point simulated `repeats` times; the mean energy drives the search
            k = len(p)
            p = torch.tensor(np.repeat(p, repeats, axis=0), dtype=torch.float32)
            with lock:
                ks, info = rl.evaluate_params(A_t, D, rule, p[:, 0], p[:, 1], p[:, 2], dev,
                                              rl.default_batch(n),
                                              seed * 1_000_003 + sub * 1009 + len(store))
            ks = ks.numpy().reshape(k, repeats, 4)
            e = ks.max(-1)  # [k, repeats]
            store.append((ks.mean(1), e.mean(1), e.std(1), info))
            return e.mean(1)

        P, E, lvl = vs.search(evaluate, lo, hi, ndraw=ndraw, nlvl=nlvl,
                              seed=seed * 1009 + sub)
        ks = np.concatenate([s[0] for s in store])  # mean KS per point
        return {"scale": scale, "subject": sub, "rule": rule, "n": n, "T": int(A_t.sum()) // 2,
                "repeats": repeats, "params": P.tolist(), "ks": ks.tolist(),
                "energy": np.concatenate([s[1] for s in store]).tolist(),
                "energy_sd": np.concatenate([s[2] for s in store]).tolist(),
                "level": lvl.tolist(),
                "valid": all(s[3]["validation"]["ok"] for s in store),
                "t_gen_s": sum(s[3]["t_gen_s"] for s in store),
                "t_eval_s": sum(s[3]["t_eval_s"] for s in store),
                "git": os.environ.get("GIT_SHA", "")}

    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(one, subjects))
    _commit_triton()
    return out


@app.function(timeout=3600, **GPU)
def recovery_remote(scale: str, subject: int, optima: dict, n_sims: int, seed: int) -> dict:
    import torch

    from src import recovery

    out = recovery.run_rule_subject(scale, subject, optima, torch.device("cuda"),
                                    n_sims=n_sims, seed=seed)
    _commit_triton()
    return out


@app.local_entrypoint()
def grids(rule: str = "add_exp", scales: str = "033 125", subjects: str = "all", run: str = "",
          seed: int = 0, eta_lo: float = -2.0, eta_hi: float = 0.0, n_eta: int = 21,
          gamma_lo: float = -10.0, gamma_hi: float = 10.0, n_gamma: int = 21,
          alpha_lo: float = 0.0, alpha_hi: float = 10.0, n_alpha: int = 21, repeats: int = 10):
    import numpy as np

    run = run or f"{rule}_{time.strftime('%y-%m-%d_%H-%M-%S')}"
    if rule != "add_exp":
        alpha_lo = alpha_hi = 1.0
        n_alpha = 1
    grid = dict(eta_lo=eta_lo, eta_hi=eta_hi, n_eta=n_eta, gamma_lo=gamma_lo,
                gamma_hi=gamma_hi, n_gamma=n_gamma, alpha_lo=alpha_lo, alpha_hi=alpha_hi,
                n_alpha=n_alpha, repeats=repeats)
    out = ROOT / "results" / "rules" / run
    out.mkdir(parents=True, exist_ok=True)
    mp = out / "meta.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        assert old["grid"] == grid and old["seed"] == seed and old["rule"] == rule, "resume mismatch"
    else:
        mp.write_text(json.dumps({"run": run, "rule": rule, "grid": grid, "seed": seed,
                                  "gpu": "H100", "git": _git_sha(), "started": time.ctime()},
                                 indent=2))
    jobs = []
    for s in scales.split():
        subs = range(70) if subjects == "all" else map(int, subjects.split())
        for sub in subs:
            if not (out / f"scale{s}" / f"sub{sub:02d}.npz").exists():
                jobs.append((run, s, sub, rule, grid, seed))
    jobs.sort(key=lambda j: -int(j[1]))
    print(f"run {run}: {len(jobs)} jobs -> {out}", flush=True)
    t0, done = time.time(), 0
    for res in grid_remote.starmap(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"FAILED: {repr(res)[:300]}", flush=True)
            continue
        m = json.loads(str(np.load(io.BytesIO(res))["meta"]))
        p = out / f"scale{m['scale']}" / f"sub{m['subject']:02d}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(res)
        done += 1
        print(f"[{done}/{len(jobs)} {time.time() - t0:6.0f}s] scale {m['scale']} sub "
              f"{m['subject']:02d}: gen {m['t_gen_s']:.1f}s eval {m['t_eval_s']:.1f}s "
              f"valid={m['validation']['ok']} R_err={m['validation']['R_max_rel_err']:.1e} "
              f"rec/step={m['row_max_recomputes_per_step']:.2f} "
              f"resync/sim={m['resyncs_per_sim']:.2f}", flush=True)
    print(f"done: {done}/{len(jobs)}, {time.time() - t0:.0f}s wall", flush=True)


@app.local_entrypoint()
def voronoi(rule: str = "add_exp", scales: str = "033", subjects: str = "all", run: str = "",
            seed: int = 0, ndraw: int = 2000, nlvl: int = 5, chunk: int = 4, repeats: int = 1):
    run = run or f"vor_{rule}"
    out = ROOT / "results" / "rules" / run
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in scales.split():
        subs = list(range(70)) if subjects == "all" else list(map(int, subjects.split()))
        subs = [x for x in subs if not (out / f"scale{s}" / f"sub{x:02d}.json").exists()]
        for i in range(0, len(subs), chunk):
            jobs.append((s, subs[i:i + chunk], rule, seed, ndraw, nlvl, repeats))
    print(f"voronoi {run}: {len(jobs)} calls", flush=True)
    t0 = time.time()
    for res in voronoi_remote.starmap(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"FAILED: {repr(res)[:300]}", flush=True)
            continue
        for r in res:
            p = out / f"scale{r['scale']}" / f"sub{r['subject']:02d}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(r))
            best = min(r["energy"])
            print(f"[{time.time() - t0:6.0f}s] scale {r['scale']} sub {r['subject']:02d}: "
                  f"best {best:.3f} valid={r['valid']} gen {r['t_gen_s']:.0f}s", flush=True)


@app.local_entrypoint()
def recovery(scales: str = "033 125", optima: str = "results/rules/optima.json",
             n_sims: int = 100, seed: int = 0, out: str = "results/rules/recovery"):
    opt = json.loads((ROOT / optima).read_text())  # {scale: {subject: {rule: [e, g, a]}}}
    od = ROOT / out
    od.mkdir(parents=True, exist_ok=True)
    jobs = [(s, int(sub), {k: v for k, v in o.items() if not k.startswith("_")}, n_sims, seed)
            for s in scales.split()
            for sub, o in opt.get(s, {}).items()]
    print(f"{len(jobs)} recovery jobs", flush=True)
    rows = {}
    for res in recovery_remote.starmap(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"FAILED: {repr(res)[:300]}", flush=True)
            continue
        rows.setdefault(res["scale"], []).append(res)
    for s, rs in rows.items():
        rs.sort(key=lambda r: r["subject"])
        with open(od / f"scale{s}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    print("done", {s: len(r) for s, r in rows.items()}, flush=True)
