"""Simulation (generation) throughput benchmark, SPEC §10.

For each implementation x n x B: setup time (state allocation, initial P / row sums), time
for T steps, peak GPU memory, ms/step and simulations/s for a realistic fit.

    python bench/bench_sim.py                                  # default grid, all impls
    python bench/bench_sim.py --impls triton_proto --n 68 1000 --B 1 64 1024
    python bench/bench_sim.py --impls mymod:MyGNM --T 5000     # any class with the contract below
    modal run bench/modal_bench.py --args "--impls triton_proto --n 1000"   # on a Modal GPU

Implementations (`--impls`):
    triton_proto    prep/kernel/kernel.py TritonGNM, persistent kernel, T steps in one launch
    rowsampled      prep/kernel/row_sampled.py, O(n)/step PyTorch, eager      (extrapolated)
    leandense       prep/kernel/lean_dense.py, the fair dense baseline        (extrapolated)
    toolbox         gnm.GenerativeNetworkModel.binary_update, unchanged       (extrapolated)
    pkg.mod:Class   any class with the TritonGNM contract:
                        sim = Class(dist [n,n] f32 diag 1, eta [B] f32, gamma [B] f32)
                        sim.run(T) -> [B, T, 2]      (or sim.step() if it has no run)
                    Module paths resolve from the repo root and prep/kernel.

"Extrapolated" implementations run --probe-steps steps after a warm-up and report
per_step * T; their rows say `extrapolated: true`. Per-step cost of the dense ones doesn't
depend on t, and RowSampled's barely does.

T defaults to the mean Griffa edge count per subject at that n (68: 610, 219: 2639,
1000: 10785), else the n = 1000 density (2.16%) times n(n-1)/2. Distances are Griffa's
template distances where n matches a Griffa scale, else synthetic (`make_dist`).

Output: results/<YY-MM-DD_HH-MM-SS>_sim/{meta.json, results.jsonl, summary.md}.
Benchmarks quoted in the presentation must come from these directories, committed.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "prep" / "kernel"
for p in (str(ROOT), str(PROTO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from bench_gnm import make_dist  # noqa: E402  (prep/kernel)

GRIFFA = {68: "033", 114: "060", 219: "125", 448: "250", 1000: "500"}
GRIFFA_T = {68: 610, 114: 1227, 219: 2639, 448: 5324, 1000: 10785}  # mean edges/subject
DENSITY_1000 = 10785 / (1000 * 999 / 2)


# ------------------------------------------------------------------ inputs
def default_T(n: int) -> int:
    return GRIFFA_T.get(n) or round(DENSITY_1000 * n * (n - 1) / 2)


def load_dist(n: int, source: str) -> tuple[torch.Tensor, str]:
    """[n, n] float32 distances with a zero diagonal, and where they came from."""
    if source in ("auto", "griffa") and n in GRIFFA:
        z = np.load(ROOT / "data" / "griffa" / f"scale{GRIFFA[n]}.npz")
        d = torch.from_numpy(z["dist_mm"].astype(np.float32))
        return d, f"griffa scale{GRIFFA[n]}"
    if source == "griffa":
        raise ValueError(f"no Griffa scale with n = {n} (have {sorted(GRIFFA)})")
    return make_dist(n, zero_diag=True), "synthetic make_dist"


def make_params(B: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One (eta, gamma) per simulation, the prototype's sweep ranges. Same for every impl."""
    g = torch.Generator().manual_seed(seed)
    eta = torch.empty(B).uniform_(-3.0, 0.0, generator=g)
    gamma = torch.empty(B).uniform_(0.1, 0.5, generator=g)
    return eta, gamma


# ------------------------------------------------------------------ implementations
class Impl:
    """build() allocates state and computes initial P; run(sim, T) runs T steps."""

    name: str
    extrapolated = False

    def build(self, dist0, eta, gamma, device):
        raise NotImplementedError

    def run(self, sim, T):
        raise NotImplementedError

    def step(self, sim):
        raise NotImplementedError


def _diag1(dist0):
    d = dist0.clone()
    d.fill_diagonal_(1.0)
    return d


class ContractImpl(Impl):
    """Any class with the TritonGNM contract (constructor + run(T), or step())."""

    def __init__(self, name, cls):
        self.name, self.cls = name, cls

    def build(self, dist0, eta, gamma, device):
        return self.cls(_diag1(dist0).to(device), eta.to(device), gamma.to(device))

    def run(self, sim, T):
        if hasattr(sim, "run"):
            return sim.run(T)
        for _ in range(T):
            sim.step()

    def step(self, sim):
        sim.step()


class RowSampledImpl(ContractImpl):
    extrapolated = True

    def __init__(self):
        from row_sampled import RowSampled

        super().__init__("rowsampled", RowSampled)

    def run(self, sim, T):
        for _ in range(T):
            sim.step()


class LeanDenseImpl(RowSampledImpl):
    def __init__(self):
        from lean_dense import LeanDense

        ContractImpl.__init__(self, "leandense", LeanDense)


class ToolboxImpl(Impl):
    name = "toolbox"
    extrapolated = True

    def build(self, dist0, eta, gamma, device):
        # The toolbox takes one (eta, gamma) per model, so a sweep is a Python loop over
        # models. Time one model with all B simulations: a lower bound on the sweep.
        from gnm import BinaryGenerativeParameters, GenerativeNetworkModel
        from gnm.generative_rules import MatchingIndex

        params = BinaryGenerativeParameters(
            eta=float(eta[0]),
            gamma=float(gamma[0]),
            lambdah=0.0,
            distance_relationship_type="powerlaw",
            preferential_relationship_type="powerlaw",
            heterochronicity_relationship_type="powerlaw",
            generative_rule=MatchingIndex(divisor="mean"),
            num_iterations=1,
        )
        return GenerativeNetworkModel(
            binary_parameters=params,
            num_simulations=eta.shape[0],
            num_nodes=dist0.shape[0],
            distance_matrix=dist0.to(device),
            device=device,
        )

    def step(self, sim):
        sim.binary_update()

    def run(self, sim, T):
        for _ in range(T):
            sim.binary_update()


def get_impl(spec: str) -> Impl:
    if spec == "triton_proto":
        from kernel import TritonGNM

        return ContractImpl("triton_proto", TritonGNM)
    if spec == "rowsampled":
        return RowSampledImpl()
    if spec == "leandense":
        return LeanDenseImpl()
    if spec == "toolbox":
        return ToolboxImpl()
    if ":" in spec:
        mod, cls = spec.split(":", 1)
        return ContractImpl(spec, getattr(importlib.import_module(mod), cls))
    raise ValueError(f"unknown implementation {spec!r}")


# ------------------------------------------------------------------ timing
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def timed(fn, device):
    sync(device)
    t = time.perf_counter()
    out = fn()
    sync(device)
    return time.perf_counter() - t, out


def check_edges(sim, B, n, T):
    """True if every simulation holds exactly T undirected edges (symmetric A, no
    self-loops); None if the implementation exposes no adjacency. Guards against timing
    a kernel that silently does nothing."""
    A = getattr(sim, "A", None)
    if A is None and hasattr(sim, "state"):
        A = sim.state().get("A")
    if A is None:
        return None
    A = A.reshape(B, n, n)
    for b in range(B):  # one simulation at a time: the check must not need B x n^2 temporaries
        a = A[b].bool()
        if a.diagonal().any() or not torch.equal(a, a.T):
            return False
        if int(a.sum(dtype=torch.int32)) != 2 * T:
            return False
    return True


def bench_one(impl, n, B, T, dist0, dist_src, args, device) -> dict:
    eta, gamma = make_params(B, args.seed)
    row = dict(impl=impl.name, n=n, B=B, T=T, dist=dist_src, extrapolated=impl.extrapolated)
    free(device)

    # warm-up: compile Triton kernels / init cuBLAS on a tiny run, outside the clock
    warm_T = min(T, 8)
    t0 = time.perf_counter()
    sim = impl.build(dist0, eta[: min(B, 2)], gamma[: min(B, 2)], device)
    impl.run(sim, warm_T)
    sync(device)
    row["warmup_s"] = time.perf_counter() - t0
    del sim
    free(device)

    setups, runs = [], []
    for _ in range(args.repeats):
        setup_s, sim = timed(lambda: impl.build(dist0, eta, gamma, device), device)
        if impl.extrapolated:
            k = min(args.probe_steps, T)
            for _ in range(min(3, T - k)):
                impl.step(sim)
            dt, _ = timed(lambda: [impl.step(sim) for _ in range(k)], device)
            run_s = dt / k * T
        else:
            run_s, _ = timed(lambda: impl.run(sim, T), device)
        setups.append(setup_s)
        runs.append(run_s)
        if device.type == "cuda":  # before check_edges, which allocates
            row["peak_mem_gb"] = max(
                row.get("peak_mem_gb", 0.0), torch.cuda.max_memory_allocated(device) / 1e9
            )
        if not impl.extrapolated and "edges_ok" not in row:
            row["edges_ok"] = check_edges(sim, B, n, T)
        del sim
        free(device)
        if impl.extrapolated and run_s > args.slow_s:
            break  # one repeat is enough for a baseline this slow

    setup_s, run_s = statistics.median(setups), statistics.median(runs)
    total = setup_s + run_s
    row.update(
        status="ok",
        repeats=len(runs),
        setup_s=setup_s,
        run_s=run_s,
        total_s=total,
        ms_per_step=run_s / T * 1e3,
        sims_per_s=B / total,
        sims_per_s_run_only=B / run_s,
        run_s_all=runs,
        setup_s_all=setups,
    )
    return row


# ------------------------------------------------------------------ output
def env_meta(device) -> dict:
    import triton

    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
        except Exception:
            return None

    meta = dict(
        date=datetime.now().isoformat(timespec="seconds"),
        host=platform.node(),
        python=platform.python_version(),
        torch=torch.__version__,
        triton=triton.__version__,
        cuda=torch.version.cuda,
        device=str(device),
        git_sha=git("rev-parse", "HEAD") or os.environ.get("GIT_SHA"),
        git_dirty=bool(git("status", "--porcelain")) if git("rev-parse", "HEAD") else None,
    )
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        meta.update(gpu=p.name, gpu_mem_gb=p.total_memory / 1e9, sm_count=p.multi_processor_count)
    return meta


def fmt(x, spec):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def summary_md(meta, rows) -> str:
    lines = [
        f"# Simulation benchmark, {meta['date']}",
        "",
        f"{meta.get('gpu', meta['device'])}, torch {meta['torch']}, triton {meta['triton']}, "
        f"git {str(meta.get('git_sha'))[:8]}{' (dirty)' if meta.get('git_dirty') else ''}",
        "",
        f"`{meta['argv']}`",
        "",
        "sims/s = B / (setup + run). `~` = extrapolated from probe steps.",
        "",
        "| impl | n | B | T | setup s | run s | ms/step | sims/s | peak GB | edges ok | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        ok = r["status"] == "ok"
        x = "~" if r.get("extrapolated") and ok else ""
        lines.append(
            f"| {r['impl']} | {r['n']} | {r['B']} | {r['T']} | "
            f"{fmt(r.get('setup_s'), '.3f')} | {x}{fmt(r.get('run_s'), '.3f')} | "
            f"{x}{fmt(r.get('ms_per_step'), '.4f')} | {x}{fmt(r.get('sims_per_s'), ',.1f')} | "
            f"{fmt(r.get('peak_mem_gb'), '.2f')} | {fmt(r.get('edges_ok'), '')} | {r['status']} |"
        )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ main
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impls", nargs="+", default=["triton_proto", "rowsampled", "leandense", "toolbox"])
    ap.add_argument("--n", nargs="+", type=int, default=[68, 219, 1000, 2000])
    ap.add_argument("--B", nargs="+", type=int, default=[1, 64, 1024])
    ap.add_argument("--T", type=int, default=None, help="steps (edges); default: realistic per n")
    ap.add_argument("--dist", choices=["auto", "griffa", "synthetic"], default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="median over this many fresh runs")
    ap.add_argument("--probe-steps", type=int, default=20, help="steps timed for extrapolated impls")
    ap.add_argument("--slow-s", type=float, default=60.0, help="extrapolated run above this: 1 repeat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None, help="default: results/<timestamp>_sim")
    ap.add_argument("--tag", default="", help="appended to the results directory name")
    return ap.parse_args(argv)


def main(argv=None) -> Path:
    args = parse(argv)
    device = torch.device(args.device)
    stamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    out = args.out or ROOT / "results" / f"{stamp}_sim{'_' + args.tag if args.tag else ''}"
    out.mkdir(parents=True, exist_ok=True)

    meta = env_meta(device)
    meta["argv"] = " ".join(["bench/bench_sim.py", *(argv if argv is not None else sys.argv[1:])])
    meta["args"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta.get(k) for k in ("gpu", "torch", "triton", "git_sha")}), flush=True)

    rows = []
    jl = (out / "results.jsonl").open("a")
    for spec in args.impls:
        impl = get_impl(spec)
        for n in sorted(args.n):
            dist0, dist_src = load_dist(n, args.dist)
            T = args.T or default_T(n)
            T = min(T, n * (n - 1) // 2)
            for B in sorted(args.B):
                try:
                    r = bench_one(impl, n, B, T, dist0, dist_src, args, device)
                except torch.OutOfMemoryError:
                    r = dict(impl=impl.name, n=n, B=B, T=T, dist=dist_src, status="oom")
                except Exception as e:  # keep the sweep going; record the failure
                    r = dict(impl=impl.name, n=n, B=B, T=T, dist=dist_src,
                             status=f"error: {type(e).__name__}: {e}"[:200],
                             traceback=traceback.format_exc())
                free(device)
                rows.append(r)
                jl.write(json.dumps(r) + "\n")
                jl.flush()
                print(summary_md(meta, [r]).splitlines()[-1], flush=True)
    jl.close()
    (out / "summary.md").write_text(summary_md(meta, rows))
    print(f"\nwrote {out}")
    return out


if __name__ == "__main__":
    main()
