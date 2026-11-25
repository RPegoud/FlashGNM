"""Simulation speed: GenerativeNetworkModel (torch backend) vs gnm.fast.FlashGNM.

    python scripts/benchmark_flash.py                         # n = 90 (default data), 400 edges
    python scripts/benchmark_flash.py --n 1000 --steps 10000 --sims 64 1024

Each simulation has its own (eta, gamma) for the flash kernel; the toolbox runs one
(eta, gamma) with all simulations (its fastest case). The toolbox is timed on its first
--toolbox-steps steps and extrapolated linearly; it is skipped when its per-step adjacency
snapshots (steps x sims x n x n floats) would exceed 20 GB. Simulation only, no evaluation.
"""

import argparse
import time

import torch

from gnm import BinaryGenerativeParameters, GenerativeNetworkModel
from gnm.defaults import get_distance_matrix
from gnm.fast import FlashGNM
from gnm.generative_rules import MatchingIndex


def distances(n: int, device) -> torch.Tensor:
    if n == 90:
        return get_distance_matrix().to(device)
    x = torch.rand(n, 3, generator=torch.Generator().manual_seed(0))
    return torch.cdist(x, x).to(device)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_toolbox(D, sims, steps, probe):
    probe = min(probe, steps)
    bp = BinaryGenerativeParameters(
        eta=-2.0, gamma=0.3, lambdah=0.0, distance_relationship_type="powerlaw",
        preferential_relationship_type="powerlaw", heterochronicity_relationship_type="powerlaw",
        generative_rule=MatchingIndex(), num_iterations=probe)
    torch.cuda.reset_peak_memory_stats()
    sync(); t0 = time.perf_counter()
    model = GenerativeNetworkModel(binary_parameters=bp, num_simulations=sims,
                                   distance_matrix=D, device=D.device)
    model.run_model()
    sync()
    seconds = (time.perf_counter() - t0) * steps / probe
    return seconds, torch.cuda.max_memory_allocated() / 2**30


def time_flash(D, sims, steps):
    eta = torch.linspace(-3, 0, sims, device=D.device)
    gamma = torch.linspace(0, 1, sims, device=D.device)
    FlashGNM(D, eta[:1], gamma[:1]).run(min(steps, 10))  # compile once
    torch.cuda.reset_peak_memory_stats()
    sync(); t0 = time.perf_counter()
    FlashGNM(D, eta, gamma).run(steps)
    sync()
    return time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**30


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=90)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--sims", type=int, nargs="+", default=[64, 1024])
    p.add_argument("--toolbox-steps", type=int, default=50)
    p.add_argument("--skip-toolbox", action="store_true")
    a = p.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    D = distances(a.n, torch.device("cuda"))
    print(f"{torch.cuda.get_device_name()}, n = {a.n}, {a.steps} steps")
    print(f"{'sims':>6} {'backend':>8} {'seconds':>10} {'sims/s':>10} {'peak GB':>8}")
    for S in a.sims:
        rows = [("flash", *time_flash(D, S, a.steps))]
        snapshots_gb = min(a.toolbox_steps, a.steps) * S * a.n * a.n * 4 / 2**30
        if a.skip_toolbox:
            pass
        elif snapshots_gb > 20:
            print(f"{S:>6} {'torch~':>8}  skipped: {snapshots_gb:.0f} GB of snapshots")
        else:
            rows.append(("torch~", *time_toolbox(D, S, a.steps, a.toolbox_steps)))
        for name, sec, gb in rows:
            print(f"{S:>6} {name:>8} {sec:>10.3f} {S / sec:>10.1f} {gb:>8.2f}")
    print("torch~: extrapolated from the first --toolbox-steps steps")


if __name__ == "__main__":
    main()
