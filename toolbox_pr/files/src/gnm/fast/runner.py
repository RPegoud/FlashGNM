r"""Run many [`RunConfig`][gnm.fitting.RunConfig]s on the flash kernel at once.

The toolbox runs one configuration at a time, with `num_simulations` networks per
configuration. The flash kernel is fastest with many simulations per launch, whatever
their parameters, so [`run_flash`][gnm.fast.run_flash] packs the simulations of all
configurations that share a distance matrix, `prob_offset` and number of steps into the
same launches, then splits the results back into one
[`Experiment`][gnm.fitting.Experiment] per configuration, in the input order.
"""

from __future__ import annotations

import gc
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from gnm.generative_rules import MatchingIndex
from gnm.model import GenerativeNetworkModel
from gnm.utils import binary_checks, weighted_checks

from .kernel import FlashGNM, flash_available

# Simulation state is 3 bytes per node pair; leave room for evaluation and outputs.
_STATE_BYTES_PER_PAIR = 3
_OUTPUT_BYTES_PER_PAIR = 4  # the float32 adjacency matrices handed to evaluations


def flash_unsupported_reason(run_config) -> Optional[str]:
    r"""Why `run_config` can't run on the flash backend, or None if it can.

    Supported: binary growth with `MatchingIndex(divisor="mean")`, power-law distance and
    preference relationships, any `lambdah` without a heterochronous matrix, any
    `prob_offset` and `binary_updates_per_iteration`, optional seed networks, and a
    single [n, n] distance matrix (or none).
    """
    bp = run_config.binary_parameters
    rule = bp.generative_rule
    if type(rule) is not MatchingIndex:
        return f"generative rule {rule} (only MatchingIndex is implemented)"
    if rule.divisor != "mean":
        return f"MatchingIndex divisor '{rule.divisor}' (only 'mean' is implemented)"
    if bp.distance_relationship_type != "powerlaw":
        return f"distance relationship '{bp.distance_relationship_type}' (only 'powerlaw')"
    if bp.preferential_relationship_type != "powerlaw":
        return f"preferential relationship '{bp.preferential_relationship_type}' (only 'powerlaw')"
    if run_config.heterochronous_matrix is not None:
        return "heterochronous_matrix (heterochronicity is not implemented)"
    # without a heterochronous matrix H = 1: H ** lambdah = 1, but exp(lambdah * H) is a
    # constant that would change the weight of prob_offset
    if bp.heterochronicity_relationship_type == "exponential" and float(bp.lambdah) != 0.0:
        return "exponential heterochronicity with lambdah != 0"
    if run_config.weighted_parameters is not None or run_config.seed_weight_matrix is not None:
        return "weighted parameters (only binary growth is implemented)"
    D = run_config.distance_matrix
    if D is not None and D.dim() != 2:
        return "a per-simulation distance matrix (the batch shares one [n, n] matrix)"
    return None


def _num_nodes(run_config) -> int:
    if run_config.distance_matrix is not None:
        return run_config.distance_matrix.shape[-1]
    if run_config.seed_adjacency_matrix is not None:
        return run_config.seed_adjacency_matrix.shape[-1]
    raise ValueError("The flash backend needs a distance matrix or a seed adjacency matrix")


def _distance(run_config, n: int, device) -> torch.Tensor:
    if run_config.distance_matrix is None:  # the toolbox's default: uniform distances
        D = torch.ones(n, n)
        D.fill_diagonal_(0.0)
        return D.to(device)
    return run_config.distance_matrix.to(device, torch.float32)


def _evaluate(
    adjacency: torch.Tensor,
    binary_evaluations,
    real_binary_matrices,
):
    # imported here: gnm.fitting imports this module (perform_sweep(backend="flash"))
    from gnm.fitting.experiment_dataclasses import EvaluationResults

    if binary_evaluations is None or real_binary_matrices is None:
        return EvaluationResults(binary_evaluations={}, weighted_evaluations={})
    return EvaluationResults(
        binary_evaluations={
            str(evaluation): evaluation(adjacency, real_binary_matrices)
            for evaluation in binary_evaluations
        },
        weighted_evaluations={},
    )


def run_flash(
    run_configs: List,
    binary_evaluations: Optional[List] = None,
    real_binary_matrices: Optional[torch.Tensor] = None,
    save_model: bool = True,
    save_run_history: bool = True,
    device: Optional[Union[torch.device, str]] = None,
    max_batch_size: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[List, List[float]]:
    r"""Run binary [`RunConfig`][gnm.fitting.RunConfig]s on the flash kernel.

    Same inputs and outputs as calling [`perform_run`][gnm.fitting.perform_run] on each
    configuration, except:

    - `run_history.adjacency_snapshots` is None. Storing every step costs
      num_steps x num_simulations x n x n floats (40 GB for n = 1000, 100 simulations and
      1,000 steps); `run_history.added_edges` has the same information, and
      [`adjacency_snapshots_from_edges`][gnm.fast.adjacency_snapshots_from_edges] rebuilds
      the snapshots when they fit in memory.
    - `clamp_count` counts the steps at which a newly computed wiring weight was inf or
      nan (and so clamped), per simulation, maximised over simulations. It is nonzero
      exactly when the toolbox's would be, but the toolbox counts every step while a
      non-finite weight exists anywhere in the batch, so the values differ.
    - Weighted evaluations are not supported (there is no weighted model).

    Args:
        run_configs: configurations; each must pass
            [`flash_unsupported_reason`][gnm.fast.flash_unsupported_reason].
        binary_evaluations, real_binary_matrices, save_model, save_run_history, device:
            as in [`perform_run`][gnm.fitting.perform_run].
        max_batch_size: simulations per kernel launch. The default fills a third of free
            GPU memory, up to 1,024 (throughput is flat beyond ~1,000 on an H100).
        verbose: show a progress bar over launches.

    Returns:
        The experiments, in the order of `run_configs`, and the time spent on each
        (each launch's time, split evenly over its simulations).
    """
    from gnm.fitting.experiment_dataclasses import Experiment, RunHistory

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if not flash_available(device):
        raise RuntimeError(
            f"The flash backend needs Triton and a CUDA device (got {device})."
        )
    for k, rc in enumerate(run_configs):
        reason = flash_unsupported_reason(rc)
        if reason is not None:
            raise ValueError(f"run_configs[{k}] is not supported by the flash backend: {reason}")
    if real_binary_matrices is not None:
        try:
            binary_checks(real_binary_matrices)
        except AssertionError as e:
            raise AssertionError(f"real_binary_matrices are not valid. {e}")
        real_binary_matrices = real_binary_matrices.to(device)
    if binary_evaluations is not None:
        for evaluation in binary_evaluations:
            assert evaluation.accepts == "binary", (
                f"Binary evaluations must accept binary matrices. "
                f"Evaluation {evaluation} accepts {evaluation.accepts}."
            )

    # -- group configurations that can share a launch
    groups: Dict[tuple, List[int]] = {}
    distances: Dict[tuple, torch.Tensor] = {}
    for k, rc in enumerate(run_configs):
        bp = rc.binary_parameters
        n = _num_nodes(rc)
        D = _distance(rc, n, device)
        weighted_checks(D.unsqueeze(0))
        steps = int(bp.num_iterations) * int(bp.binary_updates_per_iteration)
        # configurations share a distance matrix if they hold the same tensor (the usual
        # case in a SweepConfig), else if the values are equal
        d_key = next(
            (key for key, other in distances.items()
             if other.shape == D.shape and (other is D or torch.equal(other, D))),
            None,
        )
        if d_key is None:
            d_key = (len(distances),)
            distances[d_key] = D
        key = (d_key, float(bp.prob_offset), steps)
        groups.setdefault(key, []).append(k)

    experiments: List = [None] * len(run_configs)
    run_times: List[float] = [0.0] * len(run_configs)
    pbar = tqdm(desc="Flash launches", disable=not verbose)

    for (d_key, offset, steps), members in groups.items():
        D = distances[d_key]
        n = D.shape[-1]
        if max_batch_size is None:
            if device.type == "cuda":
                free = torch.cuda.mem_get_info(device)[0]
                per_sim = (_STATE_BYTES_PER_PAIR + _OUTPUT_BYTES_PER_PAIR) * n * n
                batch = int(max(1, min(1024, free // 3 // per_sim)))
            else:
                batch = 64
        else:
            batch = max_batch_size

        # flatten to (configuration, replicate) and fill launches in order
        sims = [
            (k, r)
            for k in members
            for r in range(run_configs[k].num_simulations or 1)
        ]
        # per configuration: pieces of (replicate slice, final A, edges, clamp counts)
        pending: Dict[int, list] = {k: [] for k in members}
        remaining = {k: run_configs[k].num_simulations or 1 for k in members}

        for lo in range(0, len(sims), batch):
            chunk = sims[lo:lo + batch]
            start = time.perf_counter()
            eta = torch.tensor(
                [float(run_configs[k].binary_parameters.eta) for k, _ in chunk]
            )
            gamma = torch.tensor(
                [float(run_configs[k].binary_parameters.gamma) for k, _ in chunk]
            )
            seeds = [run_configs[k].seed_adjacency_matrix for k, _ in chunk]
            seed = None
            if any(s is not None for s in seeds):
                seed = torch.stack([
                    torch.zeros(n, n, device=device) if s is None
                    else (s.to(device) if s.dim() == 2 else s[r].to(device))
                    for s, (_, r) in zip(seeds, chunk)
                ])
            sim = FlashGNM(D, eta.to(device), gamma.to(device),
                           seed_adjacency_matrix=seed, prob_offset=offset)
            edges = sim.run(steps)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            # split the launch back into configurations
            pos = 0
            while pos < len(chunk):
                k = chunk[pos][0]
                end = pos
                while end < len(chunk) and chunk[end][0] == k:
                    end += 1
                pending[k].append((
                    sim.A[pos:end].to(torch.float32),
                    edges[pos:end].to(torch.int64),
                    sim.clamp_count[pos:end],
                ))
                run_times[k] += elapsed * (end - pos) / len(chunk)
                remaining[k] -= end - pos
                if remaining[k] == 0:
                    experiments[k] = _finish(
                        run_configs[k], pending.pop(k), binary_evaluations,
                        real_binary_matrices, save_model, save_run_history,
                        Experiment, RunHistory,
                    )
                pos = end
            del sim, edges
            pbar.update(1)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pbar.close()
    return experiments, run_times


def _finish(run_config, pieces, binary_evaluations, real_binary_matrices,
            save_model, save_run_history, Experiment, RunHistory):
    A = torch.cat([p[0] for p in pieces])
    edges = torch.cat([p[1] for p in pieces])  # [S, T, 2]
    clamp = int(torch.cat([p[2] for p in pieces]).max().item())

    evaluation_results = _evaluate(A, binary_evaluations, real_binary_matrices)

    model = None
    if save_model:
        model = GenerativeNetworkModel(
            binary_parameters=run_config.binary_parameters,
            num_simulations=A.shape[0],
            seed_adjacency_matrix=run_config.seed_adjacency_matrix,
            distance_matrix=run_config.distance_matrix,
            device=torch.device("cpu"),
            verbose=False,
        )
        model.adjacency_matrix = A.cpu()
        model._clamp_count = clamp

    run_history = None
    if save_run_history:
        run_history = RunHistory(
            added_edges=edges.transpose(0, 1).contiguous(),  # the toolbox's [T, S, 2]
            adjacency_snapshots=None,
            weight_snapshots=None,
        )

    experiment = Experiment(
        run_config=run_config,
        model=model,
        run_history=run_history,
        evaluation_results=evaluation_results,
        clamp_count=clamp,
    )
    experiment.to_device("cpu")
    return experiment


def adjacency_snapshots_from_edges(
    added_edges: torch.Tensor,
    num_nodes: int,
    seed_adjacency_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Rebuild the toolbox's adjacency snapshots from `run_history.added_edges`.

    Args:
        added_edges: [T, S, 2] edges, as in [`RunHistory`][gnm.fitting.RunHistory].
        num_nodes: n.
        seed_adjacency_matrix: optional [n, n] or [S, n, n] starting networks.

    Returns:
        [T, S, n, n] float32: the adjacency matrices after each step. This needs
        4 T S n^2 bytes; use it on a subset of steps or simulations at large n.
    """
    T, S, _ = added_edges.shape
    A = torch.zeros(S, num_nodes, num_nodes, device=added_edges.device)
    if seed_adjacency_matrix is not None:
        A = A + seed_adjacency_matrix.to(A)
    out = torch.empty(T, S, num_nodes, num_nodes, device=added_edges.device)
    s = torch.arange(S, device=added_edges.device)
    for t in range(T):
        i, j = added_edges[t, :, 0], added_edges[t, :, 1]
        A[s, i, j] = 1.0
        A[s, j, i] = 1.0
        out[t] = A
    return out
