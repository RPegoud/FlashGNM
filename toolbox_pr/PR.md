# Add a flash backend: fused GPU kernel for binary matching-index GNMs

Following our emails in November, this adds the GPU kernel from the FlashGNM project as an
**optional** backend. Nothing changes unless you ask for it:

```python
experiments = perform_sweep(sweep_config, binary_evaluations=..., real_binary_matrices=...,
                            backend="flash")   # "auto": flash where supported, torch elsewhere
```

## What it does

For binary models with `MatchingIndex(divisor="mean")` and power-law distance and
preference terms, the kernel samples from the same distribution as
`GenerativeNetworkModel.binary_update`. It uses the same offset handling, the same divisor
and the same clamping of overflowing weights. Each step costs O(n) instead of O(n³):

- **Two-stage sampling:** a row by its row sum of P, then a column within that row.
- **Incremental updates:** adding edge {i, j} changes only rows and columns i and j of A,
  AᵀA, the degrees and P.
- **One GPU program per simulation:** it runs all steps of that simulation, and P is never
  stored, so the state is 3 bytes per node pair.

Simulations in one launch can have different (η, γ), so a whole grid fills the GPU.
`perform_sweep` packs all supported configurations into large launches and returns the
same `Experiment` objects, in the same order.

Measured on an H100 at the hackathon (simulation only): at n = 1000 with 10,785 edges,
~1,850 simulations/s, against ~0.1/s for the current implementation. At n = 68, ~6,600/s
against ~340/s. Memory at n = 1000 with 1,024 simulations is 3.2 GB instead of ~50 GB.
`scripts/benchmark_flash.py` reproduces the comparison on any GPU.

## What's supported

- Matching index with the mean divisor; power-law distance and preference relationships.
- Any `prob_offset` and `binary_updates_per_iteration`.
- Seed networks.
- Any `lambdah` when no heterochronous matrix is given.

Not supported: other rules, exponential relationships, heterochronous matrices, weighted
models, Bayesian sweeps. `backend="flash"` raises with the reason;
`backend="auto"` falls back to the torch backend for those configurations.

Differences in the results (documented in `docs/user-guide/flash-backend.md`):

- `run_history.adjacency_snapshots` is `None`. Storing every step does not fit at large
  n; `added_edges` is kept, and `gnm.fast.adjacency_snapshots_from_edges` rebuilds the
  snapshots.
- `clamp_count` is nonzero exactly when the torch backend's would be, but counts
  differently.

## Changes

- `src/gnm/fast/`: the kernel (`FlashGNM`), the runner that packs `RunConfig`s into
  launches (`run_flash`), and `flash_unsupported_reason`. Triton is imported lazily and is
  optional; it ships with torch on Linux.
- `src/gnm/fitting/sweep.py`: a `backend` argument for `perform_run` and `perform_sweep`
  (default `"torch"`, unchanged behaviour). The grid loop now runs flash configurations
  first and torch ones after, keeping the order.
- `tests/test_flash_kernel.py`, `tests/test_flash_sweep.py`: see below.
- `docs/user-guide/flash-backend.md`, `docs/api-reference/fast.md`, README section,
  `scripts/benchmark_flash.py`.

## How to check it

The reference for every test is the toolbox itself:

```bash
pytest tests/test_flash_kernel.py tests/test_flash_sweep.py -v        # on a CUDA GPU
GNM_RUN_SLOW=1 pytest tests/test_flash_kernel.py -v -k toolbox         # vs GenerativeNetworkModel
```

- **State:** the starting state, and the state after any sequence of added edges, equals
  a from-scratch recomputation with `MatchingIndex` and the operations of
  `binary_update`. This holds for empty and seeded networks, zero distances, a non-default
  offset and η = 0.
- **Sampling:** given the same uniform random numbers, each sampled edge is the one
  inverse-CDF sampling picks from the toolbox's weights. With the kernel's own RNG, a
  chi-square test against P / ΣP passes.
- **Edges:** they are always new and off-diagonal, and networks can be grown to
  completion.
- **Overflow:** with γ = −20, weights are clamped as in the toolbox and counted.
- **Distribution (slow test):** 300 networks grown by `GenerativeNetworkModel` and 300 by
  the kernel, on the default distance matrix with 400 edges, have the same distributions
  of max degree, degree spread, mean clustering, mean edge length and isolated nodes
  (two-sample KS).
- **Integration:** `perform_sweep(backend="flash")` returns the same experiments,
  evaluation keys and shapes as `backend="torch"`. The tests also cover seeds,
  `binary_updates_per_iteration`, configurations split across launches, reproducibility
  under `torch.manual_seed`, `"auto"` fallback and the error messages.

Without a GPU, `TRITON_INTERPRET=1 pytest tests/test_flash_*.py` runs everything except the
slow test in Triton's CPU interpreter.

Unrelated to this PR: on master, the four tests in `tests/test_experiment_saving.py` fail
with `IndexError: list index out of range`, with or without this change.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
