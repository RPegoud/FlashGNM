# Flash backend (GPU)

`gnm.fast` is an optional backend for **binary** generative network models. It simulates
the same model as [`GenerativeNetworkModel`](../api-reference/model.md), samples from the same
distribution, and is typically 20× to 10,000× faster per simulation on a GPU, with far less
memory. It is switched on with one argument:

```python
from gnm.fitting import perform_sweep

experiments = perform_sweep(
    sweep_config,
    binary_evaluations=binary_evaluations,
    real_binary_matrices=real_binary_matrices,
    backend="flash",   # or "auto": flash where supported, torch elsewhere
)
```

`perform_run` takes the same `backend` argument. The default, `backend="torch"`, is
unchanged.

## What it supports

| | Flash backend |
|---|---|
| Generative rule | `MatchingIndex(divisor="mean")` |
| Distance and preference relationships | `"powerlaw"` |
| Heterochronicity | no heterochronous matrix (any `lambdah` with `"powerlaw"`; `lambdah = 0` with `"exponential"`) |
| Seed networks | yes, shared `[n, n]` or per simulation `[S, n, n]` |
| Distance matrix | one `[n, n]` matrix per run (or none: uniform distances) |
| `prob_offset`, `binary_updates_per_iteration` | any |
| Weighted models | no |
| Sweep method | grid (`method="bayesian"` uses the torch backend) |
| Hardware | a CUDA GPU (Triton ships with torch on Linux) |

`backend="flash"` raises a clear error for anything else; `backend="auto"` runs those
configurations with the torch backend. `gnm.fast.flash_unsupported_reason(run_config)`
says why a configuration is not supported.

## What differs in the results

The experiments have the same structure as with the torch backend (same order, same
evaluation keys and shapes, `experiment.model.adjacency_matrix` holds the final networks),
with two differences:

- **`run_history.adjacency_snapshots` is `None`.** The torch backend stores the whole
  network after every step: `num_steps × num_simulations × n × n` floats, 40 GB for
  n = 1000, 100 simulations and 1,000 steps. `run_history.added_edges` holds the same
  information, and `gnm.fast.adjacency_snapshots_from_edges` rebuilds the snapshots when
  they fit in memory.
- **`clamp_count`** is nonzero exactly when the torch backend's is (some wiring weight
  overflowed and was clamped, see the warning printed by `perform_sweep`), but counts
  differently: steps at which a newly computed weight was clamped, maximised over
  simulations.

Random draws come from Triton's generator, seeded from torch's, so results are
reproducible under `torch.manual_seed` but are not the same networks as the torch
backend's for the same seed. They are draws from the same distribution.

## How it works

At each step the torch backend recomputes the whole matching-index matrix,
$K = A^T A / \text{mean degree}$, which costs $O(n^3)$ for the matrix product, then the
$n \times n$ wiring probabilities, and samples one entry. The flash backend does the same
step in $O(n)$:

1. **Two-stage sampling.** Keep the row sums $R_i = \sum_j P_{ij}$. Pick a row $i$ with
   probability $R_i / \sum_k R_k$, then a column $j$ with probability $P_{ij} / R_i$. The
   pair $(i, j)$ then has probability $P_{ij} / \sum P$, as with `torch.multinomial` over
   the flattened matrix.
2. **Incremental update.** Adding the edge $\{i, j\}$ changes only rows and columns $i$ and
   $j$ of $A$, of the shared-neighbour counts $C = A^T A$, and of the degrees. So only rows
   $i$ and $j$ of $P$ change (P is symmetric). They are recomputed before and after the
   edge, and $R$ is updated by the differences, in float64.
3. **One fused kernel.** One GPU program runs one simulation for all of its steps, so
   there is one launch per batch instead of several per step. $P$ is never stored: the
   state is $A$ (bool) and $C$ (int16), 3 bytes per node pair. Each simulation in a batch
   has its own $(\eta, \gamma)$ and seed network, so a whole parameter grid fills the GPU.

## Performance

Measured on an NVIDIA H100 (torch 2.8, Triton 3.4), one $(\eta, \gamma)$ per simulation,
T = the number of edges of a Lausanne connectome at that resolution. Simulation only, without
evaluation. Toolbox times were extrapolated from its first steps (`~`).

| n | simulations per batch | steps | toolbox (sims/s) | flash (sims/s) |
|---:|---:|---:|---:|---:|
| 68 | 1,024 | 610 | ~336 | 6,607 |
| 219 | 1,024 | 2,639 | ~7.3 | 5,811 |
| 1000 | 64 | 10,785 | ~0.1 | 671 |
| 1000 | 1,024 | 10,785 | ~0.1 | 1,854 |

At n = 1000 and 1,024 simulations the toolbox needs ~50 GB of GPU memory and the flash
backend 3.2 GB. `scripts/benchmark_flash.py` measures both on your GPU.

With the flash backend, evaluation (the KS criteria, especially betweenness) usually costs
more time than simulation.

## Checking it on your machine

The tests compare the kernel with the toolbox's own maths:

```bash
pytest tests/test_flash_kernel.py tests/test_flash_sweep.py -v       # ~1 min on a GPU
GNM_RUN_SLOW=1 pytest tests/test_flash_kernel.py -v -k toolbox        # vs GenerativeNetworkModel
```

- the starting state, and the state after any sequence of added edges, equals a
  from-scratch recomputation with `MatchingIndex` (exact for $A$, $C$ and the degrees;
  $10^{-5}$ relative for the row sums);
- given the same uniform random numbers, every sampled edge is the one inverse-CDF
  sampling picks from the toolbox's wiring weights;
- the sampled edges follow $P / \sum P$ (chi-square), edges are always new, and networks
  can be grown to completion;
- overflowing weights are clamped as in the toolbox;
- networks grown by the kernel and by `GenerativeNetworkModel` have the same distributions of
  degree, clustering and edge length (two-sample KS; the slow test).

Without a GPU, `TRITON_INTERPRET=1 pytest tests/test_flash_*.py` runs the same tests in
Triton's CPU interpreter, slowly; the slow test is skipped there.
