r"""Flash backend: a fused GPU kernel for binary GNMs with O(n) work per step.

For the matching index (mean divisor) with power-law distance and preference terms, the
kernel samples from the same distribution as
[`GenerativeNetworkModel`][gnm.GenerativeNetworkModel], with far less work and memory per
step, and runs many parameter combinations in one launch. It needs a CUDA GPU and Triton
(installed with torch on Linux).

Most users only need the `backend` argument of the sweep functions:

    >>> from gnm.fitting import perform_sweep
    >>> experiments = perform_sweep(sweep_config, binary_evaluations=..., real_binary_matrices=...,
    ...                             backend="flash")   # or "auto": flash where supported

Lower-level pieces:

- [`FlashGNM`][gnm.fast.FlashGNM]: a batch of simulations, each with its own (eta, gamma).
- [`run_flash`][gnm.fast.run_flash]: many [`RunConfig`][gnm.fitting.RunConfig]s, packed
  into large launches.
- [`flash_unsupported_reason`][gnm.fast.flash_unsupported_reason]: whether a run
  configuration can use the backend.
"""

from .kernel import FlashGNM, flash_available
from .runner import adjacency_snapshots_from_edges, flash_unsupported_reason, run_flash

__all__ = [
    "FlashGNM",
    "flash_available",
    "flash_unsupported_reason",
    "run_flash",
    "adjacency_snapshots_from_edges",
]
