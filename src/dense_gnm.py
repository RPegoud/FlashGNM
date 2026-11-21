"""Dense batched binary GNM in plain PyTorch: an independent reference with switchable
conventions and seed networks (for diagnosing the fitted gamma, see src/gamma_diagnostics.py).

Each step recomputes C = A @ A and the full wiring weights over the upper triangle, and
samples one non-edge per simulation with the Gumbel-max trick (exact). O(n^3) per step:
fine at n = 68, not meant for large n.

Matching index (both conventions): K_ij = 2 C_ij / (d_i + d_j - 2 A_ij), 0 when the
denominator is 0. This is the GNM toolbox's MatchingIndex(divisor="mean") and BCT's
matching index (shared neighbours over the summed degrees, each excluding the pair).

Conventions for zeros (the only difference between the two):
    toolbox  (GenerativeNetworkModels, model.binary_update):
             K == 0 -> 1e-6, then P = D^eta K^gamma + 1e-6
    bct      (BCT generative_model.m, fcn_matching, epsilon = 1e-5):
             P = D^eta (K + 1e-5)^gamma, no additive offset
Weights are formed in log space (no under/overflow at extreme eta, gamma).
"""

from __future__ import annotations

import math

import torch

CONVENTIONS = ("toolbox", "bct")


def log_weights(A_pairs, deg, C_pairs, iu, logD_pairs, eta, gamma, convention):
    """Log wiring weights [b, P] over upper-triangle pairs; -inf on existing edges.

    A_pairs, C_pairs [b, P]; deg [b, n]; iu [2, P]; logD_pairs [P]; eta, gamma [b, 1].
    """
    denom = deg[:, iu[0]] + deg[:, iu[1]] - 2 * A_pairs
    K = torch.where(denom > 0, 2 * C_pairs / denom.clamp_min(1), 0.0)
    if convention == "toolbox":
        K = torch.where(K == 0, 1e-6, K)
        logw = torch.logaddexp(eta * logD_pairs + gamma * K.log(),
                               torch.tensor(math.log(1e-6), device=K.device))
    elif convention == "bct":
        logw = eta * logD_pairs + gamma * (K + 1e-5).log()
    else:
        raise ValueError(convention)
    return logw.masked_fill(A_pairs > 0, -math.inf)


@torch.no_grad()
def simulate(D: torch.Tensor, eta: torch.Tensor, gamma: torch.Tensor, T: int,
             convention: str = "toolbox", seed: torch.Tensor | None = None,
             chunk: int = 8192, generator: torch.Generator | None = None) -> torch.Tensor:
    """B networks [B, n, n] bool with T edges each (seed edges included).

    D [n, n] distances (diagonal ignored); eta, gamma [B]; seed [n, n] bool or None.
    """
    dev = D.device
    n = D.shape[0]
    B = eta.shape[0]
    iu = torch.triu_indices(n, n, 1, device=dev)
    logD = D.double().clamp_min(1e-12).log().float()[iu[0], iu[1]]
    n_seed = 0 if seed is None else int(torch.triu(seed.bool(), 1).sum())
    steps = T - n_seed
    assert steps >= 0, "seed has more edges than the target"
    out = torch.empty(B, n, n, dtype=torch.bool, device=dev)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False  # C must be exact counts
    try:
        for lo in range(0, B, chunk):
            hi = min(B, lo + chunk)
            b = hi - lo
            A = torch.zeros(b, n, n, dtype=torch.bool, device=dev)
            if seed is not None:
                A |= seed.to(dev).bool()
            e = eta[lo:hi].to(dev, torch.float32).view(-1, 1)
            g = gamma[lo:hi].to(dev, torch.float32).view(-1, 1)
            rows = torch.arange(b, device=dev)
            for _ in range(steps):
                Af = A.float()
                C = torch.bmm(Af, Af)[:, iu[0], iu[1]]
                lw = log_weights(Af[:, iu[0], iu[1]], Af.sum(-1), C, iu, logD, e, g, convention)
                u = torch.rand(lw.shape, device=dev, generator=generator).clamp_(1e-20, 1.0)
                k = torch.argmax(lw - torch.log(-torch.log(u)), dim=-1)
                i, j = iu[0][k], iu[1][k]
                A[rows, i, j] = True
                A[rows, j, i] = True
            out[lo:hi] = A
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return out
