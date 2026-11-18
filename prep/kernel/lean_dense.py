"""Lean dense binary GNM: the reference spec and the credible O(n^2) baseline.

Same maths as the toolbox's `binary_update` (matching index, 'mean' divisor, powerlaw
rules, lambdah = 0), without the per-step O(n^2) validation, and sampling by
cumsum + searchsorted instead of torch.multinomial (about 12x faster on CPU at n = 1000).
Batches over a different (eta, gamma) per simulation. Device follows `dist`.
"""

import torch

OFFSET = 1e-6  # toolbox default prob_offset


class LeanDense:
    def __init__(self, dist, eta, gamma):
        B, n = eta.shape[0], dist.shape[0]
        dev = dist.device
        self.B, self.n = B, n
        self.A = torch.zeros(B, n, n, device=dev)
        # off-diagonal distances only matter; diagonal is masked, so avoid 0^eta = inf
        d = dist.clone()
        d.fill_diagonal_(1.0)
        self.D = d.unsqueeze(0).pow(eta.to(dev).view(B, 1, 1))
        self.gamma = gamma.to(dev).view(B, 1, 1)
        self.notdiag = ~torch.eye(n, dtype=torch.bool, device=dev)
        self.b = torch.arange(B, device=dev)
        # A[idx] = 1 with a Python scalar does a blocking host -> device copy each time
        self.one = torch.ones((), device=dev)

    def probs(self):
        """Unnormalised wiring probabilities P, shape [B, n, n]."""
        A = self.A
        deg = A.sum(-1)
        denom = (deg.unsqueeze(-2) + deg.unsqueeze(-1) - 2 * A) * 0.5
        denom.masked_fill_(denom == 0, 1.0)
        K = (A @ A).div_(denom)  # A symmetric, so A^T A = A A
        K.masked_fill_(K == 0, OFFSET)
        return (
            K.pow_(self.gamma).mul_(self.D).add_(OFFSET).mul_(1 - A).mul_(self.notdiag)
        )

    def step(self):
        P = self.probs().view(self.B, -1)
        # float64 cdf: in float32, entries below ~1e-7 of the total would never be drawn at large n
        cdf = P.cumsum(-1, dtype=torch.float64)
        u = torch.rand(self.B, 1, dtype=torch.float64, device=P.device) * cdf[:, -1:]
        idx = torch.searchsorted(cdf, u).squeeze(-1).clamp_(max=self.n * self.n - 1)
        a, c = idx // self.n, idx % self.n
        self.A[self.b, a, c] = self.one
        self.A[self.b, c, a] = self.one
        return a, c
