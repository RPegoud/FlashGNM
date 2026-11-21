"""Dense fp64 reference for the Oldham et al. rule forms (SPEC §4.1, H9).

Written from the MATLAB of Oldham, Fornito & Ball 2025 ("Coming up short"),
https://github.com/StuartJO/ComingUpShort @ 800bea9, code/GNM/GNM.m and matching.m:

    Df  = exp(eta * D)            (cost 'exp')     or D .^ eta    (cost 'power')
    Fk  = (K + epsilon) .^ gamma                   epsilon = 1e-6 (fitGNM.m default)
    K   = 2 * shared / (k_i + k_j - 2 A_ij),       0 / 0 -> 0
    add:  P = Df / max(Df over non-edges) + alpha * Fk / max(Fk over non-edges)
    mult: P = Df / max(...) * Fk / max(...)        (the maxima only rescale P)
    P = 0 on edges and the diagonal; no other offset. Maxima over current non-edges,
    recomputed after every edge. Empty seed; one edge per step; T = target edge count.

Their 2022 code (StuartJO/GenerativeNetworkModel, gen_model_add_normmax.m) uses
epsilon = 0 and sets Inf (0 ** negative gamma) to 0: pass epsilon=0 for that convention.

Sampling follows the kernel contract (prep/kernel/test_kernel.py): a row by inverse CDF of
the row sums, then a column of that row. P is symmetric, so pair {i, j} is drawn with
probability 2 P_ij / sum(P), the same as MATLAB's draw over the upper triangle.

Every step recomputes everything from A: O(n^2) per step, for tests only.
"""

from __future__ import annotations

import torch

EPSILON = 1e-6


class DenseRule:
    def __init__(self, dist, eta, gamma, alpha=None, form: str = "add", cost: str = "exp",
                 epsilon: float = EPSILON):
        assert form in ("add", "mult") and cost in ("exp", "power")
        B, n = eta.shape[0], dist.shape[0]
        dev = dist.device
        self.B, self.n, self.form, self.cost, self.eps = B, n, form, cost, epsilon
        self.D = dist.to(dev, torch.float64)
        self.eta = eta.to(dev, torch.float64).view(B, 1, 1)
        self.gamma = gamma.to(dev, torch.float64).view(B, 1, 1)
        alpha = torch.ones(B) if alpha is None else alpha
        self.alpha = alpha.to(dev, torch.float64).view(B, 1, 1)
        self.A = torch.zeros(B, n, n, dtype=torch.float64, device=dev)
        self.b = torch.arange(B, device=dev)
        self.eye = torch.eye(n, dtype=torch.bool, device=dev)
        self.refresh()

    # -- full recompute
    def matching(self):
        A = self.A
        C = A @ A
        deg = A.sum(-1)
        den = deg.unsqueeze(-1) + deg.unsqueeze(-2) - 2 * A
        K = torch.where(den > 0, 2 * C / den.clamp_min(1), torch.zeros_like(C))
        return K, C, deg

    def refresh(self):
        K, C, deg = self.matching()
        self.K, self.C, self.deg = K, C, deg
        nonedge = (self.A == 0) & ~self.eye
        self.nonedge = nonedge
        if self.cost == "exp":
            Dexp = self.eta * self.D  # log of Df
        else:
            D1 = torch.where(self.eye, torch.ones_like(self.D), self.D)
            Dexp = self.eta * torch.log(D1)
        if self.eps > 0:
            Kexp = self.gamma * torch.log(K + self.eps)
        else:  # 2022: 0 ** gamma -> Inf (gamma < 0) set to 0, i.e. excluded
            Kexp = torch.where(K > 0, self.gamma * torch.log(K.clamp_min(1e-300)),
                               torch.where(self.gamma > 0, -torch.inf, 0.0 * self.gamma))
            Kexp = torch.where((K == 0) & (self.gamma < 0), -torch.inf, Kexp)
        neg = torch.full_like(Dexp, -torch.inf)
        # log of the maxima over non-edges, per simulation
        self.logMa = torch.where(nonedge, Dexp, neg).amax((-1, -2), keepdim=True)
        self.logMb = torch.where(nonedge, Kexp, neg).amax((-1, -2), keepdim=True)
        a = torch.exp(Dexp - self.logMa)
        # no finite topology term on any non-edge (only with epsilon = 0): MATLAB's
        # NaN -> 0 (2022) / maxFK == 0 guard (2025) make the term 0
        bk = torch.where(torch.isfinite(self.logMb), torch.exp(Kexp - self.logMb),
                         torch.zeros_like(Kexp))
        if self.form == "add":
            P = a + self.alpha * bk
        else:
            P = a * bk
        self.Pa = torch.where(nonedge, a, 0.0)
        self.Pb = torch.where(nonedge, bk, 0.0)
        self.Dexp, self.Kexp = Dexp, Kexp
        self.P = torch.where(nonedge, P, 0.0)
        self.r = self.P.sum(-1)

    @staticmethod
    def sample_cdf(w, u):
        cdf = w.cumsum(-1, dtype=torch.float64)
        t = u.double().unsqueeze(-1) * cdf[..., -1:]
        return (cdf <= t).sum(-1)

    def step(self, u=None, edges=None):
        if edges is not None:
            e = edges.to(self.A.device).long()
            i, j = e[:, 0], e[:, 1]
        else:
            if u is None:
                u = torch.rand(self.B, 2, device=self.A.device)
            i = self.sample_cdf(self.r, u[:, 0])
            j = self.sample_cdf(self.P[self.b, i], u[:, 1])
        self.A[self.b, i, j] = 1.0
        self.A[self.b, j, i] = 1.0
        self.refresh()
        return torch.stack((i, j), -1)

    def state(self) -> dict:
        """Dense state; maxima as logs. 'ra'/'rb' are row sums of the two normalised
        terms (additive form), 'r' of P."""
        return {"A": self.A, "C": self.C, "deg": self.deg, "r": self.r, "P": self.P,
                "ra": self.Pa.sum(-1), "rb": self.Pb.sum(-1),
                "logMa": self.logMa.view(-1), "logMb": self.logMb.view(-1)}
