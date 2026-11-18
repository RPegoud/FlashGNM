import torch
from bench_gnm import OFFSET, make_dist, time_steps


class RowSampled:
    def __init__(self, dist, eta, gamma, resync_every=500):
        B, n = eta.shape[0], dist.shape[0]
        dev = dist.device
        self.B, self.n = B, n
        self.A = torch.zeros(B, n, n, device=dev)
        self.C = torch.zeros(B, n, n, device=dev)
        self.deg = torch.zeros(B, n, device=dev)
        self.D = dist.unsqueeze(0).pow(eta.to(dev).view(B, 1, 1))
        self.gamma = gamma.to(dev).view(B, 1)
        self.b = torch.arange(B, device=dev)
        self.ar = torch.arange(n, device=dev)
        # x[idx] = 1 with a Python scalar copies the scalar host -> device first, which is
        # not allowed during CUDA graph capture; on-device constants avoid that copy.
        self.zero = torch.zeros((), device=dev)
        self.one = torch.ones((), device=dev)
        self.P = self.full_P()
        self.r = self.P.sum(-1, dtype=torch.float64)
        self.resync_every, self.t = resync_every, 0

    # full recompute, used at init, for periodic resync, and for testing
    def full_P(self):
        A, deg = self.A, self.deg
        denom = (deg.unsqueeze(-2) + deg.unsqueeze(-1) - 2 * A) / 2
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        K = self.C / denom
        K = torch.where(K == 0, K + OFFSET, K)
        P = self.D * K.pow(self.gamma.unsqueeze(-1)) + OFFSET
        P = P * (1 - A)
        P[:, self.ar, self.ar] = 0.0
        return P

    def row_P(self, i):
        """Recompute row i (shape [B]) of P: O(n)."""
        b = self.b
        Ai, degi = self.A[b, i], self.deg[b, i].unsqueeze(-1)
        denom = (degi + self.deg - 2 * Ai) / 2
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        K = self.C[b, i] / denom
        K = torch.where(K == 0, K + OFFSET, K)
        row = self.D[b, i] * K.pow(self.gamma) + OFFSET
        row = row * (1 - Ai)
        row[b, i] = self.zero
        return row

    @staticmethod
    def sample(w):
        """One index per row of w, with probability proportional to w.

        argmax(w / E), E ~ Exp(1), is the trick torch.multinomial uses for a single
        sample, minus its input check, which reads a value back to the host and so
        can't be captured in a CUDA graph.
        """
        return (w / torch.empty_like(w).exponential_()).argmax(-1)

    @staticmethod
    def sample_cdf(w, u):
        """Inverse-CDF sample per row of w: the first i with cdf_i > u * cdf_{n-1}.

        This is the sampling spec shared with the Triton kernel (test_kernel.py):
        given the same uniforms u in [0, 1), both must pick the same index, up to ties
        where u * total lands within float rounding of a cdf boundary. Zero-weight
        entries can't be picked: their cdf equals the previous one.
        """
        cdf = w.cumsum(-1, dtype=torch.float64)
        t = u.double().unsqueeze(-1) * cdf[..., -1:]
        return (cdf <= t).sum(-1)

    def step(self, u=None, edges=None):
        """Add one edge per simulation and return it as [B, 2] (row, col).

        u:     [B, 2] uniforms in [0, 1): inverse-CDF sampling (row from u[:, 0], col
               from u[:, 1]), reproducible across implementations.
        edges: [B, 2] forced edges (teacher forcing): skip sampling, apply these.
        Neither: sample with torch's RNG (the graph-capturable production path).
        """
        if edges is not None:
            rows, cols = edges[:, 0], edges[:, 1]
        elif u is not None:
            rows = self.sample_cdf(self.r, u[:, 0])
            cols = self.sample_cdf(self.P[self.b, rows], u[:, 1])
        else:
            rows = self.sample(self.r)
            cols = self.sample(self.P[self.b, rows])
        self.add_edge(rows, cols)
        self.t += 1
        if self.resync_every and self.t % self.resync_every == 0:
            self.resync()
        return torch.stack((rows, cols), -1)

    def state(self):
        """Dense float views of the state, the layout tests compare against."""
        return {"A": self.A, "C": self.C, "deg": self.deg, "P": self.P, "r": self.r}

    def resync(self):
        """Recompute r from P to bound float drift. In place, so a captured graph sees it."""
        self.r.copy_(self.P.sum(-1, dtype=torch.float64))

    def add_edge(self, a, c):
        b, A, C, P = self.b, self.A, self.C, self.P
        Aa, Ac = A[b, a].clone(), A[b, c].clone()
        C[b, :, c] += Aa
        C[b, :, a] += Ac
        C[b, c, :] += Aa
        C[b, a, :] += Ac
        C[b, a, a] += 1
        C[b, c, c] += 1
        A[b, a, c] = self.one
        A[b, c, a] = self.one
        self.deg[b, a] += 1
        self.deg[b, c] += 1

        old_a, old_c = P[b, :, a].clone(), P[b, :, c].clone()
        new_a, new_c = self.row_P(a), self.row_P(c)
        P[b, a, :] = new_a
        P[b, :, a] = new_a
        P[b, c, :] = new_c
        P[b, :, c] = new_c
        # every row's sum changes only through its entries in columns a and c
        self.r += (P[b, :, a] - old_a + P[b, :, c] - old_c).double()
        self.r[b, a] = P[b, a].sum(-1, dtype=torch.float64)
        self.r[b, c] = P[b, c].sum(-1, dtype=torch.float64)


def graphed_steps(m, k, device, warmup=3):
    """Record k steps of m into one CUDA graph and return its replay function.

    Capture records kernels and the addresses they read and write; it doesn't run them.
    Replay re-runs exactly those kernels on whatever those addresses now hold, so all
    state must be updated in place and the Python control flow taken during capture is
    frozen in (hence resync_every=0 on m, and resyncing outside the graph).
    """
    assert not m.resync_every, "resync inside a graph would be frozen at capture time"
    # Warm up on a side stream so one-off setup (allocator, cuBLAS handles, ...) isn't captured.
    s = torch.cuda.Stream(device)
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        for _ in range(warmup):
            m.step()
    torch.cuda.current_stream(device).wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(k):
            m.step()
    return g.replay


def check(n=60, B=4, steps=300, device=None, graph_k=0):
    """graph_k > 0: run the steps as replays of a k-step CUDA graph (steps rounded down)."""
    dist = make_dist(n).to(device or "cpu")
    eta, gamma = torch.full((B,), -2.0), torch.full((B,), 0.3)
    if graph_k:
        m = RowSampled(dist, eta, gamma, resync_every=0)
        replay = graphed_steps(m, graph_k, device)
        for _ in range(steps // graph_k):
            replay()
    else:
        m = RowSampled(dist, eta, gamma)
        for _ in range(steps):
            m.step()
    P_ref = m.full_P()
    rel_P = ((m.P - P_ref).abs().max() / P_ref.abs().max()).item()
    rel_r = (
        (m.r - P_ref.sum(-1, dtype=torch.float64)).abs().max()
        / P_ref.sum(-1).abs().max()
    ).item()
    K_ok = torch.equal(m.C, m.A.transpose(-2, -1) @ m.A)
    return K_ok, rel_P, rel_r


def time_rowsampled(n, B, steps, device):
    dist = make_dist(n).to(device)
    m = RowSampled(
        dist, torch.empty(B).uniform_(-3, 0), torch.empty(B).uniform_(0.1, 0.5)
    )
    return time_steps(m.step, steps, device)


def time_rowsampled_graph(n, B, steps, device, k=50, resync_every=500):
    """Seconds per step with k steps per CUDA graph replay, resyncing r between replays."""
    assert resync_every % k == 0
    dist = make_dist(n).to(device)
    m = RowSampled(
        dist,
        torch.empty(B).uniform_(-3, 0),
        torch.empty(B).uniform_(0.1, 0.5),
        resync_every=0,
    )
    replay = graphed_steps(m, k, device)
    replays = 0

    def run():
        nonlocal replays
        replay()
        replays += 1
        if replays % (resync_every // k) == 0:
            m.resync()

    return time_steps(run, max(1, steps // k), device) / k
