"""Compare the GNM toolbox's binary update (full recompute of the matching index
every step) with an incremental update that maintains the shared-neighbour count
matrix C = A^T A and patches it in O(n) per added edge.

Also batches over *different* (eta, gamma) per simulation, which the toolbox does not
(it batches only repeats of one parameter setting; sweeps loop in Python).
"""

import time

import torch

from gnm import BinaryGenerativeParameters, GenerativeNetworkModel
from gnm.generative_rules import MatchingIndex

torch.manual_seed(0)
torch.set_num_threads(torch.get_num_threads())
OFFSET = 1e-6


def make_dist(n, zero_diag=False):
    g = torch.Generator().manual_seed(n)
    xyz = torch.rand(n, 3, generator=g) * 100
    d = torch.cdist(xyz, xyz)
    d = 0.5 * (d + d.T)
    # toolbox requires a zero diagonal; ours masks the diagonal, so use 1 to avoid 0^eta
    d.fill_diagonal_(0.0 if zero_diag else 1.0)
    return d


# ---------------------------------------------------------------- timing
def time_steps(step, steps, device, warmup=3):
    """Seconds per call of `step`. CUDA runs asynchronously, so sync before reading the clock."""
    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t = time.perf_counter()
    for _ in range(steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - t) / steps


# ---------------------------------------------------------------- toolbox
def toolbox_step_time(n, B, steps, device, eta=-2.0, gamma=0.3):
    dist = make_dist(n, zero_diag=True)
    params = BinaryGenerativeParameters(
        eta=eta,
        gamma=gamma,
        lambdah=0.0,
        distance_relationship_type="powerlaw",
        preferential_relationship_type="powerlaw",
        heterochronicity_relationship_type="powerlaw",
        generative_rule=MatchingIndex(divisor="mean"),
        num_iterations=steps,
    )
    model = GenerativeNetworkModel(
        binary_parameters=params,
        num_simulations=B,
        num_nodes=n,
        distance_matrix=dist,
        device=device,
    )
    return time_steps(model.binary_update, steps, device)


# ---------------------------------------------------------------- incremental
class Incremental:
    """Batched binary GNM with matching index ('mean' divisor), powerlaw rules.
    eta, gamma: tensors of shape [B] (one parameter setting per simulation).
    Device follows `dist`."""

    def __init__(self, dist, eta, gamma):
        B, n = eta.shape[0], dist.shape[0]
        dev = dist.device
        self.B, self.n = B, n
        self.A = torch.zeros(B, n, n, device=dev)
        self.C = torch.zeros(B, n, n, device=dev)  # shared-neighbour counts, A^T A
        self.deg = torch.zeros(B, n, device=dev)
        self.D = dist.unsqueeze(0).pow(eta.to(dev).view(B, 1, 1))  # distance factor
        self.gamma = gamma.to(dev).view(B, 1, 1)
        self.diag = torch.eye(n, dtype=torch.bool, device=dev)
        self.b = torch.arange(B, device=dev)

    def affinity(self):
        A, deg = self.A, self.deg
        denom = (deg.unsqueeze(-2) + deg.unsqueeze(-1) - 2 * A) / 2
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        K = self.C / denom
        K = K.masked_fill(self.diag, 0.0)
        return K

    def step(self):
        K = self.affinity()
        K = torch.where(K == 0, K + OFFSET, K)
        P = self.D * K.pow(self.gamma) + OFFSET
        P = P * (1 - self.A)
        P = P.masked_fill(self.diag, 0.0)
        idx = torch.multinomial(P.view(self.B, -1), 1).squeeze(-1)
        a, c = idx // self.n, idx % self.n
        self.add_edges(a, c)

    def add_edges(self, a, c):
        b, A, C = self.b, self.A, self.C
        Aa, Ac = A[b, a].clone(), A[b, c].clone()  # rows before the update, [B, n]
        # C' = C + (A_a e_c^T + A_c e_a^T) + transpose + e_a e_a^T + e_c e_c^T
        C[b, :, c] += Aa
        C[b, :, a] += Ac
        C[b, c, :] += Aa
        C[b, a, :] += Ac
        C[b, a, a] += 1
        C[b, c, c] += 1
        A[b, a, c] = 1
        A[b, c, a] = 1
        self.deg[b, a] += 1
        self.deg[b, c] += 1


def incremental_step_time(n, B, steps, device):
    dist = make_dist(n).to(device)
    eta = torch.empty(B).uniform_(-3, 0)
    gamma = torch.empty(B).uniform_(0.1, 0.5)
    m = Incremental(dist, eta, gamma)
    return time_steps(m.step, steps, device)


# ---------------------------------------------------------------- correctness
def check_correctness(n=60, B=4, steps=150):
    dist = make_dist(n)
    m = Incremental(dist, torch.full((B,), -2.0), torch.full((B,), 0.3))
    for _ in range(steps):
        m.step()
    C_ref = m.A.transpose(-2, -1) @ m.A
    K_ref = MatchingIndex(divisor="mean")(m.A)
    err_C = (m.C - C_ref).abs().max().item()
    err_K = (m.affinity() - K_ref).abs().max().item()
    return err_C, err_K


if __name__ == "__main__":
    device = torch.device("cpu")
    print("threads:", torch.get_num_threads())
    print("correctness (max abs err C, K):", check_correctness())
    print(
        f"{'n':>6} {'B':>4} {'toolbox ms/step':>16} {'incremental ms/step':>20} {'speedup':>8}"
    )
    for n, B, steps in [
        (68, 1, 200),
        (68, 64, 100),
        (200, 1, 100),
        (200, 16, 30),
        (500, 1, 30),
        (1000, 1, 10),
        (1000, 4, 5),
    ]:
        t_tb = toolbox_step_time(n, B, steps, device=device)
        t_inc = incremental_step_time(n, B, steps, device=device)
        print(
            f"{n:>6} {B:>4} {t_tb * 1e3:>16.2f} {t_inc * 1e3:>20.2f} {t_tb / t_inc:>8.1f}x",
            flush=True,
        )
