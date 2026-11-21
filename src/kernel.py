# type: ignore
"""Fused Triton kernel for the binary GNM (matching index, power-law rules), v2.

One program per simulation, persistent over T steps. Per step: pick edge (i, j) (sampled
in two stages, rows by R then a column by P[i], or given for teacher forcing), update A,
C = A^T A and the degrees in O(n), recompute rows i and j of P, and patch the fp64 row
sums R by deltas.

State per simulation (contract: prep/kernel/test_kernel.py):
    A    [n, n] bool       adjacency
    C    [n, n] int16      shared-neighbour counts (<= max degree < n)
    deg  [n]    int16      degrees
    R    [n]    float64    row sums of the wiring weights P
P is not stored: rows i and j, old and new, are recomputed from C, A, deg and D each step.

Changes from the prototype (prep/kernel/kernel.py), SPEC §4.0: P not stored (state 9 ->
3 bytes per pair, no P row/column stores); C int16; one launch path for step() and run();
teacher forcing in run() too.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

OFFSET = 1e-6  # toolbox default prob_offset


# ------------------------------------------------------------------ device helpers
@triton.jit
def _row_P(C_row, A_row, Dist_row, Deg, deg_x, x, Eta, Gamma, cols, mask, OFFSET):
    """Row x of P from row x of C and A, all degrees and row x of the distances:
    P = D^eta * K^gamma + OFFSET with K = C / ((deg_x + deg - 2 A) / 2) and zero K
    replaced by OFFSET; zero on edges, the diagonal and padded lanes."""
    denom = (deg_x + Deg - 2 * A_row) / 2
    # a zero denominator only occurs on an edge lane, where P is zeroed anyway
    denom = tl.where(denom == 0, 1, denom)
    K_row = C_row / denom
    K_row = tl.where(K_row == 0, OFFSET, K_row)
    # x^y = exp(y * log(x))
    D_eta = tl.exp(Eta * tl.log(Dist_row))
    K_gamma = tl.exp(Gamma * tl.log(K_row))
    P_row = D_eta * K_gamma + OFFSET
    return tl.where(mask & (A_row == 0) & (cols != x), P_row, 0)


@triton.jit
def _sample_cdf(W, U):
    """Index of the first entry whose inclusive cdf exceeds U * sum(W) (inverse CDF, fp64)."""
    cdf = tl.cumsum(W.to(tl.float64), axis=-1)
    t = U.to(tl.float64) * tl.max(cdf)
    return tl.sum(cdf <= t, axis=-1)


@triton.jit
def _uniform(U_ptr, t, k, SEED, pid, SHARED_UNIFORM: tl.constexpr):
    """Draw k (0: row, 1: column) of step t: from U if given, else the kernel's RNG."""
    if SHARED_UNIFORM:
        u = tl.load(U_ptr + 2 * t + k)
    else:
        u = tl.rand(SEED + t, 2 * pid + k)
    return u


@triton.jit
def _load_row(A_ptr, C_ptr, Dist_ptr, Deg, x, cols, mask, N):
    """Row x of A, C and the distances (padded with 1 so log stays finite), and deg[x]."""
    offs = x * N + cols
    A_row = tl.load(A_ptr + offs, mask=mask, other=0)
    C_row = tl.load(C_ptr + offs, mask=mask, other=0)
    Dist_row = tl.load(Dist_ptr + offs, mask=mask, other=1.0)
    deg_x = tl.sum(tl.where(cols == x, Deg, 0))
    return A_row, C_row, Dist_row, deg_x


@triton.jit
def _store_sym(ptr, x, row, old_row, cols, mask, N):
    """Write row x and, where it changed, column x (strided: one cache line per lane)."""
    tl.store(ptr + x * N + cols, value=row, mask=mask)
    tl.store(ptr + x + cols * N, value=row, mask=mask & (old_row != row))


# ------------------------------------------------------------------ kernel
@triton.jit(do_not_specialize=["T", "SEED"])
def binary_gnm_kernel(
    A_ptr,  # [B, N, N] bool
    C_ptr,  # [B, N, N] int16
    R_ptr,  # [B, N] float64
    Deg_ptr,  # [B, N] int16
    Eta_ptr,  # [B] float32
    Gamma_ptr,  # [B] float32
    Dist_ptr,  # [N, N] float32, shared by the batch
    U_ptr,  # [B, T, 2] float32 if SHARED_UNIFORM
    Edges_ptr,  # [B, T, 2] int32 if EDGES_GIVEN
    Out_ptr,  # [B, T, 2] int32: the edges added
    N,
    T,
    OFFSET,
    SEED,
    BLOCK_N: tl.constexpr,  # next_power_of_2(N): a whole row per program
    EDGES_GIVEN: tl.constexpr,  # teacher forcing: apply Edges instead of sampling
    SHARED_UNIFORM: tl.constexpr,  # sample with U instead of tl.rand
):
    pid = tl.program_id(0)
    sim = pid.to(tl.int64) * N * N  # int64: B * N^2 overflows int32
    A_ptr += sim
    C_ptr += sim
    R_ptr += pid * N
    Deg_ptr += pid * N
    U_ptr += pid * 2 * T
    Edges_ptr += pid * 2 * T
    Out_ptr += pid * 2 * T

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    Eta = tl.load(Eta_ptr + pid)
    Gamma = tl.load(Gamma_ptr + pid)
    R = tl.load(R_ptr + cols, mask=mask, other=0.0)
    Deg = tl.load(Deg_ptr + cols, mask=mask, other=0)

    for t in tl.range(T):
        # -- 1. row i, and the old rows of i
        if EDGES_GIVEN:
            i = tl.load(Edges_ptr + 2 * t).to(tl.int32)
        else:
            i = _sample_cdf(R, _uniform(U_ptr, t, 0, SEED, pid, SHARED_UNIFORM))
        old_Ai, old_Ci, Dist_i, deg_i = _load_row(A_ptr, C_ptr, Dist_ptr, Deg, i, cols, mask, N)
        old_Pi = _row_P(old_Ci, old_Ai, Dist_i, Deg, deg_i, i, Eta, Gamma, cols, mask, OFFSET)

        # -- 2. column j, and the old rows of j
        if EDGES_GIVEN:
            j = tl.load(Edges_ptr + 2 * t + 1).to(tl.int32)
        else:
            j = _sample_cdf(old_Pi, _uniform(U_ptr, t, 1, SEED, pid, SHARED_UNIFORM))
        old_Aj, old_Cj, Dist_j, deg_j = _load_row(A_ptr, C_ptr, Dist_ptr, Deg, j, cols, mask, N)
        old_Pj = _row_P(old_Cj, old_Aj, Dist_j, Deg, deg_j, j, Eta, Gamma, cols, mask, OFFSET)

        # -- 3. add edge (i, j): degrees, A rows (registers), C rows and columns
        Deg = tl.where((cols == i) | (cols == j), Deg + 1, Deg)
        deg_i += 1
        deg_j += 1
        Ai = old_Ai + tl.where(cols == j, 1, 0)
        Aj = old_Aj + tl.where(cols == i, 1, 0)
        # C' = C + A_i e_j^T + A_j e_i^T + transpose + e_i e_i^T + e_j e_j^T (old A rows)
        Ci = old_Ci + old_Aj + tl.where(cols == i, 1, 0)
        Cj = old_Cj + old_Ai + tl.where(cols == j, 1, 0)
        _store_sym(C_ptr, i, Ci, old_Ci, cols, mask, N)
        _store_sym(C_ptr, j, Cj, old_Cj, cols, mask, N)

        # -- 4. new rows of P (registers only), R patched by deltas (fp64); rows i and j
        # summed exactly
        Pi = _row_P(Ci, Ai, Dist_i, Deg, deg_i, i, Eta, Gamma, cols, mask, OFFSET)
        Pj = _row_P(Cj, Aj, Dist_j, Deg, deg_j, j, Eta, Gamma, cols, mask, OFFSET)
        R += (Pi.to(tl.float64) - old_Pi.to(tl.float64)) + (
            Pj.to(tl.float64) - old_Pj.to(tl.float64)
        )
        sum_i = tl.sum(Pi.to(tl.float64))
        sum_j = tl.sum(Pj.to(tl.float64))
        R = tl.where(cols == i, sum_i, tl.where(cols == j, sum_j, R))

        tl.store(Out_ptr + 2 * t, value=i)
        tl.store(Out_ptr + 2 * t + 1, value=j)

        # -- 5. A[i, j] = A[j, i] = 1. These stores don't depend on other threads' loads of
        # the old A rows, so they must come after a reduction over those rows (the sums
        # above), else they can land first (SPEC §4, concurrency hazard).
        tl.store(A_ptr + i * N + j, value=1)
        tl.store(A_ptr + j * N + i, value=1)
        tl.debug_barrier()  # all writes of step t visible before step t + 1 loads

    tl.store(Deg_ptr + cols, value=Deg, mask=mask)
    tl.store(R_ptr + cols, value=R, mask=mask)


# ------------------------------------------------------------------ host
def default_num_warps(n: int) -> int:
    """One warp per 128 columns (4 per thread), clamped to [2, 16]. Best or within 6% of
    best in the 2026-11-02 H100 sweep at n = 68, 200 (2), 1000 (8) and 2000 (16), B = 1 to
    1024. Fewer warps spill: at n = 2000, 4 warps spill 58 registers and 2 warps 550."""
    return max(2, min(16, triton.next_power_of_2(n) // 128))


class TritonGNM:
    """Batch of B binary GNM simulations, one (eta, gamma) each.

    dist [n, n] float32, symmetric, diagonal 1 (masked; avoids 0 ** eta)
    eta, gamma [B] float32
    """

    def __init__(self, dist, eta, gamma, num_warps: int | None = None):
        device = dist.device
        B, N = eta.shape[0], dist.shape[0]
        self.device, self.B, self.N = device, B, N
        self.num_warps = num_warps or default_num_warps(N)
        self.base_seed = torch.randint(0, 2**30, (1,)).item()
        self.t = 0

        self.D = dist.contiguous()
        self.eta = eta.to(device, torch.float32).contiguous()
        self.gamma = gamma.to(device, torch.float32).contiguous()
        self.A = torch.zeros((B, N, N), device=device, dtype=torch.bool)
        self.C = torch.zeros((B, N, N), device=device, dtype=torch.int16)
        self.deg = torch.zeros((B, N), device=device, dtype=torch.int16)
        self.R = self._init_R()

    # -- initial state
    def _sim_P(self, b: int) -> torch.Tensor:
        """Full recompute of P for simulation b."""
        A, deg = self.A[b], self.deg[b]
        denom = (deg.unsqueeze(-2) + deg.unsqueeze(-1) - 2 * A) / 2
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        K = self.C[b] / denom
        K = torch.where(K == 0, K + OFFSET, K)
        P = (self.D.pow(self.eta[b]) * K.pow(self.gamma[b]) + OFFSET) * ~A
        return P.fill_diagonal_(0.0)

    def _init_R(self):
        """fp64 row sums of P, one simulation at a time (B x n x n temporaries OOM at
        n = 2000, B = 1024); P itself is discarded. The per-simulation Python loop is most
        of the setup cost at small n (results/26-11-21_14-41-53_sim_triton_proto)."""
        R = torch.empty(self.B, self.N, device=self.device, dtype=torch.float64)
        for b in range(self.B):
            R[b] = self._sim_P(b).sum(-1, dtype=torch.float64)
        return R

    # -- stepping
    def _launch(self, T: int, u=None, edges=None) -> torch.Tensor:
        """Run T steps in one launch; returns the edges added, [B, T, 2] int32."""
        out = torch.empty((self.B, T, 2), dtype=torch.int32, device=self.device)
        if u is not None:
            u = u.to(self.device, torch.float32).reshape(self.B, T, 2).contiguous()
        if edges is not None:
            edges = edges.to(self.device, torch.int32).reshape(self.B, T, 2).contiguous()
        binary_gnm_kernel[(self.B,)](
            A_ptr=self.A,
            C_ptr=self.C,
            R_ptr=self.R,
            Deg_ptr=self.deg,
            Eta_ptr=self.eta,
            Gamma_ptr=self.gamma,
            Dist_ptr=self.D,
            U_ptr=out if u is None else u,  # placeholder when unused
            Edges_ptr=out if edges is None else edges,
            Out_ptr=out,
            N=self.N,
            T=T,
            OFFSET=OFFSET,
            SEED=self.base_seed + self.t,
            BLOCK_N=triton.next_power_of_2(self.N),
            EDGES_GIVEN=edges is not None,
            SHARED_UNIFORM=u is not None,
            num_warps=self.num_warps,
        )
        self.t += T
        return out

    def step(self, u=None, edges=None) -> torch.Tensor:
        """One step. u [B, 2] uniforms, or edges [B, 2] to teacher-force. Returns [B, 2]."""
        return self._launch(1, u=u, edges=edges)[:, 0]

    def run(self, T: int, u=None, edges=None) -> torch.Tensor:
        """T steps in one launch. u [B, T, 2], or edges [B, T, 2]. Returns [B, T, 2]."""
        return self._launch(T, u=u, edges=edges)

    def state(self) -> dict[str, torch.Tensor]:
        return {"A": self.A, "C": self.C, "deg": self.deg, "r": self.R}
