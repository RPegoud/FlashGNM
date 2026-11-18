# type: ignore
from __future__ import annotations

from functools import wraps

import torch
import triton
import triton.language as tl
from bench_gnm import OFFSET


def ensure_contiguous(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        new_args = [
            arg.contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args
        ]
        new_kwargs = {
            k: (v.contiguous() if isinstance(v, torch.Tensor) else v)
            for k, v in kwargs.items()
        }
        return func(*new_args, **new_kwargs)

    return wrapper


def default_num_warps(n: int) -> int:
    """One warp per 128 columns (4 per thread), clamped to [2, 16]. Best or within 6% of
    best in the 2026-11-02 H100 sweep (results/26-11-02_08-09-25) at n = 68, 200 (2),
    1000 (8) and 2000 (16), B = 1 to 1024. Fewer warps spill: at n = 2000, 4 warps spill
    58 registers and 2 warps 550."""
    return max(2, min(16, triton.next_power_of_2(n) // 128))


class TritonGNM:
    def __init__(
        self,
        dist: torch.Tensor,
        eta: torch.Tensor,
        gamma: torch.Tensor,
        num_warps: int | None = None,
    ) -> None:
        device = dist.device
        self.base_seed = torch.randint(0, 2**30, (1,)).item()
        B, N = eta.shape[0], dist.shape[0]

        self.device = device
        self.B, self.N = B, N
        self.offset = OFFSET
        self.num_warps = num_warps or default_num_warps(N)

        self.eta = eta  # [B] float32
        self.gamma = gamma.to(device).view(B, 1)  # [B, 1] float32
        self.deg = torch.zeros(
            (self.B, self.N), device=device, dtype=torch.int16
        )  # [B, N] int16

        self.A = torch.zeros(
            (B, N, N), device=device, dtype=torch.bool
        )  # [B, N, N] binary
        self.C = torch.zeros(
            (B, N, N), device=device, dtype=torch.int32
        )  # [B, N, N] int32
        self.D = dist  # [N, N] float 32
        self.P, self.R = self._full_P()  # [B, N, N] float32, [B, N] float64

        self.t = 0

    def _sim_P(self, b: int) -> torch.Tensor:
        """Full recompute of P for simulation b."""
        A, deg = self.A[b], self.deg[b]
        denom = (deg.unsqueeze(-2) + deg.unsqueeze(-1) - 2 * A) / 2
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        D_eta = self.D.pow(self.eta[b])
        K = self.C[b] / denom
        K = torch.where(K == 0, K + self.offset, K)
        P = (D_eta * K.pow(self.gamma[b]) + self.offset) * ~A  # zero-out edges
        return P.fill_diagonal_(0.0)

    def _full_P(self):
        """P and its fp64 row sums, one simulation at a time: B x n x n temporaries OOM
        at n = 2000, B = 1024. P.sum(-1, dtype=float64) on all of P first copies P to
        fp64 (32.8 GB there), and that block, once fragmented, failed the next allocation.
        """
        P = torch.empty(self.B, self.N, self.N, device=self.device)
        R = torch.empty(self.B, self.N, device=self.device, dtype=torch.float64)
        for b in range(self.B):
            P[b] = self._sim_P(b)
            R[b] = P[b].sum(-1, dtype=torch.float64)
        return P, R

    @ensure_contiguous
    def step(
        self,
        u: torch.Tensor | None = None,  # [B, 2]
        edges: torch.Tensor | None = None,  # [B, 2]
    ) -> torch.Tensor:
        T = 1
        EDGES_GIVEN = edges is not None
        SHARED_UNIFORM = u is not None

        edges = (
            edges if EDGES_GIVEN else torch.empty((self.B, T, 2), device=self.device)
        )
        u = u if SHARED_UNIFORM else torch.empty((self.B, 2), device=self.device)

        BLOCK_N = triton.next_power_of_2(self.N)

        step_edges = torch.empty((self.B, T, 2), dtype=torch.int32, device=self.device)
        binary_gnm_step[(self.B,)](
            A_ptr=self.A,
            C_ptr=self.C,
            P_ptr=self.P,
            R_ptr=self.R,
            Gamma_ptr=self.gamma,
            Eta_ptr=self.eta,
            Deg_ptr=self.deg,
            Dist_ptr=self.D,
            U_ptr=u,
            Edges_ptr=edges,
            Out_ptr=step_edges,
            N=self.N,
            T=T,
            SEED=self.base_seed + self.t,
            OFFSET=OFFSET,
            BLOCK_N=BLOCK_N,
            EDGES_GIVEN=EDGES_GIVEN,
            SHARED_UNIFORM=SHARED_UNIFORM,
            num_warps=self.num_warps,
        )

        self.t += 1

        if EDGES_GIVEN:
            return edges
        return step_edges[:, 0]

    @ensure_contiguous
    def run(
        self,
        T: int,
        u: torch.Tensor | None = None,  # [B, T, 2]
    ) -> torch.Tensor:
        SHARED_UNIFORM = u is not None
        u = u if SHARED_UNIFORM else torch.empty((self.B, 2), device=self.device)
        edges = torch.empty((self.B, T, 2), device=self.device)
        BLOCK_N = triton.next_power_of_2(self.N)

        step_edges = torch.empty((self.B, T, 2), dtype=torch.int32, device=self.device)
        self.kernel = binary_gnm_step[(self.B,)](
            A_ptr=self.A,
            C_ptr=self.C,
            P_ptr=self.P,
            R_ptr=self.R,
            Gamma_ptr=self.gamma,
            Eta_ptr=self.eta,
            Deg_ptr=self.deg,
            Dist_ptr=self.D,
            U_ptr=u,
            Edges_ptr=edges,
            Out_ptr=step_edges,
            N=self.N,
            T=T,
            SEED=self.base_seed + self.t,
            OFFSET=OFFSET,
            BLOCK_N=BLOCK_N,
            EDGES_GIVEN=False,
            SHARED_UNIFORM=SHARED_UNIFORM,
            num_warps=self.num_warps,
        )

        self.t += T

        return step_edges

    def state(
        self,
    ) -> dict[str, torch.Tensor]:
        return {
            "A": self.A,
            "C": self.C,
            "deg": self.deg,
            "r": self.R,
            "P": self.P,
        }


@triton.jit
def _row_P(
    C_row,
    A_row,
    Dist_row,
    Deg,
    deg_x,
    x,
    Eta,
    Gamma,
    cols,
    mask,
    OFFSET,
):
    """
    Compute the new P rows:
    1: compute denom: [N] = (deg_r + degrees - 2Arc) / 2
    2: Kx = Cx / denom
    3: Add OFFSET to zeros in K
    4: new_Pr = pow(Dist[r], Eta) * pow(K, Gamma) + OFFSET
    5: new_Pr *= 1-A_r (zero-out the existing edges)
    6: new_Pr[r] = 0 (zero-out diagonal)
    """
    denom = (deg_x + Deg - 2 * A_row) / 2
    # a zero denominator can only happen on a lane that's an edge, where P is zero
    # change 0 to 1 to avoid NaNs, this doesn't affect the result
    denom = tl.where(denom == 0, 1, denom)
    K_row = C_row / denom
    K_row = tl.where(K_row == 0, OFFSET, K_row)
    # triton doesn't support `pow`, replace with x^y = (exp(log(x)))^y = exp(y * log(x))
    D_Eta_r = tl.exp(Eta * tl.log(Dist_row))
    K_row = tl.exp(Gamma * tl.log(K_row))
    P_row = D_Eta_r * K_row + OFFSET
    # zero-out existing edges, diagonal and padded elements
    P_row = tl.where(mask & (A_row == 0) & (cols != x), P_row, 0)
    return P_row


@triton.jit
def sample_cdf(W, U):
    """
    Sample from an unnormalised distribution W using a pre-computed uniform scalar U.

    Args:
        W: Unnormalised distribution, [N,] float64
        U: x~U(0,1), float32

    Returns:
        int: The sampled scalar index.
    """
    cdf = tl.cumsum(W.to(tl.float64), axis=-1)
    t = U.to(tl.float64) * tl.max(cdf)
    return tl.sum(cdf <= t, axis=-1)


@triton.jit(do_not_specialize=["T", "SEED"])
def binary_gnm_step(
    A_ptr,
    C_ptr,
    P_ptr,
    R_ptr,
    Eta_ptr,
    Gamma_ptr,
    Deg_ptr,
    Dist_ptr,
    U_ptr,
    Edges_ptr,
    Out_ptr,
    N,
    T,
    OFFSET,
    SEED,
    BLOCK_N: tl.constexpr,
    EDGES_GIVEN: tl.constexpr,
    SHARED_UNIFORM: tl.constexpr,
):
    """
    Args:
        A_ptr: Adjacency matrix ptr, [B, N, N] bool
        C_ptr: C matrix ptr, [B, N, N] int32
        P_ptr: P matrix ptr, [B, N, N] float32
        R_ptr: R vector ptr, [B, N] float64
        Eta_ptr: Eta vector ptr, [B,] float32
        Gamma_ptr: Gamma vector ptr, [B, 1] float32
        Deg_ptr: Degrees vector ptr, [B, N] int16
        Dist_ptr: Dist matrix ptr, [N, N] float32
        U_ptr: Pre-computed uniform ptr, [B, T, 2] float32
        Edges_ptr: Pre-computed edges ptr, [B, 2] int32
        N (int): Number of rows/columns
        T (int): Number of steps
        OFFSET (float): Small constant for numerical stability, set by `gnm`
        SEED (int): The step's seed, used to generate random draws
        BLOCK_N (tl.constexpr): Block size along the N dimension, `BLOCK_N=next_power_of_2(N)`
        EDGES_GIVEN (tl.constexpr): Whether the edges have already been sampled
        SHARED_UNIFORM (tl.constexpr): Whether the kernel should use a pre-computed
          uniform for sampling
    """
    pid = tl.program_id(0)
    row_stride = N
    sim_offs = (
        pid.to(tl.int64) * N * row_stride
    )  # cast to int64 to prevent ptr overflow

    # offset pointers to the base of the program's simulation (i.e. the current batch element)
    A_ptr = A_ptr + sim_offs
    C_ptr = C_ptr + sim_offs
    P_ptr = P_ptr + sim_offs
    Deg_ptr = Deg_ptr + pid * N
    R_ptr = R_ptr + pid * N
    Edges_ptr = Edges_ptr + pid * 2 * T
    U_ptr = U_ptr + pid * 2 * T
    Out_ptr = Out_ptr + pid * 2 * T

    # all matrices are [N, N] and use the same mask/offsets
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    Eta = tl.load(Eta_ptr + pid)
    Gamma = tl.load(Gamma_ptr + pid)

    R = tl.load(R_ptr + cols, mask=mask, other=0.0)
    Deg = tl.load(Deg_ptr + cols, mask=mask, other=0.0)

    for t in tl.range(T):
        if EDGES_GIVEN:  # skip sampling
            i = tl.load(Edges_ptr + 2 * t)
            j = tl.load(Edges_ptr + 2 * t + 1)
            old_Pi = tl.load(P_ptr + i * row_stride + cols, mask=mask, other=0.0)

        else:
            U1 = (
                tl.load(U_ptr + 2 * t) if SHARED_UNIFORM else tl.rand(SEED + t, 2 * pid)
            )
            i = sample_cdf(W=R, U=U1)

            old_Pi = tl.load(P_ptr + i * row_stride + cols, mask=mask, other=0.0)
            U2 = (
                tl.load(U_ptr + 2 * t + 1)
                if SHARED_UNIFORM
                else tl.rand(SEED + t, 2 * pid + 1)
            )
            j = sample_cdf(W=old_Pi, U=U2)

        i_offs = i * row_stride + cols
        j_offs = j * row_stride + cols

        old_Pj = tl.load(P_ptr + j_offs, mask=mask, other=0.0)

        # load A rows before updates, they are used in the C update
        old_Ai = tl.load(A_ptr + i_offs, mask=mask, other=0)
        old_Aj = tl.load(A_ptr + j_offs, mask=mask, other=0)

        # update Degrees with sampled nodes
        Deg = tl.where((cols == i) | (cols == j), Deg + 1, Deg)

        # update in registers
        Ai = old_Ai + tl.where(cols == j, 1, 0)
        Aj = old_Aj + tl.where(cols == i, 1, 0)

        # Same with C, load only rows
        Ci_ptr = C_ptr + i_offs
        Cj_ptr = C_ptr + j_offs
        old_Ci = tl.load(Ci_ptr, mask=mask, other=0)
        old_Cj = tl.load(Cj_ptr, mask=mask, other=0)

        # 4 writes to C:
        # coalesced: C[j, :] = Cj + Ai + (cols == j) / C[i, :] = Ci + Aj + (cols == i)
        # uncoalesced: same values for columns -> mask unchanged to reduce cost
        Ci = old_Ci + old_Aj + tl.where(cols == i, 1, 0)
        Cj = old_Cj + old_Ai + tl.where(cols == j, 1, 0)
        tl.store(Ci_ptr, value=Ci, mask=mask)
        tl.store(Cj_ptr, value=Cj, mask=mask)
        tl.store(C_ptr + i + cols * row_stride, value=Ci, mask=mask & (old_Ci != Ci))
        tl.store(C_ptr + j + cols * row_stride, value=Cj, mask=mask & (old_Cj != Cj))

        # pad with 1.0 to avoid -inf on log dist
        Dist_i = tl.load(Dist_ptr + i_offs, mask=mask, other=1.0)
        Dist_j = tl.load(Dist_ptr + j_offs, mask=mask, other=1.0)
        deg_i = tl.sum(tl.where(cols == i, Deg, 0))
        deg_j = tl.sum(tl.where(cols == j, Deg, 0))

        Pi = _row_P(Ci, Ai, Dist_i, Deg, deg_i, i, Eta, Gamma, cols, mask, OFFSET)
        Pj = _row_P(Cj, Aj, Dist_j, Deg, deg_j, j, Eta, Gamma, cols, mask, OFFSET)

        # update P: P[i, :] = new_Pi / P[:, i] = new_Pi / P[j, :] = new_Pj / P[:, j] = new_Pj
        tl.store(P_ptr + i_offs, value=Pi, mask=mask)
        tl.store(P_ptr + j_offs, value=Pj, mask=mask)
        tl.store(P_ptr + i + cols * row_stride, value=Pi, mask=mask & (old_Pi != Pi))
        tl.store(P_ptr + j + cols * row_stride, value=Pj, mask=mask & (old_Pj != Pj))

        # update R: R+= new_Pi - old_Pi + new_Pj - old_Pj
        R = (
            R
            + (Pi.to(tl.float64) - old_Pi.to(tl.float64))
            + (Pj.to(tl.float64) - old_Pj.to(tl.float64))
        )
        sum_i = tl.sum(Pi.to(tl.float64))
        sum_j = tl.sum(Pj.to(tl.float64))
        R = tl.where(cols == i, sum_i, tl.where(cols == j, sum_j, R))

        tl.store(Out_ptr + 2 * t, value=i)
        tl.store(Out_ptr + 2 * t + 1, value=j)

        # update A with sampled nodes, this update is delayed after the reductions
        # so every thread has already loaded the old A rows
        tl.store(A_ptr + j + i * row_stride, value=1)  # A[i, j] = 1
        tl.store(A_ptr + i + j * row_stride, value=1)  # A[j, i] = 1

        tl.debug_barrier()  # sync before next iteration

    tl.store(Deg_ptr + cols, value=Deg, mask=mask)
    tl.store(R_ptr + cols, value=R, mask=mask)
