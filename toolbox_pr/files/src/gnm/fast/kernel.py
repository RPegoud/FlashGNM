# type: ignore  (Triton kernels are not jaxtyped)
r"""Fused Triton kernel for the binary GNM: matching index (mean divisor), power-law
distance and preference terms.

This computes the same model as
[`GenerativeNetworkModel.binary_update`][gnm.GenerativeNetworkModel.binary_update]
with `MatchingIndex(divisor="mean")` and power-law relationships:

$$
    P_{ij} = D_{ij}^{\eta} K_{ij}^{\gamma} + \epsilon,
$$

where $K_{ij} = 0$ is first replaced by $\epsilon$ (`prob_offset`), and $P_{ij} = 0$ on
edges and the diagonal. Non-finite weights are clamped as in the toolbox
(`nan -> 0`, `inf -> 1e30`).

It computes it differently. The toolbox recomputes the whole $n \times n$ matrix every
step ($O(n^3)$ per step for $A^T A$). Here one GPU program runs one simulation for all
its steps, and each step costs $O(n)$:

1. Sample a row $i$ with probability $R_i / \sum_k R_k$, where $R_i = \sum_j P_{ij}$, then
   a column $j$ with probability $P_{ij} / R_i$. This is the same distribution as sampling
   the pair $(i, j)$ from $P / \sum P$ (the toolbox's `multinomial` over the flattened
   matrix).
2. Add the edge $\{i, j\}$. Only rows (and, by symmetry, columns) $i$ and $j$ of
   $C = A^T A$, of the degrees and therefore of $P$ change.
3. Recompute rows $i$ and $j$ of $P$ before and after the edge, and update $R$ by the
   differences.

$P$ is never stored. The state per simulation is $A$ (bool), $C$ (int16) and the degrees,
3 bytes per node pair, and $R$ (float64, one value per node). Simulations in a batch can
have different $(\eta, \gamma)$ and seed networks, and share the distance matrix.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover: CPU-only installs
    triton = None
    _TRITON_IMPORT_ERROR = e

CLAMP_MAX = 1e30  # the toolbox's nan_to_num(posinf=1e30)


def flash_available(device: Optional[torch.device | str] = None) -> bool:
    r"""True if the kernel can run on `device` (default: CUDA if available).

    It needs Triton and a CUDA device, or Triton's CPU interpreter (set
    `TRITON_INTERPRET=1` before importing triton; slow, for tests only).
    """
    if triton is None:
        return False
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.is_available()
    return os.environ.get("TRITON_INTERPRET") == "1"


if triton is not None:

    @triton.jit
    def _row_P(C_row, A_row, Dist_row, Deg, deg_x, x, Eta, Gamma, cols, mask, OFFSET):
        """Row x of P from row x of C and A, all degrees and row x of the distances.

        Returns the row (zero on edges, the diagonal and padded lanes) and the number of
        entries that were inf or nan before clamping.
        """
        # mean divisor: (deg_x + deg_y - A_xy - A_yx) / 2; zero only on an edge lane
        denom = (deg_x + Deg - 2 * A_row) / 2
        denom = tl.where(denom == 0, 1, denom)
        K_row = C_row / denom
        K_row = tl.where(K_row == 0, OFFSET, K_row)
        # x ** y = exp(y * log(x)); 0 ** 0 = 1 as in torch.pow
        D_eta = tl.where(Eta == 0, 1.0, tl.exp(Eta * tl.log(Dist_row)))
        K_gamma = tl.exp(Gamma * tl.log(K_row))
        P_row = D_eta * K_gamma + OFFSET
        valid = mask & (A_row == 0) & (cols != x)
        nonfinite = (P_row * 0.0) != 0.0  # inf * 0 and nan * 0 are nan
        # the toolbox's nan_to_num(nan=0, posinf=1e30) (CLAMP_MAX)
        P_row = tl.where(P_row != P_row, 0.0, tl.where(nonfinite, 1e30, P_row))
        bad = (valid & nonfinite).to(tl.int32)
        return tl.where(valid, P_row, 0.0), tl.sum(bad)

    @triton.jit
    def _sample_cdf(W, U):
        """Index of the first entry whose inclusive cdf exceeds U * sum(W) (fp64)."""
        cdf = tl.cumsum(W.to(tl.float64), axis=-1)
        t = U.to(tl.float64) * tl.max(cdf)
        return tl.sum((cdf <= t).to(tl.int32), axis=-1)

    @triton.jit
    def _uniform(U_ptr, t, k, SEED, pid, SHARED_UNIFORM: tl.constexpr):
        """Draw k (0: row, 1: column) of step t: from U if given, else the kernel RNG."""
        if SHARED_UNIFORM:
            u = tl.load(U_ptr + 2 * t + k)
        else:
            u = tl.rand(SEED + t, 2 * pid + k)
        return u

    @triton.jit
    def _load_row(A_ptr, C_ptr, Dist_ptr, Deg, x, cols, mask, N):
        """Row x of A, C and the distances (padded with 1, so log stays finite), deg[x]."""
        offs = x * N + cols
        A_row = tl.load(A_ptr + offs, mask=mask, other=0)
        C_row = tl.load(C_ptr + offs, mask=mask, other=0)
        Dist_row = tl.load(Dist_ptr + offs, mask=mask, other=1.0)
        deg_x = tl.sum(tl.where(cols == x, Deg, 0))
        return A_row, C_row, Dist_row, deg_x

    @triton.jit
    def _store_sym(ptr, x, row, old_row, cols, mask, N):
        """Write row x and, where it changed, column x."""
        tl.store(ptr + x * N + cols, value=row, mask=mask)
        tl.store(ptr + x + cols * N, value=row, mask=mask & (old_row != row))

    @triton.jit(do_not_specialize=["T", "SEED"])
    def _binary_gnm_kernel(
        A_ptr,  # [B, N, N] bool
        C_ptr,  # [B, N, N] int16
        R_ptr,  # [B, N] float64
        Deg_ptr,  # [B, N] int16
        Clamp_ptr,  # [B] int32: steps with a clamped weight (accumulated)
        Eta_ptr,  # [B] float32
        Gamma_ptr,  # [B] float32
        Dist_ptr,  # [N, N] float32, shared by the batch, diagonal 1
        U_ptr,  # [B, T, 2] float32 if SHARED_UNIFORM
        Edges_ptr,  # [B, T, 2] int32 if EDGES_GIVEN
        Out_ptr,  # [B, T, 2] int32: the edges added
        N,
        T,
        OFFSET,
        SEED,
        BLOCK_N: tl.constexpr,  # next_power_of_2(N): one whole row per program
        EDGES_GIVEN: tl.constexpr,  # teacher forcing: apply Edges instead of sampling
        SHARED_UNIFORM: tl.constexpr,  # sample with U instead of tl.rand
    ):
        pid = tl.program_id(0)
        pid64 = pid.to(tl.int64)  # B * N^2 and B * 2T overflow int32 at large B
        sim = pid64 * N * N
        A_ptr += sim
        C_ptr += sim
        R_ptr += pid64 * N
        Deg_ptr += pid64 * N
        U_ptr += pid64 * 2 * T
        Edges_ptr += pid64 * 2 * T
        Out_ptr += pid64 * 2 * T

        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        Eta = tl.load(Eta_ptr + pid)
        Gamma = tl.load(Gamma_ptr + pid)
        R = tl.load(R_ptr + cols, mask=mask, other=0.0)
        Deg = tl.load(Deg_ptr + cols, mask=mask, other=0)
        clamped = tl.load(Clamp_ptr + pid)

        for t in tl.range(T):
            # 1. row i, and the old rows of i
            if EDGES_GIVEN:
                i = tl.load(Edges_ptr + 2 * t).to(tl.int32)
            else:
                i = _sample_cdf(R, _uniform(U_ptr, t, 0, SEED, pid, SHARED_UNIFORM))
            old_Ai, old_Ci, Dist_i, deg_i = _load_row(
                A_ptr, C_ptr, Dist_ptr, Deg, i, cols, mask, N
            )
            old_Pi, _ = _row_P(
                old_Ci, old_Ai, Dist_i, Deg, deg_i, i, Eta, Gamma, cols, mask, OFFSET
            )

            # 2. column j, and the old rows of j
            if EDGES_GIVEN:
                j = tl.load(Edges_ptr + 2 * t + 1).to(tl.int32)
            else:
                j = _sample_cdf(old_Pi, _uniform(U_ptr, t, 1, SEED, pid, SHARED_UNIFORM))
            old_Aj, old_Cj, Dist_j, deg_j = _load_row(
                A_ptr, C_ptr, Dist_ptr, Deg, j, cols, mask, N
            )
            old_Pj, _ = _row_P(
                old_Cj, old_Aj, Dist_j, Deg, deg_j, j, Eta, Gamma, cols, mask, OFFSET
            )

            # 3. add edge (i, j): degrees, A rows (registers), C rows and columns.
            # C' = C + A_i e_j^T + A_j e_i^T + transpose + e_i e_i^T + e_j e_j^T (old A)
            Deg = tl.where((cols == i) | (cols == j), Deg + 1, Deg)
            deg_i += 1
            deg_j += 1
            Ai = old_Ai + tl.where(cols == j, 1, 0)
            Aj = old_Aj + tl.where(cols == i, 1, 0)
            Ci = old_Ci + old_Aj + tl.where(cols == i, 1, 0)
            Cj = old_Cj + old_Ai + tl.where(cols == j, 1, 0)
            _store_sym(C_ptr, i, Ci, old_Ci, cols, mask, N)
            _store_sym(C_ptr, j, Cj, old_Cj, cols, mask, N)

            # 4. new rows of P, R patched by deltas (fp64); rows i and j summed exactly.
            # Every entry of P that changes is in row (or, by symmetry, column) i or j.
            Pi, bad_i = _row_P(Ci, Ai, Dist_i, Deg, deg_i, i, Eta, Gamma, cols, mask, OFFSET)
            Pj, bad_j = _row_P(Cj, Aj, Dist_j, Deg, deg_j, j, Eta, Gamma, cols, mask, OFFSET)
            clamped += ((bad_i + bad_j) > 0).to(tl.int32)
            R += (Pi.to(tl.float64) - old_Pi.to(tl.float64)) + (
                Pj.to(tl.float64) - old_Pj.to(tl.float64)
            )
            sum_i = tl.sum(Pi.to(tl.float64))
            sum_j = tl.sum(Pj.to(tl.float64))
            R = tl.where(cols == i, sum_i, tl.where(cols == j, sum_j, R))

            tl.store(Out_ptr + 2 * t, value=i)
            tl.store(Out_ptr + 2 * t + 1, value=j)

            # 5. A[i, j] = A[j, i] = 1. Other threads may still be reading the old A rows,
            # so these stores must come after a reduction over those rows (the sums above).
            tl.store(A_ptr + i * N + j, value=1)
            tl.store(A_ptr + j * N + i, value=1)
            tl.debug_barrier()  # all writes of step t visible before step t + 1 loads

        tl.store(Deg_ptr + cols, value=Deg, mask=mask)
        tl.store(R_ptr + cols, value=R, mask=mask)
        tl.store(Clamp_ptr + pid, value=clamped)


def default_num_warps(n: int) -> int:
    """One warp per 128 columns, clamped to [2, 16] (best or within 6% of best on an H100
    at n = 68 to 2000; fewer warps spill registers at n = 2000)."""
    return max(2, min(16, triton.next_power_of_2(n) // 128))


class FlashGNM:
    r"""A batch of B binary GNM simulations on the GPU, each with its own $(\eta, \gamma)$.

    This is the low-level interface. Most users want
    `perform_sweep(..., backend="flash")` or
    [`run_flash`][gnm.fast.run_flash], which build these batches from
    [`RunConfig`][gnm.fitting.RunConfig] objects.

    Args:
        distance_matrix: [n, n] symmetric, non-negative distances, shared by the batch.
            The diagonal is ignored.
        eta: [B] distance exponents.
        gamma: [B] matching-index exponents.
        seed_adjacency_matrix: optional [n, n] or [B, n, n] binary, symmetric, zero
            diagonal. Defaults to empty networks.
        prob_offset: $\epsilon$, as in
            [`BinaryGenerativeParameters`][gnm.BinaryGenerativeParameters].
        num_warps: Triton warps per simulation; the default suits n = 68 to 2000.

    Examples:
        >>> import torch
        >>> from gnm.defaults import get_distance_matrix
        >>> from gnm.fast import FlashGNM
        >>> D = get_distance_matrix().cuda()
        >>> eta = torch.tensor([-2.0, -1.0, -3.0]).cuda()
        >>> gamma = torch.tensor([0.3, 0.5, 0.2]).cuda()
        >>> sim = FlashGNM(D, eta, gamma)
        >>> edges = sim.run(400)            # [3, 400, 2]: the edges added
        >>> sim.adjacency_matrix().shape    # float32, like the toolbox
        torch.Size([3, 90, 90])
    """

    # Bytes of temporaries per (simulation, pair) when computing the initial row sums.
    INIT_BYTES_PER_PAIR = 40

    def __init__(
        self,
        distance_matrix: torch.Tensor,
        eta: torch.Tensor,
        gamma: torch.Tensor,
        seed_adjacency_matrix: Optional[torch.Tensor] = None,
        prob_offset: float = 1e-6,
        num_warps: Optional[int] = None,
    ):
        if triton is None:
            raise ImportError(
                "The flash backend needs Triton (installed with torch on Linux with CUDA)."
            ) from _TRITON_IMPORT_ERROR
        device = distance_matrix.device
        if not flash_available(device):
            raise RuntimeError(
                f"The flash backend needs a CUDA device (got {device}); on CPU set "
                "TRITON_INTERPRET=1 before importing triton (slow, for tests only)."
            )
        eta = torch.as_tensor(eta).reshape(-1)
        gamma = torch.as_tensor(gamma).reshape(-1)
        B, N = eta.shape[0], distance_matrix.shape[-1]
        if distance_matrix.shape != (N, N):
            raise ValueError(f"distance_matrix must be [n, n], got {tuple(distance_matrix.shape)}")
        if gamma.shape[0] != B:
            raise ValueError(f"eta and gamma must have the same length, got {B} and {gamma.shape[0]}")
        if N > 32767:
            raise ValueError("n must be < 32768 (int16 shared-neighbour counts)")

        self.device, self.B, self.N = device, B, N
        self.prob_offset = float(prob_offset)
        self.num_warps = num_warps or default_num_warps(N)
        self.base_seed = int(torch.randint(0, 2**30, (1,)).item())  # follows torch.manual_seed
        self.t = 0

        D = distance_matrix.to(device, torch.float32).clone()
        D.fill_diagonal_(1.0)  # masked anyway; avoids log(0) on the diagonal
        self.D = D.contiguous()
        self.eta = eta.to(device, torch.float32).contiguous()
        self.gamma = gamma.to(device, torch.float32).contiguous()

        if seed_adjacency_matrix is None:
            self.A = torch.zeros((B, N, N), device=device, dtype=torch.bool)
            self.C = torch.zeros((B, N, N), device=device, dtype=torch.int16)
            self.deg = torch.zeros((B, N), device=device, dtype=torch.int16)
        else:
            seed = seed_adjacency_matrix.to(device)
            if seed.dim() == 2:
                seed = seed.unsqueeze(0).expand(B, N, N)
            if seed.shape != (B, N, N):
                raise ValueError(
                    f"seed_adjacency_matrix must be [n, n] or [B, n, n], got {tuple(seed.shape)}"
                )
            self.A = (seed != 0).contiguous()
            if not torch.equal(self.A, self.A.transpose(-1, -2)):
                raise ValueError("seed_adjacency_matrix must be symmetric")
            if self.A.diagonal(dim1=-2, dim2=-1).any():
                raise ValueError("seed_adjacency_matrix must have a zero diagonal")
            self.C = torch.empty((B, N, N), device=device, dtype=torch.int16)
            for s in range(B):  # exact in fp32 up to 2^24 shared neighbours
                a = self.A[s].to(torch.float32)
                self.C[s] = (a @ a).to(torch.int16)
            self.deg = self.A.sum(-1, dtype=torch.int16)

        self.clamp_count = torch.zeros(B, device=device, dtype=torch.int32)
        self.R = self._init_R()

    # -- initial state
    def _block_P(self, s: int, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Full P for simulations s:e, [e - s, n, n] float32, and per-simulation counts of
        clamped entries."""
        A = self.A[s:e]
        deg = self.deg[s:e].to(torch.float32)
        denom = (deg.unsqueeze(-2) + deg.unsqueeze(-1) - 2 * A.to(torch.float32)) / 2
        denom.masked_fill_(denom == 0, 1.0)
        K = self.C[s:e].to(torch.float32).div_(denom)
        del denom
        K.masked_fill_(K == 0, self.prob_offset)
        P = K.pow_(self.gamma[s:e].view(-1, 1, 1))
        P.mul_(self.D.pow(self.eta[s:e].view(-1, 1, 1))).add_(self.prob_offset)
        P.masked_fill_(A, 0.0)
        P.diagonal(dim1=-2, dim2=-1).zero_()
        bad = (~torch.isfinite(P)).flatten(1).any(-1).to(torch.int32)
        torch.nan_to_num_(P, nan=0.0, posinf=CLAMP_MAX, neginf=0.0)
        return P, bad

    def _init_R(self) -> torch.Tensor:
        """float64 row sums of P, in blocks of simulations to bound temporary memory."""
        if self.device.type == "cuda":
            free = torch.cuda.mem_get_info(self.device)[0]
            budget = min(free // 4, 8 * 2**30)
        else:
            budget = 2**30
        block = max(1, min(self.B, budget // (self.INIT_BYTES_PER_PAIR * self.N * self.N)))
        R = torch.empty(self.B, self.N, device=self.device, dtype=torch.float64)
        for s in range(0, self.B, block):
            e = min(s + block, self.B)
            P, bad = self._block_P(s, e)
            R[s:e] = P.sum(-1, dtype=torch.float64)
            self.clamp_count[s:e] += bad
        return R

    # -- stepping
    def _launch(self, T: int, u=None, edges=None) -> torch.Tensor:
        out = torch.empty((self.B, T, 2), dtype=torch.int32, device=self.device)
        if T == 0:
            return out
        if u is not None:
            u = u.to(self.device, torch.float32).reshape(self.B, T, 2).contiguous()
        if edges is not None:
            edges = edges.to(self.device, torch.int32).reshape(self.B, T, 2).contiguous()
        _binary_gnm_kernel[(self.B,)](
            A_ptr=self.A,
            C_ptr=self.C,
            R_ptr=self.R,
            Deg_ptr=self.deg,
            Clamp_ptr=self.clamp_count,
            Eta_ptr=self.eta,
            Gamma_ptr=self.gamma,
            Dist_ptr=self.D,
            U_ptr=out if u is None else u,  # placeholder when unused
            Edges_ptr=out if edges is None else edges,
            Out_ptr=out,
            N=self.N,
            T=T,
            OFFSET=self.prob_offset,
            SEED=self.base_seed + self.t,
            BLOCK_N=triton.next_power_of_2(self.N),
            EDGES_GIVEN=edges is not None,
            SHARED_UNIFORM=u is not None,
            num_warps=self.num_warps,
        )
        self.t += T
        return out

    def run(self, num_steps: int, u=None, edges=None) -> torch.Tensor:
        r"""Add `num_steps` edges to every simulation in one kernel launch.

        Args:
            num_steps: edges to add. Must not exceed the number of missing edges.
            u: optional [B, num_steps, 2] uniforms in [0, 1) (row draw, column draw),
                instead of the kernel's random numbers. For testing.
            edges: optional [B, num_steps, 2] edges to add instead of sampling
                (teacher forcing). For testing.

        Returns:
            [B, num_steps, 2] int32, the edges added, (row, column) as sampled.
        """
        missing = self.N * (self.N - 1) // 2 - int(self.deg.sum(-1).max().item()) // 2
        if edges is None and num_steps > missing:
            raise ValueError(f"num_steps = {num_steps} exceeds the {missing} missing edges")
        return self._launch(num_steps, u=u, edges=edges)

    def step(self, u=None, edges=None) -> torch.Tensor:
        """One step; u or edges [B, 2]. Returns [B, 2]."""
        return self.run(1, u=u, edges=edges)[:, 0]

    def adjacency_matrix(self) -> torch.Tensor:
        """[B, n, n] float32 adjacency matrices (the toolbox's dtype)."""
        return self.A.to(torch.float32)

    def state(self) -> dict[str, torch.Tensor]:
        """The internal state, for tests: A, C = A^T A, degrees and row sums of P."""
        return {"A": self.A, "C": self.C, "deg": self.deg, "r": self.R}
