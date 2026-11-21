# type: ignore
"""Fused Triton kernel for the Oldham et al. rule forms (SPEC §4.1, H9).

    form 'add':  P = a / max(a) + alpha * b / max(b)      form 'mult':  P = a * b / max(a * b)
    a = exp(eta * D)  (cost 'exp')  or  D ** eta  (cost 'power')
    b = (K + eps) ** gamma,  K = 2 shared / (k_i + k_j - 2 A_ij)  (0 / 0 -> 0)
    Maxima over current non-edges, after every edge. Conventions of Oldham, Fornito & Ball
    2025 (StuartJO/ComingUpShort @ 800bea9, GNM.m), where the product form normalises each
    factor by its own max; that only rescales P, so normalising the product by its max
    samples the same distribution and keeps every entry <= 1. Dense reference:
    src/reference_rules.py.

Exponents, in fp64: ea = eta * x (x = D, or log D) and eb = gamma * log(K + eps) (add), or
eq = ea + eb (mult). The terms are exp(ea - Sa), exp(eb - Sb) (add) or exp(eq - Sq) (mult),
with S the current maximum of the exponent over non-edges. fp64 because the exponents
reach ~|eta D| + |gamma log eps| ~ 480: in fp32, rounding of the exponent alone costs
~3e-5 relative per term, and the product's max can sit far below the factors' maxima.

Per simulation, persistent over T steps (one program each), as in src/kernel.py:
- rows i and j are recomputed old and new; row sums are patched by deltas (fp64), using
  symmetry for the column changes. Ra: sums of the cost term (add) or of P (mult); Rb:
  sums of the topology term (add);
- per-row maxima over non-edges: MaxS of ea (static entries, add only: changes only in
  rows i and j, recomputed) and MaxY of the dynamic exponent y = eb (add) or eq (mult),
  with ArgY its column. y changes in rows and columns i and j: increases are applied with
  max(); a row whose argmax column (i or j) decreased is recomputed (~1.1-1.8 rows per
  step at n = 68-1000, bench/additive_max_recompute.py);
- y can only exceed its global max in rows i and j: the shift is raised to cover them
  before any term is formed, and the row sums rescaled. After the step the maxima are
  re-read from the row maxima; when they fall, the row sums are rescaled in fp64, unless a
  maximum has fallen by more than RESYNC_LOG since the last exact computation: then every
  row sum is recomputed (O(n^2), rare), since rescaling amplifies accumulated absolute
  error.

The topology term needs eps > 0 (2025 convention); the 2022 convention (eps = 0, Inf -> 0)
is in the reference only.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

EPS = 1e-6
RESYNC_LOG = 20.0


@triton.jit
def _row_exp(C_row, A_row, Dist_row, Deg, deg_x, x, Eta, Gamma, cols, mask, EPS,
             COST_EXP: tl.constexpr, MULT: tl.constexpr):
    """Exponents of row x: (ea, y) with y = eb (add) or ea + eb (mult), fp64, and the
    non-edge mask."""
    valid = mask & (A_row == 0) & (cols != x)
    d = Dist_row.to(tl.float64)
    if COST_EXP:
        ea = Eta * d
    else:
        ea = Eta * tl.log(d)
    den = (deg_x + Deg - 2 * A_row).to(tl.float64)
    K = tl.where(den > 0, 2.0 * C_row.to(tl.float64) / tl.maximum(den, 1.0), 0.0)
    eb = Gamma * tl.log(K + EPS)
    if MULT:
        y = ea + eb
    else:
        y = eb
    return ea, y, valid


@triton.jit
def _terms(ea, y, valid, Sa, Sy, MULT: tl.constexpr):
    """(a, b) for add; (p, 0) for mult. fp64."""
    if MULT:
        a = tl.where(valid, tl.exp(y - Sy), 0.0)
        b = tl.zeros_like(a)
    else:
        a = tl.where(valid, tl.exp(ea - Sa), 0.0)
        b = tl.where(valid, tl.exp(y - Sy), 0.0)
    return a, b


@triton.jit
def _sample_cdf(W, U):
    cdf = tl.cumsum(W.to(tl.float64), axis=-1)
    t = U.to(tl.float64) * tl.max(cdf)
    return tl.sum(cdf <= t, axis=-1)


@triton.jit
def _uniform(U_ptr, t, k, SEED, pid, SHARED_UNIFORM: tl.constexpr):
    if SHARED_UNIFORM:
        u = tl.load(U_ptr + 2 * t + k)
    else:
        u = tl.rand(SEED + t, 2 * pid + k)
    return u


@triton.jit
def _load_row(A_ptr, C_ptr, Dist_ptr, Deg, x, cols, mask, N):
    offs = x * N + cols
    A_row = tl.load(A_ptr + offs, mask=mask, other=1).to(tl.int32)  # padding = "edge"
    C_row = tl.load(C_ptr + offs, mask=mask, other=0).to(tl.int32)
    Dist_row = tl.load(Dist_ptr + offs, mask=mask, other=1.0)
    deg_x = tl.sum(tl.where(cols == x, Deg, 0))
    return A_row, C_row, Dist_row, deg_x


@triton.jit
def _store_sym(ptr, x, row, old_row, cols, mask, N):
    tl.store(ptr + x * N + cols, value=row, mask=mask)
    tl.store(ptr + x + cols * N, value=row, mask=mask & (old_row != row))


@triton.jit
def _resync(A_ptr, C_ptr, Dist_ptr, Deg, Eta, Gamma, Sa, Sy, cols, mask, N, EPS,
            COST_EXP: tl.constexpr, MULT: tl.constexpr):
    """Exact row sums of every row at shifts Sa, Sy."""
    Ra = tl.zeros(cols.shape, dtype=tl.float64)
    Rb = tl.zeros(cols.shape, dtype=tl.float64)
    for k in tl.range(N):
        A_row, C_row, Dist_row, deg_k = _load_row(A_ptr, C_ptr, Dist_ptr, Deg, k, cols, mask, N)
        ea, y, valid = _row_exp(C_row, A_row, Dist_row, Deg, deg_k, k, Eta, Gamma, cols, mask,
                                EPS, COST_EXP, MULT)
        a, b = _terms(ea, y, valid, Sa, Sy, MULT)
        Ra = tl.where(cols == k, tl.sum(a), Ra)
        Rb = tl.where(cols == k, tl.sum(b), Rb)
    return Ra, Rb


@triton.jit(do_not_specialize=["T", "SEED"])
def rule_gnm_kernel(
    A_ptr,  # [B, N, N] bool
    C_ptr,  # [B, N, N] int16
    Deg_ptr,  # [B, N] int16
    Ra_ptr,  # [B, N] float64: row sums of the cost term (add) or of P (mult)
    Rb_ptr,  # [B, N] float64: row sums of the topology term (add)
    MaxS_ptr,  # [B, N] float64: row max of ea over non-edges (add)
    MaxY_ptr,  # [B, N] float64: row max of y over non-edges
    ArgY_ptr,  # [B, N] int32: its column
    Ref_ptr,  # [B, 2] float64: highest Sa, Sy since the last exact row sums
    Stats_ptr,  # [B, 2] int32: row-max recomputes, full resyncs (cumulative)
    Eta_ptr, Gamma_ptr, Alpha_ptr,  # [B] float32
    Dist_ptr,  # [N, N] float32
    U_ptr, Edges_ptr, Out_ptr,  # [B, T, 2]
    Eps_ptr,  # [1] float64: epsilon (a scalar argument would be passed as fp32)
    N, T, RESYNC_LOG, SEED,
    BLOCK_N: tl.constexpr,
    COST_EXP: tl.constexpr,
    MULT: tl.constexpr,
    EDGES_GIVEN: tl.constexpr,
    SHARED_UNIFORM: tl.constexpr,
):
    pid = tl.program_id(0)
    pid64 = pid.to(tl.int64)
    sim = pid64 * N * N
    A_ptr += sim
    C_ptr += sim
    Deg_ptr += pid64 * N
    Ra_ptr += pid64 * N
    Rb_ptr += pid64 * N
    MaxS_ptr += pid64 * N
    MaxY_ptr += pid64 * N
    ArgY_ptr += pid64 * N
    Ref_ptr += pid64 * 2
    Stats_ptr += pid64 * 2
    U_ptr += pid64 * 2 * T
    Edges_ptr += pid64 * 2 * T
    Out_ptr += pid64 * 2 * T

    NEG = -float("inf")
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    Eta = tl.load(Eta_ptr + pid).to(tl.float64)
    Gamma = tl.load(Gamma_ptr + pid).to(tl.float64)
    Alpha = tl.load(Alpha_ptr + pid).to(tl.float64)
    EPS = tl.load(Eps_ptr)
    Deg = tl.load(Deg_ptr + cols, mask=mask, other=0).to(tl.int32)
    Ra = tl.load(Ra_ptr + cols, mask=mask, other=0.0)
    Rb = tl.load(Rb_ptr + cols, mask=mask, other=0.0)
    MaxS = tl.load(MaxS_ptr + cols, mask=mask, other=NEG)
    MaxY = tl.load(MaxY_ptr + cols, mask=mask, other=NEG)
    ArgY = tl.load(ArgY_ptr + cols, mask=mask, other=-1)
    RefA = tl.load(Ref_ptr)
    RefY = tl.load(Ref_ptr + 1)
    n_rec = tl.load(Stats_ptr)
    n_sync = tl.load(Stats_ptr + 1)
    Sa = tl.max(MaxS)
    Sy = tl.max(MaxY)
    if MULT:
        Sa = tl.zeros_like(Sa)

    for t in tl.range(T):
        # -- 1. row i (row weights relative to the current maxima)
        if MULT:
            W = Ra
        else:
            W = Ra + Alpha * Rb
        if EDGES_GIVEN:
            i = tl.load(Edges_ptr + 2 * t).to(tl.int32)
        else:
            i = _sample_cdf(W, _uniform(U_ptr, t, 0, SEED, pid, SHARED_UNIFORM)).to(tl.int32)
        oA_i, oC_i, D_i, deg_i = _load_row(A_ptr, C_ptr, Dist_ptr, Deg, i, cols, mask, N)
        ea_i, oy_i, ok_i = _row_exp(oC_i, oA_i, D_i, Deg, deg_i, i, Eta, Gamma, cols, mask, EPS,
                                    COST_EXP, MULT)

        # -- 2. column j, from P[i] at the current shifts
        if EDGES_GIVEN:
            j = tl.load(Edges_ptr + 2 * t + 1).to(tl.int32)
        else:
            pa, pb = _terms(ea_i, oy_i, ok_i, Sa, Sy, MULT)
            j = _sample_cdf(pa + Alpha * pb,
                            _uniform(U_ptr, t, 1, SEED, pid, SHARED_UNIFORM)).to(tl.int32)
        oA_j, oC_j, D_j, deg_j = _load_row(A_ptr, C_ptr, Dist_ptr, Deg, j, cols, mask, N)
        ea_j, oy_j, ok_j = _row_exp(oC_j, oA_j, D_j, Deg, deg_j, j, Eta, Gamma, cols, mask, EPS,
                                    COST_EXP, MULT)

        # -- 3. add edge (i, j): degrees, A rows (registers), C rows and columns
        Deg = tl.where((cols == i) | (cols == j), Deg + 1, Deg)
        deg_i += 1
        deg_j += 1
        A_i = tl.where(cols == j, 1, oA_i)
        A_j = tl.where(cols == i, 1, oA_j)
        C_i = oC_i + tl.where(mask, oA_j, 0) + tl.where(cols == i, 1, 0)
        C_j = oC_j + tl.where(mask, oA_i, 0) + tl.where(cols == j, 1, 0)
        _store_sym(C_ptr, i, C_i.to(tl.int16), oC_i.to(tl.int16), cols, mask, N)
        _store_sym(C_ptr, j, C_j.to(tl.int16), oC_j.to(tl.int16), cols, mask, N)

        # -- 4. new rows i, j. y can only exceed Sy in rows/columns i, j: raise the shift
        # to cover them before forming any term, then patch the row sums by deltas.
        _, y_i, k_i = _row_exp(C_i, A_i, D_i, Deg, deg_i, i, Eta, Gamma, cols, mask, EPS,
                               COST_EXP, MULT)
        _, y_j, k_j = _row_exp(C_j, A_j, D_j, Deg, deg_j, j, Eta, Gamma, cols, mask, EPS,
                               COST_EXP, MULT)
        yi = tl.where(k_i, y_i, NEG)
        yj = tl.where(k_j, y_j, NEG)
        mY_i = tl.max(yi)
        mY_j = tl.max(yj)
        Sy_up = tl.maximum(Sy, tl.maximum(mY_i, mY_j))
        if Sy_up > Sy:
            f = tl.exp(Sy - Sy_up)
            if MULT:
                Ra = Ra * f
            else:
                Rb = Rb * f
            RefY = tl.maximum(RefY, Sy_up)
            Sy = Sy_up
        oa_i, ob_i = _terms(ea_i, oy_i, ok_i, Sa, Sy, MULT)
        oa_j, ob_j = _terms(ea_j, oy_j, ok_j, Sa, Sy, MULT)
        a_i, b_i = _terms(ea_i, y_i, k_i, Sa, Sy, MULT)
        a_j, b_j = _terms(ea_j, y_j, k_j, Sa, Sy, MULT)
        Ra += (a_i - oa_i) + (a_j - oa_j)
        Ra = tl.where(cols == i, tl.sum(a_i), tl.where(cols == j, tl.sum(a_j), Ra))
        if not MULT:
            Rb += (b_i - ob_i) + (b_j - ob_j)
            Rb = tl.where(cols == i, tl.sum(b_i), tl.where(cols == j, tl.sum(b_j), Rb))

        # -- 5. maxima. ea is static: only rows i, j lose an entry (i, j).
        if not MULT:
            mS_i = tl.max(tl.where(k_i, ea_i, NEG))
            mS_j = tl.max(tl.where(k_j, ea_j, NEG))
            MaxS = tl.where(cols == i, mS_i, tl.where(cols == j, mS_j, MaxS))
        # y: rows i and j exactly; other rows k see entries (k, i) = y_i[k] and (k, j) =
        # y_j[k] by symmetry, still non-edges iff k_i[k] (resp. k_j[k])
        aY_i = tl.argmax(yi, axis=0).to(tl.int32)
        aY_j = tl.argmax(yj, axis=0).to(tl.int32)
        other = mask & (cols != i) & (cols != j)
        flag = other & (((ArgY == i) & (yi < MaxY)) | ((ArgY == j) & (yj < MaxY)))
        up_i = other & (yi > MaxY)
        MaxY = tl.where(up_i, yi, MaxY)
        ArgY = tl.where(up_i, i, ArgY)
        up_j = other & (yj > MaxY)
        MaxY = tl.where(up_j, yj, MaxY)
        ArgY = tl.where(up_j, j, ArgY)
        MaxY = tl.where(cols == i, mY_i, tl.where(cols == j, mY_j, MaxY))
        ArgY = tl.where(cols == i, aY_i, tl.where(cols == j, aY_j, ArgY))
        # flagged rows: recompute (their C rows were patched above: make those visible)
        tl.debug_barrier()
        n_flag = tl.sum(flag.to(tl.int32))
        n_rec += n_flag
        while n_flag > 0:
            k = tl.min(tl.where(flag, cols, BLOCK_N))
            A_k, C_k, D_k, deg_k = _load_row(A_ptr, C_ptr, Dist_ptr, Deg, k, cols, mask, N)
            _, y_k, k_k = _row_exp(C_k, A_k, D_k, Deg, deg_k, k, Eta, Gamma, cols, mask, EPS,
                                   COST_EXP, MULT)
            yk = tl.where(k_k, y_k, NEG)
            MaxY = tl.where(cols == k, tl.max(yk), MaxY)
            ArgY = tl.where(cols == k, tl.argmax(yk, axis=0).to(tl.int32), ArgY)
            flag = flag & (cols != k)
            n_flag -= 1

        # -- 6. maxima after the step (they can only have fallen): rescale or resync
        nSy = tl.max(MaxY)
        nSy = tl.where(nSy == NEG, Sy, nSy)  # no non-edges left
        if MULT:
            nSa = Sa
        else:
            nSa = tl.max(MaxS)
            nSa = tl.where(nSa == NEG, Sa, nSa)

        tl.store(Out_ptr + 2 * t, value=i)
        tl.store(Out_ptr + 2 * t + 1, value=j)
        # A[i, j] = A[j, i] = 1 only after every reduction over the old A rows (SPEC §4)
        tl.store(A_ptr + i * N + j, value=1)
        tl.store(A_ptr + j * N + i, value=1)
        tl.debug_barrier()

        if ((RefA - nSa) > RESYNC_LOG) | ((RefY - nSy) > RESYNC_LOG):
            Ra, Rb = _resync(A_ptr, C_ptr, Dist_ptr, Deg, Eta, Gamma, nSa, nSy, cols, mask, N,
                             EPS, COST_EXP, MULT)
            RefA = nSa
            RefY = nSy
            n_sync += 1
        else:
            if MULT:
                Ra = Ra * tl.exp(Sy - nSy)
            else:
                Ra = Ra * tl.exp(Sa - nSa)
                Rb = Rb * tl.exp(Sy - nSy)
        Sa = nSa
        Sy = nSy

    tl.store(Deg_ptr + cols, value=Deg.to(tl.int16), mask=mask)
    tl.store(Ra_ptr + cols, value=Ra, mask=mask)
    tl.store(Rb_ptr + cols, value=Rb, mask=mask)
    tl.store(MaxS_ptr + cols, value=MaxS, mask=mask)
    tl.store(MaxY_ptr + cols, value=MaxY, mask=mask)
    tl.store(ArgY_ptr + cols, value=ArgY, mask=mask)
    tl.store(Ref_ptr, value=RefA)
    tl.store(Ref_ptr + 1, value=RefY)
    tl.store(Stats_ptr, value=n_rec)
    tl.store(Stats_ptr + 1, value=n_sync)


def default_num_warps(n: int) -> int:
    return max(2, min(16, triton.next_power_of_2(n) // 128))


class TritonRuleGNM:
    """B simulations of one rule form, one (eta, gamma, alpha) each.

    dist [n, n] float32 symmetric (diagonal ignored); eta, gamma, alpha [B] float32.
    form 'add' | 'mult', cost 'exp' | 'power'. Same interface as src/kernel.TritonGNM:
    step(u=None, edges=None), run(T, u=None, edges=None) -> [B, T, 2], state().
    """

    def __init__(self, dist, eta, gamma, alpha=None, form: str = "add", cost: str = "exp",
                 eps: float = EPS, num_warps: int | None = None):
        assert form in ("add", "mult") and cost in ("exp", "power") and eps > 0
        dev = dist.device
        B, N = eta.shape[0], dist.shape[0]
        self.device, self.B, self.N, self.form, self.cost, self.eps = dev, B, N, form, cost, eps
        self.num_warps = num_warps or default_num_warps(N)
        self.base_seed = torch.randint(0, 2**30, (1,)).item()
        self.t = 0
        D = dist.to(dev, torch.float32).clone()
        D.fill_diagonal_(1.0)  # masked; keeps log finite for the power cost
        self.D = D.contiguous()
        f = lambda x: x.to(dev, torch.float32).contiguous()  # noqa: E731
        self.eta, self.gamma = f(eta), f(gamma)
        self.alpha = f(torch.ones(B) if alpha is None else alpha)
        self.A = torch.zeros((B, N, N), device=dev, dtype=torch.bool)
        self.C = torch.zeros((B, N, N), device=dev, dtype=torch.int16)
        self.deg = torch.zeros((B, N), device=dev, dtype=torch.int16)
        self.stats = torch.zeros((B, 2), device=dev, dtype=torch.int32)
        self.eps64 = torch.tensor([eps], device=dev, dtype=torch.float64)
        self._init_state()

    # -- exact state from A, in torch (initial state; brute-force checks in tests)
    def exps(self, s: int, e: int):
        """ea, y [e - s, n, n] float64 and the non-edge mask, as the kernel computes them."""
        A = self.A[s:e]
        Ai = A.to(torch.int64)
        deg = self.deg[s:e].to(torch.int64)
        n = self.N
        valid = ~A & ~torch.eye(n, dtype=torch.bool, device=self.device)
        d = self.D.double()
        x = d if self.cost == "exp" else torch.log(d)
        ea = self.eta[s:e].double().view(-1, 1, 1) * x
        den = (deg.unsqueeze(-1) + deg.unsqueeze(-2) - 2 * Ai).double()
        K = torch.where(den > 0, 2.0 * self.C[s:e].double() / den.clamp_min(1.0),
                        torch.zeros_like(den))
        eb = self.gamma[s:e].double().view(-1, 1, 1) * torch.log(K + self.eps)
        y = ea + eb if self.form == "mult" else eb
        return ea, y, valid

    def exact(self, s: int, e: int) -> dict:
        ea, y, valid = self.exps(s, e)
        neg = torch.tensor(-math.inf, device=self.device, dtype=torch.float64)
        sm = torch.where(valid, ea, neg)
        ym = torch.where(valid, y, neg)
        MaxS, MaxY = sm.amax(-1), ym.amax(-1)
        ArgY = ym.argmax(-1).to(torch.int32)
        Sy = MaxY.amax(-1, keepdim=True)
        Sa = MaxS.amax(-1, keepdim=True)
        if self.form == "mult":
            Ra = torch.where(valid, torch.exp(y - Sy.unsqueeze(-1)), 0.0).sum(-1)
            Rb = torch.zeros_like(Ra)
            Sa = torch.zeros_like(Sa)
        else:
            Ra = torch.where(valid, torch.exp(ea - Sa.unsqueeze(-1)), 0.0).sum(-1)
            Rb = torch.where(valid, torch.exp(y - Sy.unsqueeze(-1)), 0.0).sum(-1)
        return {"MaxS": MaxS, "MaxY": MaxY, "ArgY": ArgY, "Ra": Ra, "Rb": Rb,
                "Sa": Sa[:, 0], "Sy": Sy[:, 0]}

    def _init_state(self, block: int | None = None):
        B, N = self.B, self.N
        dev = self.device
        self.Ra = torch.empty(B, N, device=dev, dtype=torch.float64)
        self.Rb = torch.empty_like(self.Ra)
        self.MaxS = torch.empty(B, N, device=dev, dtype=torch.float64)
        self.MaxY = torch.empty_like(self.MaxS)
        self.ArgY = torch.empty(B, N, device=dev, dtype=torch.int32)
        self.Ref = torch.empty(B, 2, device=dev, dtype=torch.float64)
        if block is None:
            free = torch.cuda.mem_get_info(dev)[0] if dev.type == "cuda" else 2**30
            block = max(1, min(B, min(free // 4, 8 * 2**30) // (64 * N * N)))
        for s in range(0, B, block):
            e = min(B, s + block)
            x = self.exact(s, e)
            for k in ("Ra", "Rb", "MaxS", "MaxY", "ArgY"):
                getattr(self, k)[s:e] = x[k]
            self.Ref[s:e, 0], self.Ref[s:e, 1] = x["Sa"], x["Sy"]

    def _launch(self, T: int, u=None, edges=None) -> torch.Tensor:
        out = torch.empty((self.B, T, 2), dtype=torch.int32, device=self.device)
        if u is not None:
            u = u.to(self.device, torch.float32).reshape(self.B, T, 2).contiguous()
        if edges is not None:
            edges = edges.to(self.device, torch.int32).reshape(self.B, T, 2).contiguous()
        rule_gnm_kernel[(self.B,)](
            self.A, self.C, self.deg, self.Ra, self.Rb, self.MaxS, self.MaxY, self.ArgY,
            self.Ref, self.stats, self.eta, self.gamma, self.alpha, self.D,
            out if u is None else u, out if edges is None else edges, out, self.eps64,
            self.N, T, RESYNC_LOG, self.base_seed + self.t,
            BLOCK_N=triton.next_power_of_2(self.N),
            COST_EXP=self.cost == "exp", MULT=self.form == "mult",
            EDGES_GIVEN=edges is not None, SHARED_UNIFORM=u is not None,
            num_warps=self.num_warps,
        )
        self.t += T
        return out

    def step(self, u=None, edges=None) -> torch.Tensor:
        return self._launch(1, u=u, edges=edges)[:, 0]

    def run(self, T: int, u=None, edges=None) -> torch.Tensor:
        return self._launch(T, u=u, edges=edges)

    def state(self) -> dict:
        """Dense state in the reference layout (src/reference_rules.DenseRule.state).
        For mult, P is normalised by the product's max, the reference by the product of
        the factors' maxima: compare mult row sums after normalising by their total."""
        if self.form == "mult":
            r = self.Ra
            out = {"r": r}
        else:
            r = self.Ra + self.alpha.double().unsqueeze(-1) * self.Rb
            out = {"r": r, "ra": self.Ra, "rb": self.Rb,
                   "logMa": self.MaxS.amax(-1), "logMb": self.MaxY.amax(-1)}
        return {"A": self.A, "C": self.C, "deg": self.deg, **out}
