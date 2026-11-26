"""Correctness tests for GNM step implementations, checked against RowSampled.

    cd prep/kernel && pytest test_kernel.py -v                        # GPU
    cd prep/kernel && TRITON_INTERPRET=1 pytest test_kernel.py -v     # CPU, kernel interpreted
    cd prep/kernel && pytest test_kernel.py -v --slow                 # + distribution vs toolbox
    modal run prep/kernel/modal_tests.py::run_tests                   # on a Modal GPU

Every test also runs against the reference itself ("reference" params), which checks
the harness. The kernel side ("triton" params) is skipped until it exists.

Contract for `kernel.TritonGNM` (internal layout and dtypes are free):

    sim = TritonGNM(dist, eta, gamma)
        dist [n, n] float32, symmetric, diagonal 1 (masked anyway; avoids 0**eta)
        eta, gamma [B] float32: one parameter setting per simulation
    sim.step(u=None, edges=None) -> [B, 2] int (row, col) of the edge added
        u     [B, 2] float32 in [0, 1): inverse-CDF sampling, RowSampled.sample_cdf:
              row = first i with cdf(r)_i > u[:, 0] * sum(r),
              col = first j with cdf(P[row])_j > u[:, 1] * sum(P[row])
        edges [B, 2] int64: teacher forcing, apply these edges instead of sampling
        neither: sample with the kernel's own RNG (tl.rand)
    sim.state() -> dict of dense tensors, converted to the reference layout:
        "A", "C" [B, n, n]; "deg", "r" [B, n]; optionally "P" [B, n, n]
    sim.run(T, u=None) -> [B, T, 2]   (optional: the persistent kernel, T steps in one launch)
"""

import pytest
import torch
from bench_gnm import make_dist
from row_sampled import RowSampled

# Integer state must match exactly; float state to these relative tolerances.
P_TOL = 1e-5
R_TOL = 1e-5
R_TOL_LONG = 1e-4  # after ~1000 steps of incremental row-sum updates
# A shared-uniform mismatch counts as a rounding tie (not a bug) if u * total was this
# close (relative to the total) to a cdf boundary in the reference.
TIE_TOL = 1e-4

# (n, B): tiny; a real parcellation size; n > 128 and not a power of two (block masking)
CASES = [(12, 3), (68, 4), (130, 2)]
STEPS = {12: 60, 68: 200, 130: 120}


def _triton_factory(dist, eta, gamma):
    from kernel import FlashGNM

    return FlashGNM(dist, eta, gamma)


@pytest.fixture(params=["reference", "triton"])
def factory(request, triton_runnable):
    if request.param == "reference":
        return RowSampled
    if not triton_runnable:
        pytest.skip("no GPU: set TRITON_INTERPRET=1 to run the kernel on CPU")
    try:
        from kernel import FlashGNM  # noqa: F401
    except (ImportError, AttributeError):
        pytest.skip("kernel.TritonGNM not implemented yet")
    return _triton_factory


# ------------------------------------------------------------------ helpers
def params(B, device):
    """A different (eta, gamma) per simulation, so per-simulation indexing is tested."""
    eta = torch.linspace(-3.0, -0.5, B, device=device)
    gamma = torch.linspace(0.1, 0.5, B, device=device)
    return eta, gamma


def uniforms(B, T, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, T, 2, generator=g).to(device)


def build(factory, n, B, device):
    dist = make_dist(n).to(device)
    eta, gamma = params(B, device)
    return factory(dist, eta, gamma), RowSampled(dist, eta, gamma)


def _first_mismatch(got, ref):
    idx = (got != ref).nonzero()[0].tolist()
    return f"first at index {idx}: got {got[tuple(idx)].item()}, expected {ref[tuple(idx)].item()}"


def assert_state_close(sim, ref, where, r_tol=R_TOL):
    got, exp = sim.state(), ref.state()
    for k in ("A", "C", "deg"):
        g, e = got[k].double(), exp[k].double()
        assert g.shape == e.shape, (
            f"{where}: {k} shape {tuple(g.shape)} != {tuple(e.shape)}"
        )
        assert torch.equal(g, e), f"{where}: {k} differs, {_first_mismatch(g, e)}"
    tols = [("r", r_tol)] + ([("P", P_TOL)] if "P" in got else [])
    for k, tol in tols:
        g, e = got[k].double(), exp[k].double()
        # Entrywise: P spans orders of magnitude, scaling by its max hides small entries.
        rel = ((g - e).abs() / e.abs().clamp_min(1e-30)).max().item()
        assert rel <= tol, f"{where}: {k} rel err {rel:.2e} > {tol:.0e}"


def margin(w, u):
    """Distance from u * total to the nearest cdf boundary, relative to the total. [B]"""
    cdf = w.double().cumsum(-1)
    t = u.double().unsqueeze(-1) * cdf[:, -1:]
    return (cdf - t).abs().min(-1).values / cdf[:, -1]


def reference_trajectory(n, B, u, device):
    """Edges [B, T, 2] from RowSampled under shared uniforms, plus per-draw tie margins."""
    dist = make_dist(n).to(device)
    ref = RowSampled(dist, *params(B, device))
    edges, margins = [], []
    for t in range(u.shape[1]):
        rows = RowSampled.sample_cdf(ref.r, u[:, t, 0])
        m_row = margin(ref.r, u[:, t, 0])
        m_col = margin(ref.P[ref.b, rows], u[:, t, 1])
        margins.append(torch.stack((m_row, m_col), -1))
        edges.append(ref.step(u=u[:, t]))
    return torch.stack(edges, 1), torch.stack(margins, 1)


def assert_same_trajectory(e_sim, e_ref, margins, min_identical=0.9):
    """Edges must match draw for draw. A sim may diverge only at a rounding tie, after
    which its states differ and it's no longer compared."""
    e_sim = e_sim.to(e_ref)
    assert e_sim.shape == e_ref.shape, (
        f"edges shape {tuple(e_sim.shape)} != {tuple(e_ref.shape)}"
    )
    neq = e_sim != e_ref
    ties = []
    for b in range(e_ref.shape[0]):
        bad = neq[b].any(-1).nonzero()
        if len(bad) == 0:
            continue
        t = bad[0].item()
        stage = 0 if neq[b, t, 0] else 1
        m = margins[b, t, stage].item()
        assert m < TIE_TOL, (
            f"sim {b}, step {t}: {'row' if stage == 0 else 'col'} differs "
            f"(got {e_sim[b, t].tolist()}, expected {e_ref[b, t].tolist()}) but u * total "
            f"was {m:.1e} from the nearest cdf boundary: not a rounding tie"
        )
        ties.append((b, t, m))
    assert len(ties) <= (1 - min_identical) * e_ref.shape[0], (
        f"too many tie divergences: {ties}"
    )


# ------------------------------------------------------------------ tests
@pytest.mark.parametrize("n,B", CASES)
def test_initial_state(factory, n, B, device):
    sim, ref = build(factory, n, B, device)
    assert_state_close(sim, ref, "init")


@pytest.mark.parametrize("n,B", CASES)
def test_teacher_forced_update(factory, n, B, device):
    """Apply the reference's edges and compare the whole state after every step. Tests
    the update (C, deg, A, P rows/cols, row sums) independently of sampling."""
    sim, ref = build(factory, n, B, device)
    u = uniforms(B, STEPS[n], device)
    for t in range(STEPS[n]):
        e = ref.step(u=u[:, t])
        sim.step(edges=e)
        assert_state_close(sim, ref, f"step {t}, edges {e.tolist()}")


@pytest.mark.parametrize("mode", ["step", "run"])
def test_shared_uniforms(factory, mode, device):
    """Same uniforms in, same edges out: tests sampling and update together."""
    n, B, T = 68, 32, 200
    u = uniforms(B, T, device, seed=1)
    e_ref, margins = reference_trajectory(n, B, u, device)
    sim, _ = build(factory, n, B, device)
    if mode == "run":
        if not hasattr(sim, "run"):
            pytest.skip("no persistent run() yet")
        e_sim = sim.run(T, u=u)
    else:
        e_sim = torch.stack([sim.step(u=u[:, t]) for t in range(T)], 1)
    assert_same_trajectory(e_sim, e_ref, margins)


@pytest.mark.parametrize("rng", ["uniforms", "internal"])
def test_run_final_state(factory, rng, device):
    """After run(), memory must hold the full state, including anything the kernel keeps
    in registers during the loop. Replaying its edges on the reference tolerates ties."""
    n, B, T = 68, 4, 60
    sim, ref = build(factory, n, B, device)
    if not hasattr(sim, "run"):
        pytest.skip("no persistent run() yet")
    e = sim.run(T, u=uniforms(B, T, device, seed=4) if rng == "uniforms" else None)
    for t in range(T):
        ref.step(edges=e[:, t].to(device).long())
    assert_state_close(sim, ref, f"after run({T})")


@pytest.mark.parametrize(
    "B,T,rng",
    [
        (8, 200, "uniforms"),
        (8, 400, "uniforms"),
        (8, 1000, "uniforms"),
        (8, 400, "internal"),
        (64, 400, "internal"),
        (1024, 400, "uniforms"),
        (1024, 400, "internal"),
    ],
)
def test_run_long_C(factory, B, T, rng, device):
    """Persistent runs past the T of test_run_final_state: C against the reference's
    incremental C and against A @ A, reporting where errors sit."""

    if device.type != "cuda" and (T > 200 or B > 8):
        pytest.skip("too slow for the interpreter")
    n = 68
    sim, ref = build(factory, n, B, device)
    if not hasattr(sim, "run"):
        pytest.skip("no persistent run() yet")
    e = sim.run(T, u=uniforms(B, T, device, seed=6) if rng == "uniforms" else None)
    for t in range(T):
        ref.step(edges=e[:, t].to(device).long())
    C = sim.state()["C"].to(ref.C)
    A = ref.A
    matmul = A @ A
    vs_ref, vs_mm, ref_mm = C != ref.C, C != matmul, ref.C != matmul
    where = (
        f"vs reference {vs_ref.sum().item()}, vs A@A {vs_mm.sum().item()}, "
        f"reference vs A@A {ref_mm.sum().item()}; of those vs reference: "
        f"diagonal {vs_ref.diagonal(dim1=-2, dim2=-1).sum().item()}, "
        f"edge entries {(vs_ref & A.bool()).sum().item()}"
    )
    assert not (vs_ref.any() or vs_mm.any()), f"C differs after run({T}): {where}"


@pytest.mark.slow
@pytest.mark.parametrize("n,B", [(1000, 16), (2000, 4)])
def test_run_large_blocks(factory, n, B, device):
    """The CASES stop at block size 256: this covers 1024 and 2048, where the kernel
    reaches the register limit and spills, against the reference."""
    if device.type != "cuda":
        pytest.skip("too slow for the interpreter")
    T = 300
    sim, ref = build(factory, n, B, device)
    if not hasattr(sim, "run"):
        pytest.skip("no persistent run() yet")
    e = sim.run(T, u=uniforms(B, T, device, seed=5))
    for t in range(T):
        ref.step(edges=e[:, t].to(device).long())
    assert_state_close(sim, ref, f"after run({T}), n={n}")


@pytest.mark.parametrize("rng", ["uniforms", "internal"])
def test_valid_edges_until_complete(factory, rng, device):
    """Grow to the complete graph: never a self-loop, a repeated edge or an index out of
    range, including the extreme uniforms 0 and the largest float32 below 1."""
    n, B = 10, 8
    sim, _ = build(factory, n, B, device)
    seen = torch.zeros(B, n, n, dtype=torch.bool, device=device)
    b = torch.arange(B, device=device)
    u = uniforms(B, n * (n - 1) // 2, device, seed=2)
    u[0] = 0.0
    u[1] = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)).item()
    for t in range(n * (n - 1) // 2):
        e = sim.step(u=u[:, t] if rng == "uniforms" else None).to(device).long()
        a, c = e[:, 0], e[:, 1]
        assert ((a >= 0) & (a < n) & (c >= 0) & (c < n)).all(), (
            f"step {t}: out of range {e.tolist()}"
        )
        assert (a != c).all(), f"step {t}: self-loop {e.tolist()}"
        assert not seen[b, a, c].any(), f"step {t}: repeated edge {e.tolist()}"
        seen[b, a, c] = seen[b, c, a] = True
    st = sim.state()
    A = st["A"].double()
    assert torch.equal(A, seen.double()), "state A doesn't match the returned edges"
    assert torch.equal(st["C"].double(), A @ A), "C != A^T A"
    assert torch.equal(st["deg"].double(), A.sum(-1)), "deg != row sums of A"


@pytest.mark.parametrize("rng", ["uniforms", "internal"])
def test_sampling_distribution(factory, rng, device):
    """From one mid-growth state copied B times, one step's (row, col) draws must follow
    P / sum(P): a chi-square test of the two-stage sampler against the flat n^2 sampler."""
    from scipy.stats import chisquare

    n, B, warm = 12, 2048, 15
    dist = make_dist(n).to(device)
    eta, gamma = (
        torch.full((B,), -2.0, device=device),
        torch.full((B,), 0.3, device=device),
    )
    ref = RowSampled(dist, eta[:1], gamma[:1])
    sim = factory(dist, eta, gamma)
    for t in range(warm):
        e = ref.step(u=uniforms(1, 1, device, seed=10 + t)[:, 0])
        sim.step(edges=e.expand(B, 2).contiguous())
    P = ref.P[0].double()
    expected = (P / P.sum()).flatten()

    u = uniforms(B, 1, device, seed=3)[:, 0] if rng == "uniforms" else None
    e = sim.step(u=u).to(device).long()
    counts = torch.bincount(e[:, 0] * n + e[:, 1], minlength=n * n).double()

    assert counts[expected == 0].sum() == 0, "drew an existing edge or the diagonal"
    exp = expected * B
    big = exp >= 5  # pool small cells so the chi-square approximation holds
    obs = torch.cat([counts[big], counts[~big & (expected > 0)].sum().view(1)])
    ex = torch.cat([exp[big], exp[~big & (expected > 0)].sum().view(1)])
    p = chisquare(obs.cpu().numpy(), ex.cpu().numpy()).pvalue
    assert p > 1e-3, f"chi-square p = {p:.1e}: draws don't follow P"


def test_long_run_drift(factory, device):
    """~1000 teacher-forced steps: integer state stays exact, row sums stay accurate
    (whatever resync policy the implementation uses)."""
    n, B, T = 68, 2, 1100
    sim, ref = build(factory, n, B, device)
    u = uniforms(B, T, device, seed=4)
    for t in range(T):
        e = ref.step(u=u[:, t])
        sim.step(edges=e)
        if (t + 1) % 100 == 0:
            assert_state_close(sim, ref, f"step {t}", r_tol=R_TOL_LONG)


@pytest.mark.slow
def test_distribution_vs_toolbox(factory, device):
    """Networks from the implementation's own RNG vs the toolbox, AAL (n=90), 400 edges,
    300 runs each: two-sample KS on per-network statistics."""
    from assess import AAL, EDGES, ETA, GAMMA, ours_dist, stats, toolbox_model
    from scipy.stats import ks_2samp

    R = 300
    torch.manual_seed(5)
    m = toolbox_model(AAL.to(device), R, device=device)
    for _ in range(EDGES):
        m.binary_update()
    eta = torch.full((R,), ETA, device=device)
    gamma = torch.full((R,), GAMMA, device=device)
    sim = factory(ours_dist(AAL).to(device), eta, gamma)
    if hasattr(sim, "run"):
        sim.run(EDGES)
    else:
        for _ in range(EDGES):
            sim.step()
    s_tb = stats(m.adjacency_matrix.cpu(), AAL)
    s_sim = stats(sim.state()["A"].float().cpu(), AAL)
    pvals = {k: ks_2samp(s_tb[k], s_sim[k]).pvalue for k in s_tb}  # type: ignore
    assert min(pvals.values()) > 1e-3, f"KS p-values: {pvals}"
