"""src/kernel_additive.py (Oldham rule forms) against src/reference_rules.py (SPEC §4.1, §6).

    TRITON_INTERPRET=1 pytest tests/test_kernel_additive.py     # CPU, kernel interpreted
    modal run tests/modal_tests.py --args "tests/test_kernel_additive.py -v -rs"

1. The reference against a literal port of the MATLAB formula (GNM.m, 2025; 2022 eps = 0).
2. Teacher forcing: same edges into kernel and reference, every step: A, C, deg exact; row
   sums and both maxima within tolerance; the kernel's maintained per-row maxima equal to
   a brute-force recompute of its own state (fp64, to the last few ulps).
3. Shared uniforms: same trajectories up to rounding ties.
4. Sampling: chi-square of the first edge against the reference P.
5. Long runs at extreme eta / gamma: row sums against an exact recompute (drift, resync).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "prep" / "kernel")):
    if p not in sys.path:
        sys.path.insert(0, p)

from bench_gnm import make_dist  # noqa: E402

from src.reference_rules import DenseRule  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
HAS_KERNEL = torch.cuda.is_available() or INTERPRET
DEV = torch.device("cuda" if torch.cuda.is_available() and not INTERPRET else "cpu")
needs_kernel = pytest.mark.skipif(not HAS_KERNEL, reason="needs CUDA or TRITON_INTERPRET=1")

R_TOL = 1e-9  # row sums, relative to the total weight (fp64 both sides)
LOG_TOL = 1e-9  # log maxima, absolute (fp64 both sides)
MAX_TOL = 1e-12  # maintained per-row maxima vs brute force (fp64), relative

# eta, gamma, alpha per simulation: extremes of Oldham's ranges and the usual region
PARAMS = [(-2.0, 10.0, 10.0), (-2.0, -10.0, 0.0), (-0.1, 3.0, 1.0), (0.0, -1.0, 0.05),
          (-1.0, 0.0, 5.0), (-0.3, -10.0, 10.0)]
RULES = [("add", "exp"), ("mult", "exp"), ("add", "power")]


def params(device, rule):
    eta, gamma, alpha = (torch.tensor(x, dtype=torch.float32, device=device)
                         for x in zip(*PARAMS))
    if rule[1] == "power":
        eta = eta * 5  # power-law range [-10, 0]
    return eta, gamma, alpha


def kernel(dist, eta, gamma, alpha, rule):
    from src.kernel_additive import TritonRuleGNM
    return TritonRuleGNM(dist, eta, gamma, alpha, form=rule[0], cost=rule[1])


# ------------------------------------------------------------------ 1. reference
def matlab_P(A, D, eta, gamma, alpha, form, cost, eps):
    """Literal port of GNM.m's probability over the upper triangle (column-major
    find(triu(ones(n),1)) order), for one network; returns the dense symmetric P."""
    n = A.shape[0]
    v, u = np.nonzero(np.triu(np.ones((n, n)), 1).T)  # MATLAB column-major order
    u, v = v, u
    a = (A[u, v] == 0).astype(float)
    d = D[u, v]
    Df = np.exp(eta * d) if cost == "exp" else d ** eta
    PDf = Df / np.max(Df * a)
    nei = A @ A
    deg = A.sum(1)
    degsum = deg[:, None] + deg[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (nei * 2) / (((degsum <= 2) & (nei != 1)) + (degsum - A * 2))
    m[np.isnan(m)] = 0
    K = m[u, v]
    with np.errstate(divide="ignore"):
        Fk = (K + eps) ** gamma
    if eps == 0:  # 2022: Inf -> 0
        Fk[np.isinf(Fk)] = 0
    mx = np.max(Fk * a)
    FK = alpha * (Fk / mx) if mx > 0 else alpha * Fk
    prob = PDf + FK if form == "add" else PDf * (Fk / mx)
    P = np.zeros((n, n))
    P[u, v] = prob * a
    return P + P.T


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("eps", [1e-6, 0.0])
def test_reference_matches_matlab_formula(rule, eps):
    n = 20
    rng = np.random.default_rng(1)
    A = np.triu(rng.random((n, n)) < 0.25, 1).astype(float)
    A = A + A.T
    D = make_dist(n).double().numpy()
    eta, gamma, alpha = params("cpu", rule)
    ref = DenseRule(torch.tensor(D), eta, gamma, alpha, form=rule[0], cost=rule[1], epsilon=eps)
    ref.A = torch.tensor(A).expand(len(PARAMS), n, n).clone()
    ref.refresh()
    for b, (e, g, a) in enumerate(zip(eta.tolist(), gamma.tolist(), alpha.tolist())):
        if rule[0] == "mult" and eps == 0 and g < 0:
            continue  # 2022 mult code has no Inf clamp; not a convention we use
        P = matlab_P(A, D, e, g, a, rule[0], rule[1], eps)
        Pr = ref.P[b].numpy()
        # MATLAB uses a single global max; compare up to the overall scale (it cancels)
        np.testing.assert_allclose(Pr / Pr.sum(), P / P.sum(), rtol=1e-9, atol=1e-300)
        if rule[0] == "add":  # the additive form doesn't cancel: compare exactly
            np.testing.assert_allclose(Pr, P, rtol=1e-9, atol=1e-300)


# ------------------------------------------------------------------ helpers
def assert_close(sim, ref, where):
    got, exp = sim.state(), ref.state()
    for k in ("A", "C", "deg"):  # reference C = A @ A, diagonal included, like the kernel's
        g, e = got[k].double().cpu(), exp[k].double().cpu()
        assert torch.equal(g, e), f"{where}: {k} differs at {(g != e).nonzero()[:3].tolist()}"
    if sim.form == "add":  # the additive form's scale is fixed by the two maxima
        for k in ("logMa", "logMb"):
            g, e = got[k].double().cpu(), exp[k].double().cpu()
            assert (g - e).abs().max() <= LOG_TOL, f"{where}: {k} {g.tolist()} vs {e.tolist()}"
        keys = ("r", "ra", "rb")
    else:  # the product form only fixes P up to scale: compare normalised row sums
        keys = ("r",)
    for k in keys:
        g, e = got[k].double().cpu(), exp[k].double().cpu()
        if sim.form == "mult":
            g = g / g.sum(-1, keepdim=True)
            e = e / e.sum(-1, keepdim=True)
        tot = e.sum(-1, keepdim=True).clamp_min(1e-300)
        err = ((g - e).abs() / tot).max().item()
        assert err <= R_TOL, f"{where}: {k} err {err:.2e} (relative to the total)"


def assert_maxima_exact(sim, where):
    """Maintained per-row maxima equal a brute-force recompute from the kernel's state
    (fp64 on both sides; Triton's and torch's log may differ in the last ulp)."""
    x = sim.exact(0, sim.B)

    def close(g, e):
        both_inf = torch.isinf(g) & torch.isinf(e) & (g.sign() == e.sign())
        return bool((both_inf | ((g - e).abs() <= MAX_TOL * e.abs().clamp_min(1))).all())

    if sim.form == "add":
        assert close(sim.MaxS, x["MaxS"]), f"{where}: MaxS"
    assert close(sim.MaxY, x["MaxY"]), f"{where}: MaxY"
    # the argmax may differ on ties; its value must be the max
    _, y, valid = sim.exps(0, sim.B)
    at = torch.gather(torch.where(valid, y, -math.inf), -1, sim.ArgY.long().unsqueeze(-1))[..., 0]
    full = torch.isfinite(x["MaxY"])
    assert close(at[full], x["MaxY"][full]), f"{where}: ArgY doesn't point at the max"


def reference_edges(dist, eta, gamma, alpha, rule, T, seed=0):
    ref = DenseRule(dist, eta, gamma, alpha, form=rule[0], cost=rule[1])
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(len(eta), T, 2, generator=g)
    edges, margins = [], []
    for t in range(T):
        rows = DenseRule.sample_cdf(ref.r, u[:, t, 0])
        m_row = margin(ref.r, u[:, t, 0])
        m_col = margin(ref.P[ref.b, rows], u[:, t, 1])
        margins.append(torch.stack((m_row, m_col), -1))
        edges.append(ref.step(u=u[:, t]))
    return torch.stack(edges, 1), u, torch.stack(margins, 1)


def margin(w, u):
    cdf = w.double().cumsum(-1)
    t = u.double().unsqueeze(-1) * cdf[:, -1:]
    return (cdf - t).abs().min(-1).values / cdf[:, -1]


# ------------------------------------------------------------------ 2. teacher forcing
@needs_kernel
@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("n,T", [(12, 50), (68, 120), (130, 80)])
def test_teacher_forced(rule, n, T):
    dist = make_dist(n)
    eta, gamma, alpha = params("cpu", rule)
    edges, _, _ = reference_edges(dist, eta, gamma, alpha, rule, T)
    ref = DenseRule(dist, eta, gamma, alpha, form=rule[0], cost=rule[1])
    sim = kernel(dist.to(DEV), eta.to(DEV), gamma.to(DEV), alpha.to(DEV), rule)
    assert_close(sim, ref, "init")
    assert_maxima_exact(sim, "init")
    for t in range(T):
        sim.step(edges=edges[:, t].to(DEV))
        ref.step(edges=edges[:, t])
        assert_close(sim, ref, f"step {t}")
        assert_maxima_exact(sim, f"step {t}")


@needs_kernel
@pytest.mark.parametrize("rule", RULES)
def test_teacher_forced_one_launch(rule):
    """run(T, edges=...): the persistent loop, the recompute loop and resyncs in one launch."""
    n, T = 68, 600
    dist = make_dist(n)
    eta, gamma, alpha = params("cpu", rule)
    edges, _, _ = reference_edges(dist, eta, gamma, alpha, rule, T)
    ref = DenseRule(dist, eta, gamma, alpha, form=rule[0], cost=rule[1])
    for t in range(T):
        ref.step(edges=edges[:, t])
    sim = kernel(dist.to(DEV), eta.to(DEV), gamma.to(DEV), alpha.to(DEV), rule)
    out = sim.run(T, edges=edges.to(DEV))
    assert torch.equal(out.cpu().long(), edges)
    assert_close(sim, ref, f"after {T} steps")
    assert_maxima_exact(sim, f"after {T} steps")


# ------------------------------------------------------------------ 3. shared uniforms
@needs_kernel
@pytest.mark.parametrize("rule", RULES)
def test_shared_uniforms(rule):
    n, T = 68, 150
    dist = make_dist(n)
    eta, gamma, alpha = params("cpu", rule)
    e_ref, u, margins = reference_edges(dist, eta, gamma, alpha, rule, T, seed=3)
    sim = kernel(dist.to(DEV), eta.to(DEV), gamma.to(DEV), alpha.to(DEV), rule)
    e_sim = sim.run(T, u=u.to(DEV)).cpu().long()
    for b in range(len(PARAMS)):
        bad = (e_sim[b] != e_ref[b]).any(-1).nonzero()
        if len(bad):
            t = bad[0].item()
            stage = 0 if e_sim[b, t, 0] != e_ref[b, t, 0] else 1
            assert margins[b, t, stage] < 1e-4, (
                f"sim {b} {PARAMS[b]}: step {t} differs, not at a rounding tie "
                f"(margin {margins[b, t, stage]:.1e})")


# ------------------------------------------------------------------ 4. sampling
@needs_kernel
@pytest.mark.parametrize("rule", RULES)
def test_sampling_chi_square(rule):
    """First edge after a teacher-forced prefix, drawn with the kernel's RNG, against
    2 P_ij / sum(P) from the reference, at every parameter setting."""
    n, prefix, draws = 12, 20, 4000
    dist = make_dist(n)
    eta, gamma, alpha = params("cpu", rule)
    edges, _, _ = reference_edges(dist, eta, gamma, alpha, rule, prefix, seed=5)
    for b in range(len(PARAMS)):
        ref = DenseRule(dist, eta[b:b + 1], gamma[b:b + 1], alpha[b:b + 1], form=rule[0],
                        cost=rule[1])
        for t in range(prefix):
            ref.step(edges=edges[b:b + 1, t])
        P = torch.triu(ref.P[0], 1)
        p = (P / P.sum()).flatten().numpy()
        rep = lambda x: x.expand(draws).contiguous().to(DEV)  # noqa: E731
        sim = kernel(dist.to(DEV), rep(eta[b:b + 1]), rep(gamma[b:b + 1]), rep(alpha[b:b + 1]),
                     rule)
        sim.run(prefix, edges=edges[b:b + 1].expand(draws, -1, -1).contiguous().to(DEV))
        e = sim.step().cpu().long()
        lo, hi = e.min(-1).values, e.max(-1).values
        counts = np.bincount((lo * n + hi).numpy(), minlength=n * n)
        assert counts[p == 0].sum() == 0, f"{PARAMS[b]}: drew an edge with P = 0"
        pval = chi_square(counts, p)
        if pval is None:
            continue  # (almost) all mass on one pair at extreme parameters: nothing to test
        # control: the same test on the reference's own sampler (guards the binning)
        g = torch.Generator().manual_seed(b)
        ref_draws = torch.multinomial(torch.tensor(p), draws, replacement=True, generator=g)
        p_ref = chi_square(np.bincount(ref_draws.numpy(), minlength=n * n), p)
        assert pval > 1e-4, f"{rule} {PARAMS[b]}: chi-square p = {pval:.1e} (control {p_ref:.1e})"


def chi_square(counts, p):
    """Chi-square goodness of fit, pooling the rarest bins until every bin expects >= 5
    draws. None if fewer than two bins remain."""
    from scipy import stats
    draws = counts.sum()
    nz = np.flatnonzero(p > 0)
    order = nz[np.argsort(p[nz])]  # rarest first
    exp_ = p[order] * draws
    obs = counts[order]
    f_obs, f_exp, acc_o, acc_e = [], [], 0, 0.0
    for o, e in zip(obs, exp_):
        acc_o += o
        acc_e += e
        if acc_e >= 5:
            f_obs.append(acc_o)
            f_exp.append(acc_e)
            acc_o, acc_e = 0, 0.0
    if acc_e > 0 and f_exp:  # leftover goes into the last bin
        f_obs[-1] += acc_o
        f_exp[-1] += acc_e
    if len(f_exp) < 2:
        return None
    f_exp = np.array(f_exp)
    f_exp *= draws / f_exp.sum()
    return stats.chisquare(np.array(f_obs), f_exp).pvalue


# ------------------------------------------------------------------ 5. drift, resync
@needs_kernel
@pytest.mark.parametrize("rule", RULES)
def test_long_run_drift(rule):
    """To 60% density with the kernel's own RNG; row sums against an exact recompute."""
    n = 68
    T = int(0.6 * n * (n - 1) / 2)
    dist = make_dist(n).to(DEV)
    eta, gamma, alpha = (x.to(DEV) for x in params("cpu", rule))
    sim = kernel(dist, eta, gamma, alpha, rule)
    sim.run(T)
    x = sim.exact(0, sim.B)
    for k in ("Ra",) + (("Rb",) if rule[0] == "add" else ()):
        g, e = getattr(sim, k), x[k]
        err = ((g - e).abs() / e.sum(-1, keepdim=True)).max().item()
        assert err <= 1e-9, f"{k} drift {err:.1e}"
    assert_maxima_exact(sim, "end")
    assert int(sim.A.sum()) == 2 * T * sim.B
    if rule == ("add", "exp"):  # eta = -2 spans e^-300: maxima fall, resyncs happen
        assert int(sim.stats[0, 1]) > 0
    assert int(sim.stats[:, 0].sum()) > 0  # some row maxima were recomputed
