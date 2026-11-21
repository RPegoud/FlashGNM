"""src/landscape.py: grid layout, exact KS storage, and (GPU) one tiny end-to-end subject."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import evaluation as ev  # noqa: E402
from src import landscape as ls  # noqa: E402

DATA = ROOT / "data" / "griffa" / "scale033.npz"


def test_grid_order_is_gamma_eta_repeat():
    g = ls.Grid(eta_lo=-2, eta_hi=0, n_eta=3, gamma_lo=0, gamma_hi=1, n_gamma=2, repeats=4)
    eta, gamma = g.params()
    eta = eta.numpy().reshape(2, 3, 4)
    gamma = gamma.numpy().reshape(2, 3, 4)
    E, G = np.meshgrid(g.etas, g.gammas, indexing="xy")
    np.testing.assert_allclose(eta, np.repeat(E[..., None], 4, -1), atol=1e-6)
    np.testing.assert_allclose(gamma, np.repeat(G[..., None], 4, -1), atol=1e-6)


def test_ks_encoding_is_exact():
    """Equal-size samples give KS = k/m: stored numerators decode to the same floats."""
    g = torch.Generator().manual_seed(0)
    n, T = 68, 560
    x = ev.Samples.full(torch.randint(0, 20, (50, n), generator=g))
    y = ev.Samples.full(torch.randint(0, 20, (1, n), generator=g))
    xl = ev.Samples.full(torch.rand(50, T, generator=g))
    yl = ev.Samples.full(torch.rand(1, T, generator=g))
    ks = torch.stack([ev.ks_2samp(x, y)[:, 0]] * 3 + [ev.ks_2samp(xl, yl)[:, 0]], -1)
    den = np.array([n, n, n, T], float)
    back = ls.encode_ks(ks, den) / den
    np.testing.assert_allclose(back, ks.numpy(), rtol=0, atol=1e-15)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton kernel)")
@pytest.mark.skipif(not DATA.exists(), reason="needs data/griffa")
def test_run_subject_end_to_end():
    grid = ls.Grid(eta_lo=-3, eta_hi=-1, n_eta=3, gamma_lo=0, gamma_hi=2, n_gamma=2, repeats=5)
    res = ls.run_subject("033", 0, grid, torch.device("cuda"), batch=7, log=lambda *a: None)
    meta = __import__("json").loads(res["meta"])
    assert meta["validation"]["ok"] and meta["validation"]["R_max_rel_err"] < 1e-4
    assert res["ks_num"].shape == (2, 3, 5, 4)
    buf = io.BytesIO(ls.to_bytes(res))
    L = ls.load(buf)
    assert np.all((L["ks"] > 0) & (L["ks"] <= 1))
    np.testing.assert_array_equal(L["energy"], L["ks"].max(-1))

    # re-simulate one batch with the same seed and score it independently from A alone
    from src.kernel import TritonGNM
    D, target, _ = ls.load_griffa("033")
    A_t = target(0)
    T = int(A_t.sum()) // 2
    eta, gamma = grid.params()
    torch.manual_seed(0 * 1_000_003 + 0 * 1009 + 0)
    D1 = D.clone().cuda()
    D1.fill_diagonal_(1.0)
    sim = TritonGNM(D1, eta[:7].cuda(), gamma[:7].cuda())
    sim.run(T)
    ref = ev.evaluate(sim.A.cpu(), D, ev.Targets.from_adjacency(A_t, D))
    np.testing.assert_allclose(L["ks"].reshape(-1, 4)[:7], ref.ks[:, 0].numpy(), atol=1e-12)
