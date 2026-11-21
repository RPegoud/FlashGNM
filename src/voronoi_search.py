"""Oldham et al.'s Voronoi landscape search (SPEC H3, H9), ported from their MATLAB.

StuartJO/ComingUpShort @ 800bea9, code/GNM/VoronoinLandScape_GNM.m and
fcn_voronoi_selectn.m (same procedure as the 2022 repo):
- level 1: `ndraw` points uniform in the parameter box;
- levels 2..nlvl: Voronoi tessellation of ALL points evaluated so far, in coordinates
  rescaled to [0, 1] per free parameter; each of `ndraw` new points picks a cell with
  probability proportional to maxKS ** -p, p = linspace(0, pow, nlvl)[level], and is drawn
  uniformly inside that cell (clipped to the unit cube);
- one simulation per point; defaults ndraw = 2000, nlvl = 5, pow = 2 (10,000 networks).

Uniform draw inside a cell: rejection from the bounding box of the cell's (clipped)
vertices, accepting a proposal whose nearest generator is the cell's own (the definition
of the Voronoi cell). Their code tests the cell's half-planes (convhulln) instead; same
region. As in theirs, after 100 * (draws from that cell) failed proposals the last one is
kept. Unbounded cells use the whole unit cube as the box.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial import Voronoi, cKDTree


def _cell_boxes(vor: Voronoi, npts: int, d: int) -> np.ndarray:
    """[npts, 2, d] bounding box per generator's cell, clipped to [0, 1]."""
    boxes = np.zeros((npts, 2, d))
    boxes[:, 1] = 1.0
    for p in range(npts):
        reg = vor.regions[vor.point_region[p]]
        if len(reg) == 0 or -1 in reg:
            continue  # unbounded: whole cube
        v = np.clip(vor.vertices[reg], 0, 1)
        boxes[p, 0], boxes[p, 1] = v.min(0), v.max(0)
    return boxes


def _draw_in_cells(pts: np.ndarray, cells: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    d = pts.shape[1]
    if d == 1:
        boxes = np.zeros((len(pts), 2, 1))
        order = np.argsort(pts[:, 0])
        s = pts[order, 0]
        mid = np.r_[0.0, (s[1:] + s[:-1]) / 2, 1.0]
        boxes[order, 0, 0], boxes[order, 1, 0] = mid[:-1], mid[1:]
    else:
        # joggle duplicates (Qhull fails on them); they'd share a cell anyway
        vor = Voronoi(pts, qhull_options="QJ" if d > 2 else "Qbb Qc Qz QJ")
        boxes = _cell_boxes(vor, len(pts), d)
    tree = cKDTree(pts)
    out = np.empty((len(cells), d))
    counts = np.bincount(cells, minlength=len(pts))
    for c in np.flatnonzero(counts):
        idx = np.flatnonzero(cells == c)
        lo, hi = boxes[c]
        got, tries, x = 0, 0, None
        max_tries = 100 * counts[c]
        while got < len(idx):
            x = rng.uniform(lo, hi, size=(max(8, 2 * (len(idx) - got)), d))
            _, nn = tree.query(x)
            ok = x[nn == c]
            take = ok[: len(idx) - got]
            out[idx[got:got + len(take)]] = take
            got += len(take)
            tries += len(x)
            if tries >= max_tries and got < len(idx):
                out[idx[got:]] = x[-1]  # as in their code: keep the last proposal
                break
    return out


def search(evaluate: Callable[[np.ndarray], np.ndarray], lo, hi, ndraw: int = 2000,
           nlvl: int = 5, pow_: float = 2.0, seed: int = 0, log=lambda *a: None):
    """evaluate(params [k, d] in original units) -> maxKS [k]. lo, hi [d]: bounds; a
    parameter with lo == hi is fixed. Returns (params [ndraw * nlvl, d], maxKS, level)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    free = hi > lo
    d = int(free.sum())
    rng = np.random.default_rng(seed)
    pows = np.linspace(0, pow_, nlvl)
    unit = np.empty((0, d))
    f = np.empty(0)
    levels = np.empty(0, int)

    def to_params(u):
        p = np.tile(lo, (len(u), 1))
        p[:, free] = lo[free] + u * (hi[free] - lo[free])
        return p

    for lvl in range(nlvl):
        if lvl == 0:
            new = rng.uniform(0, 1, size=(ndraw, d))
        else:
            w = f ** -pows[lvl]
            cells = rng.choice(len(unit), size=ndraw, p=w / w.sum())
            new = _draw_in_cells(unit, cells, rng)
        fn = np.asarray(evaluate(to_params(new)), float)
        unit = np.vstack([unit, new])
        f = np.r_[f, fn]
        levels = np.r_[levels, np.full(ndraw, lvl)]
        log(f"  level {lvl + 1}/{nlvl}: best maxKS {f.min():.4f}")
    return to_params(unit), f, levels
