# Rule forms (SPEC H9): wide replicated grids, Griffa n = 68 and 219, 70 subjects

Kernel: `src/kernel_additive.py` (Oldham, Fornito & Ball 2025 conventions, checked against
their MATLAB, StuartJO/ComingUpShort @ 800bea9: maxima over current non-edges after every
edge, eps = 1e-6 added to K before the power, no offset, empty seed). 61 GPU tests pass
(`tests/test_kernel_additive.py`). Every batch passes full-run validation (row sums
within 1e-14 of an exact recompute). Per-subject `.npz`: Modal volume
`flashgnm-landscapes/{add_exp_wide, mult_exp_wide, mult_pow_wide}`.

Grids span Oldham et al.'s published ranges: eta in [-2, 0] (exponential) or [-10, 0] (power
law), gamma in [-10, 10], alpha in [0, 10].
- **add_exp:** 21 x 21 x 21 points (eta step 0.1, gamma step 1, alpha step 0.5), 10 repeats.
- **mult_exp, mult_pow:** 41 x 41 points (eta step 0.05 or 0.25, gamma step 0.5), 20 repeats.

`summary.json` per run; `python src/analysis_rules.py results/rules/<run>`.

| rule | n | best mean energy | eta opt [IQR] | gamma opt | alpha opt [IQR] | near-opt cells | interior | binding KS (most common) |
|---|---:|---:|---|---|---|---:|---:|---|
| add_exp | 68 | 0.215 | -0.10 [-0.47, -0.10] | 2.0 | 3.0 [0.5, 4.5] | 62 / 9261 | 63/70 | betweenness 46, clustering 17 |
| add_exp | 219 | 0.173 | -0.10 | 2.0 | 9.5 [9.0, 10.0] | 8 / 9261 | 44/70 | betweenness 65 |
| mult_exp | 68 | 0.269 | -0.10 | 0.5 | - | 2 / 1681 | 69/70 | edge length 33, clustering 31 |
| mult_exp | 219 | 0.308 | -0.10 [-0.24, -0.10] | 0.0 [-0.5, 0.0] | - | 1 / 1681 | 70/70 | edge length 48, clustering 21 |
| mult_pow | 68 | 0.258 | -5.0 [-5.75, -4.5] | 0.5 | - | 11 / 1681 | 70/70 | clustering 35, edge length 23 |
| mult_pow | 219 | 0.280 | -3.6 [-4.9, -3.3] | 0.5 [0.0, 0.5] | - | 4 / 1681 | 70/70 | degree 29, edge length 23 |

For comparison, the toolbox-convention power law (`../landscape_griffa_full`): 0.229
(n = 68) and 0.268 (n = 219), at eta -1.8 / -1.5 and gamma 2.3 / 1.85.

## Preliminary readings (coarse grids; not final)

- **The additive rule fits best on max(KS), and better at n = 219 than at n = 68.** That
  is the opposite of every multiplicative form, which get worse with resolution. With it,
  edge length stops binding and betweenness becomes the limit.
  - Caveat: the additive minimum is taken over 9261 noisy cells (10 repeats), the 2-D
    ones over 1681 (20 repeats), so it is biased lower. A held-out energy at each optimum
    (fresh networks, `recovery.run_rule_subject`) is implemented but not yet run.
- **Its optimum is poorly identified.**
  - alpha: the near-optimal set spans 9 to 9.5 of the 10 alpha units.
  - eta: it spans 1.8 units at n = 68.
  - At n = 219, 26/70 optima sit on the boundary at alpha = 10, so the published range
    truncates the optimum.
- **The exponential distance rules choose eta = -0.1/mm,** the grid point next to 0. That
  is a ~10 mm decay length; most of the published range [-2, 0] is irrelevant on this data.
- **Conventions move the optimum a lot.** The same multiplicative power-law rule fitted
  with Oldham's conventions lands at eta -5.0, gamma 0.5, energy 0.258. With the
  toolbox's (+1e-6 offset on P, K = 0 replaced by 1e-6) it lands at eta -1.8, gamma 2.3,
  energy 0.229. A likely cause:
  - with gamma ~ 2 and an empty start, the toolbox's D^eta K^gamma term is ~1e-14, so the
    offset dominates and the first edges are placed almost uniformly at random;
  - Oldham's version has no offset, so its first edges follow distance.
  This is not verified yet.

## Not run (Modal workspace disabled: credits exhausted)

1. Zoomed grids around each optimum:
   - add_exp: eta [-0.3, 0], gamma [0.5, 3.5], alpha [0, 30];
   - mult_exp: eta [-0.3, 0], gamma [-1.5, 2];
   - mult_pow: eta [-8, -1], gamma [-0.5, 1.5].
2. Held-out energy and recovery by distance at each rule's optimum
   (`modal run src/modal_rules.py::recovery`, optima in `optima.json`; rerun
   `analysis_rules.py` on the zoom runs first to update it).
3. Oldham's Voronoi search (`modal run src/modal_rules.py::voronoi`), one run per point,
   10,000 per subject, against the replicated grids.
