# Rule forms (SPEC H9): Griffa n = 68 and 219, 70 subjects

**Kernel:** `src/kernel_additive.py`.
- Conventions are those of Oldham, Fornito & Ball 2025, checked against their MATLAB
  (StuartJO/ComingUpShort @ 800bea9): maxima over current non-edges after every edge,
  eps = 1e-6 added to K before the power, no offset, empty seed.
- 61 GPU tests pass (`tests/test_kernel_additive.py`).
- Every batch passes full-run validation: row sums within 1e-14 of an exact recompute,
  maxima against brute force.
- Per-subject `.npz` are on the Modal volume `flashgnm-landscapes/<run>`.

**Rules:** additive exponential (`add_exp`, eta, gamma, alpha), multiplicative exponential
(`mult_exp`), multiplicative power law (`mult_pow`).

**Pipeline:**
1. wide replicated grids over the published ranges (`*_wide`);
2. zoomed replicated grids around the optima (`*_zoom`);
3. held-out energy and recovery by distance from 100 fresh networks at each zoom optimum
   (`recovery/`);
4. Oldham et al.'s Voronoi search, one simulation per point, 10,000 per subject
   (`vor_*`), with held-out energy at its optima (`recovery_vor/`).

Summaries: `python src/analysis_rules.py results/rules/<run>`, then
`python src/compare_voronoi.py compare`.

## 1. Grids

| rule | n | wide grid: best mean energy | zoom grid: best mean energy | zoom optimum (median) | near-opt cells (zoom) | near-opt width (zoom): eta / gamma / alpha |
|---|---:|---:|---:|---|---:|---|
| add_exp | 68 | 0.215 | 0.197 | eta -0.06, gamma 2.3, alpha 12 | 126 / 9261 | 0.18 / 2.55 / 28.5 |
| add_exp | 219 | 0.173 | 0.153 | eta -0.12, gamma 1.7, alpha 6 | 54 / 9261 | 0.21 / 1.05 / 25.5 |
| mult_exp | 68 | 0.269 | 0.212 | eta -0.075, gamma 0.42 | 5 / 1681 | 0.023 / 0.09 |
| mult_exp | 219 | 0.308 | 0.214 | eta -0.06, gamma 0.34 | 2 / 1681 | 0.008 / 0 |
| mult_pow | 68 | 0.258 | 0.204 | eta -4.0, gamma 0.40 | 13 / 1681 | 1.4 / 0.10 |
| mult_pow | 219 | 0.280 | 0.194 | eta -2.9, gamma 0.40 | 3 / 1681 | 0.18 / 0 |

Zoom ranges:
- **add_exp:** eta [-0.3, 0], gamma [0.5, 3.5], alpha [0, 30] (alpha beyond the published
  10, since the wide optimum sat on that edge), 21^3 points, 10 repeats.
- **mult_exp:** eta [-0.3, 0], gamma [-1.5, 2], 41^2 points, 20 repeats.
- **mult_pow:** eta [-8, -1], gamma [-0.5, 1.5], 41^2 points, 20 repeats.

All zoom optima are interior except 2/70 add_exp at n = 68.

- **The multiplicative rules sit on a narrow ridge in gamma (~0.1 wide).** The wide grid's
  gamma step (0.5) missed it, and their best energy dropped 0.05-0.09 once zoomed.
- **The additive rule is broad in alpha.** The near-optimal set covers 85-95% of [0, 30].

## 2. Held-out energy and recovery at the zoom optima (100 fresh networks each)

| rule | n | held-out energy [IQR] | grid min -> held-out | KS deg / clu / bet / len | recovery | > 60 mm top-25% hit vs base |
|---|---:|---|---|---|---:|---|
| add_exp | 68 | 0.233 [0.223, 0.246] | +0.036 | .159 / .194 / .182 / .136 | 0.36 | 0.21 vs 0.16 |
| mult_exp | 68 | 0.220 [0.208, 0.228] | +0.008 | .174 / .178 / .168 / .181 | 0.47 | 0.24 vs 0.16 |
| mult_pow | 68 | 0.219 [0.205, 0.234] | +0.015 | .166 / .188 / .167 / .164 | 0.48 | 0.24 vs 0.16 |
| add_exp | 219 | 0.180 [0.172, 0.185] | +0.027 | .124 / .142 / .144 / .084 | 0.28 | 0.072 vs 0.050 |
| mult_exp | 219 | 0.216 [0.205, 0.229] | +0.002 | .140 / .177 / .115 / .203 | 0.34 | 0.078 vs 0.050 |
| mult_pow | 219 | 0.196 [0.191, 0.202] | +0.002 | .175 / .142 / .141 / .177 | 0.34 | 0.073 vs 0.050 |

Paired comparisons of held-out energy (Wilcoxon over 70 subjects):

| n | add_exp vs mult_exp | add_exp vs mult_pow | mult_pow vs mult_exp |
|---|---|---|---|
| 68 | **add worse** in 58/70 (p 6e-9) | **add worse** in 53/70 (p 7e-7) | no difference (p 0.11) |
| 219 | **add better** in 70/70 (p 4e-13) | **add better** in 62/70 (p 7e-11) | pow better in 70/70 (p 4e-13) |

- **Which rule fits best depends on resolution.** The additive rule's apparent advantage at
  n = 68 is selection bias: its flat 3-D landscape inflates the grid minimum by 0.036,
  against 0.008-0.015 for the 2-D rules. On fresh networks it is worse than both
  multiplicative forms there. At n = 219 it is clearly best, mainly because it fits edge
  length (KS 0.084 vs 0.18-0.20).
- **No rule form recovers which edges exist.** Recall equals chance (the model's own edge
  density) in every distance bin, for every rule, at both resolutions:

  | n | rule | recall, bins 0-20 / 20-40 / 40-60 / 60-80 mm | chance |
  |---|---|---|---|
  | 219 | add_exp | .654 .396 .197 .099 | .657 .382 .185 .093 |
  | 219 | mult_pow | .776 .572 .237 .066 | .781 .563 .227 .067 |

  The top 25% most probable > 60 mm pairs exist barely above the base rate for all three
  forms. Leave-one-out group consensus reaches 0.57 (n = 68) and 0.19 (n = 219) on the
  same test. **The additive form improves the max(KS) fit, not which connections are
  recovered;** Oldham et al.'s conclusion holds across rule forms.
- Node degree doesn't correspond for any rule (Spearman -0.01 to -0.08).

## 3. Oldham et al.'s Voronoi search vs replicated grids (`voronoi_vs_grid.md`)

Same rules, published ranges, 5 levels x 2,000 points, one simulation per point.

| rule | n | search: reported energy | search: held-out at its optimum | grid: held-out at its optimum | search worse (subjects) | p |
|---|---:|---:|---:|---:|---:|---:|
| add_exp | 68 | 0.147 | 0.279 | 0.233 | 64/70 | 3e-12 |
| mult_exp | 68 | 0.191 | 0.251 | 0.220 | 56/70 | 1e-9 |
| mult_pow | 68 | 0.147 | 0.251 | 0.219 | 59/70 | 5e-10 |
| add_exp | 219 | 0.114 | 0.204 | 0.180 | 56/70 | 3e-9 |
| mult_exp | 219 | 0.221 | 0.243 | 0.216 | 56/70 | 2e-9 |
| mult_pow | 219 | 0.169 | 0.209 | 0.196 | 51/70 | 5e-7 |

- **The search's reported fit is optimistic by 0.02-0.13:** it reports the luckiest of
  10,000 single runs.
- **Its optima are worse than the replicated grid's** for 51-64/70 subjects, in every case.
  - For add_exp at n = 68 they are worse by 0.046 in median.
  - Part of that is the range: the grid optimum has alpha ~12, outside the searched [0, 10].
- **It can manufacture cross-subject structure.** mult_exp at n = 68:
  - 2/70 searches landed in a distant basin (eta ~ -2, gamma ~ -4 to -6);
  - those two make the cross-subject eta-gamma correlation +0.99 (grid: -0.47);
  - without them it is -0.59 (Spearman, all 70: -0.42).
  A Pearson correlation of single-search optima, as used for individual differences, can
  be flipped by two subjects.

## Caveats

- Griffa (deterministic tractography, Lausanne), Euclidean template distances; Oldham et
  al. used HCP Schaefer-400 with mean vertex-pair distances. Same model code conventions,
  different data.
- Held-out energies use 100 networks at one point per subject (SE ~0.003).
- n = 448 and 1000 not run for the rule forms (cost: ~$5 and ~$25 per replicated grid
  pass with the fp64 kernel).
