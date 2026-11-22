# Findings: Griffa, 70 subjects x 5 scales (H3, H5, H6, H7, H8 cost), 2026-11-21

All on H100 via Modal; every landscape passes full-run validation (490/490 new runs).
Details: `analysis_individual/README.md` (H3, H5 bootstrap, H6), `analysis_null/compare.md`,
`landscape_griffa_fine{250,500}/summary.md`, `recovery/summary.json`.

| run | what | where (per-subject .npz on Modal volume `flashgnm-landscapes`) |
|---|---|---|
| griffa_null | 70 synthetic subjects x 5 scales: one model network at the scale's median optimum (`NULL_PARAMS`) with each real subject's T; same 51 x 51 x 20 grid | `griffa_null` |
| griffa_fine250/500 | 31 x 31 grid, eta step 0.02, gamma step 0.05, 20 repeats | `griffa_fine250`, `griffa_fine500` |
| recovery | 100 networks at each subject's optimum and at its distance-only optimum (gamma = 0), plus leave-one-out group consensus | `recovery/scale*.jsonl` (in git) |

## H5: individual parameters vs estimation noise (synthetic-subject null)

| scale | n | gamma SD of optima, real / null | eta SD, real / null | corr(eta, gamma), real / null | F p (real gamma var > null) |
|---|---:|---|---|---|---:|
| 033 | 68 | 0.38 / 0.71 | 0.059 / 0.133 | +0.79 / +0.90 | 1.0 |
| 060 | 114 | 0.22 / 0.48 | 0.034 / 0.086 | +0.79 / +0.85 | 1.0 |
| 125 | 219 | 0.22 / 0.26 | 0.049 / 0.065 | +0.81 / +0.66 | 0.90 |
| 250 | 448 | 0.13 / 0.11 | 0.017 / 0.031 | +0.21 / +0.22 | 0.06 |
| 500 | 1000 | 0.13 / 0.07 | 0.033 / 0.026 | +0.19 / +0.35 | 8e-7 |

- **Subjects with identical true parameters reproduce the cross-subject eta-gamma
  correlation** (+0.90 at n = 68), and spread more than the real cohort at n <= 219. At
  Akarca-like resolution, nothing in the optima distinguishes individuals from noise.
- Only at n = 1000 does the real gamma spread exceed the null's (about 1.8x in SD). That
  is the one place individual parameters are supported here.
- The null fit is biased at low n: true gamma 2.3, fitted median 2.55, mean bias +0.37 at
  n = 68; negligible at n >= 219.
- Bootstrapping over repeats underestimates within-subject uncertainty (Cochran Q is
  significant on the null too), because the target is one random network. Noise-aware
  estimates need target-level resampling, not just repeat-level resampling.
- Caveats: one null parameter set per scale, at the cohort's median; the null's noise is
  model-network variability, which may be larger or smaller than real scan-to-scan
  variability. The null uses the coarse grid (step 0.1), so it can't test eta at n >= 448;
  a fine-grid null would (~$3).

## Fine grids at n = 448 and 1000

| scale | eta opt median [IQR] | gamma opt median [IQR] | best energy | near-opt cells (coarse grid) |
|---|---|---|---:|---:|
| 250 | -1.42 [-1.44, -1.40] | 1.50 [1.45, 1.60] | 0.298 | 18 (4) |
| 500 | -1.32 [-1.34, -1.30] | 1.20 [1.15, 1.25] | 0.326 | 13 (3) |

All 140 optima are interior. The coarse optima were right to within one coarse cell.

## H6: noise floor

The empirical floor (max KS between real subjects) falls from 0.16 to 0.08 between n = 68
and 1000 while the model's best energy rises from 0.23 to 0.33 (1.5x to 4.3x the floor).
Edge-length KS is ~0.04 between subjects at every scale, against 0.19 to 0.33 for the model.

## H7: connection recovery by distance (cf. Oldham et al. 2025)

Recall per distance bin; chance = the model's own edge density in that bin (edges placed
at random within the bin). Medians over 70 subjects.

| scale | n | recovery: homophily / distance-only / group | >60 mm top-25% hit rate vs base: homophily / group | degree Spearman: homophily / group |
|---|---:|---|---|---|
| 033 | 68 | 0.33 / 0.46 / 0.83 | 0.20 vs 0.16 / 0.57 | -0.09 / 0.89 |
| 060 | 114 | 0.27 / 0.42 / 0.77 | 0.13 vs 0.10 / 0.36 | -0.10 / 0.75 |
| 125 | 219 | 0.18 / 0.35 / 0.69 | 0.06 vs 0.05 / 0.19 | -0.07 / 0.65 |
| 250 | 448 | 0.10 / 0.28 / 0.61 | 0.024 vs 0.021 / 0.077 | -0.09 / 0.60 |
| 500 | 1000 | 0.05 / 0.20 / 0.50 | 0.008 vs 0.007 / 0.026 | -0.04 / 0.54 |

n = 1000, recall vs chance per bin (0-20, 20-40, 40-60, 60-80, 80-100, 100-120, >120 mm):

| model | recall | chance |
|---|---|---|
| homophily | .090 .051 .030 .021 .015 .012 .009 | .086 .048 .031 .021 .016 .012 .010 |
| distance only | .581 .094 .019 .008 .005 .004 .003 | .493 .074 .018 .008 .005 .004 .003 |
| group consensus | .857 .539 .299 .201 .201 .125 .146 | .495 .087 .018 .006 .004 .001 .002 |

- **At its fitted optimum, the homophily model is at chance in every distance bin, at every
  resolution.** All its overlap with the target comes from matching the distance profile.
  A distance-only model recovers more edges, and has some lift at short range.
- **Long range: Oldham et al.'s result holds at 1000 regions, whole brain.** The 25% most
  probable > 60 mm pairs exist at the base rate (0.8% vs 0.7%). Resolution does not fix it.
- **The long-range edges are predictable**: leave-one-out group consensus is above chance
  in every bin beyond 60 mm, by ~3x at n = 68 and 30-125x at n = 1000. The failure is the model's, not noise in the targets.
- Nodal degree doesn't correspond either (Spearman ~ -0.1): max(KS) matches degree
  distributions, not which regions are hubs.
- Distances are Euclidean template centroids, and Oldham et al. fit an additive
  exponential rule; their rule form (SPEC H9) is the remaining difference.

## H8 cost: dense targets at n = 1000 (B = 1024, H100)

| T (density) | ms/step | sims/s | peak GB |
|---|---:|---:|---:|
| 99,900 (20%) | 0.040 | 251 | 5.7 |
| 149,850 (30%) | 0.043 | 157 | 5.7 |
| 249,750 (50%) | 0.052 | 79 | 5.7 |

The SPEC's extrapolation (220 / 150 / 90 sims/s) holds to within 15%. A 50 x 50 x 20 grid at 50%
is ~11 min of generation per (resolution, density), shared across subjects. Evaluation
not included.

## H9: rule form (additive vs multiplicative), 2026-11-22 (`rules/README.md`)

- Rule kernel with Oldham et al. 2025 conventions, checked against their MATLAB.
- Wide and zoomed replicated grids on 70 subjects at n = 68 and 219.
- Held-out energy from 100 fresh networks at every optimum; their Voronoi search on the
  same subjects.

| | n = 68 | n = 219 |
|---|---|---|
| held-out energy: additive / mult exp / mult power | 0.233 / 0.220 / 0.219 | **0.180** / 0.216 / 0.196 |
| additive vs best multiplicative | worse in 53/70 (p 7e-7) | better in 62/70 (p 7e-11) |
| any rule's recall above chance in any distance bin | no | no |
| Voronoi search (one run per point) worse than replicated grid | 56-64/70 per rule | 51-56/70 per rule |

- **Which rule fits best depends on resolution.** The additive rule's lower grid minimum at
  n = 68 is selection bias from a flat 3-D landscape (+0.036 from grid minimum to held-out).
  At n = 219 it fits genuinely better, through edge length.
- **No rule form recovers which edges exist,** short or long. The additive rule fixes the
  distribution match, not the wiring.
- **Coarse searches miss the multiplicative rules' narrow gamma ridge** (~0.1 wide): best
  energy dropped by 0.05-0.09 when zoomed.
- **Oldham's single-run Voronoi search:**
  - it reports fits 0.02-0.13 better than its optimum delivers on fresh networks;
  - its optima are worse than the replicated grid's for most subjects in every
    rule/scale;
  - 2/70 outlying searches turned a cross-subject eta-gamma correlation from -0.59 into
    +0.99.
- **Conventions matter:**
  - the same multiplicative power law under Oldham vs toolbox conventions: optimum (-4.0,
    0.40) vs (-1.8, 2.3), held-out 0.219 vs ~0.23 at n = 68;
  - at n = 219, 0.196 vs 0.268 (grid minimum).
