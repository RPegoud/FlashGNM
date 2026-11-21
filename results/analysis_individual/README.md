# Individual differences, noise floor (SPEC H3, H5, H6): Griffa, 70 subjects x 5 scales

No new simulations: the per-repeat landscapes of `landscape_griffa_full` (51 x 51 grid,
step 0.1, 20 repeats; `modal volume get flashgnm-landscapes griffa_full results/griffa_raw`)
and the real connectomes.

    python src/analysis_individual.py results/griffa_raw     # -> summary.json, per_subject.npz
    python src/noise_floor.py --mem-gb 0.4                    # -> ../noise_floor/

## H5: between-subject spread of the optimum vs within-subject noise

Within = bootstrap SD of the argmin over repeats (500 resamples), plus grid quantisation
(step^2 / 12). Noise share = within^2 / between^2 (capped at 1). Q: Cochran's heterogeneity
test, df 69 (small p = subjects differ beyond noise).

| scale | n | eta between SD | eta within SD | eta noise share | eta Q p | gamma between SD | gamma within SD | gamma noise share | gamma Q p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 033 | 68 | 0.059 | 0.058 | 96% | 0.15 | 0.38 | 0.33 | 76% | 1e-4 |
| 060 | 114 | 0.034 | 0.039 | 100% | 0.99 | 0.22 | 0.18 | 65% | 5e-5 |
| 125 | 219 | 0.049 | 0.041 | 69% | 9e-7 | 0.22 | 0.14 | 41% | 8e-26 |
| 250 | 448 | 0.017 | 0.030 | 100% | 1.00 | 0.13 | 0.09 | 51% | 2e-9 |
| 500 | 1000 | 0.033 | 0.030 | 85% | 0.38 | 0.13 | 0.07 | 32% | 1e-21 |

- eta: subjects are indistinguishable from estimation noise at 4 of 5 scales. At n >= 448
  the within SD is mostly grid quantisation (step 0.1): a finer grid is needed before
  reading anything into eta there (planned: `griffa_fine250/500`).
- gamma: Q flags heterogeneity at every scale, BUT Q is miscalibrated: it is equally
  significant on the synthetic-subject null (p ~ 1e-20, `../analysis_null/`), where every
  "subject" has the same true parameters. The bootstrap over repeats misses the main noise
  source, which is that the target is itself one random network. Use the null comparison
  (`../FINDINGS.md`) instead: real subjects spread no more than the null at n <= 219, and
  more than it only at n = 1000 (gamma SD 0.13 vs 0.07, F p 8e-7).

## eta-gamma correlation across subjects (cf. Akarca et al. 2021)

| scale | corr, replicated optima | corr, single-run optima (mean [min, max] over 20 runs) | median within-subject bootstrap corr |
|---|---:|---|---:|
| 033 | +0.79 | +0.54 [+0.05, +0.84] | +0.88 |
| 060 | +0.79 | +0.59 [+0.29, +0.74] | +0.51 |
| 125 | +0.81 | +0.46 [+0.27, +0.64] | +0.47 |
| 250 | +0.21 (p 0.08) | +0.38 [+0.03, +0.56] | +0.26 (9 subjects defined) |
| 500 | +0.19 (p 0.11) | +0.14 [+0.00, +0.34] | 0.00 |

At n = 68 the noise in a single subject's optimum runs along the same direction (+0.88) as
the cross-subject correlation (+0.79), so the correlation is what noise along the valley
would produce. The synthetic-subject null (`modal_landscape.py --null`) tests this directly.
Sign: Akarca et al. report a negative eta-gamma correlation; ours is positive. Check the
conventions (matching-index denominator, offset) before comparing.

## H3: one simulation per cell, and coarse grids

Share of single-run argmins (one repeat per cell) inside the replicated near-optimal set;
median |error| of gamma.

| scale | full grid (0.1) | every 2nd cell (0.2) | every 5th cell (0.5) |
|---|---|---|---|
| 033 | 44%, 0.5 | 48%, 0.5 | 7%, 1.2 |
| 060 | 46%, 0.3 | 24%, 0.4 | 0%, 1.1 |
| 125 | 59%, 0.2 | 47%, 0.2 | 53%, 0.3 (a grid point falls on the optimum) |
| 250 | 59%, 0.1 | 71%, 0.1 | 3%, 0.4 |
| 500 | 64%, 0.1 | 11%, 0.1 | 0%, 0.3 |

## H6: empirical noise floor (`../noise_floor/summary.json`)

max(KS) between every pair of real subjects (same distances and evaluation code), against
the model's best mean energy per subject.

| scale | n | empirical median [q10, q90] | model median | model / empirical (per subject, median) | edge-length KS, empirical median | subjects with model below their empirical median |
|---|---:|---|---:|---:|---:|---:|
| 033 | 68 | 0.162 [0.118, 0.235] | 0.229 | 1.5 | 0.044 | 1/70 |
| 060 | 114 | 0.140 [0.096, 0.219] | 0.236 | 1.8 | 0.040 | 1/70 |
| 125 | 219 | 0.110 [0.073, 0.178] | 0.268 | 2.6 | 0.041 | 0/70 |
| 250 | 448 | 0.092 [0.058, 0.145] | 0.303 | 3.5 | 0.041 | 0/70 |
| 500 | 1000 | 0.081 [0.049, 0.132] | 0.330 | 4.3 | 0.041 | 0/70 |

The empirical floor falls with resolution (as Oldham et al. report), while the model's
best energy rises, driven by edge length (model edge-length KS 0.19 -> 0.33 against 0.04
between subjects). The model is not at the noise ceiling at any scale; the gap grows with n.
