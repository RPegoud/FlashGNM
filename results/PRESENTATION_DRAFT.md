# Presentation draft: 5 min (3 min intro + kernel, 2 min science)

Data: Griffa et al., 70 subjects, Lausanne 68 to 1000 regions, deterministic tractography,
Euclidean template distances. Every number below is in `results/` (sources in brackets).

## 1. Claims, graded

**A: say it plainly.** Large effects, paired over 70 subjects, simple metrics, validated
runs.

| # | Claim | Numbers | Caveat to say if asked |
|---|---|---|---|
| A1 | The kernel is orders of magnitude faster than the toolbox on the same GPU | n = 1000, B = 1024: 2,217 vs ~0.1 sims/s (~20,000x); n = 68: 237k vs 336 sims/s (~700x) | Toolbox times are extrapolated from timed probe steps; same H100 [`26-11-21_*_sim_*`] |
| A2 | Every result run is validated, and the kernel matches the reference implementations | Full-run checks on every batch; 61 GPU tests for the rule kernel; reference = literal port of Oldham's MATLAB to 1e-9; classic kernel statistically indistinguishable from the toolbox (300 networks each, all KS p > 1e-3) | Toolbox check: AAL, 90 regions, one parameter setting. Oldham check: our port of their formula, not their code run |
| A3 | Fitted models match the *distributions*, not the *wiring*: recall equals chance in every distance bin | Classic rule, n = 68 to 1000; all three rule forms at n = 68, 219; leave-one-out group consensus is 3x to 125x above chance at > 60 mm [`recovery/`, `rules/recovery/`]. The consensus network recovers ~10x more edges yet has about the same energy as the model (n = 1000: 0.327 vs 0.330); at n = 219 the additive model *beats* it on energy for 50/70 subjects [`consensus_energy/`] | "Chance" = the model's own edge density in that bin. Euclidean distances. |
| A4 | Single-run fitting overstates the fit, most for the flexible rule; replicating each point fixes it | Oldham-style search: finds the right region, but the energy it reports is 0.02 to 0.14 lower than fresh networks deliver (median per subject); the optimism is largest for the additive rule (+0.143 at n = 68, +0.086 at n = 219 vs +0.019 to +0.104), so reported energies exaggerate its advantage. **The same search with 20 simulations per point** (200k per subject, enabled by the kernel): optimism +0.015, and its pick matches the replicated grid (additive, n = 219: held-out 0.179 vs 0.180, p 0.57). At equal budget (5 x 400 x 5), replication makes fits honest (optimism +0.002 to +0.046) but can miss narrow optima (multiplicative-exponential rule much worse) [`rules/voronoi_vs_grid.md`, `rules/README.md` §4] | Our port of their search, on our data. 20-repeat search run for the additive rule only. Don't claim it finds the wrong region: it doesn't |

**B: say it with the qualifier attached.**

| # | Claim (with qualifier) | Numbers |
|---|---|---|
| B1 | At 68 regions, synthetic subjects with *identical* parameters reproduce the cross-subject eta-gamma correlation and spread | Three null placements (cohort median, 10th and 90th percentile of gamma): correlation +0.78 to +0.90 (real +0.79); gamma SD 0.63 to 0.71 (real 0.38); real spread never exceeds the null (p = 1.0). Real spread exceeds the null only at 1000 regions (0.13 vs 0.07, p 8e-7) [`analysis_null*/`]. Qualifiers: toolbox conventions; our correlation is *positive*, Akarca's negative, so this is about the method, not a replication of their result. |
| B2 | The fit gets worse relative to real between-subject variability as resolution rises, for every rule form tested | Classic rule: 1.5x the floor at n = 68, 4.3x at n = 1000 [`noise_floor/`]. Oldham's rules: 1.35-1.44x at n = 68, 1.64-1.97x at n = 219. Qualifier: rule forms tested only up to 219. |
| B3 | Which rule fits best depends on resolution | Held-out energy, additive vs best multiplicative: worse at n = 68 (53/70, p 7e-7), better at n = 219 (62/70, p 7e-11) [`rules/README.md`]. Qualifier: two resolutions only. |
| B4 | Coarse grids miss the multiplicative rules' optimum | The ridge in gamma is ~0.1 wide; zooming improved the best energy by 0.05-0.09. |

**C: don't claim (or only as an aside).**

- "Akarca's individual differences are noise". We didn't use their data, conventions or
  sign. Say B1 instead.
- "The additive rule fits best". Only at n = 219, and its n = 68 lead was selection bias.
- "Oldham's long-range result holds at 1000 regions for *their* rule". Only the classic
  rule was run at 1000; their rule up to 219.
- The toolbox-vs-Oldham convention difference "is caused by the offset". The optimum
  moves, but the mechanism is unverified.
- Anything from the Cochran Q test: retracted (miscalibrated, see the null).

## 2. Structure and script

**0:00-3:00 Intro and kernel: the existing clips** (`presentation-anims/presentation/output`)

| Time | Clip | Line |
|---|---|---|
| 0:00 | clip1_wiring | "Brains wire by cost and by similarity. GNMs grow a network edge by edge to test which rules reproduce real connectomes." |
| 0:30 | clip2_naive | "Each step samples one edge from n^2 probabilities: fitting needs millions of networks, so studies use one run per parameter." |
| 1:00 | clip3_chain, clip4_matrices | "Only rows i and j change per step: keep row sums, sample a row, then a column. O(n) per step." |
| 1:45 | clip5_kernel | "One fused GPU kernel per simulation, state cut from 9 to 3 bytes per pair, thousands of simulations in parallel." |
| 2:30 | clip6_bench | "~20,000x the toolbox at 1000 regions [A1]. Everything validated [A2]. So we can replicate every fit." |

**3:00-5:00 Science: two results and one line**

| Time | Beat | Animation | Line |
|---|---|---|---|
| 3:00 | **One run per point fools you [A4, B1]** | S1, then S2 | "We grew 70 synthetic subjects with *identical* parameters. Fitted the usual way, they show the same eta-gamma correlation as real subjects. And Oldham's single-run search finds the right region, but reports a fit fresh networks can't reproduce, most of all for the flexible additive rule. Run the same search with 20 simulations per point, which our kernel makes cheap, and the reported fit becomes honest and matches a full replicated grid." |
| 4:05 | **Distributions, not wiring [A3]** | S3 | "At the best fit, every model, classic or additive, at every resolution, places edges no better than chance in each distance bin. Other people's brains predict the edges ten times better, yet score the same energy. The metric rewards statistics, not wiring." |
| 4:45 | Close [B2, B3] | static | "And the gap to real variability grows with resolution. Questions we can now afford to ask: dense tractography, per-subject uncertainty." |

## 3. Animations for the science part (3 picks, same style as clips 1-6)

All numbers should come from an export script reading `results/`, as in `bench.tsx`.

**S1: "Synthetic twins" (~20 s, B1)**
- **Beat A:** 70 dots, the real subjects' optima at n = 68, in (eta, gamma).
  - Correlation counter +0.79.
  - Thin bootstrap ellipses show each subject's own uncertainty: they are elongated along
    the same diagonal.
- **Beat B:** a star at the true parameters (-1.8, 2.3).
  - 70 dots burst from the star and settle where the fits land: same spread along the
    valley, correlation counter +0.90.
- **Beat C (optional):** the same at n = 1000: real dots spread wider than the null's
  (SD 0.13 vs 0.07). "Here, individuals do differ."
- **Data:**
  - `results/analysis_individual/per_subject.npz` and `results/analysis_null/per_subject.npz`
    (keys `033_eta`, `033_gamma`, `033_boot_eta`, `033_boot_gamma`; `500_*`);
  - `NULL_PARAMS` in `src/landscape.py`.

**S2: "Luckiest of 10,000" (~25 s, A4)**

Revised: the earlier version ended on "the grid picks a better point". On the full-range
plane the two optima overlap (subject 42: search (-0.197, 1.15), grid (-0.225, 1.40)),
which reads as agreement. The search does find the right region; the critique is the
reported fit, and that it flatters the flexible rule most.
- **Subject:** subject 42, n = 219, additive rule (typical: median gap and median
  optimism).
- **Beat A (keep):** five waves of 2,000 points fill the (eta, gamma) plane, coloured by
  level, concentrating near the valley.
- **Beat B (keep):** the best single point pulses, labelled "reported 0.119".
- **Beat C (keep):** 100 fresh networks at that point drop as a band: mean 0.205. Caption:
  "its reported fit was luck".
- **Beat D (replace the grid-optimum beat):** say it plainly, "the search found the right
  region". Then three pairs of bars, one per rule at n = 219: reported energy vs fresh
  networks.
  - additive 0.114 -> 0.204 (+0.090); power law 0.169 -> 0.209 (+0.040); exponential
    0.221 -> 0.243 (+0.023).
  - Caption: "luck flatters the flexible rule most: its advantage looks 3x bigger than it
    is".
- **Beat E (optional, n = 68):** the reported ranking (additive tied best) flips on fresh
  networks (additive worst: 0.233 vs 0.219 and 0.220).
- **Beat F (closing, the positive note):** "same search, 20 simulations per point". Re-run
  the waves for subject 42 with each dot averaged; the best point pulses "reported 0.160";
  fresh networks land at 0.171, so the two bars almost touch.
  - Caption: "replicate every point and the search tells the truth: optimism 0.015
    (median), same optimum as a full replicated grid".
  - Numbers per subject 42: single run 0.119 -> 0.205; 20 repeats 0.160 -> 0.171.
  - Data: `results/rules/vorB_add_exp/scale125/sub42.json` (`params`, `energy`,
    `energy_sd`, `level`; also on the Modal volume `vorB_add_exp`),
    `_reported` in `results/rules/optima_vorB.json`, held-out in
    `results/rules/recovery_vorB/scale125.jsonl`.
  - Optional footnote: at their 10,000-simulation budget, replication alone isn't enough;
    with fewer points the search can miss narrow optima. Speed provides both.
- **Data:**
  - `results/rules/vor_add_exp/scale125/sub42.json` (`params`, `ks`, `level`; also on the
    Modal volume `vor_add_exp`);
  - reported energies: `_reported` in `results/rules/optima_vor.json`;
  - held-out at the search's optima: `results/rules/recovery_vor/scale{033,125}.jsonl`;
    at the grid's optima: `results/rules/recovery/scale{033,125}.jsonl`. Take medians
    over 70 subjects.
  - Only the mean and SD of held-out energies are stored. Draw mean ± SD, or regenerate the
    100 per-network energies (seconds on GPU) if a histogram is wanted.

**S3: "Recall = chance" (~25 s, A3)**
- **Beat A:** distance bins on x (0-20 ... >120 mm). Grey bars show chance; model recall
  bars grow and stop exactly at the grey tops.
- **Beat B:** cycle with a short crossfade: classic rule n = 68 -> 219 -> 1000, then
  additive and multiplicative at n = 219. The model bars track the grey every time.
- **Beat C:** group-consensus bars shoot far above chance, most strikingly at long range.
  - Caption: "> 60 mm: top-25% predicted edges exist 0.8% vs 0.7% base rate (model) vs
    2.6% (other brains), n = 1000".
- **Data:**
  - `results/recovery/summary.json` (`recall_per_bin`, `chance_per_bin` per model and
    scale);
  - `results/rules/recovery/scale{033,125}.jsonl` (per subject: take medians).

**Optional, if time:** a "replicated landscape" bridge (~10 s): 20 single-run landscapes
flicker, then settle into their mean with the near-optimal set outlined. It sets up S1 and
S2. Data: per-subject `.npz` on the volume `flashgnm-landscapes/griffa_full`.

## 4. Likely questions

- **"Is chance the right baseline?"** It is the model's own density per distance bin, so it
  asks whether the model knows *which* pairs at a given distance connect. A pure
  distance-only model is also at chance; group consensus is far above it.
- **"Your conventions differ from Akarca / the toolbox."** Yes. Toolbox conventions for the
  classic rule; Oldham's verified conventions for H9. The same rule under the two
  conventions moves the optimum from (-1.8, 2.3) to (-4.0, 0.40).
- **"Does the null depend on where you put it?"** No, at n = 68: three placements (the
  cohort median, and the 10th and 90th percentiles of gamma) all give correlations of
  +0.78 to +0.90 and spreads larger than the real cohort.
- **"Isn't group consensus cheating?"** It's a ceiling, not a competitor: it shows the
  wiring is predictable from other brains. It also shows the energy can't tell the
  difference: the consensus network has about the same energy as the model while
  recovering ~10x more edges.
- **"Why not their data?"** HCP Schaefer-400 isn't in the repo. MICA-MICs is packaged and
  would be next.
- **"Speedup vs MATLAB?"** We compared against the PyTorch toolbox, not their MATLAB; don't
  quote a MATLAB number.
