# SPEC: fast generative network model fitting on GPU

## 1. Goal

Make fitting generative network models (GNMs) to individual connectomes cheap enough to:

- map the whole energy landscape over (eta, gamma) per subject, with repeats, instead of
  searching it sparsely;
- do it for every subject in a cohort;
- do it at high parcellation resolution (~1000 regions), where current tools are impractical.

Then use that capacity for a scientific question: how well are the parameters identified,
how do they vary across subjects, and do they depend on resolution and modelling choices?

## 2. The model

A binary GNM grows a network one edge at a time, starting empty or from a seed network.

- Wiring probability for a missing edge (i, j): `P_ij ∝ D_ij^eta · K_ij^gamma`, plus a small
  offset. P is restricted to non-edges off the diagonal.
- `D` is the inter-regional distance. `K` is the matching index, with numerator `(AᵀA)_ij`
  (shared neighbours) and a degree-based denominator.
- There are two common conventions for the denominator: the sum of the degrees (BCT) or
  their mean (the GNM toolbox). The handling of zeros and the offset also differs between
  implementations. Pick one, state it, and make the others options.
- A simulation stops after T edges, where T is the target network's edge count.

**Fit.** For each (eta, gamma), compare the simulated networks with the target. The energy
is the maximum of four two-sample KS statistics: degree, clustering, betweenness and edge
length.

## 3. Baselines

- **Reference implementation:** the existing toolbox (PyTorch), unchanged, on the same GPU.
- **Fair dense baseline:** the same maths as the toolbox with validation and host
  syncs removed, so the speedup isn't just "we removed overhead".
- **Credible CPU baseline for evaluation:** a mainstream graph library (e.g. igraph)
  run on all cores.
- **Report wall-clock time for a realistic fit**, not only time per step: simulations per
  second at realistic n, B (batch) and T.

## 4. Simulation: reshaping the algorithm

**Hypothesis H1: sampling, not the matching index, is the bottleneck.** The reference
recomputes `AᵀA`, K and P over all n² entries every step, then samples over n². Updating
the matching index incrementally is known (BCT, Oldham et al.), so don't claim it. We
expect it alone to give little speedup, because sampling over n² still dominates. Profile
this before optimising.

Directions:

1. **Keep `C = AᵀA` up to date incrementally.** Adding edge (a, c) changes only rows and
   columns a and c, by the old rows of A (plus the diagonal). Cost per step is O(n).
2. **Keep P and its row sums up to date.** Only rows and columns a and c of K and P change.
   Store P symmetric and update the row sums by deltas.
3. **Sample in two stages:** pick a row in proportion to the row sums, then a column in
   proportion to that row of P. This gives the same distribution as sampling over all n²
   entries, at O(n) cost.
4. **Use a sampler that suits the GPU.** Either inverse-CDF sampling on a row, or
   `argmax(w / Exp(1))`. Both are exact. The second needs no host sync, so it works under
   graph capture.
5. **Accumulate row-sum deltas in fp64**, casting before subtracting, so drift stays
   negligible without periodic resyncs. Measure the drift over long runs to confirm.
6. **Batch over parameters.** Each simulation in a batch has its own (eta, gamma). This
   turns a sweep into one large batch instead of a Python loop over configurations.
7. **Fuse the step.** In plain PyTorch, an O(n) step is dozens of tiny kernels, so it is
   bound by launch overhead. Write one kernel with one program per simulation, persistent
   over all T steps.
   - Tune the number of warps against n. Check register spills.
   - Expect high-B throughput to be limited by occupancy.
8. **Memory.** Keep dense per-simulation state as small as possible: a boolean A and a narrow
   integer C. P doesn't need to be stored (§4.0). Store the edge list rather than adjacency snapshots, and rebuild
   networks only when needed.
9. **Setup is part of the cost.** Building the initial P one simulation at a time can
   dominate wall time at small n. Do it in blocks of simulations, with block size limited
   by memory at large n.

**Concurrency hazard to design for.** In a kernel where many threads read rows of A and then
write A, source order only orders a single thread. A store with no data dependency on
other threads' loads can land before those loads. Place writes after a reduction that
depends on all of the old values, or add an explicit barrier. This kind of bug never
reproduces in an interpreter and only shows up under high contention on the GPU, so
validate full runs on the GPU (§6).

### 4.0 Changes to the prototype kernel at the event

The prototype (see `PRE_EVENT.md`) stores P and keeps whole rows in registers. Planned changes,
in order:

1. **Stop storing P.** Each row of P can be computed from C, A, the degrees and D, so the
   stored P is only a cache.
   - Memory: P is 4 of the 9 bytes per pair, so dropping it cuts state by ~45%.
   - Bandwidth: removes two row loads, two row stores and two strided column stores per
     step. Strided column writes touch one cache line per element, so they are the most
     expensive accesses in the kernel.
   - Cost: the R update needs old and new values for rows a and c, so each step computes
     four rows instead of loading two. Expected to be cheaper than the memory traffic;
     measure it.
   - Initialise R in one batched operation. The prototype's `_full_P` loops over the batch
     in Python (~1,024 small launches), which may be the unexplained 10–15 ms fixed cost
     per launch. Check this first.
2. **Smaller types.** C as int16: shared-neighbour counts never exceed the maximum degree
   (< n). Precompute log D once, shared across the batch (`exp(eta · logD)`). With step 1,
   state goes from 9 bytes per pair to 3 (bool A plus int16 C).
3. **An edge count T per simulation,** so several subjects can share a batch (Griffa's edge
   counts vary by ±700 at n = 1000). Not needed for fixed-density targets (§7.1).
4. **Cleanups:** `run()` allocates an unused fp32 `edges` buffer. Make it int32 or drop it.
5. **The additive rule (§4.1), the main new engineering at the event.** It was never
   prototyped. Maintain per-row sums and maxima of the cost and feature terms over
   non-edges, with exact O(n) updates where possible, and recompute rows only when a row's
   argmax decreases (§4.1). Also add the exponential distance term as an option.
6. **Stretch: tiled rows for large n.** Loop over column tiles of fixed width instead of
   holding whole rows in registers. Sample in two levels (per-tile sums, then within the
   tile) for both the row draw from R and the column draw. Keep R in global or shared
   memory. Only worth it for n ≥ 2000 (spills start there).

Dense targets (H8) multiply T by 9–23×, so changes 1 and 2 matter most there.

### 4.1 Wiring-rule variants

The literature uses at least three forms of the rule. The design in §4 (maintained
per-row structures plus two-stage sampling) extends to all of them.

| Form | Rule | Used by |
|---|---|---|
| multiplicative, power law | `w = D^eta · K^gamma` | Betzel et al. 2016, Akarca et al. 2021 |
| multiplicative, exponential | `w = e^(−eta·D) · F^gamma` | Oldham et al. 2025 (alternative) |
| additive, max-normalised | `w = e^(−eta·D) / max e^(−eta·D) + alpha · F^gamma / max F^gamma` | Oldham et al. 2022, 2025 (primary) |

In the additive rule, both maxima run over the current non-edges. Here F is the matching
index K.

**Multiplicative forms.** Normalising by maxima multiplies every entry by the same constant,
so it cancels in P. Switching from a power law to an exponential only changes the static
cost matrix, and the kernel stays as it is. Don't maintain maxima for these forms.

**The additive form doesn't cancel, but it fits the same design.**

- **Two terms.** Write `w = a / M_a + alpha · b / M_b`, where:
  - `a` is the cost term: static, and only masked as edges form;
  - `b = F^gamma` is the dynamic term;
  - `M_a` and `M_b` are global scalars: maxima over all non-edges, not per row.

  The overall scale cancels in P, so the only quantity needed each step is
  `rho = M_a / M_b`, giving `P ∝ a + alpha · rho · b`.
- **Maintain per row, over non-edges:** the fp64 row sums of `a` and of `b`, and the row
  maxima of `a` and of `b`, with their argmax.
- **Per step:**
  - get the global maxima by reducing over the row maxima (O(n));
  - give row i the weight `Ra_i + alpha · rho · Rb_i`;
  - sample a row, then a column from `a + alpha · rho · b`, computed on the fly from the
    stored terms.
- **Keeping the maxima exact:**
  - **Cost term `a`:** it only loses entries, and only rows a and c lose one when edge
    (a, c) forms. The kernel already rewrites both rows, so recompute their maxima there:
    exact and O(n).
  - **Dynamic term `b`:** it changes in rows a and c and in columns a and c of every row.
    Values can go down as well as up, because a matching-index denominator grows with
    degree.
    - If an entry increases, update the row maximum with `max()`.
    - If the row's current argmax decreases, recompute that row.
    - The worst case is O(n²) per step, when many rows have their argmax in columns a
      or c. Measure how often that happens.
  - **If recomputes dominate,** keep a per-row tournament tree over columns. That costs
    O(log n) per changed entry, so O(n log n) per step in the worst case, for about 2× the
    memory of P.
  - **A stale upper bound** on the maximum would not be exact. Avoid it unless you can show
    it doesn't change the sampled distribution.
- **Numerics:**
  - An exponential cost term spans an enormous dynamic range. In fp32, `e^(−eta·D)` underflows
    to zero once eta·D exceeds about 87 (D in the units of the data). Under the additive rule
    those pairs stay reachable through the F term; under the multiplicative rule they don't.
    Decide how to handle this, and match the reference.
  - The argument that fp64 row-sum deltas don't drift needs the entries to span less than
    about 2^29. That fails here, so add a resync: recompute a row when its sum cancels down
    to a small fraction of its running maximum. Run the drift test at extreme eta.
  - Oldham et al. allow negative gamma for topological rules (range −10 to 10), so
    `F = 0` gives `F^gamma = inf`. The epsilon convention must match the reference exactly.
- **Tests:**
  - teacher-force each rule against a dense reference written from the original MATLAB
    code;
  - check every maintained maximum and row sum against brute force after every step;
  - for the additive rule, add chi-square sampling tests at extreme alpha and eta.

## 5. Evaluation: reshaping the scoring

**Hypothesis H2: once generation is fast, evaluation dominates, and betweenness dominates
evaluation.**

- **Get the cheap statistics from simulation state:**
  - degree is the degree vector;
  - clustering comes from A and C (the triangle count per node);
  - edge length comes from the edge list.
- **Betweenness:** Brandes' algorithm with all sources at once, in linear-algebra form, so
  each BFS level is one batched matmul over (network, source, vertex).
  - It wastes FLOPs on sparse graphs but runs dense matmuls.
  - Its cost is set by the deepest network in a chunk.
  - Avoid TF32 unless it is shown not to change the KS results.
- **Batched KS against one target:** sort each network's samples and evaluate both ECDFs at
  the pooled points with a binary search. Memory grows with the number of samples. Don't
  build a networks × n × distinct-values table: that runs out of memory at n = 1000.
- **Decision rule:** build a custom betweenness kernel only if evaluation stays well above
  generation time after the batched version.

Known quirk to check in any reference: the toolbox's betweenness may be a constant multiple
of the standard normalised value. KS between two samples computed the same way is
unaffected, but comparisons across libraries are not.

## 6. Tests and validation

Correctness is the first judging criterion, so the tests are part of the deliverable.

**Ownership:** the teammate owns validation and evaluation end to end: the tests in this
section (including the kernel tests, against a reference the teammate writes or chooses),
full-run validation, the evaluation pipeline (§5), and the decisions on pruning and on
whether to write more kernels (e.g. betweenness). Ryan owns the simulation kernel (§4.0)
and hands over a small interface: inputs (D, eta, gamma, alpha, T per simulation, seed)
and outputs (edge list in order, plus final A, degree and C for checking).

**Simulation kernel, against a plain PyTorch reference:**

- **Teacher forcing:** feed both the same edge sequence and check that C, P, row sums and
  degree match after every step.
- **Shared uniforms:** with the same random numbers, check that the two produce the same
  trajectories.
- **Sampling:** chi-square test of the sampled edges against P.
- **Completion and edge cases:** the network fills up, the denominator is zero, the
  diagonal stays masked, and there are no duplicate edges or self-loops.
- **Drift:** row sums against a full recompute over long runs.

**Full-run validation on the GPU,** for every benchmarked or experimental run (at least on
a subset of simulations):

- rebuild A from the edge list;
- check A, degree and C exactly;
- check P and the row sums against a full recompute, within tolerance.

**Distributional equivalence with the reference.** At matched (eta, gamma), over a few
hundred runs each, run two-sample KS tests on the degree, clustering, betweenness and
edge-length distributions. Do this at small n, and at least one large n if feasible.

**Evaluation:**

- **Betweenness:** against an established library, on the same networks.
- **KS values:** the batched KS and the cheap statistics against the reference's own
  criteria, on a subset of networks.
- **Faster setup paths:** check any faster setup path against the original one (bitwise
  if possible).

**Benchmark hygiene:**

- synchronise before starting and before stopping the clock;
- exclude queued setup work unless you are timing it on purpose;
- take the median of repeats after a warm-up;
- record the GPU, library versions, n, B, T and the parameter ranges;
- watch for a fixed per-launch cost that makes short runs look slow, and don't
  extrapolate baselines from short runs without saying so.

## 7. Data

- **Cohort:** individual structural connectomes at several resolutions (e.g. Lausanne
  scales from ~68 to ~1000 cortical regions), with public licensing. A second, independent
  dataset at ~1000 regions with tract lengths would strengthen claims.
- **Preprocessing:** drop self-loops; binarise; keep track of the region order and the
  hemisphere convention.
- **Distances:** Euclidean distances between template centroids are the simplest choice.
  Per-subject distances or tract lengths are an alternative, so state which one is used.
- **Check the region order without labels:**
  - distance against connection strength should be clearly more negative than under
    within-hemisphere permutations;
  - most nodes should connect to one of their nearest neighbours;
  - edge-weighted mean distance should be well below the permuted value.
- **Target choice matters.** Uniform consensus thresholding removes long edges at high
  resolution and halves density. Use individual subjects, or distance-dependent consensus,
  for anything about long-range connections.

### 7.1 Dense targets from probabilistic tractography

Griffa et al. used deterministic tractography (density 2–27% depending on resolution).
Probabilistic tractography gives much denser networks. The toolbox maintainers suggested
fitting these, since compute has been the barrier (F. Poli, email, Nov 2026). Target
densities: 20%, 30% and 50%, plus 10% as an anchor to deterministic densities.

**Primary dataset: MICA-MICs** (Royer et al. 2022; OSF `j532r`, CC0; cite the paper).
- 50 healthy adults. micapipe v0.1.1, MRtrix iFOD2 with 40M streamlines and SIFT2 weights.
- Per subject: SC (`desc-sc`), tract length (`desc-edgeLength`) and geodesic distance
  (`anat/geo_dist`, `desc-gd`), for Schaefer 100–1000, Glasser 360, aparc and others. Only SC,
  tract length and centroid distances are packaged here.
- Packaged in this repo: `data/mica/schaefer{100,200,400,1000}.npz` (plus `_length.npz`),
  built and validated by `prep/mica_build.py` from `micapipe_MICs_v1.1.zip` (2.7 GB; OSF
  doesn't serve range requests, so rebuilding needs the whole zip). Region order: CBIG
  Schaefer 7-network centroids. Centroid distance tracks each subject's tract length at
  Spearman 0.79–0.84, against 0.23–0.26 (sd ≤ 0.026) under within-hemisphere permutations;
  the 17-network order scores at chance (`data/mica/validation.json`).
- **Layout** (checked at Schaefer 100, 400 and 1000): (p + 16) × (p + 16), upper triangle
  only. Rows 0–13 are subcortical, row 14 and row 15 + p/2 are the medial walls (all zero).
  Left cortex is rows 15 … 14 + p/2, right cortex rows 16 + p/2 … 15 + p. The diagonal is
  nonzero: drop it.

**Measured on all 50 subjects (cortex only, 2026-11-18; `prep/mica_density.py`):**

| Schaefer | raw density | > 90 mm (tract length), raw | interhemispheric, raw | interhemispheric at top 20% / 50% |
| --- | --- | --- | --- | --- |
| 100 | 99.5% ± 1.5 | 70% | 50% | 18.5% / 31.6% |
| 200 | 96.3% ± 2.9 | 72% | 49% | 20.7% / 32.4% |
| 400 | 85.7% ± 4.4 (min 62.8%) | 71% | 46% | 22.4% / 34.3% |
| 1000 | 58.6% ± 4.5 (min 38.3%) | 64% | 40% | 23.8% / 37.0% |

- **Every subject reaches 20% and 30% at every scale. 50% is reached by all subjects at
  Schaefer ≤ 400 and by 48/50 at Schaefer 1000.** Exclude those two at 50%, or cap them
  at their raw density, and say so.
- At Schaefer 1000 the median nonzero weight is ~2.5 (SIFT2 units). A 50% target
  includes many edges supported by one or two streamlines, so expect more noise and more
  false positives. Weights are consistent across subjects (median correlation of log
  weights 0.85 at Schaefer 400, 0.78 at 1000).
- **Thresholding is a modelling choice.** Keep each subject's strongest k% of edges.
  Weight thresholding prefers short edges: at Schaefer 1000 the > 90 mm share goes from
  64% (raw) to 36% (top 20%). Report the long-range results against the thresholded
  target, and consider distance-dependent thresholding as a check.
- **At a fixed density, T is the same for every subject** (k% of n(n − 1)/2). So one set
  of simulations per (resolution, density) serves all 50 subjects. This is cheaper than
  Griffa, where T varies by subject.
- **Distances.** Tract lengths exist only where streamlines exist (complete at
  Schaefer ≤ 200, 41% of pairs missing at 1000), so they can't be the full GNM distance
  matrix at high resolution. Default: Euclidean distance between CBIG Schaefer centroids
  (`dist_mm` in the packaged files, order validated). Alternative: tract length where it is
  complete. Tract lengths avoid the
  midline artefact of Euclidean distance (medial homologues look a few mm apart). Note that
  "> 90 mm" in tract length is not comparable with "> 90 mm" in Euclidean distance (the
  Griffa figures).

## 8. Experiment: energy landscapes

For each subject:

- a fine (eta, gamma) grid with R repeats per cell, with T equal to that subject's edge
  count;
- store all four KS components per repeat, not just the energy.

Plots per subject (one PDF):

- the mean of each energy component, 4 panels;
- the standard deviation of each component, 4 panels;
- the combined energy landscape (the mean of the max-KS), with the optimum marked.

Use `meshgrid(indexing="xy")` so that arrays are [gamma, eta] and plot with eta on the x
axis. Use sequential single-hue colormaps.

Practical directions:

- **Sweep wide first.** Narrow the range only after checking that the optimum is interior.
  Report the energy along the grid edges, so an optimum truncated by the range is visible.
- **Account for noise.** With per-cell SD σ and R repeats, cell means are only good to
  about σ/√R. The minimum over many noisy cells is biased downward. Report the set of
  cells within noise of the minimum, not just the argmin.
- **Run one subject per GPU** and report progress from the local side, since remote
  progress bars garble.

## 9. Hypotheses to test

Engineering:

- **H1 (simulation):** O(n)/step with two-stage sampling, batched over parameters and fused
  into one kernel, gives orders of magnitude more throughput than the dense baseline at
  large n and B. At small B, launch overhead and a fixed per-step floor dominate.
- **H2 (evaluation):** evaluation, mainly betweenness, becomes the bottleneck after H1.
  Batched dense Brandes on GPU is competitive with multi-core CPU libraries at the sizes of
  interest.

Science. Sources: Akarca et al. 2021 (Nat Commun), Oldham, Fornito & Ball 2025 (Netw
Neurosci, "Coming up short"), and suggestions from the toolbox maintainers. Each needs more
simulations than those studies could afford.

- **H3 (identifiability):**
  - The low-energy region is an extended valley or ridge, not a point. Expect one parameter
    (distance, eta) to be well identified and the other (gamma) poorly identified along the
    ridge. Measure the size and shape of the region within noise of the minimum.
  - Narrow ridges, especially in clustering, can be missed or misestimated by coarse or
    single-run searches. Test by subsampling a fine, replicated landscape to coarse grids and
    single runs, and recording where the argmin lands.
- **H4 (individuality):** landscapes and optima vary across subjects. Compare them directly:
  the spread of the optima, the overlap of near-optimal regions, and the correlation between
  landscapes.
- **H5 (individual differences vs estimation noise):**
  - Akarca et al. report a narrow low-energy "crevasse", a negative η–γ correlation across
    subjects, and an association of η with age.
  - Each subject's optimum is an argmin over landscapes with one simulation per point.
    By their own report, energy at the optimum varies with an SD of about 0.045 between
    runs.
  - Hypothesis: the within-subject uncertainty of the argmin is comparable to the spread
    between subjects. A curved or L-shaped valley then produces a negative η–γ
    correlation from noise alone.
  - Test:
    - bootstrap each subject's argmin over the repeats of a replicated landscape;
    - run a null with "subjects" that are synthetic targets generated by the model itself
      at fixed parameters, and check whether the cross-subject η–γ correlation appears
      anyway;
    - report the ratio of between-subject to within-subject variance, and test whether
      subjects' parameters differ significantly.
  - The toolbox maintainers know of no per-subject estimate of the energy's mean and
    variance. If the hypothesis holds, analyses that relate parameters to age or cognition
    need noise-aware estimates, e.g. the mean over the near-optimal set rather than the
    argmin.
- **H6 (noise ceiling):**
  - Oldham et al. find that max(KS) between individuals' empirical networks shrinks at
    finer parcellations.
  - Compute the empirical–empirical max(KS) distribution per resolution and per target
    density on our own cohorts, and report model energy relative to it.
  - A model inside that range can't be improved further on max(KS). What remains has to
    be judged on topography.
- **H7 (long-range connections and resolution):**
  - Oldham et al. fitted individuals only at Schaefer 400 (left hemisphere), and a
    consensus network at 200 left-hemisphere regions, citing compute.
  - They find that recovery depends strongly on distance:
    - short connections are recovered well;
    - mid- and long-range recovery is near chance;
    - beyond 60 mm, the 25% most probable connections are no more likely to exist than
      random ones.
  - Test whether this holds at ~1000 regions and for the whole brain, and whether the
    optimum, the best energy and the long-range recovery of the fitted optimum change with
    resolution. At high resolution, clustering and edge length may become impossible to
    match together.
  - The maintainers doubt that resolution alone fixes long-range recovery. A clear
    negative result is a valid outcome.
- **H8 (dense targets from probabilistic tractography):**
  - Suggested by the toolbox maintainers: deterministic connectomes are 10–30% dense,
    probabilistic ones up to ~95%, and models may fit better on those. Compute has been
    the barrier. Their recent work (Heterochronicity shapes brain network formation, PNAS
    2026) improves topography but still misses long-range connections.
  - Hypothesis: fitted to denser probabilistic targets, the homophily model recovers
    long-range and interhemispheric connections better than when fitted to deterministic
    targets.
  - Test, on MICA-MICs (§7.1):
    - fit at 10%, 20%, 30% and 50% density, at Schaefer 100, 400 and 1000;
    - report, per density: the optimum, the near-optimal set, the empirical–empirical
      noise floor (H6) and connection recovery binned by distance (H7).
  - **Expect identifiability to fall with density:** as the target fills, the energy
    landscape flattens, since every rule converges on the complete graph. Betweenness and
    clustering distributions also compress. Report the noise floor at every density before
    reading any change in the optimum.
  - **Cost:** at n = 1000, T is 99.9k / 149.9k / 249.8k edges at 20 / 30 / 50%, against
    ~10.8k for Griffa (9–23× more steps). Extrapolating the measured 0.046 ms/step at
    B = 1024 gives ~4.6 / 6.9 / 11.5 s per 1,024 simulations (~220 / 150 / 90 sims/s). A
    50 × 50 grid with 20 repeats is then 4–10 min per (resolution, density), shared across
    all subjects. That ms/step was measured at T ≤ 11k and rose with density (0.036 at
    T = 5k), so measure at T = 250k before planning on it.
  - **Evaluation gets relatively cheaper:** dense networks have diameter 2–3, so dense
    Brandes runs few BFS levels. Generation dominates, so the kernel changes that cut
    per-step memory traffic (no stored P, int16 C) matter most here.
  - **Baselines are out of reach:** at ~1.5 s/step (toolbox, B = 1024) a 50% batch at
    n = 1000 is ~100 h. Report it as an extrapolation.
  - **Correctness at long T:** run the drift test (fp64 row sums) and the full-run checks
    at T = 250k. Matching-index values and degrees grow with density (max degree < n, so
    int16 C is safe).
- **H9 (rule form):**
  - Oldham et al. 2022/2025 use an additive rule with exponential distance decay as their
    primary form, `w = e^(−ηD) / max e^(−ηD) + α · K^γ / max K^γ` over non-edges, citing
    a better fit to the trade-offs between parameters. Most homophily work (Betzel 2016,
    Akarca 2021, the GNM toolbox) uses the multiplicative power law.
  - Their search: 2,000 random points in (η, γ, α), then four rounds of 2,000 points each,
    sampled by Voronoi cell according to fit. One simulation per point. Ranges: η ∈ [0, 2]
    (exponential) or [0, 10] (power law), γ ∈ [−10, 10] for topological rules, α ∈ [0, 10].
  - Hypothesis: the rule form changes the optimum, the size and shape of the near-optimal
    set, and long-range recovery. Multiplicative and additive fits are not interchangeable
    descriptions of the same subject.
  - Test, on the same subjects and targets:
    - multiplicative power law (η, γ), multiplicative exponential, additive exponential
      (η, γ, α), with replicated grids over their published ranges (3-D for the additive
      form);
    - compare per rule: best energy against the noise floor (H6), the near-optimal set, the
      between- vs within-subject variance (H5), and long-range recovery binned by distance
      (H7);
    - reproduce their search (one run per point, Voronoi refinement) on the same budget
      and compare its optimum with the replicated grid's (H3).
  - Numerics to settle first (§4.1): the exponential underflows in fp32 once η·D > ~87
    (η = 2 at D = 170 mm gives 340), and K = 0 with negative γ. Match the reference
    convention (Oldham et al.'s MATLAB code) exactly.

## 10. Measurements to report

- **Simulation throughput:** simulations/s and ms/step against n ∈ {68, ~200, 1000, 2000}
  and B ∈ {1, 64, 1024}, for each implementation, plus peak memory.
- **Evaluation:** the cost per batch of each statistic, against generation time.
- **End to end:** wall time for one subject's full landscape and for the whole cohort,
  against a credible extrapolated baseline.
- **Correctness:** pass/fail and tolerances for every test in §6.
- **Science:**
  - per subject: the optimum, the near-optimal set, the per-cell SD and which components
    bind the energy;
  - across subjects: the spread of the optima;
  - across resolutions and rule forms: how the optimum moves;
  - across target densities (H8): how the optimum, the noise floor and long-range
    recovery change between 10% and 50%.
  - beyond max(KS): connection recovery (overall and binned by distance), nodal degree
    correlation with the target, and the empirical–empirical max(KS) range (H6). All of
    them are cheap to compute from A and the target.

## 11. Novelty and framing

- **Not new:** updating the matching index incrementally. Faster *fitting* through fewer
  simulations (landscape search, surrogates or SBI) is complementary, not a competitor.
- **Claim only what can be checked:** O(n)/step from maintained row sums and two-stage
  sampling applied to GNMs; batching over parameters; a fused GPU kernel; edge-list
  memory; full replicated landscapes per subject; fits at ~1000 regions.
- **Expect experts to ask about** seed networks, matching-index conventions and distance
  definitions. Know how our conventions (§2) differ from published fits.
- **Keep a clear record** of what existed before the event and what was built during it.
  Confirm the organisers' rules on prior code.

## 12. Schedule at the event

Two people: Ryan (simulation kernel, rule forms, integration, benchmarks) and a teammate
(evaluation, validation and tests, landscapes and analysis; owns those decisions, §6).
GPUs are available for the whole event; plan ~6 h of sleep, with long jobs running overnight.

**Goals, by priority.**
- *Engineering (cannot be cut):* the rebuilt kernel (§4.0 items 1–4), validated on the
  event GPU (§6); evaluation, validation and the landscape pipeline, built by the
  teammate (§5, §6); one end-to-end fit with wall-clock time against LeanDense, the toolbox
  and igraph (§10).
- *Engineering, new at the event:* the additive rule with maintained maxima (§4.0 item 5,
  §4.1), which enables H9.
- *Science 1, individual differences (H3, H4, H5, H6):* per-subject replicated landscapes on
  Griffa n = 68, between- vs within-subject variance, the empirical noise floor. The toolbox
  maintainers know of no per-subject mean/variance estimate.
- *Science 2, dense probabilistic targets (H8):* MICA-MICs at 20/30/50%.
- *Science 3, long range against resolution (H7):* Griffa at n = 1000. The maintainers doubt
  resolution alone fixes long-range recovery; a well-explained negative result is fine.
- *Science 4, rule form (H9):* multiplicative vs additive on the same subjects, and
  Oldham et al.'s search against replicated grids.
- *Stretch:* tiled rows (§4.0 item 6).

| Time | Ryan (kernel) | Teammate (evaluation) | Checkpoint |
| --- | --- | --- | --- |
| Sat 11:00–12:30 (check-in, talks) | Confirm GPU type and access; run the prototype harness as a smoke test | Load the exported network sample on the event GPU | Environment works |
| 13:00–15:00 | Rebuild the kernel with §4.0 items 1–4; check the fixed launch cost first | Evaluation (statistics, betweenness, KS, energy) and the validation suite, on the sample; check against igraph and the toolbox | Kernel passes the teammate's tests; evaluation matches the references |
| 15:00–16:00 | Integrate generate → evaluate | Landscape pipeline and plots; full-run validation on GPU | One Griffa n = 68 subject end to end |
| 16:00–18:00 | Benchmarks vs LeanDense and toolbox; time a T = 250k run (H8 cost) | Noise floor (H6); variance-ratio code (H5); decide on pruning and a betweenness kernel | **Headline numbers frozen** |
| 18:00–19:00 (dinner) | Launch all 70 Griffa n = 68 subjects | | |
| 19:00–late | Additive rule with maintained maxima (§4.0 item 5); teacher-forcing tests come from the teammate | Individual-differences analysis on the n = 68 results; additive-rule tests (maxima vs brute force, chi-square at extreme α, η) | **Before sleeping: launch overnight jobs:** MICA 20/30/50% at Schaefer 100/400/1000, Griffa n = 1000, and the additive-rule grids at n = 68 if its tests pass |
| Sun 09:30–12:30 | Check overnight runs; fill gaps; Oldham-style search on the same budget (H9) | H8 figures; H7 vs resolution; H9 rule-form comparison | Science results in hand |
| 13:00–14:45 | Final validation reruns; pre-event vs event record | Figures, slides, assumptions list | **No new features after 12:30** |
| 14:45–16:30 | Demo setup and demo | | |

**Cut order if behind:** tiled rows → H7 at n = 1000 → H9 at
n = 1000 (keep n = 68) → MICA Schaefer 1000 (keep 100 and 400). The rebuilt kernel, validation, benchmarks and the n = 68
individual-differences result form a complete submission on their own.

**Before the event:** export the network sample for evaluation work; pin dependencies and
test outside Modal; finish `PRE_EVENT.md`.

**Write-up (Sunday):** one slide with the timeline (before the event: algorithm, prototype
kernel, data; at the event: kernel changes, additive rule, evaluation, validation and landscapes built,
all results), then speedups with caveats, correctness evidence, landscapes and
individual differences, dense-target results, negative results explained, and a clear split
between pre-event and event work.
