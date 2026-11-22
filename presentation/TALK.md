# 5-minute talk: running order (v2, after `draft_talk_feedback_v2.md`)

Story: connectome → how one model grows → how a person is fitted → why the reference is
slow / how we make it fast → benchmark → cost of a replicated study → three scientific
results → TL;DR.

One click = one file. Each file ends on a held frame; talk over it, then click.
Files are in `presentation/output/<clip>/`. Seconds = motion + speaking time (estimate).
Every number on screen comes from `results/` via `scripts/talk_export.py`,
`scripts/bench_export.py` and `scripts/science_export.py`. The distribution curves and
the landscape in the fitting beat (row 9) are schematic and labelled as such.

| # | Part | File | Say (roughly) | s |
|---|---|---|---|---:|
| 1 | 1 Connectomes | `talkA_connectome/01_B_regions.mp4` | "MRI divides the brain into regions, here 1,000 of them." | 10 |
| 2 | | `talkA_connectome/02_C_fibres.mp4` | "Diffusion MRI traces the fibres between them. We represent the brain as regions joined by white-matter fibres, and ask whether a simple growth rule can recreate this wiring." | 14 |
| 3 | | `talkA_connectome/03_D_matrix.mp4` | "As data, it's a 1,000 by 1,000 matrix." | 7 |
| 4 | | `talkA_connectome/04_E_why.mp4` | "Wiring differs between people and can help predict cognition and schizophrenia; connectomes suit established graph-modelling techniques; researchers test wiring rules by simulating connectome growth. But first we need to fit the model." | 16 |
| 5 | 2 Model & fit | `talkM_model/01_A_grow.mp4` | "A model network starts with no connections and adds one at a time." | 10 |
| 6 | | `talkM_model/02_B_rule.mp4` | "Each missing connection gets a probability: distance to the power η times matching index to the power γ. The two parameters trade off short connections against connections between regions with similar neighbours." | 20 |
| 7 | | `talkM_model/03_C_toT.mp4` | "Sample one, add it, repeat until the network has as many connections as the person's. Every run is a different random network." | 10 |
| 8 | | `talkM_model/04_D_fit.mp4` | "To fit a person, we compare four distributions with theirs, degree, clustering, betweenness, edge length, and take the worst mismatch as the energy." | 16 |
| 9 | | `talkM_model/05_E_landscape.mp4` | "One network is random, so each point of a parameter grid is simulated many times, and we keep the lowest mean energy. That's 52,020 networks per person." | 16 |
| 10 | 3 Speed | `talkB_parallel/01_A_sampling.mp4` | "The reference draws each connection from all n² pairs. We pick a row from the row sums, then a column: same distribution, about 2n reads." | 15 |
| 11 | | `talkB_parallel/02_B_update.mp4` | "Adding edge (i, j) only changes rows and columns i and j, so we patch those and the row sums instead of rebuilding the matrix." | 15 |
| 12 | | `talkB_parallel/03_C_tricks.mp4` | "One fused kernel per simulation, probabilities recomputed instead of stored, 3 bytes of state per pair." | 10 |
| 13 | | `talkB_parallel/04_D_side_by_side.mp4` | "So in the same GPU memory, 16× more simulations run at once: a whole parameter sweep in one launch." | 10 |
| 14 | 4 Benchmark & cost | `clip6_bench/02_B_impls.mp4` | "Every implementation on the same H100: about 24,900 times the reference's simulation throughput at 1,000 regions." | 16 |
| 15 | | `talkC_payoff/01_A_count.mp4` | "One fit: 51 values of η, 51 of γ, 20 repeats: 52,020 simulations per person." | 12 |
| 16 | | `talkC_payoff/02_B_person.mp4` | "For one person at 1,000 regions that took about a week with the reference; we do it in 76 seconds." | 8 |
| 17 | | `talkC_payoff/03_C_cohort.mp4` | "For a whole study of 70 people: 1.3 years becomes an hour and a half." | 12 |
| 18 | 5 Papers | `talkD_papers/01_A_intro.mp4` | "We re-ran three published analyses on 70 people, with 20 simulations per grid point." | 6 |
| 19 | | `talkD_papers/02_B_akarca.mp4` | "Can the η–γ correlation appear without individual differences? Seventy synthetic twins sharing one parameter pair show a strong correlation at 68 regions. Only at 1,000 regions do real people vary more." | 22 |
| 20 | | `talkD_papers/03_C_oldham2022.mp4` | "A single-run search selects the luckiest simulation. Grow 100 fresh networks there and the apparent minimum disappears; for 56 of 70 people a replicated grid finds a better expected fit." | 20 |
| 21 | | `talkD_papers/04_D_oldham2025.mp4` | "Their main analysis used 200 left-hemisphere regions, partly for compute. We extended it to 1,000 regions, whole brain: the model is at chance at every distance; other brains predict the wiring up to 90× better." | 22 |
| 22 | 6 TL;DR | `talkE_enables/01_A_tldr.mp4` | Read the headline, the three results, the next steps. | 20 |
| | | | **Total** | **~5:10** |

If running long: shorten row 3 (matrix) and row 12 (kernel details). Do not cut rows 5–9,
15, or 20: they are the causal spine of the talk.

## Say if asked (or aloud, briefly)

- **24,900×** is simulation throughput (2,217 vs ~0.089 sims/s, n = 1,000, B = 1,024,
  same H100). The reference time is extrapolated from timed steps and the reference runs
  one (η, γ) per batch, which favours it. It is not the end-to-end study speedup.
- **6.8 days → 76 s, 1.3 years → 1.5 h**: reference extrapolated, simulation only; ours
  measured, including evaluation. End to end ≈ 7,400× (cohort totals).
- **Akarca et al.**: a null showing the correlation *can* arise from fitting; not an exact
  replication. Their correlation and ours have opposite signs; data, density and
  conventions differ. Interpretation (untested): distance and matching enter both the
  growth rule and the fitted statistics.
- **Oldham et al. 2022**: 56/70 is the additive rule at 219 regions (other rules and
  scales: 51–64/70). This concerns the search's internal minimum; the paper's headline
  results used cross-validation.
- **Oldham, Fornito & Ball 2025**: only the classic rule was run at 1,000 regions; their
  additive rule was tested up to 219. The 90× is the largest bin of the plotted analysis
  (`science_s3.json`). Their quote: "We primarily used data for the left hemisphere, that
  is, 200 cortical regions in total as to: (a) reduce computational complexity...".
- **Model start**: simulations start from an empty network (`src/landscape.py`).
- **"Associated with age, cognition and schizophrenia"**: per Liu et al. 2023 (citing
  Betzel 2016, Akarca 2021, Zhang 2021).

## Backup slides (questions)

- How the model works in detail: `clip1_wiring/`
- Why n² sampling is slow: `clip2_naive/`
- The two-stage sampling trick: `clip3_chain/`
- Kernel state and step: `clip4_matrices/`
- Memory layout, coalesced loads: `clip5_kernel/`
- Benchmarks in full: `clip6_bench/`
- Memory per simulation and batch size: `talkC_payoff` previous version (git history), `clip6_bench/05_E_memory.mp4`
- Full science clips: `clip7_twins/`, `clip8_luckiest/`, `clip9_chance/`
