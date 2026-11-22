# 5-minute talk: running order

One click = one file. Each file ends on a held frame; talk over it, then click.
Files are in `presentation/output/<clip>/`. Seconds = motion + speaking time (estimate).
Every number on screen comes from `results/` via `scripts/talk_export.py`,
`scripts/bench_export.py` and `scripts/science_export.py`.

| # | Part | File | Say (roughly) | s |
|---|---|---|---|---:|
| 1 | Opener | `talkA_connectome/01_A_open.mp4` | "Fitting a brain model to one person at 1,000 regions took about a week. We do it in 76 seconds." | 12 |
| 2 | 1 Connectomes | `talkA_connectome/02_B_regions.mp4` | "MRI divides the brain into regions, here 1,000 of them." | 10 |
| 3 | | `talkA_connectome/03_C_fibres.mp4` | "Diffusion MRI traces the fibres between them. Regions plus fibres: a connectome." | 14 |
| 4 | | `talkA_connectome/04_D_matrix.mp4` | "As data, it's a 1,000 by 1,000 matrix." | 8 |
| 5 | | `talkA_connectome/05_E_why.mp4` | "Wiring differs between people, through development, in disorders. Generative models ask which simple rules grow a brain-like network. Fitting one person means simulating 52,020 networks." | 18 |
| 6 | 2 GPU | `talkB_parallel/01_A_step.mp4` | "Each step adds one connection. The reference redoes all n² pairs; we update only the few rows that changed: a million vs 4,000 operations." | 16 |
| 7 | | `talkB_parallel/02_B_side_by_side.mp4` | "Because each simulation is that small, a GPU runs thousands side by side: same memory, 16× more simulations." | 12 |
| 8 | | `talkB_parallel/03_C_sweep.mp4` | "And each one gets its own parameters, so a whole sweep runs in one launch." | 7 |
| 9 | | `talkB_parallel/04_D_payoff.mp4` | "Together: about 25,000 times the reference's throughput." | 6 |
| 10 | 3 Speedups | `clip6_bench/02_B_impls.mp4` | "Every implementation, same GPU, 1,000 regions. Each new bar rescales the axis: the baselines become slivers." | 16 |
| 11 | | `talkC_payoff/01_A_memory.mp4` | "Each simulation needs about 15× less memory. At 2,000 regions the reference runs out of memory; we run 4,096 at once." | 14 |
| 12 | | `talkC_payoff/02_B_cohort.mp4` | "A whole study, 70 people at 1,000 regions: 1.3 years becomes an hour and a half." | 10 |
| 13 | 4 Papers | `talkD_papers/01_A_intro.mp4` | "So we re-ran three published analyses, with every fit replicated 20 times, on 70 people." | 6 |
| 14 | | `talkD_papers/02_B_akarca.mp4` | "Akarca et al. fitted parameters per person. Synthetic twins sharing one parameter set show the same spread and correlation at 68 regions; only at 1,000 do real people differ more." | 22 |
| 15 | | `talkD_papers/03_C_oldham2022.mp4` | "Oldham et al. used a search with one network per point. Its best fit was luck: fresh networks there do worse than a replicated grid, for 56 of 70 people." | 20 |
| 16 | | `talkD_papers/04_D_oldham2025.mp4` | "Oldham et al. showed models miss long-range connections. At 1,000 regions, the best model places connections at chance at every distance; other brains do up to 90× better." | 22 |
| 17 | 5 Conclusion | `talkE_enables/01_A_enables.mp4` | "What this enables: whole studies with replicated fits, dense connectomes, honest uncertainty, fair model comparison." | 18 |
| 18 | | `talkE_enables/02_B_close.mp4` | "Every brain, the whole landscape, replicated. Thank you." | 6 |
| | | | **Total** | **~4:50** |

## Backup slides (questions)

- How the model works: `clip1_wiring/`
- Why n² sampling is slow: `clip2_naive/`
- The two-stage sampling trick: `clip3_chain/`
- Kernel state and step: `clip4_matrices/`
- Memory layout and coalescing: `clip5_kernel/`
- Benchmarks in full (scaling with n, big batches, memory): `clip6_bench/`
- Full science clips: `clip7_twins/`, `clip8_luckiest/`, `clip9_chance/`

## Caveats to say if asked

- Reference times are extrapolated from timed steps; the reference runs one (η, γ) per
  batch, which favours it. Same H100 throughout.
- Twins result (B1): one null parameter set per scale; our η–γ correlation is positive,
  Akarca et al.'s negative, so it's a statement about the fitting method, not a
  replication.
- "Chance" = the model's own edge density in that distance bin.
