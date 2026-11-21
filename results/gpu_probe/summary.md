# GPU cost per landscape subject (2026-11-21)

One Griffa subject, 51 x 51 grid x 20 repeats (52k simulations), kernel v2 + evaluation
(fp64 betweenness), no pruning. Warm runs where available; H100 = medians of the 70-subject
main run (results/landscape_griffa_full). Cost = (gen + eval) x Modal per-second price
(modal.com/pricing, 2026-11-21: T4 $0.000164/s, A10 $0.000306, L40S $0.000542,
A100-40GB $0.000583, H100 $0.001097). Excludes container start-up and idle time.

| n | T4 | A10 | L40S | A100-40GB | H100 |
|---:|---|---|---|---|---|
| 68 | 11 s, $0.0018 | | | | 0.3 s, $0.0003 |
| 114 | 22 s, $0.0036 | 12 s, $0.0037 | | | 0.7 s, $0.0008 |
| 219 | | 67 s, $0.021 | 29 s, $0.016 | 7.5 s, $0.0044 | 3.4 s, $0.0037 |
| 448 | | | 184 s, $0.10 | 35 s, $0.020 | 14 s, $0.016 |
| 1000 | | | | 186 s, $0.11 | 76 s, $0.083 |

- Evaluation dominates on every card without fast fp64 (T4, A10, L4, L40S: fp64 at 1/32 to
  1/64 of fp32), because betweenness runs in fp64. With fp64 the H100 is the cheapest per
  job at every size; the A100 costs 20-25% more at n >= 448.
- Results are bitwise identical across T4, A100 and H100 for the same seed.
- At n <= 114 jobs take seconds, so container start-up and idle time (billed) dominate:
  a cheap card with few containers is cheapest for the sweep as a whole.
- Not measured (Modal spend limit reached): fp32 betweenness (`--bc-dtype float32`) on
  L40S/A10/L4, which should speed evaluation 10-30x there; it must first be shown not to
  change the KS values against fp64. L4 never got capacity during the probes.
