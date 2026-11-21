# Simulation benchmark, 2026-11-21T21:57:28

NVIDIA H100 80GB HBM3, torch 2.8.0+cu128, triton 3.4.0, git aee73ad7

`bench/bench_sim.py --impls src.kernel:TritonGNM --n 1000 --B 1024 --T 249750 --dist griffa --repeats 2 --tag dense_T249750`

sims/s = B / (setup + run). `~` = extrapolated from probe steps.

| impl | n | B | T | setup s | run s | ms/step | sims/s | peak GB | edges ok | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| src.kernel:TritonGNM | 1000 | 1024 | 249750 | 0.077 | 12.959 | 0.0519 | 78.6 | 5.66 | True | ok |
