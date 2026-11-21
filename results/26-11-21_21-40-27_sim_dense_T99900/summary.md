# Simulation benchmark, 2026-11-21T21:40:27

NVIDIA H100 80GB HBM3, torch 2.8.0+cu128, triton 3.4.0, git aee73ad7

`bench/bench_sim.py --impls src.kernel:TritonGNM --n 1000 --B 1024 --T 99900 --dist griffa --repeats 2 --tag dense_T99900`

sims/s = B / (setup + run). `~` = extrapolated from probe steps.

| impl | n | B | T | setup s | run s | ms/step | sims/s | peak GB | edges ok | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| src.kernel:TritonGNM | 1000 | 1024 | 99900 | 0.071 | 4.016 | 0.0402 | 250.6 | 5.66 | True | ok |
