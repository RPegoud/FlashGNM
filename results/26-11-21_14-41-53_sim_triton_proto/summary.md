# Simulation benchmark, 2026-11-21T14:41:54

NVIDIA H100 80GB HBM3, torch 2.8.0+cu128, triton 3.4.0, git 747857c9

`bench/bench_sim.py --impls triton_proto --tag triton_proto`

sims/s = B / (setup + run). `~` = extrapolated from probe steps.

| impl | n | B | T | setup s | run s | ms/step | sims/s | peak GB | edges ok | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| triton_proto | 68 | 1 | 610 | 0.001 | 0.002 | 0.0027 | 411.9 | 0.00 | True | ok |
| triton_proto | 68 | 64 | 610 | 0.011 | 0.002 | 0.0027 | 5,129.5 | 0.00 | True | ok |
| triton_proto | 68 | 1024 | 610 | 0.131 | 0.003 | 0.0042 | 7,678.3 | 0.05 | True | ok |
| triton_proto | 219 | 1 | 2639 | 0.001 | 0.011 | 0.0042 | 82.9 | 0.00 | True | ok |
| triton_proto | 219 | 64 | 2639 | 0.010 | 0.012 | 0.0044 | 3,019.2 | 0.03 | True | ok |
| triton_proto | 219 | 1024 | 2639 | 0.144 | 0.048 | 0.0181 | 5,331.6 | 0.49 | True | ok |
| triton_proto | 1000 | 1 | 10785 | 0.003 | 0.057 | 0.0053 | 16.7 | 0.03 | True | ok |
| triton_proto | 1000 | 64 | 10785 | 0.012 | 0.071 | 0.0066 | 769.9 | 0.60 | True | ok |
| triton_proto | 1000 | 1024 | 10785 | 0.146 | 0.494 | 0.0458 | 1,600.7 | 9.41 | True | ok |
| triton_proto | 2000 | 1 | 43162 | 0.005 | 0.373 | 0.0086 | 2.6 | 0.14 | True | ok |
| triton_proto | 2000 | 64 | 43162 | 0.030 | 0.504 | 0.0117 | 119.8 | 2.41 | True | ok |
| triton_proto | 2000 | 1024 | 43162 | 0.338 | 5.623 | 0.1303 | 171.8 | 37.61 | True | ok |
