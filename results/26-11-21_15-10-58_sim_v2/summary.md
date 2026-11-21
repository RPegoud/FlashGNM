# Simulation benchmark, 2026-11-21T15:10:59

NVIDIA H100 80GB HBM3, torch 2.8.0+cu128, triton 3.4.0, git 699f8249

`bench/bench_sim.py --impls triton_proto src.kernel:TritonGNM --tag v2`

sims/s = B / (setup + run). `~` = extrapolated from probe steps.

| impl | n | B | T | setup s | run s | ms/step | sims/s | peak GB | edges ok | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| triton_proto | 68 | 1 | 610 | 0.001 | 0.002 | 0.0027 | 391.2 | 0.00 | True | ok |
| triton_proto | 68 | 64 | 610 | 0.012 | 0.002 | 0.0028 | 4,513.1 | 0.00 | True | ok |
| triton_proto | 68 | 1024 | 610 | 0.181 | 0.003 | 0.0041 | 5,580.0 | 0.05 | True | ok |
| triton_proto | 219 | 1 | 2639 | 0.001 | 0.011 | 0.0042 | 81.4 | 0.00 | True | ok |
| triton_proto | 219 | 64 | 2639 | 0.013 | 0.011 | 0.0043 | 2,650.5 | 0.03 | True | ok |
| triton_proto | 219 | 1024 | 2639 | 0.180 | 0.047 | 0.0179 | 4,515.9 | 0.49 | True | ok |
| triton_proto | 1000 | 1 | 10785 | 0.003 | 0.057 | 0.0053 | 16.7 | 0.03 | True | ok |
| triton_proto | 1000 | 64 | 10785 | 0.015 | 0.070 | 0.0065 | 754.1 | 0.60 | True | ok |
| triton_proto | 1000 | 1024 | 10785 | 0.189 | 0.494 | 0.0458 | 1,499.6 | 9.41 | True | ok |
| triton_proto | 2000 | 1 | 43162 | 0.007 | 0.372 | 0.0086 | 2.6 | 0.14 | True | ok |
| triton_proto | 2000 | 64 | 43162 | 0.028 | 0.501 | 0.0116 | 121.0 | 2.41 | True | ok |
| triton_proto | 2000 | 1024 | 43162 | 0.339 | 5.607 | 0.1299 | 172.2 | 37.61 | True | ok |
| src.kernel:TritonGNM | 68 | 1 | 610 | 0.001 | 0.002 | 0.0029 | 378.9 | 0.00 | True | ok |
| src.kernel:TritonGNM | 68 | 64 | 610 | 0.010 | 0.002 | 0.0029 | 5,449.3 | 0.00 | True | ok |
| src.kernel:TritonGNM | 68 | 1024 | 610 | 0.152 | 0.003 | 0.0041 | 6,606.8 | 0.02 | True | ok |
| src.kernel:TritonGNM | 219 | 1 | 2639 | 0.001 | 0.012 | 0.0045 | 77.6 | 0.00 | True | ok |
| src.kernel:TritonGNM | 219 | 64 | 2639 | 0.011 | 0.013 | 0.0048 | 2,740.5 | 0.01 | True | ok |
| src.kernel:TritonGNM | 219 | 1024 | 2639 | 0.156 | 0.020 | 0.0076 | 5,810.7 | 0.17 | True | ok |
| src.kernel:TritonGNM | 1000 | 1 | 10785 | 0.003 | 0.073 | 0.0068 | 13.2 | 0.03 | True | ok |
| src.kernel:TritonGNM | 1000 | 64 | 10785 | 0.012 | 0.083 | 0.0077 | 670.8 | 0.22 | True | ok |
| src.kernel:TritonGNM | 1000 | 1024 | 10785 | 0.156 | 0.396 | 0.0367 | 1,854.3 | 3.18 | True | ok |
| src.kernel:TritonGNM | 2000 | 1 | 43162 | 0.007 | 0.330 | 0.0076 | 3.0 | 0.11 | True | ok |
| src.kernel:TritonGNM | 2000 | 64 | 43162 | 0.025 | 0.368 | 0.0085 | 162.8 | 0.87 | True | ok |
| src.kernel:TritonGNM | 2000 | 1024 | 43162 | 0.318 | 2.975 | 0.0689 | 310.9 | 12.68 | True | ok |
