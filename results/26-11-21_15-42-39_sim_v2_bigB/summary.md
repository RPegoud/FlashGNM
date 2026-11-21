# Simulation benchmark, 2026-11-21T15:42:39

NVIDIA H100 80GB HBM3, torch 2.8.0+cu128, triton 3.4.0, git 4cc1b271

`bench/bench_sim.py --impls src.kernel:TritonGNM --B 1024 2048 4096 8192 16384 --tag v2_bigB`

sims/s = B / (setup + run). `~` = extrapolated from probe steps.

| impl | n | B | T | setup s | run s | ms/step | sims/s | peak GB | edges ok | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| src.kernel:TritonGNM | 68 | 1024 | 610 | 0.002 | 0.003 | 0.0041 | 236,742.7 | 0.07 | True | ok |
| src.kernel:TritonGNM | 68 | 2048 | 610 | 0.004 | 0.005 | 0.0076 | 243,242.3 | 0.15 | True | ok |
| src.kernel:TritonGNM | 68 | 4096 | 610 | 0.002 | 0.008 | 0.0134 | 390,277.0 | 0.29 | True | ok |
| src.kernel:TritonGNM | 68 | 8192 | 610 | 0.005 | 0.016 | 0.0254 | 399,831.3 | 0.58 | True | ok |
| src.kernel:TritonGNM | 68 | 16384 | 610 | 0.007 | 0.030 | 0.0489 | 445,258.4 | 1.16 | True | ok |
| src.kernel:TritonGNM | 219 | 1024 | 2639 | 0.005 | 0.020 | 0.0075 | 41,165.3 | 0.74 | True | ok |
| src.kernel:TritonGNM | 219 | 2048 | 2639 | 0.009 | 0.039 | 0.0147 | 43,018.0 | 1.48 | True | ok |
| src.kernel:TritonGNM | 219 | 4096 | 2639 | 0.016 | 0.077 | 0.0291 | 43,904.0 | 2.96 | True | ok |
| src.kernel:TritonGNM | 219 | 8192 | 2639 | 0.028 | 0.151 | 0.0570 | 45,974.8 | 3.78 | True | ok |
| src.kernel:TritonGNM | 219 | 16384 | 2639 | 0.051 | 0.294 | 0.1114 | 47,549.8 | 4.98 | True | ok |
| src.kernel:TritonGNM | 1000 | 1024 | 10785 | 0.066 | 0.396 | 0.0367 | 2,217.2 | 5.66 | True | ok |
| src.kernel:TritonGNM | 1000 | 2048 | 10785 | 0.127 | 0.784 | 0.0726 | 2,248.3 | 8.74 | True | ok |
| src.kernel:TritonGNM | 1000 | 4096 | 10785 | 0.245 | 1.547 | 0.1434 | 2,286.7 | 14.90 | True | ok |
| src.kernel:TritonGNM | 1000 | 8192 | 10785 | 0.491 | 3.036 | 0.2815 | 2,322.8 | 27.23 | True | ok |
| src.kernel:TritonGNM | 1000 | 16384 | 10785 | 1.058 | 6.054 | 0.5613 | 2,303.8 | 51.89 | True | ok |
| src.kernel:TritonGNM | 2000 | 1024 | 43162 | 0.253 | 2.983 | 0.0691 | 316.4 | 14.87 | True | ok |
| src.kernel:TritonGNM | 2000 | 2048 | 43162 | 0.495 | 5.955 | 0.1380 | 317.5 | 27.18 | True | ok |
| src.kernel:TritonGNM | 2000 | 4096 | 43162 | 1.142 | 11.856 | 0.2747 | 315.1 | 51.80 | True | ok |
| src.kernel:TritonGNM | 2000 | 8192 | 43162 |  |  |  |  |  |  | oom |
| src.kernel:TritonGNM | 2000 | 16384 | 43162 |  |  |  |  |  |  | oom |
