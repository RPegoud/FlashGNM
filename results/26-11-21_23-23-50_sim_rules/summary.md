# Simulation benchmark, 2026-11-21T23:23:50

NVIDIA H100 80GB HBM3, torch 2.8.0+cu128, triton 3.4.0, git bba36f22

`bench/bench_sim.py --impls src.kernel:TritonGNM src.kernel_additive:TritonRuleGNM --n 68 219 448 1000 --B 1024 4096 --dist griffa --repeats 2 --tag rules`

sims/s = B / (setup + run). `~` = extrapolated from probe steps.

| impl | n | B | T | setup s | run s | ms/step | sims/s | peak GB | edges ok | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| src.kernel:TritonGNM | 68 | 1024 | 610 | 0.009 | 0.003 | 0.0043 | 91,996.2 | 0.07 | True | ok |
| src.kernel:TritonGNM | 68 | 4096 | 610 | 0.004 | 0.008 | 0.0135 | 345,776.3 | 0.29 | True | ok |
| src.kernel:TritonGNM | 219 | 1024 | 2639 | 0.009 | 0.020 | 0.0075 | 35,494.3 | 0.74 | True | ok |
| src.kernel:TritonGNM | 219 | 4096 | 2639 | 0.015 | 0.077 | 0.0293 | 44,351.4 | 2.96 | True | ok |
| src.kernel:TritonGNM | 448 | 1024 | 5324 | 0.014 | 0.075 | 0.0140 | 11,597.7 | 3.09 | True | ok |
| src.kernel:TritonGNM | 448 | 4096 | 5324 | 0.051 | 0.290 | 0.0545 | 12,006.6 | 5.06 | True | ok |
| src.kernel:TritonGNM | 1000 | 1024 | 10785 | 0.073 | 0.398 | 0.0369 | 2,175.3 | 5.66 | True | ok |
| src.kernel:TritonGNM | 1000 | 4096 | 10785 | 0.243 | 1.555 | 0.1442 | 2,277.4 | 14.90 | True | ok |
| src.kernel_additive:TritonRuleGNM | 68 | 1024 | 610 | 0.003 | 0.019 | 0.0319 | 45,116.4 | 0.25 | True | ok |
| src.kernel_additive:TritonRuleGNM | 68 | 4096 | 610 | 0.010 | 0.074 | 0.1210 | 48,887.4 | 1.02 | True | ok |
| src.kernel_additive:TritonRuleGNM | 219 | 1024 | 2639 | 0.012 | 0.212 | 0.0804 | 4,571.1 | 2.61 | True | ok |
| src.kernel_additive:TritonRuleGNM | 219 | 4096 | 2639 | 0.037 | 0.815 | 0.3088 | 4,806.1 | 7.34 | True | ok |
| src.kernel_additive:TritonRuleGNM | 448 | 1024 | 5324 | 0.036 | 0.513 | 0.0964 | 1,862.3 | 7.34 | True | ok |
| src.kernel_additive:TritonRuleGNM | 448 | 4096 | 5324 | 0.132 | 1.977 | 0.3714 | 1,941.9 | 9.26 | True | ok |
| src.kernel_additive:TritonRuleGNM | 1000 | 1024 | 10785 | 0.155 | 1.906 | 0.1767 | 496.8 | 9.84 | True | ok |
| src.kernel_additive:TritonRuleGNM | 1000 | 4096 | 10785 | 0.605 | 7.563 | 0.7012 | 501.5 | 19.17 | True | ok |
