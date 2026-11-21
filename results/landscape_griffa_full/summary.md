# Landscapes: landscape_griffa_full

Grid: {'eta_lo': -4.0, 'eta_hi': 1.0, 'n_eta': 51, 'gamma_lo': -1.0, 'gamma_hi': 4.0, 'n_gamma': 51, 'repeats': 20}, seed 0, git a08725ec173eff50bcc3a4e0db9fcca441d6e157

Medians over subjects (IQR for the optimum). Near-optimal: cells within 2 SE of the best mean. Interior: every grid-edge cell is worse than the optimum.

| scale | n | subjects | eta opt | gamma opt | energy | SD at opt | near-opt cells | interior | all valid | gen s | eval s |
|---|---:|---:|---|---|---:|---:|---:|---:|---|---:|---:|
| 033 | 68 | 70 | -1.80 [-1.80, -1.70] | 2.30 [2.00, 2.50] | 0.229 | 0.034 | 15 | 70/70 | True | 0.1 | 0.2 |
| 060 | 114 | 70 | -1.70 [-1.70, -1.70] | 2.00 [1.83, 2.10] | 0.236 | 0.025 | 8 | 70/70 | True | 0.2 | 0.5 |
| 125 | 219 | 70 | -1.50 [-1.60, -1.50] | 1.85 [1.60, 2.00] | 0.268 | 0.020 | 8 | 70/70 | True | 1.1 | 2.3 |
| 250 | 448 | 70 | -1.40 [-1.40, -1.40] | 1.50 [1.40, 1.60] | 0.303 | 0.013 | 4 | 70/70 | True | 4.3 | 9.9 |
| 500 | 1000 | 70 | -1.30 [-1.30, -1.30] | 1.10 [1.00, 1.20] | 0.330 | 0.010 | 3 | 70/70 | True | 23.7 | 52.1 |
