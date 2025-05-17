# Benchmark Report

The comprehensive benchmark was run on one year of ERCOT data across 16 settlement points. Results were saved in the `results/ercot` directory.

## Relative Performance (% of oracle)

| Model | Mean % | Std Dev |
|-------|-------:|-------:|
| online_mpc_ridge | 56.74 | 5.80 |
| online_quartile_p45 | 45.53 | 3.72 |
| online_quartile_p25 | 41.28 | 2.00 |
| online_quartile_p10 | 35.31 | 3.76 |

## Average Runtime (seconds)

| Model | Mean Runtime |
|-------|-------------:|
| oracle_lp | 8.49 |
| online_quartile_p45 | 20.79 |
| online_quartile_p25 | 21.85 |
| online_quartile_p10 | 34.33 |
| online_mpc_ridge | 38966.22 |

The `oracle_lp` model represents the theoretical optimum with perfect knowledge of future prices. `online_mpc_ridge` uses model predictive control with a ridge regression forecaster. The quartile models dispatch based on historical percentiles without explicit forecasts. While `online_mpc_ridge` achieved the highest average revenue relative to the oracle, it also required significantly more computation time.

Benchmarks were launched via the scripts in the `benchmarks/` directory and summary files are available under `results/summary`.
