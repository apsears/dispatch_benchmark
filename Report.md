# Benchmark Report

The comprehensive benchmark was run on one year of ERCOT data across 16 settlement points. Results were saved in the `results/ercot` directory.

## ERCOT Summary

### Average Revenue by Model

| Model | Mean Revenue ($) | Std Dev |
|-------|------------------|---------|
| oracle_lp | 2,777,047.59 | 590,425.23 |
| online_mpc_ridge | 1,474,517.48 | 192,055.81 |
| online_quartile_p45 | 1,274,187.35 | 376,776.08 |
| online_quartile_p25 | 1,151,840.94 | 294,962.20 |
| online_quartile_p10 | 988,846.24 | 275,144.44 |

### Relative Performance (% of optimal)

| Model | Mean % | Std Dev |
|-------|--------|---------|
| online_mpc_ridge | 54.25 | 8.24 |
| online_quartile_p45 | 45.53 | 3.72 |
| online_quartile_p25 | 41.28 | 2.00 |
| online_quartile_p10 | 35.31 | 3.76 |

### Average Runtime by Model

| Model | Mean Runtime (seconds) |
|-------|------------------------|
| oracle_lp | 8.84 |
| online_quartile_p45 | 21.02 |
| online_quartile_p10 | 21.74 |
| online_quartile_p25 | 21.93 |
| online_mpc_ridge | 1,698.96 |

**Output Files:**
- CSV files saved to `results/summary/ercot/ercot_summary`
- Visualizations saved to `results/summary/ercot`

**Results Location:**
- ERCOT: `results/ercot`
- ERCOT summary: `results/summary/ercot`

## Benchmark Timing Summary

| Event | Time |
|-------|------|
| Started | 2025-05-17 13:18:05 |
| Completed | 2025-05-17 13:48:01 |
| ERCOT benchmark | 0h 29m 55s |
| **Total time elapsed** | **0h 29m 56s** |

## Model Descriptions

The `oracle_lp` model represents the theoretical optimum with perfect knowledge of future prices. `online_mpc_ridge` uses model predictive control with a ridge regression forecaster. The quartile models dispatch based on historical percentiles without explicit forecasts. While `online_mpc_ridge` achieved the highest average revenue relative to the oracle, it also required significantly more computation time.

Benchmarks were launched via the scripts in the `benchmarks/` directory and summary files are available under `results/summary`.
