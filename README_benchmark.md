# ERCOT Battery Dispatch Model Benchmark

This benchmark script tests multiple battery dispatch models on nodal price data from ERCOT to compare their performance and revenue generation.

## Models Included

1. **Oracle LP** (`oracle_LP.py`): Omniscient model with perfect price knowledge
2. **Oracle MPC** (`oracle_mpc.py`): Rolling horizon MPC with perfect price knowledge 
3. **Online MPC** (`online_mpc.py`): Non-clairvoyant MPC with different forecast models:
   - EWMA: Exponentially weighted moving average
   - Ridge: Ridge regression with calendar features
4. **Online Quartile** (`online_quartile.py`): Percentile-based threshold model testing different percentiles:
   - 10th/90th percentiles
   - 25th/75th percentiles
   - 45th/55th percentiles

## Usage

```bash
python benchmark_models.py [options]
```

### Options

- `--prices PATH`: Path to the prices_wide.csv file (default: data/prices_wide.csv)
- `--start-date DATE`: Start date for the week in YYYY-MM-DD format (default: first date in data)
- `--nodes NODE1 NODE2...`: Specific nodes to benchmark (default: first 3 nodes in data)
- `--output-dir DIR`: Directory to save results (default: benchmark_results)

### Examples

Run on default settings (first 3 nodes):
```bash
python benchmark_models.py
```

Run for specific nodes:
```bash
python benchmark_models.py --nodes ALP_BESS_RN HB_NORTH_RN
```

Run for a specific week:
```bash
python benchmark_models.py --start-date 2023-06-01
```

## Results

Results are saved in two formats:

1. Individual node results: `{output_dir}/{node}_results.json`
2. Combined results: `{output_dir}/combined_results.json`

The results contain:
- Model name and configuration
- Total revenue generated
- Runtime in seconds

Results are saved after each model run to prevent data loss if the script is interrupted.

## Battery Configuration

The benchmark uses a standard battery configuration:
- 25 MW power capacity
- 200 MWh energy capacity
- 95% charging efficiency
- 15-minute intervals (0.25 hours)

These settings can be modified in the script if needed. 