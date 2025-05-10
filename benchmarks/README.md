# Virtual Energy Benchmarking System

This directory contains scripts for benchmarking different energy forecasting and optimization models across multiple ISOs (ERCOT and NYISO).

## Benchmark Scripts

### 1. `run_all_benchmarks.sh`

The main entry point for running a full benchmark across both ERCOT and NYISO data sources.

**Usage:**
```bash
cd benchmarks
./run_all_benchmarks.sh
```

This script will:
- Run benchmarks on ERCOT and NYISO data in parallel
- Generate summary statistics and visualizations
- Produce a comparison analysis between the two ISOs

### 2. `run_benchmark.sh`

Simplified script to run the comprehensive benchmark on ERCOT data only.

**Usage:**
```bash
cd benchmarks
./run_benchmark.sh
```

## Core Implementation

### 1. `comprehensive_benchmark.py`

The core benchmark engine that runs multiple optimization models on electricity price data.

**Models benchmarked:**
- Oracle LP (theoretical optimum with perfect knowledge)
- Online MPC with ridge regression forecasting
- Online MPC with naive forecasting (baseline)
- Online Quartile with various percentiles (10%, 25%, 45%)

**Usage:**
```bash
cd benchmarks
python3 comprehensive_benchmark.py \
    --prices-path ../data/ercot/2024_RealTime_SPP.csv \
    --data-format tidy \
    --output-dir results/ercot \
    --max-nodes 10 \
    --n-jobs 4
```

### 2. `summarize_benchmarks.py`

Analyzes benchmark results and produces summary statistics, CSV files, and visualizations.

**Features:**
- Calculates average revenue by model
- Measures performance relative to oracle (theoretical maximum)
- Compares ridge vs. naive forecasting performance
- Generates visualizations (revenue, relative performance, runtime)
- Creates CSV files for deeper analysis

**Usage:**
```bash
cd benchmarks
python3 summarize_benchmarks.py \
    --results-dir results/ercot \
    --output-dir results/summary/ercot \
    --iso ERCOT
```

## Data Formats

The scripts handle two different data formats:

### ERCOT Format (../data/ercot/2024_RealTime_SPP.csv)
```
node,timestamp,price
7RNCHSLR_ALL,2024-01-01 00:00:00,12.59
...
```

### NYISO Format (../data/nyiso/2024_DayAhead_LBMP.csv)
```
timestamp,zone,price
2024-01-01 00:00:00,CAPITL,25.02
...
```

## Output Directory Structure

Benchmark results are organized in the following directory structure:

```
benchmarks/
└── results/
    ├── ercot/       # ERCOT benchmark results
    ├── nyiso/       # NYISO benchmark results
    ├── combined/    # Combined results
    ├── plots/       # Visualizations
    │   ├── ercot/
    │   ├── nyiso/
    │   └── combined/
    └── summary/     # Summary statistics
        ├── ercot/
        ├── nyiso/
        └── combined/
```

## Output Files

- Individual benchmark results are saved in JSON format
- Summary statistics are presented in console tables
- CSV files are generated for detailed analysis
- Visualizations are created as PNG files

## Requirements

Required Python packages:
- pandas
- numpy
- matplotlib
- tabulate
- tqdm

Install with:
```bash
source ../.venv/bin/activate
uv pip install matplotlib tabulate
```

## Legacy Scripts

Additional specialized benchmark scripts are available in the `legacy/` directory for specific use cases but are not part of the primary benchmarking workflow. 