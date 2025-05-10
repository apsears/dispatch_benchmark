#!/bin/bash
# Simple benchmark script for ERCOT only

# Setup
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if not already active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source ../.venv/bin/activate
fi

# Create results directories
ERCOT_OUTPUT="results/ercot"
ERCOT_SUMMARY="results/summary/ercot"
PLOTS_DIR="results/plots/ercot"

mkdir -p "$ERCOT_OUTPUT" "$ERCOT_SUMMARY" "$PLOTS_DIR"

# Display setup information
echo "Setting up benchmark environment..."
echo "Python: $(which python3)"
echo "Working directory: $(pwd)"

# Check for required data files
ERCOT_DATA="../data/ercot/2024_RealTime_SPP.csv"

if [[ ! -f "$ERCOT_DATA" ]]; then
    echo "Error: ERCOT data file not found at $ERCOT_DATA"
    exit 1
fi

echo "Data file found. Starting benchmark..."

# Run ERCOT benchmark
echo "Running ERCOT benchmark..."
python3 comprehensive_benchmark.py \
    --prices-path "$ERCOT_DATA" \
    --data-format tidy \
    --output-dir "$ERCOT_OUTPUT" \
    --max-nodes 10 \
    --n-jobs 4 \
    --iso ERCOT

# Generate summary
echo "Generating ERCOT summary..."
python3 summarize_benchmarks.py \
    --results-dir "$ERCOT_OUTPUT" \
    --output-dir "$ERCOT_SUMMARY" \
    --iso ERCOT

echo "Benchmark and summary completed successfully!"
echo "Results available in:"
echo "  - ERCOT: $ERCOT_OUTPUT"
echo "  - ERCOT summary: $ERCOT_SUMMARY" 