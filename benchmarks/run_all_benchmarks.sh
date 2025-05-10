#!/bin/bash
# Main benchmark script that runs benchmarks for both ERCOT and NYISO

# Setup
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd .. && pwd)"

# Activate virtual environment if not already active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Create results directories
ERCOT_OUTPUT="results/ercot"
NYISO_OUTPUT="results/nyiso"
SUMMARY_OUTPUT="results/summary/combined"
ERCOT_SUMMARY="results/summary/ercot"
NYISO_SUMMARY="results/summary/nyiso"
PLOTS_DIR="results/plots"

# Settings
MAX_NODES=100
N_JOBS=32    

mkdir -p "$ERCOT_OUTPUT" "$NYISO_OUTPUT" "$SUMMARY_OUTPUT" "$ERCOT_SUMMARY" "$NYISO_SUMMARY" "$PLOTS_DIR"

# Display setup information
echo "Setting up benchmark environment..."
echo "Python: $(which python3)"
echo "Working directory: $(pwd)"
echo "Repository root: $REPO_ROOT"

# Make sure the virtual_energy package is in the Python path
export PYTHONPATH=$REPO_ROOT/src:$PYTHONPATH

# Check for required data files
ERCOT_DATA="$REPO_ROOT/data/ercot/2024_RealTime_SPP.csv"
NYISO_DATA="$REPO_ROOT/data/nyiso/2024_DayAhead_LBMP.csv"

if [[ ! -f "$ERCOT_DATA" ]]; then
    echo "Error: ERCOT data file not found at $ERCOT_DATA"
    exit 1
fi

if [[ ! -f "$NYISO_DATA" ]]; then
    echo "Error: NYISO data file not found at $NYISO_DATA"
    exit 1
fi

echo "Data files found. Starting benchmarks..."

# Run ERCOT and NYISO benchmarks in sequence (not parallel) to diagnose issues
echo "Starting ERCOT benchmark..."
python3 comprehensive_benchmark.py \
    --prices-path "$ERCOT_DATA" \
    --data-format tidy \
    --output-dir "$ERCOT_OUTPUT" \
    --max-nodes "$MAX_NODES" \
    --n-jobs "$N_JOBS"

if [[ $? -ne 0 ]]; then
    echo "ERCOT benchmark failed"
    exit 1
fi

echo "Starting NYISO benchmark..."
python3 comprehensive_benchmark.py \
    --prices-path "$NYISO_DATA" \
    --data-format tidy \
    --output-dir "$NYISO_OUTPUT" \
    --max-nodes "$MAX_NODES" \
    --n-jobs "$N_JOBS"

if [[ $? -ne 0 ]]; then
    echo "NYISO benchmark failed"
    exit 1
fi

echo "Benchmarks completed successfully."

# Generate summaries
echo "Generating ERCOT summary..."
python3 summarize_benchmarks.py \
    --results-dir "$ERCOT_OUTPUT" \
    --output-dir "$ERCOT_SUMMARY" \
    --iso ERCOT

echo "Generating NYISO summary..."
python3 summarize_benchmarks.py \
    --results-dir "$NYISO_OUTPUT" \
    --output-dir "$NYISO_SUMMARY" \
    --iso NYISO

echo "Generating combined summary..."
python3 summarize_benchmarks.py \
    --results-dir "results" \
    --output-dir "$SUMMARY_OUTPUT" \
    --iso BOTH

echo "All benchmarks and summaries completed successfully!"
echo "Results available in:"
echo "  - ERCOT: $ERCOT_OUTPUT"
echo "  - NYISO: $NYISO_OUTPUT"
echo "  - Combined summary: $SUMMARY_OUTPUT"
echo "  - ERCOT summary: $ERCOT_SUMMARY"
echo "  - NYISO summary: $NYISO_SUMMARY" 