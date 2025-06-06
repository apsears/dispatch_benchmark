#!/bin/bash
# ERCOT benchmark script

# Setup
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd .. && pwd)"

# Record start time
START_TIME=$(date +%s)
START_DATETIME=$(date "+%Y-%m-%d %H:%M:%S")

# Create results directories
ERCOT_OUTPUT="results/ercot"
ERCOT_SUMMARY="results/summary/ercot"

# Settings
MAX_NODES=${1:-16}
N_JOBS=${2:-16}

mkdir -p "$ERCOT_OUTPUT" "$ERCOT_SUMMARY"

# Display setup information
echo "Setting up ERCOT benchmark environment..."
echo "Python: $(which python3)"
echo "Working directory: $(pwd)"
echo "Repository root: $REPO_ROOT"

# Make sure the dispatch_benchmark package is in the Python path
export PYTHONPATH=$REPO_ROOT/src:$PYTHONPATH

# Check for required data files
ERCOT_DATA="$REPO_ROOT/data/ercot/2024_RealTime_SPP.csv"

if [[ ! -f "$ERCOT_DATA" ]]; then
    echo "Error: ERCOT data file not found at $ERCOT_DATA"
    exit 1
fi

echo "Data files found. Starting ERCOT benchmark..."

# Run ERCOT benchmarks
echo "Starting ERCOT benchmark..."
ERCOT_START_TIME=$(date +%s)

python3 comprehensive_benchmark.py \
    --prices-path "$ERCOT_DATA" \
    --data-format tidy \
    --output-dir "$ERCOT_OUTPUT" \
    --max-nodes "$MAX_NODES" \
    --n-jobs "$N_JOBS" \
    --start-date "2024-01-01"
    # --end-date "2024-01-14"  # Limit to first week of 2024 for faster results

if [[ $? -ne 0 ]]; then
    echo "ERCOT benchmark failed"
    exit 1
fi

ERCOT_END_TIME=$(date +%s)
ERCOT_ELAPSED=$((ERCOT_END_TIME - ERCOT_START_TIME))
ERCOT_HOURS=$((ERCOT_ELAPSED / 3600))
ERCOT_MINUTES=$(( (ERCOT_ELAPSED % 3600) / 60 ))
ERCOT_SECONDS=$((ERCOT_ELAPSED % 60))
echo "ERCOT benchmark completed in ${ERCOT_HOURS}h ${ERCOT_MINUTES}m ${ERCOT_SECONDS}s"

# Generate summary
echo "Generating ERCOT summary..."
python3 summarize_benchmarks.py \
    --results-dir "$ERCOT_OUTPUT" \
    --output-dir "$ERCOT_SUMMARY" \
    --iso ERCOT

echo "ERCOT benchmark and summary completed successfully!"
echo "Results available in:"
echo "  - ERCOT: $ERCOT_OUTPUT"
echo "  - ERCOT summary: $ERCOT_SUMMARY"

# Calculate and display elapsed time
END_TIME=$(date +%s)
END_DATETIME=$(date "+%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(( (ELAPSED_TIME % 3600) / 60 ))
SECONDS=$((ELAPSED_TIME % 60))

echo ""
echo "===================================="
echo "ERCOT BENCHMARK TIMING SUMMARY"
echo "===================================="
echo "Started:             ${START_DATETIME}"
echo "Completed:           ${END_DATETIME}"
echo "-----------------------------------"
echo "ERCOT benchmark:     ${ERCOT_HOURS}h ${ERCOT_MINUTES}m ${ERCOT_SECONDS}s"
echo "-----------------------------------"
echo "Total time elapsed:  ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "===================================="
