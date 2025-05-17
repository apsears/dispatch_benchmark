#!/bin/bash
# NYISO benchmark script

# Setup
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd .. && pwd)"

# Record start time
START_TIME=$(date +%s)
START_DATETIME=$(date "+%Y-%m-%d %H:%M:%S")

# Create results directories
NYISO_OUTPUT="results/nyiso"
NYISO_SUMMARY="results/summary/nyiso"

# Settings
MAX_NODES=${1:-16}
N_JOBS=${2:-16}

mkdir -p "$NYISO_OUTPUT" "$NYISO_SUMMARY"

# Display setup information
echo "Setting up NYISO benchmark environment..."
echo "Python: $(which python3)"
echo "Working directory: $(pwd)"
echo "Repository root: $REPO_ROOT"

# Make sure the dispatch_benchmark package is in the Python path
export PYTHONPATH=$REPO_ROOT/src:$PYTHONPATH

# Check for required data files
NYISO_DATA="$REPO_ROOT/data/nyiso/2024_RealTime_LBMP.csv"

if [[ ! -f "$NYISO_DATA" ]]; then
    echo "Error: NYISO data file not found at $NYISO_DATA"
    exit 1
fi

echo "Data files found. Starting NYISO benchmark..."

# Run NYISO benchmarks
echo "Starting NYISO benchmark..."
NYISO_START_TIME=$(date +%s)

python3 comprehensive_benchmark.py \
    --prices-path "$NYISO_DATA" \
    --data-format tidy \
    --data-frequency "15T" \
    --output-dir "$NYISO_OUTPUT" \
    --max-nodes "$MAX_NODES" \
    --n-jobs "$N_JOBS" \
    --start-date "2024-01-01"
    # --end-date "2024-01-07"  # Limit to first week of 2024 for faster results

if [[ $? -ne 0 ]]; then
    echo "NYISO benchmark failed"
    exit 1
fi

NYISO_END_TIME=$(date +%s)
NYISO_ELAPSED=$((NYISO_END_TIME - NYISO_START_TIME))
NYISO_HOURS=$((NYISO_ELAPSED / 3600))
NYISO_MINUTES=$(( (NYISO_ELAPSED % 3600) / 60 ))
NYISO_SECONDS=$((NYISO_ELAPSED % 60))
echo "NYISO benchmark completed in ${NYISO_HOURS}h ${NYISO_MINUTES}m ${NYISO_SECONDS}s"

# Generate summary
echo "Generating NYISO summary..."
python3 summarize_benchmarks.py \
    --results-dir "$NYISO_OUTPUT" \
    --output-dir "$NYISO_SUMMARY" \
    --iso NYISO

echo "NYISO benchmark and summary completed successfully!"
echo "Results available in:"
echo "  - NYISO: $NYISO_OUTPUT"
echo "  - NYISO summary: $NYISO_SUMMARY"

# Calculate and display elapsed time
END_TIME=$(date +%s)
END_DATETIME=$(date "+%Y-%m-%d %H:%M:%S")
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(( (ELAPSED_TIME % 3600) / 60 ))
SECONDS=$((ELAPSED_TIME % 60))

echo ""
echo "===================================="
echo "NYISO BENCHMARK TIMING SUMMARY"
echo "===================================="
echo "Started:             ${START_DATETIME}"
echo "Completed:           ${END_DATETIME}"
echo "-----------------------------------"
echo "NYISO benchmark:     ${NYISO_HOURS}h ${NYISO_MINUTES}m ${NYISO_SECONDS}s"
echo "-----------------------------------"
echo "Total time elapsed:  ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "===================================="
