#!/bin/bash
# Main benchmark script that runs benchmarks for both ERCOT and NYISO

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
NYISO_OUTPUT="results/nyiso"
SUMMARY_OUTPUT="results/summary/combined"
ERCOT_SUMMARY="results/summary/ercot"
NYISO_SUMMARY="results/summary/nyiso"
PLOTS_DIR="results/plots"

# Settings
MAX_NODES=16
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
NYISO_DATA="$REPO_ROOT/data/nyiso/2024_RealTime_LBMP.csv"

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
ERCOT_START_TIME=$(date +%s)

python3 comprehensive_benchmark.py \
    --prices-path "$ERCOT_DATA" \
    --data-format tidy \
    --output-dir "$ERCOT_OUTPUT" \
    --max-nodes "$MAX_NODES" \
    --n-jobs "$N_JOBS" \
    --start-date "2024-01-01" \
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

echo "Starting NYISO benchmark..."
NYISO_START_TIME=$(date +%s)

python3 comprehensive_benchmark.py \
    --prices-path "$NYISO_DATA" \
    --data-format tidy \
    --data-frequency "15T" \
    --output-dir "$NYISO_OUTPUT" \
    --max-nodes "$MAX_NODES" \
    --n-jobs "$N_JOBS" \
    --start-date "2024-01-01" \
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

echo "Benchmarks completed successfully."

# Generate summaries
echo "Generating summaries..."
SUMMARY_START_TIME=$(date +%s)

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

SUMMARY_END_TIME=$(date +%s)
SUMMARY_ELAPSED=$((SUMMARY_END_TIME - SUMMARY_START_TIME))
SUMMARY_MINUTES=$(( SUMMARY_ELAPSED / 60 ))
SUMMARY_SECONDS=$((SUMMARY_ELAPSED % 60))
echo "Summary generation completed in ${SUMMARY_MINUTES}m ${SUMMARY_SECONDS}s"

echo "All benchmarks and summaries completed successfully!"
echo "Results available in:"
echo "  - ERCOT: $ERCOT_OUTPUT"
echo "  - NYISO: $NYISO_OUTPUT"
echo "  - Combined summary: $SUMMARY_OUTPUT"
echo "  - ERCOT summary: $ERCOT_SUMMARY"
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
echo "BENCHMARK TIMING SUMMARY"
echo "===================================="
echo "Started:             ${START_DATETIME}"
echo "Completed:           ${END_DATETIME}"
echo "-----------------------------------"
echo "ERCOT benchmark:     ${ERCOT_HOURS}h ${ERCOT_MINUTES}m ${ERCOT_SECONDS}s"
echo "NYISO benchmark:     ${NYISO_HOURS}h ${NYISO_MINUTES}m ${NYISO_SECONDS}s"
echo "Summary generation:  ${SUMMARY_MINUTES}m ${SUMMARY_SECONDS}s"
echo "-----------------------------------"
echo "Total time elapsed:  ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "====================================" 