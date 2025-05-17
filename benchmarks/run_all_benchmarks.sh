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
SUMMARY_OUTPUT="results/summary/combined"
PLOTS_DIR="results/plots"

# Settings
MAX_NODES=16
N_JOBS=16

mkdir -p "$SUMMARY_OUTPUT" "$PLOTS_DIR"

# Display setup information
echo "Setting up benchmark environment..."
echo "Python: $(which python3)"
echo "Working directory: $(pwd)"
echo "Repository root: $REPO_ROOT"

# Make the individual benchmark scripts executable
chmod +x run_ercot_benchmark.sh run_nyiso_benchmark.sh

# Run ERCOT and NYISO benchmarks in sequence
echo "Starting ERCOT benchmark..."
ERCOT_START_TIME=$(date +%s)

./run_ercot_benchmark.sh $MAX_NODES $N_JOBS

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

./run_nyiso_benchmark.sh $MAX_NODES $N_JOBS

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

# Generate combined summary
echo "Generating combined summary..."
SUMMARY_START_TIME=$(date +%s)

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
echo "  - ERCOT: results/ercot"
echo "  - NYISO: results/nyiso"
echo "  - Combined summary: $SUMMARY_OUTPUT"
echo "  - ERCOT summary: results/summary/ercot"
echo "  - NYISO summary: results/summary/nyiso"

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