#!/usr/bin/env python3
"""
Enhanced benchmark script that includes and compares the naive forecaster performance.

This script runs node-by-node tests across a specified set of ERCOT nodes,
comparing the naive forecaster against more sophisticated models and theoretical optimums.
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path
import traceback

# Import the benchmark function from the models module
from virtual_energy.models.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmarks including naive baseline"
    )
    parser.add_argument(
        "--prices-path",
        type=str,
        default="data/ercot/2024_RealTime_SPP.csv",
        help="Path to price data file (defaults to tidy format)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--data-format",
        type=str,
        choices=["tidy", "wide"],
        default="tidy",
        help="Format of the input data: tidy (default) or wide",
    )
    parser.add_argument(
        "--data-frequency",
        type=str,
        default=None,
        help="Target frequency for resampling data (e.g., '15T' for 15 minutes). Uses original frequency if not specified.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for benchmarking (YYYY-MM-DD). Uses first date in dataset if not specified.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for benchmarking (YYYY-MM-DD). Uses last date in dataset if not specified.",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        nargs="+",
        help="Specific nodes to benchmark (default: use nodes defined in benchmark.py)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=20,
        help="Maximum number of nodes to benchmark",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Print benchmark information
    print("=== COMPREHENSIVE BENCHMARK (INCLUDING NAIVE BASELINE) ===")
    print(f"Data source: {args.prices_path}")
    print(f"Data format: {args.data_format}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max nodes: {args.max_nodes}")
    print(f"Parallel jobs: {args.n_jobs}")

    # Add data frequency to print output if specified
    if args.data_frequency:
        print(f"Data frequency: {args.data_frequency} (resampling enabled)")

    # Add a note about the naive forecaster
    print("\nThis benchmark includes the naive forecaster as a baseline.")
    print("The naive forecaster predicts future prices = last observed price.")
    print("It provides a reference point for evaluating more sophisticated models.")

    # Verify that the price file exists
    if not Path(args.prices_path).exists():
        print(f"Error: Price data file not found: {args.prices_path}")
        print("Please download or specify the correct path to ERCOT price data.")
        sys.exit(1)

    # Run the benchmark with the parsed arguments
    results_summary = run_benchmark(
        prices_path=args.prices_path,
        start_date=args.start_date,  # Pass start_date to run_benchmark
        end_date=args.end_date,  # Pass end_date to run_benchmark
        output_dir=args.output_dir,
        nodes=args.nodes,
        max_nodes=args.max_nodes,
        n_jobs=args.n_jobs,
        data_format=args.data_format,
        data_frequency=args.data_frequency,
    )

    # Print some analysis specifically highlighting the naive model performance
    print("\n=== NAIVE BASELINE ANALYSIS ===")
    try:
        # Load the combined results file if it exists
        results_file = Path(args.output_dir) / "combined_results.csv"
        if results_file.exists():
            results_df = pd.read_csv(results_file)

            # Filter to just naive and ridge models for comparison
            comparison = results_df[
                results_df["model"].isin(
                    ["online_mpc_naive", "online_mpc_ridge", "oracle_lp"]
                )
            ]

            # Group by node and calculate metrics
            metrics = []
            for node, group in comparison.groupby("node"):
                naive_row = group[group["model"] == "online_mpc_naive"]
                ridge_row = group[group["model"] == "online_mpc_ridge"]
                oracle_row = group[group["model"] == "oracle_lp"]

                if not naive_row.empty and not ridge_row.empty and not oracle_row.empty:
                    naive_revenue = naive_row["revenue"].iloc[0]
                    ridge_revenue = ridge_row["revenue"].iloc[0]
                    oracle_revenue = oracle_row["revenue"].iloc[0]

                    # Calculate performance metrics
                    naive_vs_ridge = (
                        (ridge_revenue - naive_revenue) / abs(naive_revenue)
                    ) * 100
                    naive_vs_oracle = (
                        (oracle_revenue - naive_revenue) / oracle_revenue
                    ) * 100
                    ridge_vs_oracle = (
                        (oracle_revenue - ridge_revenue) / oracle_revenue
                    ) * 100

                    metrics.append(
                        {
                            "node": node,
                            "naive_revenue": naive_revenue,
                            "ridge_revenue": ridge_revenue,
                            "oracle_revenue": oracle_revenue,
                            "naive_vs_ridge": naive_vs_ridge,
                            "naive_vs_oracle": naive_vs_oracle,
                            "ridge_vs_oracle": ridge_vs_oracle,
                        }
                    )

            if metrics:
                # Convert to DataFrame and calculate summary statistics
                metrics_df = pd.DataFrame(metrics)

                print("Average performance across all nodes:")
                print(
                    f"  Ridge forecaster vs. Naive: {metrics_df['naive_vs_ridge'].mean():.1f}% (+ better, - worse)"
                )
                print(
                    f"  Gap to oracle: Naive: {metrics_df['naive_vs_oracle'].mean():.1f}%, Ridge: {metrics_df['ridge_vs_oracle'].mean():.1f}%"
                )

                # Calculate how often naive outperforms ridge
                naive_better_count = sum(metrics_df["naive_vs_ridge"] < 0)
                total_nodes = len(metrics_df)
                print(
                    f"\nNaive outperforms Ridge in {naive_better_count} out of {total_nodes} nodes ({naive_better_count/total_nodes*100:.1f}%)"
                )

                # Save the metrics to a CSV file
                metrics_file = Path(args.output_dir) / "naive_comparison_metrics.csv"
                metrics_df.to_csv(metrics_file, index=False)
                print(f"\nDetailed metrics saved to: {metrics_file}")
    except Exception as e:
        print(f"Error analyzing naive baseline: {e}")
        import traceback

        traceback.print_exc()

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
