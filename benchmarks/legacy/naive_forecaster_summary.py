#!/usr/bin/env python3
"""
Generate a summary report for the naive forecaster performance.

This script analyzes benchmark results and creates a detailed report
showing how the naive forecaster compares to other forecasting methods.
"""

import json
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt

# Configuration
RESULTS_DIR = Path("benchmark_results")
OUTPUT_DIR = Path("benchmark_results")
OUTPUT_FILE = OUTPUT_DIR / "naive_forecaster_summary.md"
CHARTS_DIR = OUTPUT_DIR / "charts"


def load_results(results_file):
    """Load benchmark results from a JSON file."""
    with open(results_file, "r") as f:
        results = json.load(f)

    # If it's the combined results format, extract just the results array
    if isinstance(results, dict):
        # Create a flat list of all results with node information
        flat_results = []
        for node, node_results in results.items():
            for result in node_results:
                result_with_node = result.copy()
                result_with_node["node"] = node
                flat_results.append(result_with_node)
        return pd.DataFrame(flat_results)
    else:
        # It's already a flat list of results for a single node
        return pd.DataFrame(results)


def calculate_metrics(df):
    """Calculate performance metrics comparing naive to other forecasters."""
    metrics = {}

    # Get revenue for each model
    oracle_revenue = df[df["model"] == "oracle_LP"]["revenue"].values[0]
    metrics["oracle_revenue"] = oracle_revenue

    # Process each forecaster
    forecasters = ["naive", "ridge"]
    for forecaster in forecasters:
        model_name = f"online_mpc_{forecaster}"
        rows = df[df["model"] == model_name]
        if not rows.empty:
            revenue = rows["revenue"].values[0]
            runtime = rows["runtime_seconds"].values[0]

            # Store basic metrics
            metrics[f"{forecaster}_revenue"] = revenue
            metrics[f"{forecaster}_runtime"] = runtime

            # Calculate gap to oracle (%)
            gap_to_oracle = ((oracle_revenue - revenue) / oracle_revenue) * 100
            metrics[f"{forecaster}_gap_to_oracle"] = gap_to_oracle

    # Calculate naive vs ridge comparison
    if "naive_revenue" in metrics and "ridge_revenue" in metrics:
        naive_revenue = metrics["naive_revenue"]
        ridge_revenue = metrics["ridge_revenue"]

        if abs(naive_revenue) > 0:  # Avoid division by zero
            # Positive means ridge is better, negative means naive is better
            pct_diff = (
                (ridge_revenue - naive_revenue) / abs(naive_revenue)
            ) * 100
            metrics["naive_vs_ridge_pct"] = pct_diff

            if pct_diff > 0:
                metrics["better_forecaster"] = "ridge"
                metrics["improvement"] = pct_diff
            else:
                metrics["better_forecaster"] = "naive"
                metrics["improvement"] = -pct_diff

    return metrics


def create_summary_report(results_file):
    """Create a markdown summary report of the naive forecaster performance."""
    try:
        # Load results data
        df = load_results(results_file)

        # Create output directories if they don't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(CHARTS_DIR, exist_ok=True)

        # Initialize the markdown content
        md_content = [
            "# Naive Forecaster Performance Summary",
            "",
            "This report shows the performance of the naive price forecaster compared to other forecasting methods.",
            "The naive forecaster predicts that future prices will be equal to the last observed price.",
            "",
            "## Performance Overview",
            "",
        ]

        # Process each node
        nodes = df["node"].unique() if "node" in df.columns else ["single_node"]

        # Prepare data for the summary table
        summary_data = []

        for node in nodes:
            # Filter to this node's data
            if "node" in df.columns:
                node_df = df[df["node"] == node]
            else:
                node_df = df

            # Calculate metrics for this node
            metrics = calculate_metrics(node_df)
            metrics["node"] = node
            summary_data.append(metrics)

            # Add node section to markdown
            md_content.extend(
                [
                    f"### Node: {node}",
                    "",
                    "| Model | Revenue ($) | Runtime (s) | Gap to Oracle (%) |",
                    "|-------|-------------|-------------|-------------------|",
                ]
            )

            # Add oracle row
            md_content.append(
                f"| Oracle (perfect foresight) | ${metrics.get('oracle_revenue', 'N/A'):,.2f} | N/A | 0% |"
            )

            # Add forecaster rows
            for forecaster in ["naive", "ridge"]:
                if f"{forecaster}_revenue" in metrics:
                    md_content.append(
                        f"| {forecaster.capitalize()} | "
                        f"${metrics[f'{forecaster}_revenue']:,.2f} | "
                        f"{metrics[f'{forecaster}_runtime']:.2f} | "
                        f"{metrics[f'{forecaster}_gap_to_oracle']:.1f}% |"
                    )

            md_content.append("")

            # Add comparison section
            if "naive_vs_ridge_pct" in metrics:
                if metrics["better_forecaster"] == "ridge":
                    md_content.append(
                        f"Ridge forecaster outperforms naive by: **+{metrics['improvement']:.1f}%**"
                    )
                else:
                    md_content.append(
                        f"Naive forecaster outperforms ridge by: **+{metrics['improvement']:.1f}%**"
                    )

                md_content.append("")

            # Generate revenue comparison chart for this node
            if "naive_revenue" in metrics and "ridge_revenue" in metrics:
                chart_filename = f"{node}_revenue_comparison.png"
                chart_path = CHARTS_DIR / chart_filename

                try:
                    plt.figure(figsize=(10, 6))
                    models = ["Oracle", "Naive", "Ridge"]
                    revenues = [
                        metrics.get("oracle_revenue", 0),
                        metrics.get("naive_revenue", 0),
                        metrics.get("ridge_revenue", 0),
                    ]

                    bars = plt.bar(models, revenues)

                    # Add value labels on top of the bars
                    for bar, revenue in zip(bars, revenues):
                        plt.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 50,
                            f"${revenue:,.0f}",
                            ha="center",
                            va="bottom",
                        )

                    plt.title(f"Revenue Comparison - {node}")
                    plt.ylabel("Revenue ($)")
                    plt.grid(axis="y", linestyle="--", alpha=0.7)

                    # Make sure the directory exists
                    os.makedirs(CHARTS_DIR, exist_ok=True)

                    plt.savefig(chart_path)
                    plt.close()

                    # Add the chart to the markdown
                    md_content.extend(
                        [
                            f"![Revenue Comparison - {node}](charts/{chart_filename})",
                            "",
                        ]
                    )
                except Exception as e:
                    print(f"Error generating chart: {e}")
                    # Add text-based comparison instead
                    md_content.extend(
                        [
                            "**Revenue Comparison:**",
                            f"- Oracle: ${metrics.get('oracle_revenue', 0):,.2f}",
                            f"- Naive: ${metrics.get('naive_revenue', 0):,.2f}",
                            f"- Ridge: ${metrics.get('ridge_revenue', 0):,.2f}",
                            "",
                        ]
                    )

        # Create summary dataframe
        summary_df = pd.DataFrame(summary_data)

        # Add overall summary section if we have multiple nodes
        if len(nodes) > 1:
            md_content.extend(
                [
                    "## Overall Summary",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                ]
            )

            # Calculate how often naive is better than ridge
            if "better_forecaster" in summary_df.columns:
                naive_better_count = sum(
                    summary_df["better_forecaster"] == "naive"
                )
                total_nodes = len(summary_df)
                naive_better_pct = (naive_better_count / total_nodes) * 100

                md_content.append(
                    f"| Nodes where naive outperforms ridge | {naive_better_count}/{total_nodes} ({naive_better_pct:.1f}%) |"
                )

            # Calculate average metrics
            if (
                "naive_gap_to_oracle" in summary_df.columns
                and "ridge_gap_to_oracle" in summary_df.columns
            ):
                avg_naive_gap = summary_df["naive_gap_to_oracle"].mean()
                avg_ridge_gap = summary_df["ridge_gap_to_oracle"].mean()

                md_content.append(
                    f"| Average gap to oracle (naive) | {avg_naive_gap:.1f}% |"
                )
                md_content.append(
                    f"| Average gap to oracle (ridge) | {avg_ridge_gap:.1f}% |"
                )

            md_content.append("")

        # Add conclusion
        md_content.extend(
            [
                "## Conclusion",
                "",
                "The naive forecaster serves as an important baseline for comparison with more sophisticated forecasting methods.",
                "In some market conditions, particularly during stable price periods, the naive approach can perform well.",
                "However, during volatile periods or when there are predictable price patterns, more sophisticated forecasters",
                "may provide better results.",
                "",
                "This comparison helps demonstrate the value of appropriate forecasting methods for different market conditions",
                "and is useful for stakeholder communication about model performance.",
                "",
            ]
        )

        # Write the markdown file
        with open(OUTPUT_FILE, "w") as f:
            f.write("\n".join(md_content))

        print(f"Summary report created: {OUTPUT_FILE}")
        return True

    except Exception as e:
        print(f"Error creating summary report: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point for the script."""
    print("Generating naive forecaster summary report...")

    # Debug directory listing
    print(f"Current working directory: {os.getcwd()}")
    print("Contents of benchmark_results directory:")
    try:
        for f in os.listdir(RESULTS_DIR):
            print(f"  - {f}")
    except Exception as e:
        print(f"Error listing directory: {e}")

    # Check for result files
    combined_results = RESULTS_DIR / "combined_results.json"
    print(f"Looking for combined results at: {combined_results}")
    print(f"File exists: {combined_results.exists()}")

    if combined_results.exists():
        print(f"Using combined results from {combined_results}")
        try:
            result = create_summary_report(combined_results)
            print(f"create_summary_report returned: {result}")
        except Exception as e:
            print(f"Error in create_summary_report: {e}")
            import traceback

            traceback.print_exc()
            result = False
    else:
        # Look for individual node result files
        result_files = list(RESULTS_DIR.glob("*_results.json"))
        if result_files:
            print(f"Using individual result file: {result_files[0]}")
            try:
                result = create_summary_report(result_files[0])
                print(f"create_summary_report returned: {result}")
            except Exception as e:
                print(f"Error in create_summary_report: {e}")
                import traceback

                traceback.print_exc()
                result = False
        else:
            print("No benchmark result files found!")
            return False

    if result:
        print("Summary generation complete!")
        return True
    else:
        print("Failed to generate summary.")
        return False


if __name__ == "__main__":
    main()
