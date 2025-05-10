#!/usr/bin/env python3
"""
Visualization script for battery dispatch benchmark results.

This script generates plots and tables from benchmark results to help compare
the performance of different battery dispatch models.
"""

import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(
    results_dir="benchmark_results", combined_file="combined_results.json"
):
    """Load benchmark results from the specified directory."""
    results_path = Path(results_dir) / combined_file

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path) as f:
        data = json.load(f)

    return data


def create_summary_df(results):
    """Create a summary dataframe from the results dictionary."""
    summary_data = []

    for node, node_results in results.items():
        for model_result in node_results:
            # Extract basic info
            record = {
                "node": node,
                "model": model_result["model"],
                "revenue": model_result["revenue"],
                "runtime_seconds": model_result["runtime_seconds"],
            }

            # Add model-specific parameters if available
            if "forecast_model" in model_result:
                record["forecast_model"] = model_result["forecast_model"]
            if "percentile" in model_result:
                record["percentile"] = model_result["percentile"]
            if "horizon" in model_result:
                record["horizon"] = model_result["horizon"]

            summary_data.append(record)

    return pd.DataFrame(summary_data)


def plot_revenue_comparison(df, output_dir=None):
    """Create a bar plot comparing revenue across models and nodes."""
    plt.figure(figsize=(12, 8))

    # Group by node and model, and create a pivot table
    revenue_pivot = df.pivot_table(
        index="model", columns="node", values="revenue", aggfunc="mean"
    )

    # Plot
    ax = revenue_pivot.plot(kind="bar", rot=45)
    plt.title("Revenue Comparison by Model and Node", fontsize=14)
    plt.ylabel("Revenue ($)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend().set_visible(False)  # Remove legend
    plt.tight_layout()

    # Save if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            Path(output_dir) / "revenue_comparison.png", dpi=300, bbox_inches="tight"
        )

    return ax


def plot_runtime_comparison(df, output_dir=None):
    """Create a bar plot comparing runtime across models."""
    plt.figure(figsize=(12, 6))

    # Group by model and get mean runtime
    runtime_by_model = df.groupby("model")["runtime_seconds"].mean().sort_values()

    # Plot
    ax = runtime_by_model.plot(kind="bar", color="skyblue", rot=45)
    plt.title("Average Runtime by Model", fontsize=14)
    plt.ylabel("Runtime (seconds)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            Path(output_dir) / "runtime_comparison.png", dpi=300, bbox_inches="tight"
        )

    return ax


def plot_revenue_vs_oracle(df, output_dir=None):
    """Create a plot showing how each model performs as % of oracle revenue."""
    # First, get the oracle_LP revenue for each node as the reference
    oracle_df = df[df["model"] == "oracle_LP"].copy()
    oracle_df = oracle_df[["node", "revenue"]].rename(
        columns={"revenue": "oracle_revenue"}
    )

    # Merge with the main dataframe
    merged_df = df.merge(oracle_df, on="node")

    # Calculate percentage of oracle revenue
    merged_df["pct_of_oracle"] = (
        merged_df["revenue"] / merged_df["oracle_revenue"] * 100
    )

    # Plot
    plt.figure(figsize=(12, 6))

    models_df = merged_df[
        merged_df["model"] != "oracle_LP"
    ]  # exclude oracle from comparison

    ax = sns.barplot(x="model", y="pct_of_oracle", hue="node", data=models_df)
    plt.title("Model Performance as % of Oracle LP Revenue", fontsize=14)
    plt.ylabel("% of Oracle Revenue", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.axhline(y=100, color="r", linestyle="--", alpha=0.7)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend().set_visible(False)  # Remove legend
    plt.tight_layout()

    # Save if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            Path(output_dir) / "revenue_vs_oracle.png", dpi=300, bbox_inches="tight"
        )

    return plt.gca()


def plot_revenue_histograms(df, output_dir=None):
    """Create histograms comparing revenue distribution vs the LP oracle."""
    # First, get the oracle_LP revenue for each node as the reference
    oracle_df = df[df["model"] == "oracle_LP"].copy()
    oracle_df = oracle_df[["node", "revenue"]].rename(
        columns={"revenue": "oracle_revenue"}
    )

    # Merge with the main dataframe
    merged_df = df.merge(oracle_df, on="node")

    # Calculate percentage of oracle revenue
    merged_df["pct_of_oracle"] = (
        merged_df["revenue"] / merged_df["oracle_revenue"] * 100
    )

    # Create a figure with subplots for each model (excluding oracle)
    models = merged_df["model"].unique().tolist()
    if "oracle_LP" in models:
        models.remove("oracle_LP")  # Remove oracle from the list of models to plot

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 6), sharey=True)

    # If there's only one model, axes won't be an array
    if len(models) == 1:
        axes = [axes]

    for i, model in enumerate(models):
        model_data = merged_df[merged_df["model"] == model]

        # Create a histogram of the percentage of oracle values across nodes
        # Each bar represents how many nodes achieved a specific percentage range
        sns.histplot(
            x=model_data["pct_of_oracle"],
            bins=15,
            kde=True,
            ax=axes[i],
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
        )

        # Add vertical line at 100% (oracle performance)
        axes[i].axvline(x=100, color="r", linestyle="--", alpha=0.7)

        # Set title and labels
        axes[i].set_title(f"{model} Model", fontsize=12)
        axes[i].set_xlabel("% of Oracle Revenue", fontsize=10)

        # Only set y-label for the first subplot
        if i == 0:
            axes[i].set_ylabel("Number of Nodes", fontsize=10)
        else:
            axes[i].set_ylabel("")

        # Ensure y-axis starts at 0
        axes[i].set_ylim(bottom=0)

        # Add mean line
        mean_pct = model_data["pct_of_oracle"].mean()
        axes[i].axvline(x=mean_pct, color="green", linestyle="-", alpha=0.7)
        axes[i].text(
            mean_pct + 1,
            axes[i].get_ylim()[1] * 0.9,
            f"Mean: {mean_pct:.1f}%",
            color="green",
            fontsize=9,
        )

    plt.tight_layout()

    # Save if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            Path(output_dir) / "revenue_histograms.png", dpi=300, bbox_inches="tight"
        )

    return fig


def create_summary_table(df, output_dir=None):
    """Create a summary table with key metrics."""
    # Group by model and calculate mean revenue and runtime
    summary = (
        df.groupby("model")
        .agg(
            {
                "revenue": ["mean", "std", "min", "max"],
                "runtime_seconds": ["mean", "std", "min", "max"],
            }
        )
        .round(2)
    )

    # Calculate percentage of oracle revenue
    oracle_mean = df[df["model"] == "oracle_LP"]["revenue"].mean()

    model_means = df.groupby("model")["revenue"].mean()
    model_pcts = (model_means / oracle_mean * 100).round(1)

    summary["pct_of_oracle"] = model_pcts

    # Save if output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        summary.to_csv(Path(output_dir) / "summary_table.csv")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Visualize battery dispatch benchmark results"
    )
    parser.add_argument(
        "--results-dir",
        default="benchmark_results",
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_plots",
        help="Directory to save visualization outputs",
    )

    args = parser.parse_args()

    # Load the results
    try:
        results = load_results(args.results_dir)
        print(f"Loaded results from {args.results_dir}/combined_results.json")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Create summary dataframe
    df = create_summary_df(results)
    print(f"Found {len(df)} model results across {df['node'].nunique()} nodes")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate visualizations
    print("Generating revenue comparison plot...")
    plot_revenue_comparison(df, args.output_dir)

    print("Generating runtime comparison plot...")
    plot_runtime_comparison(df, args.output_dir)

    print("Generating revenue vs oracle plot...")
    plot_revenue_vs_oracle(df, args.output_dir)

    print("Generating revenue histograms...")
    plot_revenue_histograms(df, args.output_dir)

    print("Generating summary table...")
    summary = create_summary_table(df, args.output_dir)
    print(summary)

    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
