#!/usr/bin/env python3
"""
Benchmark script to compare selected battery dispatch models.

This script runs the following models on a set of nodes from prices_wide.csv:
- oracle_LP: Omniscient LP model with perfect price knowledge
- online_mpc (ridge): Non-clairvoyant MPC with ridge regression forecasting
- online_quartile: Percentile-based model with no forecasting (10%, 25%, 45%)

Results are saved after each model to prevent data loss.
Multiprocessing is used to speed up execution across nodes.
"""

import argparse
import json
import os
import time
import pandas as pd
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Import models from the new package structure
from virtual_energy.optimisers.battery_config import BatteryConfig
from virtual_energy.optimisers.oracle_lp import (
    create_and_solve_model as oracle_lp_solve,
    tidy,
)
from virtual_energy.optimisers.online_mpc import (
    run_mpc as online_mpc_run,
    FORECASTERS as online_mpc_forecasters,
)
from virtual_energy.forecasters.quartile import quartile_dispatch

# Global battery config for multiprocessing
DEFAULT_CFG = BatteryConfig(
    delta_t=0.25,
    eta_chg=0.95,
    p_max_mw=25,
    e_max_mwh=200,
)


def load_prices(prices_path, node=None, max_nodes=100):
    """Load price data for a specific node or all nodes."""
    print(f"Loading prices from {prices_path}")

    # Load concatenated data from CSV
    raw_df = pd.read_csv(prices_path)

    # Check if it's already in wide format or needs transformation
    if "deliveryDate" in raw_df.columns and "settlementPoint" in raw_df.columns:
        print("Converting from long format to wide format...")

        # Create timestamp column from date, hour, and interval
        raw_df["timestamp"] = pd.to_datetime(
            raw_df["deliveryDate"]
            + " "
            + (raw_df["deliveryHour"] - 1).astype(str)
            + ":"
            + ((raw_df["deliveryInterval"] - 1) * 15).astype(str).str.zfill(2)
        )

        # Get the unique settlement points
        settlement_points = raw_df["settlementPoint"].unique()
        print(f"Found {len(settlement_points)} unique settlement points")

        # Take only the first max_nodes settlement points
        if len(settlement_points) > max_nodes:
            settlement_points = settlement_points[:max_nodes]
            print(f"Filtering to first {max_nodes} settlement points")
            raw_df = raw_df[raw_df["settlementPoint"].isin(settlement_points)]

        # Check for duplicates in the timestamp and settlementPoint combination
        dup_count = raw_df.duplicated(
            subset=["timestamp", "settlementPoint"]
        ).sum()
        if dup_count > 0:
            print(
                f"Found {dup_count} duplicate timestamp-settlementPoint combinations. Dropping duplicates..."
            )
            raw_df = raw_df.drop_duplicates(
                subset=["timestamp", "settlementPoint"], keep="first"
            )

        # Pivot to wide format
        df = raw_df.pivot(
            index="timestamp",
            columns="settlementPoint",
            values="settlementPointPrice",
        ).reset_index()

        print(
            f"Created wide format with {len(df)} rows and {len(df.columns)} columns"
        )
    else:
        # Already in wide format
        df = raw_df

    # If no node specified, return the dataframe with all nodes
    if node is None:
        return df

    # Check if node exists
    if node not in df.columns:
        raise ValueError(
            f"Node '{node}' not found in price data. Available nodes: {[col for col in df.columns if col != 'timestamp']}"
        )

    # Create a tidy dataframe for the specified node
    return tidy(df, node)


def filter_week(df, start_date=None):
    """Filter dataframe to include just one week of data."""
    if start_date is None:
        # Use the first date in the dataset
        start_date = df["timestamp"].min().date()
    else:
        start_date = pd.to_datetime(start_date).date()

    end_date = start_date + timedelta(days=7)

    # Filter to the week
    return df[
        (df["timestamp"].dt.date >= start_date)
        & (df["timestamp"].dt.date < end_date)
    ]


def run_oracle_lp(prices_df, cfg):
    """Run the oracle LP model with perfect knowledge."""
    print("Running Oracle LP model...")
    start_time = time.time()

    _, dispatch = oracle_lp_solve(prices_df, cfg)

    elapsed = time.time() - start_time
    revenue = dispatch["Revenue$"].sum()

    return {
        "model": "oracle_LP",
        "revenue": float(revenue),
        "runtime_seconds": elapsed,
        "dispatch": dispatch.to_dict(orient="records"),
    }


def run_online_mpc(prices_df, cfg, forecast_model="ridge", horizon=32):
    """Run the online MPC model with the specified forecast model."""
    print(f"Running Online MPC with {forecast_model} model...")
    start_time = time.time()

    # Convert to series format needed by online_mpc
    price_series = prices_df.set_index("timestamp")["SettlementPointPrice"]

    dispatch = online_mpc_run(
        price_series, horizon, forecaster=online_mpc_forecasters[forecast_model]
    )

    elapsed = time.time() - start_time
    revenue = dispatch["Revenue$"].sum()

    return {
        "model": f"online_mpc_{forecast_model}",
        "forecast_model": forecast_model,
        "horizon": horizon,
        "revenue": float(revenue),
        "runtime_seconds": elapsed,
        "dispatch": dispatch.to_dict(orient="records"),
    }


def run_online_quartile(prices_df, pct=10, window=672):
    """Run the online quartile model."""
    print(f"Running Online Quartile model (pct={pct}, window={window})...")
    start_time = time.time()

    # Convert to series format needed by online_quartile
    price_series = prices_df.set_index("timestamp")["SettlementPointPrice"]

    dispatch = quartile_dispatch(price_series, pct, window)

    elapsed = time.time() - start_time
    revenue = dispatch["Revenue$"].sum()

    return {
        "model": f"online_quartile_p{pct}",
        "percentile": pct,
        "window": window,
        "revenue": float(revenue),
        "runtime_seconds": elapsed,
        "dispatch": dispatch.to_dict(orient="records"),
    }


def save_results(results, output_file):
    """Save results to a file."""
    # Remove the dispatch data to make the results file smaller
    # We'll keep just the summary for each model
    summary_results = []
    for model_result in results:
        summary = {k: v for k, v in model_result.items() if k != "dispatch"}
        summary_results.append(summary)

    # Convert to JSON serializable format
    json_results = json.dumps(summary_results, indent=2, default=str)

    with open(output_file, "w") as f:
        f.write(json_results)

    return summary_results


# Define worker function for multiprocessing outside of run_benchmark
def process_node(node, all_prices, start_date, output_dir):
    print(f"\nBenchmarking models for node: {node}")

    try:
        # Get tidy price data for this node
        prices_df = tidy(all_prices, node)

        # Filter to one week
        weekly_prices = filter_week(prices_df, start_date)
        print(
            f"Using data from {weekly_prices['timestamp'].min()} to {weekly_prices['timestamp'].max()}"
        )

        # Initialize results for this node
        node_results = []
        results_file = Path(output_dir) / f"{node}_results.json"

        # 1. Run Oracle LP
        try:
            result = run_oracle_lp(weekly_prices, DEFAULT_CFG)
            node_results.append(result)
            # Save intermediate results
            save_results(node_results, results_file)
            print(
                f"Oracle LP: ${result['revenue']:,.2f} in {result['runtime_seconds']:.2f}s"
            )
        except Exception as e:
            print(f"Error running Oracle LP: {e}")

        # 2. Run Online MPC with ridge model only
        try:
            result = run_online_mpc(weekly_prices, DEFAULT_CFG, "ridge")
            node_results.append(result)
            # Save intermediate results
            save_results(node_results, results_file)
            print(
                f"Online MPC (ridge): ${result['revenue']:,.2f} in {result['runtime_seconds']:.2f}s"
            )
        except Exception as e:
            print(f"Error running Online MPC (ridge): {e}")

        # 2.1 Run Online MPC with naive forecaster (baseline)
        try:
            result = run_online_mpc(weekly_prices, DEFAULT_CFG, "naive")
            node_results.append(result)
            # Save intermediate results
            save_results(node_results, results_file)
            print(
                f"Online MPC (naive): ${result['revenue']:,.2f} in {result['runtime_seconds']:.2f}s"
            )
        except Exception as e:
            print(f"Error running Online MPC (naive): {e}")

        # 3. Run Online Quartile with different percentiles
        for pct in [10, 25, 45]:
            try:
                result = run_online_quartile(weekly_prices, pct)
                node_results.append(result)
                # Save intermediate results
                save_results(node_results, results_file)
                print(
                    f"Online Quartile (p{pct}): ${result['revenue']:,.2f} in {result['runtime_seconds']:.2f}s"
                )
            except Exception as e:
                print(f"Error running Online Quartile (p{pct}): {e}")

        return node, node_results

    except Exception as e:
        print(f"Error processing node {node}: {e}")
        return node, []


def run_benchmark(
    prices_path,
    start_date=None,
    nodes=None,
    output_dir="benchmark_results",
    n_jobs=None,
    max_nodes=100,
    data_format="wide",
):
    """Run benchmark for all models on specified nodes.

    Parameters:
    -----------
    prices_path : str
        Path to the prices CSV file
    start_date : str, optional
        Start date for the week (YYYY-MM-DD)
    nodes : list, optional
        Specific nodes to benchmark
    output_dir : str, default="benchmark_results"
        Directory to save results
    n_jobs : int, optional
        Number of processes to use
    max_nodes : int, default=100
        Maximum number of settlement points to use
    data_format : str, default="wide"
        Format of the input data ('wide' or 'tidy')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load all price data based on format
    if data_format.lower() == "tidy":
        # For tidy format, we need to load and transform
        print(f"Loading tidy format data from {prices_path}")
        raw_df = pd.read_csv(prices_path)
        
        # Check for different tidy format variants
        if all(col in raw_df.columns for col in ["node", "timestamp", "price"]):
            # ERCOT simplified format (node, timestamp, price)
            print("Detected ERCOT simplified tidy format with node, timestamp, price columns")
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_dtype(raw_df["timestamp"]):
                raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
                
            # Filter to max_nodes if specified
            unique_nodes = raw_df["node"].unique()
            print(f"Found {len(unique_nodes)} unique nodes")
            if len(unique_nodes) > max_nodes:
                unique_nodes = unique_nodes[:max_nodes]
                print(f"Filtering to first {max_nodes} nodes")
                raw_df = raw_df[raw_df["node"].isin(unique_nodes)]
                
            # Drop duplicates if necessary
            dup_count = raw_df.duplicated(subset=["timestamp", "node"]).sum()
            if dup_count > 0:
                print(f"Found {dup_count} duplicate timestamp-node combinations. Dropping duplicates...")
                raw_df = raw_df.drop_duplicates(subset=["timestamp", "node"], keep="first")
                
            # Pivot to wide format
            all_prices = raw_df.pivot(
                index="timestamp", 
                columns="node", 
                values="price"
            ).reset_index()
            
            print(f"Converted ERCOT tidy to wide format with {len(all_prices)} rows and {len(all_prices.columns)} columns")
        
        elif all(col in raw_df.columns for col in ["zone", "timestamp", "price"]):
            # NYISO format (zone, timestamp, price)
            print("Detected NYISO tidy format with zone, timestamp, price columns")
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_dtype(raw_df["timestamp"]):
                raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
                
            # Filter to max_nodes if specified
            unique_zones = raw_df["zone"].unique()
            print(f"Found {len(unique_zones)} unique zones")
            if len(unique_zones) > max_nodes:
                unique_zones = unique_zones[:max_nodes]
                print(f"Filtering to first {max_nodes} zones")
                raw_df = raw_df[raw_df["zone"].isin(unique_zones)]
                
            # Drop duplicates if necessary
            dup_count = raw_df.duplicated(subset=["timestamp", "zone"]).sum()
            if dup_count > 0:
                print(f"Found {dup_count} duplicate timestamp-zone combinations. Dropping duplicates...")
                raw_df = raw_df.drop_duplicates(subset=["timestamp", "zone"], keep="first")
                
            # Pivot to wide format
            all_prices = raw_df.pivot(
                index="timestamp", 
                columns="zone", 
                values="price"
            ).reset_index()
            
            print(f"Converted NYISO tidy to wide format with {len(all_prices)} rows and {len(all_prices.columns)} columns")
            
        elif all(col in raw_df.columns for col in ["deliveryDate", "settlementPoint", "settlementPointPrice", "deliveryHour", "deliveryInterval"]):
            # ERCOT standard format
            print("Detected standard ERCOT tidy format")
            
            # Verify it's actually in tidy format
            required_columns = [
                "deliveryDate",
                "settlementPoint",
                "settlementPointPrice",
                "deliveryHour",
                "deliveryInterval",
            ]
            if not all(col in raw_df.columns for col in required_columns):
                raise ValueError(
                    f"Tidy format requires columns: {required_columns}"
                )

            # Create timestamp column from date, hour, and interval
            raw_df["timestamp"] = pd.to_datetime(
                raw_df["deliveryDate"]
                + " "
                + (raw_df["deliveryHour"] - 1).astype(str)
                + ":"
                + ((raw_df["deliveryInterval"] - 1) * 15).astype(str).str.zfill(2)
            )

            # Filter to max_nodes if specified
            settlement_points = raw_df["settlementPoint"].unique()
            print(f"Found {len(settlement_points)} unique settlement points")
            if len(settlement_points) > max_nodes:
                settlement_points = settlement_points[:max_nodes]
                print(f"Filtering to first {max_nodes} settlement points")
                raw_df = raw_df[raw_df["settlementPoint"].isin(settlement_points)]

            # Drop duplicates if necessary
            dup_count = raw_df.duplicated(
                subset=["timestamp", "settlementPoint"]
            ).sum()
            if dup_count > 0:
                print(
                    f"Found {dup_count} duplicate timestamp-settlementPoint combinations. Dropping duplicates..."
                )
                raw_df = raw_df.drop_duplicates(
                    subset=["timestamp", "settlementPoint"], keep="first"
                )

            # Pivot to wide format
            all_prices = raw_df.pivot(
                index="timestamp",
                columns="settlementPoint",
                values="settlementPointPrice",
            ).reset_index()

            print(
                f"Converted ERCOT standard tidy to wide format with {len(all_prices)} rows and {len(all_prices.columns)} columns"
            )
        else:
            raise ValueError(
                "Tidy format not recognized. Expected one of:\n"
                "1. node, timestamp, price columns (ERCOT simplified), or\n"
                "2. zone, timestamp, price columns (NYISO format), or\n"
                "3. ERCOT standard format: deliveryDate, settlementPoint, settlementPointPrice, deliveryHour, deliveryInterval"
            )
    else:  # Default to wide format
        all_prices = load_prices(prices_path, max_nodes=max_nodes)

    # Get list of nodes if not specified
    if nodes is None:
        nodes = [col for col in all_prices.columns if col != "timestamp"]
        print(f"Found {len(nodes)} nodes after filtering")
    elif isinstance(nodes, str):
        nodes = [nodes]

    # Determine number of processes to use
    if n_jobs is None:
        n_jobs = min(mp.cpu_count(), len(nodes))

    print(f"Using {n_jobs} processes for {len(nodes)} nodes")

    # Run benchmarks in parallel with progress bar
    all_results = {}

    with mp.Pool(processes=n_jobs) as pool:
        process_func = partial(
            process_node,
            all_prices=all_prices,
            start_date=start_date,
            output_dir=output_dir,
        )

        # Using imap to process nodes in order and display progress
        for node, results in tqdm(
            pool.imap(process_func, nodes),
            total=len(nodes),
            desc="Processing nodes",
        ):
            if results:  # Only add if we got valid results
                all_results[node] = results

    # Save combined results
    combined_file = Path(output_dir) / "combined_results.json"
    with open(combined_file, "w") as f:
        # Convert to JSON serializable format with default str conversion for Timestamps
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark battery dispatch models"
    )
    parser.add_argument(
        "--prices",
        default="concatenated_all_data.csv",
        help="Path to the prices CSV file",
    )
    parser.add_argument(
        "--start-date",
        help="Start date for the week (YYYY-MM-DD), defaults to first date in data",
    )
    parser.add_argument(
        "--nodes",
        nargs="+",
        help="Specific nodes to benchmark (default: first N nodes)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of processes to use (default: number of CPU cores)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=100,
        help="Maximum number of settlement points to use",
    )
    parser.add_argument(
        "--data-format",
        choices=["wide", "tidy"],
        default="wide",
        help="Format of the input data (wide or tidy)",
    )

    args = parser.parse_args()

    run_benchmark(
        prices_path=args.prices,
        start_date=args.start_date,
        nodes=args.nodes,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        max_nodes=args.max_nodes,
        data_format=args.data_format,
    )


if __name__ == "__main__":
    main()
