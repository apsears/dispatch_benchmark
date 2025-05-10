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
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Import models
from virtual_energy.models.model_config import BatteryConfig
from virtual_energy.optimisers.oracle_lp import create_and_solve_model as oracle_lp_solve
from virtual_energy.optimisers.online_mpc import run_mpc as online_mpc_run, FORECASTERS as online_mpc_forecasters
from virtual_energy.models.online_quartile import quartile_dispatch
from virtual_energy.utils.ercot_utils import tidy

# Import the new configuration system
from virtual_energy.config import get_battery_config, get_optimisers, get_forecaster_config, get_benchmark_config

# Custom JSON encoder to handle timestamps and other special objects
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle pandas Timestamps and other special objects."""
    def default(self, obj):
        # Convert pandas Timestamp to ISO format string
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        # Convert numpy types
        elif hasattr(obj, 'item'):
            return obj.item()
        # Convert numpy arrays
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        # Let the base class handle other types or raise TypeError
        return super().default(obj)

# Get battery config for multiprocessing from our new config system
DEFAULT_CFG = get_battery_config()


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
        dup_count = raw_df.duplicated(subset=["timestamp", "settlementPoint"]).sum()
        if dup_count > 0:
            print(
                f"Found {dup_count} duplicate timestamp-settlementPoint combinations. Dropping duplicates..."
            )
            raw_df = raw_df.drop_duplicates(
                subset=["timestamp", "settlementPoint"], keep="first"
            )

        # Pivot to wide format
        df = raw_df.pivot(
            index="timestamp", columns="settlementPoint", values="settlementPointPrice"
        ).reset_index()

        print(f"Created wide format with {len(df)} rows and {len(df.columns)} columns")
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


def filter_date_range(df, start_date=None, end_date=None):
    """Filter dataframe to include data within the specified date range."""
    # Process start date
    if start_date is None:
        # Use the first date in the dataset
        start_date = df["timestamp"].min()
    else:
        # Handle YYYY-MM format
        if len(start_date.split("-")) == 2:
            start_date = f"{start_date}-01"  # Add default day
        start_date = pd.to_datetime(start_date)

    # Process end date
    if end_date is None:
        # Use start date + 1 week by default
        end_date = start_date + timedelta(days=7)
    else:
        # Handle YYYY-MM format
        if len(end_date.split("-")) == 2:
            # Add a month to get to the first day of the next month
            year, month = map(int, end_date.split("-"))
            if month == 12:
                end_date = f"{year + 1}-01-01"
            else:
                end_date = f"{year}-{month + 1:02d}-01"
        end_date = pd.to_datetime(end_date)

    # Filter to the date range
    filtered_df = df[(df["timestamp"] >= start_date) & (df["timestamp"] < end_date)]
    
    if len(filtered_df) == 0:
        print(f"Warning: No data found between {start_date} and {end_date}")
        
    return filtered_df


def run_oracle_lp(prices_df, cfg):
    """Run the oracle LP model with perfect knowledge."""
    print("Running Oracle LP model...")
    start_time = time.time()

    _, dispatch = oracle_lp_solve(prices_df, cfg)

    elapsed = time.time() - start_time
    revenue = dispatch["Revenue$"].sum()

    # For JSON serialization, convert to basic types
    dispatch_dict = {
        col: list(dispatch[col]) if isinstance(dispatch[col], pd.Series) else dispatch[col]
        for col in dispatch.columns
    }

    return {
        "model": "oracle_LP",
        "revenue": float(revenue),
        "runtime_seconds": elapsed,
        "dispatch_summary": {
            "timestamps": len(dispatch),
            "mean_price": float(dispatch["SettlementPointPrice"].mean()),
            "mean_revenue": float(dispatch["Revenue$"].mean()),
            "total_mwh_deployed": float(dispatch["MWhDeployed"].sum()),
        }
    }


def run_online_mpc(prices_df, cfg, forecast_model="ridge", horizon=32):
    """Run the online MPC model with the specified forecast model."""
    # Get forecaster config from our configuration system
    forecaster_config = get_forecaster_config(forecast_model)
    
    # Use horizon from config if available
    if "horizon" in forecaster_config:
        horizon = forecaster_config.get("horizon", horizon)
        
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
        "dispatch_summary": {
            "timestamps": len(dispatch),
            "mean_price": float(dispatch["SettlementPointPrice"].mean()),
            "mean_revenue": float(dispatch["Revenue$"].mean()),
            "total_mwh_deployed": float(dispatch["MWhDeployed"].sum()),
        }
    }


def run_online_quartile(prices_df, pct=10, window=672):
    """Run the online quartile model."""
    # Get quartile config from our configuration system
    quartile_config = get_forecaster_config("quartile")
    
    # Use percentiles and window_sizes from config if available
    percentiles = quartile_config.get("percentiles", [pct])
    window_sizes = quartile_config.get("window_sizes", [window])
    
    # If the provided pct is not in the config, use it anyway
    if pct not in percentiles:
        percentiles.append(pct)
        
    # If the provided window is not in the config, use it anyway
    if window not in window_sizes:
        window_sizes.append(window)
        
    # Use the provided pct and window or the first ones from config
    pct = pct if pct else percentiles[0]
    window = window if window else window_sizes[0]
    
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
        "window_size": window,
        "revenue": float(revenue),
        "runtime_seconds": elapsed,
        "dispatch_summary": {
            "timestamps": len(dispatch),
            "mean_price": float(dispatch["SettlementPointPrice"].mean()),
            "mean_revenue": float(dispatch["Revenue$"].mean()),
            "total_mwh_deployed": float(dispatch["MWhDeployed"].sum()),
        }
    }


def save_results(results, output_file):
    """Save results to a JSON file."""
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save results using our custom JSON encoder
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4, cls=CustomJSONEncoder)

    print(f"Results saved to {output_file}")


def process_node(node, df_all, start_date, end_date, output_dir):
    """Process a single node and run all models."""
    # Filter to the specific node
    try:
        node_df = tidy(df_all, node)
    except Exception as e:
        print(f"Error processing node {node}: {str(e)}")
        return None

    # Filter to the date range
    node_df = filter_date_range(node_df, start_date, end_date)

    if len(node_df) == 0:
        print(f"No data found for node {node}")
        return None

    print(f"\nBenchmarking models for node: {node}")
    print(f"Using data from {node_df['timestamp'].min()} to {node_df['timestamp'].max()}")

    # Initialize results
    results = []
    output_file = os.path.join(output_dir, f"{node}_results.json")

    # Get the enabled optimisers from our configuration system
    enabled_optimisers = get_optimisers()

    # Run the oracle LP model if enabled
    if "oracle_lp" in enabled_optimisers:
        try:
            result = run_oracle_lp(node_df, DEFAULT_CFG)
            print(f"Oracle LP: ${result['revenue']:.2f} in {result['runtime_seconds']:.2f}s")

            results.append(result)
            save_results(results, output_file)
        except Exception as e:
            print(f"Error running Oracle LP model: {str(e)}")

    # Run the online MPC model with ridge forecasting if enabled
    if "online_mpc" in enabled_optimisers:
        # Get ridge config from our configuration system
        ridge_config = get_forecaster_config("ridge")
        
        try:
            result = run_online_mpc(node_df, DEFAULT_CFG, forecast_model="ridge", horizon=ridge_config.get("horizon", 32))
            print(f"Online MPC (ridge): ${result['revenue']:.2f} in {result['runtime_seconds']:.2f}s")

            results.append(result)
            save_results(results, output_file)
        except Exception as e:
            print(f"Error running Online MPC model: {str(e)}")

    # Run the online quartile model with various percentiles if enabled
    if "online_quartile" in enabled_optimisers:
        # Get quartile config from our configuration system
        quartile_config = get_forecaster_config("quartile")
        percentiles = quartile_config.get("percentiles", [10, 25, 45])
        window_size = quartile_config.get("window_sizes", [672])[0]
        
        for pct in percentiles:
            try:
                result = run_online_quartile(node_df, pct=pct, window=window_size)
                print(f"Online Quartile (p{pct}): ${result['revenue']:.2f} in {result['runtime_seconds']:.2f}s")

                results.append(result)
                save_results(results, output_file)
            except Exception as e:
                print(f"Error running Online Quartile model (p{pct}): {str(e)}")

    return results


def run_benchmark(
    prices_path,
    start_date=None,
    end_date=None,
    nodes=None,
    output_dir=None,
    n_jobs=None,
    max_nodes=None,
):
    """
    Run benchmark on selected nodes.
    
    Args:
        prices_path: Path to CSV with price data
        start_date: Start date for analysis (YYYY-MM-DD)
        end_date: End date for analysis (YYYY-MM-DD)
        nodes: List of nodes to benchmark
        output_dir: Directory to save results
        n_jobs: Number of processes to use
        max_nodes: Maximum number of nodes to load from prices file
    """
    # Get benchmark config from our configuration system
    benchmark_config = get_benchmark_config()
    
    # Use config values if not provided
    output_dir = output_dir or benchmark_config.get("output_dir", "benchmark_results")
    n_jobs = n_jobs or benchmark_config.get("n_jobs", -1)
    max_nodes = max_nodes or benchmark_config.get("max_nodes", 100)
    
    # Load price data
    df_all = load_prices(prices_path, max_nodes=max_nodes)

    # If no nodes specified, use all columns except timestamp
    if not nodes:
        nodes = [col for col in df_all.columns if col != "timestamp"]
        print(f"No nodes specified. Using all {len(nodes)} nodes.")

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Determine number of processes to use
    if n_jobs is None or n_jobs < 1:
        n_jobs = mp.cpu_count()
    n_jobs = min(n_jobs, len(nodes))
    print(f"Using {n_jobs} processes for {len(nodes)} nodes")
    print(f"Date range: {start_date or 'beginning'} to {end_date or 'one week later'}")

    # Process each node
    if n_jobs == 1:
        # Single process mode
        all_results = {}
        for node in tqdm(nodes, desc="Processing nodes"):
            all_results[node] = process_node(node, df_all, start_date, end_date, output_dir)
    else:
        # Multiprocessing mode
        process_partial = partial(
            process_node, df_all=df_all, start_date=start_date, end_date=end_date, output_dir=output_dir
        )
        with mp.Pool(processes=n_jobs) as pool:
            all_results = dict(
                zip(nodes, pool.map(process_partial, nodes))
            )

    # Save combined results
    combined_file = os.path.join(output_dir, "combined_results.json")
    save_results(all_results, combined_file)


def main():
    """Parse command-line arguments and run the benchmark."""
    # Get benchmark config from our configuration system
    benchmark_config = get_benchmark_config()
    
    parser = argparse.ArgumentParser(
        description="Benchmark battery dispatch strategies on historical ERCOT price data"
    )
    parser.add_argument(
        "--prices",
        default="concatenated_all_data.csv",
        help="Path to the prices CSV file",
    )
    parser.add_argument(
        "--nodes",
        help="Comma-separated list of nodes to benchmark (e.g., 'HB_HOUSTON,HB_NORTH')",
    )
    parser.add_argument(
        "--nodes-file",
        help="Path to a file containing nodes to benchmark (one per line)",
    )
    parser.add_argument(
        "--start",
        help="Start date for backtesting (YYYY-MM or YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        help="End date for backtesting (YYYY-MM or YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        default=benchmark_config.get("output_dir", "benchmark_results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=benchmark_config.get("n_jobs", -1),
        help="Number of processes to use (default: from config or all cores)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=benchmark_config.get("max_nodes", 100),
        help="Maximum number of settlement points to use",
    )
    args = parser.parse_args()

    # Process nodes from CLI or file
    nodes = []
    if args.nodes:
        nodes = args.nodes.split(",")
    elif args.nodes_file:
        with open(args.nodes_file, "r") as f:
            nodes = [line.strip() for line in f if line.strip()]

    # Run the benchmark
    run_benchmark(
        prices_path=args.prices,
        start_date=args.start,
        end_date=args.end,
        nodes=nodes,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        max_nodes=args.max_nodes,
    )


if __name__ == "__main__":
    main()
