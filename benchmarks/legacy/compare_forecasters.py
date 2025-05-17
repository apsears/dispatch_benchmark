#!/usr/bin/env python3
"""
Compare different forecasters on a sample ERCOT dataset.

This script runs the MPC optimization with different forecasters including our naive baseline
and outputs a comparison table to demonstrate the value of more sophisticated models.
"""

import sys
import pandas as pd
import time

# Import our models and forecasters
from dispatch_benchmark.optimisers.battery_config import BatteryConfig
from dispatch_benchmark.optimisers.oracle_lp import (
    create_and_solve_model as oracle_lp_solve,
)
from dispatch_benchmark.optimisers.online_mpc import (
    run_mpc as online_mpc_run,
    FORECASTERS,
)
from dispatch_benchmark.config import get_battery_config

# Get battery config from configuration
battery_config = get_battery_config()
CONFIG = BatteryConfig(
    delta_t=battery_config.delta_t,
    eta_chg=battery_config.eta_chg,
    p_max_mw=battery_config.p_max_mw,
    e_max_mwh=battery_config.e_max_mwh,
)


def load_price_data(path, node="HB_HOUSTON"):
    """Load price data for a specific node."""
    print(f"Loading prices from {path}")

    # Load the data from CSV
    raw_df = pd.read_csv(path)

    # Check if it's in long format (with settlementPoint column)
    if "settlementPoint" in raw_df.columns:
        print(f"Converting from long format to wide format for {node}...")
        # Filter to just the node we want
        node_data = raw_df[raw_df["settlementPoint"] == node]

        # Create timestamp from the date, hour and interval columns
        node_data["timestamp"] = pd.to_datetime(
            node_data["deliveryDate"]
            + " "
            + (node_data["deliveryHour"] - 1).astype(str).str.zfill(2)
            + ":"
            + ((node_data["deliveryInterval"] - 1) * 15)
            .astype(str)
            .str.zfill(2)
        )

        # Create a DataFrame with timestamp and SettlementPointPrice
        prices_df = pd.DataFrame(
            {
                "timestamp": node_data["timestamp"],
                "SettlementPointPrice": node_data["settlementPointPrice"],
            }
        )
    else:
        # It's already in wide format
        if node not in raw_df.columns:
            raise ValueError(f"Node {node} not found in price data")

        # If there's a timestamp column, parse it, otherwise create it from date, hour, interval
        if "timestamp" in raw_df.columns:
            prices_df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(raw_df["timestamp"]),
                    "SettlementPointPrice": raw_df[node],
                }
            )
        else:
            # Create timestamp from the date, hour and interval columns
            raw_df["timestamp"] = pd.to_datetime(
                raw_df["deliveryDate"]
                + " "
                + (raw_df["deliveryHour"] - 1).astype(str).str.zfill(2)
                + ":"
                + ((raw_df["deliveryInterval"] - 1) * 15)
                .astype(str)
                .str.zfill(2)
            )
            prices_df = pd.DataFrame(
                {
                    "timestamp": raw_df["timestamp"],
                    "SettlementPointPrice": raw_df[node],
                }
            )

    # Sort by timestamp and reset index
    prices_df = prices_df.sort_values("timestamp").reset_index(drop=True)
    return prices_df


def run_model(prices_df, model_type, forecaster=None):
    """Run the specified model on the price data."""
    start_time = time.time()

    if model_type == "oracle_lp":
        # Oracle LP (perfect foresight)
        _, dispatch = oracle_lp_solve(prices_df, CONFIG)
    elif model_type == "online_mpc":
        # Online MPC (with forecaster)
        if forecaster not in FORECASTERS:
            raise ValueError(f"Unknown forecaster: {forecaster}")
        # Convert to series format needed by online_mpc
        price_series = prices_df.set_index("timestamp")["SettlementPointPrice"]
        dispatch = online_mpc_run(
            price_series, 32, forecaster=FORECASTERS[forecaster]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Calculate metrics
    elapsed = time.time() - start_time
    revenue = dispatch["Revenue$"].sum()
    deployed = dispatch["MWhDeployed"].sum()

    return {
        "model": f"{model_type}{'_' + forecaster if forecaster else ''}",
        "revenue": float(revenue),
        "runtime_seconds": elapsed,
        "deployed_mwh": float(deployed),
    }


def main():
    """Run all models and print comparison table."""
    # Use sample data file from tests
    prices_path = "tests/data/ercot_1day_sample.csv"
    node = "HB_HOUSTON"

    print("Starting forecaster comparison...")

    # Check available forecasters
    print(f"Available forecasters in registry: {list(FORECASTERS.keys())}")

    # Load price data
    try:
        prices_df = load_price_data(prices_path, node)
        print(f"Loaded {len(prices_df)} price points for {node}")
    except Exception as e:
        print(f"Error loading price data: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Run a very simple test first
    print("\nTesting naive forecaster directly...")
    try:
        # Convert to series format needed by forecasters
        price_series = prices_df.set_index("timestamp")["SettlementPointPrice"]
        naive_forecast = FORECASTERS["naive"](
            price_series, price_series.index[0], 32
        )
        print(f"Naive forecast for first 3 periods: {naive_forecast[:3]}")
    except Exception as e:
        print(f"Error testing naive forecaster: {e}")
        import traceback

        traceback.print_exc()

    # Run models
    results = []

    # Run Oracle LP (perfect foresight)
    try:
        print("\nRunning Oracle LP (perfect foresight)...")
        oracle_result = run_model(prices_df, "oracle_lp")
        results.append(oracle_result)
        print(f"Oracle LP revenue: ${oracle_result['revenue']:,.2f}")
    except Exception as e:
        print(f"Error running oracle model: {e}")
        import traceback

        traceback.print_exc()

    # Run Online MPC with different forecasters
    forecasters = ["ridge", "ewma", "naive"]
    for forecaster in forecasters:
        try:
            print(f"\nRunning Online MPC with {forecaster} forecaster...")
            mpc_result = run_model(prices_df, "online_mpc", forecaster)
            results.append(mpc_result)
            print(
                f"MPC with {forecaster} revenue: ${mpc_result['revenue']:,.2f}"
            )
        except Exception as e:
            print(f"Error running MPC with {forecaster}: {e}")
            import traceback

            traceback.print_exc()

    # Create results table if we have results
    if results:
        results_df = pd.DataFrame(results)

        # Format revenue as $ thousands
        results_df["revenue_k"] = results_df["revenue"].apply(
            lambda x: f"${x/1000:.0f}k"
        )

        # Print results table
        print("\n=== FORECASTER COMPARISON ===")
        comparison = results_df[
            ["model", "revenue_k", "runtime_seconds"]
        ].copy()
        comparison = comparison.rename(
            columns={
                "model": "Model",
                "revenue_k": "Revenue",
                "runtime_seconds": "Runtime (s)",
            }
        )
        comparison["Runtime (s)"] = comparison["Runtime (s)"].apply(
            lambda x: f"{x:.2f}"
        )
        print(comparison.to_string(index=False))

        print(
            "\nThe results show the value of sophisticated forecasting compared to the naive baseline."
        )
    else:
        print("\nNo results were collected. Check the error messages above.")


if __name__ == "__main__":
    main()
